from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from icalens import ICALens
from icalens.profiling import _final_norm


class TinyTokenizer:
    def convert_ids_to_tokens(self, token_id: int) -> str:
        return f"T{token_id}"

    def decode(self, token_ids: list[int]) -> str:
        return f" token-{token_ids[0]}"


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.norm = torch.nn.Identity()
        self.head = torch.nn.Linear(2, 4, bias=False)

    def get_output_embeddings(self) -> torch.nn.Module:
        return self.head


def test_final_norm_supports_gpt_neox_layout() -> None:
    model = torch.nn.Module()
    model.gpt_neox = torch.nn.Module()
    model.gpt_neox.final_layer_norm = torch.nn.LayerNorm(2)

    assert _final_norm(model) is model.gpt_neox.final_layer_norm


def test_profiles_and_round_trips_component_metadata(tmp_path, monkeypatch) -> None:
    signals = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.2], [0.1, -1.0]], dtype=np.float32)
    lens = ICALens(model_id="example/model", model_revision="revision").fit(
        signals, layer=1, n_components=2, max_iter=2
    )
    lens._analysis_model = TinyModel()
    lens._analysis_tokenizer = TinyTokenizer()
    result = SimpleNamespace(
        scores=torch.tensor([[2.0, -1.0], [-3.0, 0.5]]),
        energy=torch.tensor([[0.8, 0.2], [0.97, 0.03]]),
        tokens=("A", "B"),
        token_texts=(" alpha", " beta"),
        token_ids=torch.tensor([10, 11]),
        positions=torch.tensor([4, 5]),
    )
    monkeypatch.setattr(lens, "analyze", lambda *args, **kwargs: result)

    profile = lens.profile_components(["example"], layer=1, min_energy=0.1, top_k_examples=2)

    assert profile["n_tokens"] == 2
    assert profile["components"][0]["sign_statistics"]["positive_fraction"] == 0.5
    assert profile["components"][0]["dominant_sign"] == "negative"
    assert profile["components"][0]["score_statistics"]["excess_kurtosis"] == pytest.approx(-2.0)
    assert profile["components"][0]["score_statistics"]["excess_kurtosis_rank"] == 1
    assert profile["components"][1]["score_statistics"]["excess_kurtosis_rank"] == 2
    assert profile["components"][0]["examples"]["positive"]["tokens"] == [
        {"text": " alpha", "count": 1}
    ]
    artifact = lens.save(tmp_path / "profiled")
    assert (artifact / "component_profiles/resid_post/layer_01.json.gz").is_file()
    lens.checkpoint_component_profile(artifact, layer=1)

    loaded = ICALens.from_pretrained(artifact)
    component = loaded.component_profile(layer=1, component=0)
    assert component["dominant_sign"] == "negative"
    assert component["logit_lens"]["method"] == "final_norm_then_unembed"
    assert component.component == 0
    assert component.layer == 1
    rendered = component._repr_html_()
    assert rendered.startswith('<iframe title="ICA Lens Component Profile"')
    assert "Component profile — C0 · layer 1 · dominant negative" in rendered
    assert "Logit-lens tokens · negative" in rendered
    assert "Excess kurtosis" in rendered
    report = component.to_html(tmp_path / "component-profile.html")
    assert report.is_file()
    assert "High-energy occurrences · negative" in report.read_text()
    assert profile["format_version"] == 1


def test_profiles_from_cached_activations_without_analyze(monkeypatch) -> None:
    signals = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.2], [0.1, -1.0]], dtype=np.float32)
    lens = ICALens(model_id="example/model", model_revision="revision").fit(
        signals, layer=1, n_components=2, max_iter=2
    )
    lens._analysis_model = TinyModel()
    lens._analysis_tokenizer = TinyTokenizer()
    lens._analysis_device = "cpu"
    monkeypatch.setattr(
        lens,
        "analyze",
        lambda *args, **kwargs: pytest.fail("cached profiling must not call analyze"),
    )
    records = [
        {
            "token": f"T{index}",
            "text": f" token-{index}",
            "token_id": index,
            "position": index,
            "context": "cached context",
            "source_index": index // 2,
        }
        for index in range(4)
    ]

    profile = lens.profile_components_from_activations(
        torch.from_numpy(signals),
        records,
        layer=1,
        batch_size=2,
        min_energy=0.0,
        device="cpu",
    )

    assert profile["n_tokens"] == 4
    assert profile["n_inputs"] == 2
    assert "excess_kurtosis" in profile["components"][0]["score_statistics"]
    assert profile["components"][0]["examples"]["positive"]["occurrences"]


def test_add_r_lens_profile_preserves_existing_information(monkeypatch) -> None:
    signals = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.2], [0.1, -1.0]], dtype=np.float32)
    lens = ICALens(model_id="example/model", model_revision="revision").fit(
        signals, layer=1, n_components=2, max_iter=2
    )
    lens._analysis_model = TinyModel()
    lens._analysis_tokenizer = TinyTokenizer()
    lens._analysis_device = "cpu"
    result = SimpleNamespace(
        scores=torch.tensor([[2.0, -1.0], [-3.0, 0.5]]),
        energy=torch.tensor([[0.8, 0.2], [0.97, 0.03]]),
        tokens=("A", "B"),
        token_texts=(" alpha", " beta"),
        token_ids=torch.tensor([10, 11]),
        positions=torch.tensor([4, 5]),
    )
    monkeypatch.setattr(lens, "analyze", lambda *args, **kwargs: result)
    original = lens.profile_components(["example"], layer=1, min_energy=0.1, top_k_examples=2)
    original_examples = original["components"][0]["examples"]

    enriched = lens.add_r_lens_profile(
        layer=1,
        r_lens={
            "J": {1: torch.eye(2)},
            "d_model": 2,
            "provenance": {
                "model_id": "example/model",
                "model_revision": "revision",
            },
        },
        top_k=2,
        batch_size=1,
        device="cpu",
    )

    assert enriched["components"][0]["examples"] == original_examples
    assert enriched["components"][0]["r_lens"]["method"] == ("relp_then_final_norm_then_unembed")
    assert len(enriched["components"][0]["r_lens"]["dominant"]["top_tokens"]) == 2
    assert enriched["selection"]["r_lens_top_k"] == 2


def test_add_r_lens_profile_records_explicit_base_to_instruct_transfer(
    monkeypatch,
) -> None:
    signals = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.2], [0.1, -1.0]], dtype=np.float32)
    lens = ICALens(
        model_id="example/model-instruct",
        model_revision="instruct-revision",
        model_type="instruct",
    ).fit(signals, layer=1, n_components=2, max_iter=2)
    lens._analysis_model = TinyModel()
    lens._analysis_tokenizer = TinyTokenizer()
    lens._analysis_device = "cpu"
    result = SimpleNamespace(
        scores=torch.tensor([[2.0, -1.0], [-3.0, 0.5]]),
        energy=torch.tensor([[0.8, 0.2], [0.97, 0.03]]),
        tokens=("A", "B"),
        token_texts=(" alpha", " beta"),
        token_ids=torch.tensor([10, 11]),
        positions=torch.tensor([4, 5]),
    )
    monkeypatch.setattr(lens, "analyze", lambda *args, **kwargs: result)
    lens.profile_components(["example"], layer=1, min_energy=0.1, top_k_examples=2)
    r_lens = {
        "J": {1: torch.eye(2)},
        "d_model": 2,
        "provenance": {
            "model_id": "example/model-base",
            "model_revision": "base-revision",
        },
    }

    with pytest.raises(ValueError, match="allow_base_model_transfer"):
        lens.add_r_lens_profile(layer=1, r_lens=r_lens, device="cpu")

    directly_profiled = lens.profile_components(
        ["example"],
        layer=1,
        min_energy=0.1,
        top_k_examples=2,
        r_lens=r_lens,
        r_lens_top_k=2,
        device="cpu",
        allow_base_model_transfer=True,
    )
    assert directly_profiled["r_lens_provenance"]["transfer"]["kind"] == ("base_to_instruct")

    enriched = lens.add_r_lens_profile(
        layer=1,
        r_lens=r_lens,
        allow_base_model_transfer=True,
        top_k=2,
        device="cpu",
    )
    provenance = enriched["r_lens_provenance"]
    assert provenance["model_id"] == "example/model-base"
    assert provenance["transfer"]["kind"] == "base_to_instruct"
    assert provenance["transfer"]["target_model_id"] == "example/model-instruct"
