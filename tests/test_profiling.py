from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from icalens import ICALens


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


def test_profiles_and_round_trips_component_metadata(tmp_path, monkeypatch) -> None:
    signals = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.2], [0.1, -1.0]], dtype=np.float32
    )
    lens = ICALens(model_id="example/model").fit(
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

    profile = lens.profile_components(
        ["example"], layer=1, min_energy=0.1, top_k_examples=2
    )

    assert profile["n_tokens"] == 2
    assert profile["components"][0]["sign_statistics"]["positive_fraction"] == 0.5
    assert profile["components"][0]["dominant_sign"] == "negative"
    assert profile["components"][0]["examples"]["positive"]["tokens"] == [
        {"text": " alpha", "count": 1}
    ]
    artifact = lens.save(tmp_path / "profiled")
    assert (artifact / "component_profiles/resid_post/layer_01.json").is_file()

    loaded = ICALens.from_pretrained(artifact)
    component = loaded.component_profile(layer=1, component=0)
    assert component["dominant_sign"] == "negative"
    assert component["logit_lens"]["method"] == "final_norm_then_unembed"
