"""Compute Layer-20 SAE/ICA steering-vector cosines and update RESULTS.md."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download

from icalens import ICALens
from icalens.experiments._saebench_worker import SAEFeatureEncoder

ROOT = Path(__file__).parent
DEFAULT_RUN = ROOT / "runs" / "all-methods-all-languages-layers19-21-current-position"
DEFAULT_LENS = Path("local-icalens-models/official/icalens-gemma-2-2b-pile10k")
SELECTED = {
    "chinese": {"feature": 13458, "component": 105},
    "french": {"feature": 13692, "component": 25},
    "japanese": {"feature": 3953, "component": 145},
    "spanish": {"feature": 3375, "component": 42},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--lens", default=str(DEFAULT_LENS))
    parser.add_argument("--results", type=Path, default=ROOT / "RESULTS.md")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print computed values without changing RESULTS.md.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lens = ICALens.from_pretrained(args.lens)
    artifact = lens._get_layer(20)
    if artifact.writing_matrix is None:
        raise ValueError("Layer 20 has no ICA writing matrix")

    payloads = {
        language: _load_pair(args.run, language, expected=selected)
        for language, selected in SELECTED.items()
    }
    sae_config = dict(payloads["chinese"][0]["protocol"]["sae"])
    for sae_payload, _ in payloads.values():
        if sae_payload["protocol"]["sae"]["checkpoint"] != sae_config["checkpoint"]:
            raise ValueError("selected SAE results use different checkpoints")
    sae_config["weights_file"] = _resolve_weights(sae_config)
    encoder = SAEFeatureEncoder(
        {
            "baselines": {"sae": sae_config},
            "hidden_size": artifact.writing_matrix.shape[0],
            "layer": 20,
            "saebench_model_name": lens.model_id,
        },
        device="cpu",
        dtype=torch.float32,
    )

    values: dict[str, float] = {}
    for language, selected in SELECTED.items():
        sae_payload, ica_payload = payloads[language]
        feature = selected["feature"]
        component = selected["component"]
        sae_offset = _candidate_offset(sae_payload, "feature", feature)
        ica_offset = _candidate_offset(ica_payload, "component", component)
        sae_vector = sae_offset * encoder.W_dec[feature].detach().cpu()
        ica_vector = ica_offset * torch.from_numpy(artifact.writing_matrix[:, component]).to(
            torch.float32
        )
        values[language] = float(
            torch.dot(sae_vector, ica_vector)
            / (torch.linalg.vector_norm(sae_vector) * torch.linalg.vector_norm(ica_vector))
        )

    for language, value in values.items():
        print(f"{language}: {value:.4f}")
    if args.dry_run:
        return

    document = args.results.read_text(encoding="utf-8")
    for language, value in values.items():
        pattern = re.compile(rf"<!-- cosine:{language} -->.*?<!-- /cosine:{language} -->")
        replacement = f"<!-- cosine:{language} -->{value:.4f}<!-- /cosine:{language} -->"
        document, count = pattern.subn(replacement, document)
        if count != 1:
            raise ValueError(f"expected one cosine marker for {language}, found {count}")
    args.results.write_text(document, encoding="utf-8")
    print(f"Updated {args.results}")


def _load_pair(
    run: Path, language: str, *, expected: dict[str, int]
) -> tuple[dict[str, Any], dict[str, Any]]:
    results = run / "results"
    sae = json.loads(
        (results / f"sae-english-to-{language}-layer20.json").read_text(encoding="utf-8")
    )
    ica = json.loads(
        (results / f"ica-english-to-{language}-layer20.json").read_text(encoding="utf-8")
    )
    _candidate_offset(sae, "feature", expected["feature"])
    _candidate_offset(ica, "component", expected["component"])
    return sae, ica


def _candidate_offset(payload: dict[str, Any], key: str, identifier: int) -> float:
    matches = [row for row in payload["candidates"] if row[key] == identifier]
    if len(matches) != 1:
        raise ValueError(f"expected one {key} {identifier}, found {len(matches)}")
    return float(matches[0]["offset"])


def _resolve_weights(sae: dict[str, Any]) -> str:
    recorded = Path(str(sae["weights_file"]))
    if recorded.is_file():
        return str(recorded)
    return hf_hub_download(
        repo_id=str(sae["repo_id"]),
        filename=str(sae["checkpoint"]),
        revision=str(sae["revision"]),
    )


if __name__ == "__main__":
    main()
