#!/usr/bin/env python3
"""Controlled duplication test for Qwen layer-16 neuron 11960."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from gb10_load_llm import load_model_to_cuda
from transformers import AutoModelForCausalLM, AutoTokenizer

import run as experiment
from icalens._activation_dataset import ActivationDataset
from icalens.experiments._run import atomic_write_json

HERE = Path(__file__).resolve().parent
DEFAULT_RUN = HERE / "runs/main"
DEFAULT_OUTPUT = HERE / "diagnostics/duplication-n11960.json"
NEURON = 11960

CONDITIONS = {
    "original_repeated_group2": """result = [
   {groups: []}, //0
   {groups: [\"group1\"]}, //1
   {groups: [\"group2\"]}, //2
   {groups: [\"group1\",\"group2\"]}, //3
]""",
    "target_without_prior_group2": """result = [
   {groups: []}, //0
   {groups: [\"group1\"]}, //1
   {groups: [\"groupX\"]}, //2
   {groups: [\"group1\",\"group2\"]}, //3
]""",
    "repeat_group3_instead": """result = [
   {groups: []}, //0
   {groups: [\"group1\"]}, //1
   {groups: [\"group3\"]}, //2
   {groups: [\"group1\",\"group3\"]}, //3
]""",
    "group2_repeated_within_target": """result = [
   {groups: []}, //0
   {groups: [\"group1\"]}, //1
   {groups: [\"groupX\"]}, //2
   {groups: [\"group2\",\"group2\"]}, //3
]""",
    "original_target_line_alone": """result = [
   {groups: [\"group1\",\"group2\"]}, //3
]""",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.force:
        raise FileExistsError(f"output exists; pass --force to replace it: {output}")
    run_root = args.run.expanduser().resolve()
    run = __import__("json").loads((run_root / "run.json").read_text(encoding="utf-8"))
    resolved = run["resolved"]
    dataset = ActivationDataset(Path(resolved["activation_dataset"]))
    tokenizer = AutoTokenizer.from_pretrained(
        dataset.model["repo_id"], revision=dataset.model.get("revision"), trust_remote_code=True
    )
    prefix_id = int(resolved["document_framing"]["token_id"])
    documents: list[Any] = []
    selected: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    next_row = 0
    for document_index, text in enumerate(CONDITIONS.values()):
        content = tokenizer(text, add_special_tokens=False)["input_ids"]
        ids = torch.tensor([prefix_id, *content], dtype=torch.long)
        documents.append(SimpleNamespace(input_ids=ids))
        positions = torch.arange(len(ids), dtype=torch.long)
        rows = torch.arange(next_row, next_row + len(ids), dtype=torch.long)
        next_row += len(ids)
        selected[document_index] = (positions, rows)
    model = load_model_to_cuda(
        AutoModelForCausalLM,
        dataset.model["repo_id"],
        revision=dataset.model.get("revision"),
        device=args.device,
        dtype=torch.bfloat16,
        touch="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    projection = experiment.down_projection(model, int(resolved["layer"]))
    values, rows = experiment.capture_coefficients(
        model,
        projection,
        documents,
        list(range(len(documents))),
        selected,
        pad_token_id=prefix_id,
    )
    if not torch.equal(rows, torch.arange(next_row)):
        raise ValueError("captured rows are not in the expected order")
    scores = values[:, NEURON].float().cpu()
    results = []
    offset = 0
    for (name, text), document in zip(CONDITIONS.items(), documents, strict=True):
        count = len(document.input_ids)
        ids = document.input_ids.tolist()
        token_rows = [
            {
                "position": position,
                "token_id": token_id,
                "text": tokenizer.decode([token_id], clean_up_tokenization_spaces=False),
                "score": float(scores[offset + position]),
            }
            for position, token_id in enumerate(ids)
        ]
        # The final object is the controlled target in every condition. Retain
        # its last 12 tokens, including the closing sequence and line comment.
        results.append(
            {
                "condition": name,
                "prompt": text,
                "target_suffix_tokens": token_rows[-12:],
                "minimum_suffix_score": min(row["score"] for row in token_rows[-12:]),
            }
        )
        offset += count
    atomic_write_json(
        output,
        {
            "model": dataset.model,
            "layer": int(resolved["layer"]),
            "neuron": NEURON,
            "document_framing": resolved["document_framing"],
            "results": results,
        },
    )
    for result in results:
        print(f"\n{result['condition']}")
        for token in result["target_suffix_tokens"]:
            print(f"  {token['score']:+8.4f}  {token['text']!r}")
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
