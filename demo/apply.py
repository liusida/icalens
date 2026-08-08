"""Apply a saved ICA Lens to fresh GPT-2 text."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from gb10_load_llm import load_model_to_cuda
from transformers import AutoModelForCausalLM, AutoTokenizer

from icalens import ICALens

DEFAULT_LENS = Path(__file__).parent / "output" / "icalens-gpt2-small"
DEFAULT_TEXT = "She deposited the check at the bank before walking along the river bank."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lens", type=Path, default=DEFAULT_LENS)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This GB10 demo requires a CUDA device.")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")

    lens = ICALens.from_pretrained(args.lens)
    if args.layer not in lens.available_layers:
        raise ValueError(
            f"layer {args.layer} is not in this lens; available layers: {lens.available_layers}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        lens.base_model,
        revision=lens.base_model_revision,
    )
    model = load_model_to_cuda(
        AutoModelForCausalLM,
        lens.base_model,
        revision=lens.base_model_revision,
        device="cuda",
        dtype=torch.bfloat16,
        touch="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()

    encoded = tokenizer(
        args.text,
        return_tensors="pt",
        truncation=True,
        max_length=tokenizer.model_max_length,
    )
    input_ids = encoded["input_ids"].to("cuda")
    attention_mask = encoded["attention_mask"].to("cuda")
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
    if outputs.hidden_states is None:
        raise RuntimeError("GPT-2 did not return hidden states.")

    activations = outputs.hidden_states[args.layer + 1][0].to(dtype=torch.float32)
    scores = lens.transform(activations, layer=args.layer)
    top_k = min(args.top_k, scores.shape[-1])
    top_indices = torch.topk(scores.abs(), k=top_k, dim=-1).indices
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    print(f"Lens: {args.lens}")
    print(f"Base model: {lens.base_model}@{lens.base_model_revision}")
    print(f"Layer: {args.layer}")
    print()
    for position, token in enumerate(tokens):
        entries = [
            f"C{component.item()}={scores[position, component].item():+.3f}"
            for component in top_indices[position]
        ]
        print(f"{position:>3} {token!r:<18} {'  '.join(entries)}")


if __name__ == "__main__":
    main()
