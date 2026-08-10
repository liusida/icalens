"""Apply a saved ICA Lens to fresh GPT-2 text."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from gb10_load_llm import load_model_to_cuda
from html_explorer import write_explorer_html
from transformers import AutoModelForCausalLM, AutoTokenizer

from icalens import ICALens

DEFAULT_LENS = Path(__file__).parent / "output" / "icalens-gpt2-small"
DEFAULT_TEXT = "She deposited the check at the bank before walking along the river bank."
DEFAULT_OUTPUT_FILE = Path(__file__).parent / "output" / "apply.html"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lens", type=Path, default=DEFAULT_LENS)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--metric",
        choices=("score", "energy"),
        default="score",
        help="Rank and display signed ICA scores or per-token energy shares.",
    )
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    return parser.parse_args()


def format_value(value: float, metric: str) -> str:
    return f"{value:+.3f}" if metric == "score" else f"{value:.2%}"


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
        lens.model_id,
        revision=lens.model_revision,
    )
    model = load_model_to_cuda(
        AutoModelForCausalLM,
        lens.model_id,
        revision=lens.model_revision,
        device="cuda",
        dtype=torch.bfloat16,
        touch="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()

    result = lens.analyze(
        args.text,
        layer=args.layer,
        model=model,
        tokenizer=tokenizer,
        context_length=tokenizer.model_max_length,
    )
    values = result.scores if args.metric == "score" else result.energy
    ranking_values = values.abs() if args.metric == "score" else values
    top_k = min(args.top_k, values.shape[-1])
    top_indices = torch.topk(ranking_values, k=top_k, dim=-1).indices
    token_ids = result.token_ids.tolist()
    tokens = result.tokens

    print(f"Lens: {args.lens}")
    print(f"Model: {lens.model_id}@{lens.model_revision} ({lens.model_type})")
    print(f"Layer: {args.layer}")
    print(f"Metric: {args.metric}")
    print()
    for position, token in enumerate(tokens):
        entries = [
            f"C{component.item()}={format_value(values[position, component].item(), args.metric)}"
            for component in top_indices[position]
        ]
        print(f"{position:>3} {token!r:<18} {'  '.join(entries)}")

    html_tokens = [
        {
            "position": position,
            "token": token,
            "token_text": tokenizer.decode([token_ids[position]]),
            "top": [
                {
                    "component": int(component),
                    "score": float(values[position, component]),
                }
                for component in top_indices[position]
            ],
        }
        for position, token in enumerate(tokens)
    ]
    output_file = write_explorer_html(
        args.output_file,
        title="ICA Lens Text Explorer",
        model=f"{lens.model_id}@{lens.model_revision}",
        layer=args.layer,
        input_text=args.text,
        token_scope="all text tokens",
        tokens=html_tokens,
        metric=args.metric,
        result_group_title="Text",
    )
    print()
    print(f"HTML explorer: {output_file}")


if __name__ == "__main__":
    main()
