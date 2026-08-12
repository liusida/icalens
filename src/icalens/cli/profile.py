"""Profile fitted ICA components from a streamed Hugging Face dataset."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from datasets import load_dataset  # type: ignore[import-untyped]
from huggingface_hub import HfApi

from icalens import ICALens


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="icalens profile", description=__doc__)
    parser.add_argument("--lens", required=True, help="Local lens directory or Hub repository.")
    layer_group = parser.add_mutually_exclusive_group(required=True)
    layer_group.add_argument("--layer", type=int, help="One fitted layer (compatibility form).")
    layer_group.add_argument(
        "--layers", help="Comma-separated fitted layers, or 'all'."
    )
    parser.add_argument("--dataset", required=True, help="Hugging Face dataset repository.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--messages-field", default="messages")
    parser.add_argument(
        "--input-type", choices=("auto", "text", "chat"), default="auto"
    )
    parser.add_argument(
        "--token-scope", choices=("assistant", "user", "content", "all"), default="all"
    )
    parser.add_argument("--max-tokens", type=int, default=100_000)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--top-k-examples", type=int, default=20)
    parser.add_argument("--min-energy", type=float, default=0.05)
    parser.add_argument("--logit-lens-top-k", type=int, default=20)
    parser.add_argument("--logit-lens-batch-size", type=int, default=64)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    lens = ICALens.from_pretrained(args.lens)
    input_type = (
        "chat" if lens.model_type == "instruct" else "text"
    ) if args.input_type == "auto" else args.input_type
    revision = HfApi().dataset_info(args.dataset).sha
    if revision is None:
        raise RuntimeError("could not resolve the dataset revision")
    layers = _parse_layers(args.layer, args.layers, lens.available_layers)
    for layer in layers:
        dataset = load_dataset(
            args.dataset, split=args.split, revision=revision, streaming=True
        )
        inputs = _dataset_inputs(
            dataset,
            input_type=input_type,
            text_field=args.text_field,
            messages_field=args.messages_field,
        )
        profile = lens.profile_components(
            inputs,
            layer=layer,
            token_scope=args.token_scope,
            max_tokens=args.max_tokens,
            top_k_examples=args.top_k_examples,
            min_energy=args.min_energy,
            logit_lens_top_k=args.logit_lens_top_k,
            logit_lens_batch_size=args.logit_lens_batch_size,
            context_length=args.context_length,
            provenance={
                "dataset": {
                    "repo_id": args.dataset,
                    "revision": str(revision),
                    "split": args.split,
                },
                "input_type": input_type,
                "token_scope": args.token_scope,
                "text_field": args.text_field if input_type == "text" else None,
                "messages_field": args.messages_field if input_type == "chat" else None,
            },
            device=args.device,
            progress=not args.no_progress,
        )
        if (args.output.expanduser() / "icalens.json").is_file():
            output = lens.checkpoint_component_profile(args.output, layer=layer)
        else:
            output = lens.save(args.output)
        print(
            f"Profiled layer {layer}: {len(profile['components'])} components from "
            f"{profile['n_tokens']} tokens."
        )
        print(f"Checkpointed profiled lens to {output}")


def _dataset_inputs(
    dataset: Iterable[dict[str, Any]],
    *,
    input_type: str,
    text_field: str,
    messages_field: str,
) -> Iterable[str | list[dict[str, str]]]:
    field = text_field if input_type == "text" else messages_field
    for row in dataset:
        if field not in row:
            raise KeyError(f"dataset row does not contain field {field!r}")
        value = row[field]
        if input_type == "text":
            if isinstance(value, str) and value:
                yield value
        elif isinstance(value, list) and value:
            yield value


def _parse_layers(
    layer: int | None, layers: str | None, available: tuple[int, ...]
) -> tuple[int, ...]:
    requested: tuple[int, ...]
    if layer is not None:
        requested = (layer,)
    elif layers == "all":
        requested = available
    else:
        assert layers is not None
        try:
            requested = tuple(int(value.strip()) for value in layers.split(","))
        except ValueError as error:
            raise ValueError("--layers must be comma-separated integers or 'all'") from error
    if not requested:
        raise ValueError("no layers were requested")
    unavailable = sorted(set(requested) - set(available))
    if unavailable:
        raise ValueError(
            f"unavailable layers: {unavailable}; available layers: {list(available)}"
        )
    return requested
