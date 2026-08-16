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
    parser.add_argument(
        "operation",
        nargs="?",
        choices=("add-r-lens",),
        help=(
            "Optional profile operation. Use 'add-r-lens' to enrich existing "
            "profiles without replaying the dataset."
        ),
    )
    parser.add_argument("--lens", required=True, help="Local lens directory or Hub repository.")
    layer_group = parser.add_mutually_exclusive_group(required=True)
    layer_group.add_argument("--layer", type=int, help="One fitted layer (compatibility form).")
    layer_group.add_argument(
        "--layers", help="Comma-separated fitted layers, or 'all'."
    )
    parser.add_argument(
        "--dataset",
        help="Hugging Face dataset repository (not used by 'add-r-lens').",
    )
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
    parser.add_argument(
        "--r-lens",
        type=Path,
        help="Compatible local R-lens artifact to add R-lens vocabulary readouts.",
    )
    parser.add_argument("--r-lens-top-k", type=int, default=20)
    parser.add_argument("--r-lens-batch-size", type=int, default=8)
    parser.add_argument(
        "--allow-base-model-transfer",
        action="store_true",
        help=(
            "Explicitly allow a dimension-compatible base-model R-lens to enrich "
            "an instruct-model ICA Lens and record the transfer in provenance."
        ),
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Destination artifact directory. By default, a local --lens directory "
            "is updated in place; Hub sources require this option."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output = _resolve_output(args.lens, args.output)
    lens = ICALens.from_pretrained(args.lens)
    layers = _parse_layers(args.layer, args.layers, lens.available_layers)
    if args.operation == "add-r-lens":
        if args.r_lens is None:
            raise ValueError("'add-r-lens' requires --r-lens")
        for layer in layers:
            profile = lens.add_r_lens_profile(
                layer=layer,
                r_lens=args.r_lens,
                top_k=args.r_lens_top_k,
                batch_size=args.r_lens_batch_size,
                device=args.device,
                progress=not args.no_progress,
                allow_base_model_transfer=args.allow_base_model_transfer,
            )
            saved_to = lens.checkpoint_component_profile(output, layer=layer)
            print(
                f"Added R-lens readouts to layer {layer}: "
                f"{len(profile['components'])} components."
            )
            print(f"Checkpointed profiled lens to {saved_to}")
        return
    if args.dataset is None:
        raise ValueError("--dataset is required for full component profiling")
    input_type = (
        "chat" if lens.model_type == "instruct" else "text"
    ) if args.input_type == "auto" else args.input_type
    revision = HfApi().dataset_info(args.dataset).sha
    if revision is None:
        raise RuntimeError("could not resolve the dataset revision")
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
            r_lens=args.r_lens,
            r_lens_top_k=args.r_lens_top_k,
            r_lens_batch_size=args.r_lens_batch_size,
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
        if (output / "icalens.json").is_file():
            saved_to = lens.checkpoint_component_profile(output, layer=layer)
        else:
            saved_to = lens.save(output)
        print(
            f"Profiled layer {layer}: {len(profile['components'])} components from "
            f"{profile['n_tokens']} tokens."
        )
        print(f"Checkpointed profiled lens to {saved_to}")


def _resolve_output(lens_source: str, requested: Path | None) -> Path:
    if requested is not None:
        return requested.expanduser().resolve()
    local_source = Path(lens_source).expanduser()
    if local_source.is_dir():
        return local_source.resolve()
    raise ValueError(
        "--output is required when --lens is a Hugging Face repository; "
        "local lens directories are profiled in place by default"
    )


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
