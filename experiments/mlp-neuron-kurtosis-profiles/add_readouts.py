#!/usr/bin/env python3
"""Add Logit-Lens and R-Lens vocabulary readouts to selected neuron profiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
from gb10_load_llm import load_model_to_cuda
from transformers import AutoModelForCausalLM, AutoTokenizer

from icalens._capture import transformer_blocks

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "runs/main/selected-neuron-profiles-context.json"
DEFAULT_OUTPUT = HERE / "runs/main/selected-neuron-profiles-readouts.json"
DEFAULT_RUN = HERE / "runs/main/run.json"
DEFAULT_R_LENS = (
    HERE.parent.parent / "local-r-lens-models/official/qwen3.5-9b-base/lens.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--r-lens", type=Path, default=DEFAULT_R_LENS)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def final_norm(model: torch.nn.Module) -> torch.nn.Module:
    for path in (
        ("model", "norm"),
        ("transformer", "ln_f"),
        ("model", "final_layernorm"),
        ("gpt_neox", "final_layer_norm"),
    ):
        value: Any = model
        for name in path:
            value = getattr(value, name, None)
            if value is None:
                break
        if isinstance(value, torch.nn.Module):
            return value
    raise ValueError("model does not expose a supported final norm")


def vocabulary_entries(
    tokenizer: Any, ids: torch.Tensor, values: torch.Tensor
) -> list[dict[str, Any]]:
    return [
        {
            "token_id": int(token_id),
            "token": str(tokenizer.convert_ids_to_tokens(int(token_id))),
            "text": str(
                tokenizer.decode(
                    [int(token_id)], clean_up_tokenization_spaces=False
                )
            ),
            "logit": float(logit),
        }
        for token_id, logit in zip(ids.tolist(), values.tolist(), strict=True)
    ]


def decode_directions(
    directions: torch.Tensor,
    *,
    norm: torch.nn.Module,
    unembed: torch.nn.Module,
    tokenizer: Any,
    top_k: int,
) -> list[dict[str, Any]]:
    parameter = next(norm.parameters())
    normalized = norm(directions.to(parameter.dtype)[:, None, :])
    logits = unembed(normalized)[:, 0].float()
    k = min(top_k, int(logits.shape[-1]))
    top_values, top_ids = torch.topk(logits, k=k, dim=-1)
    bottom_values, bottom_ids = torch.topk(logits, k=k, dim=-1, largest=False)
    return [
        {
            "top_tokens": vocabulary_entries(tokenizer, top_ids[row], top_values[row]),
            "bottom_tokens": vocabulary_entries(
                tokenizer, bottom_ids[row], bottom_values[row]
            ),
        }
        for row in range(int(directions.shape[0]))
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    source = args.input.expanduser().resolve()
    destination = args.output.expanduser().resolve()
    run_path = args.run.expanduser().resolve()
    r_lens_path = args.r_lens.expanduser().resolve()
    if destination.exists() and not args.force:
        raise FileExistsError(f"output exists; pass --force to replace it: {destination}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    profiles = payload.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("input contains no selected-neuron profiles")
    resolved = json.loads(run_path.read_text(encoding="utf-8"))["resolved"]
    model_spec = resolved["model"]
    layer = int(resolved["layer"])

    tokenizer = AutoTokenizer.from_pretrained(
        model_spec["repo_id"],
        revision=model_spec.get("revision"),
        trust_remote_code=True,
    )
    model = load_model_to_cuda(
        AutoModelForCausalLM,
        model_spec["repo_id"],
        revision=model_spec.get("revision"),
        device=args.device,
        dtype=torch.bfloat16,
        touch="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    projection = transformer_blocks(model)[layer].mlp.down_proj
    neurons = [int(profile["neuron"]) for profile in profiles]
    parameter = next(model.parameters())
    directions = projection.weight[:, neurons].T.to(
        device=parameter.device, dtype=torch.float32
    )
    norm = final_norm(model)
    unembed = model.get_output_embeddings()
    if unembed is None:
        raise ValueError("model does not expose output embeddings")

    r_lens_artifact = torch.load(r_lens_path, map_location="cpu", weights_only=True)
    source_map = r_lens_artifact.get("J", {}).get(
        layer, r_lens_artifact.get("J", {}).get(str(layer))
    )
    if source_map is None:
        raise ValueError(f"R-Lens artifact has no source map for layer {layer}")
    source_map = torch.as_tensor(
        source_map, device=parameter.device, dtype=torch.float32
    )
    if tuple(source_map.shape) != (directions.shape[1], directions.shape[1]):
        raise ValueError(
            f"R-Lens map shape {tuple(source_map.shape)} is incompatible with "
            f"direction width {directions.shape[1]}"
        )

    with torch.inference_mode():
        logit_positive = decode_directions(
            directions,
            norm=norm,
            unembed=unembed,
            tokenizer=tokenizer,
            top_k=args.top_k,
        )
        logit_negative = decode_directions(
            -directions,
            norm=norm,
            unembed=unembed,
            tokenizer=tokenizer,
            top_k=args.top_k,
        )
        mapped = torch.einsum("oi,bi->bo", source_map, directions)
        r_positive = decode_directions(
            mapped,
            norm=norm,
            unembed=unembed,
            tokenizer=tokenizer,
            top_k=args.top_k,
        )
        r_negative = decode_directions(
            -mapped,
            norm=norm,
            unembed=unembed,
            tokenizer=tokenizer,
            top_k=args.top_k,
        )

    for row, profile in enumerate(profiles):
        dominant = str(profile["dominant_sign"])
        profile["logit_lens"] = {
            "method": "final_norm_then_unembed",
            "positive": logit_positive[row],
            "negative": logit_negative[row],
            "dominant": logit_positive[row] if dominant == "positive" else logit_negative[row],
        }
        profile["r_lens"] = {
            "method": "relp_then_final_norm_then_unembed",
            "positive": r_positive[row],
            "negative": r_negative[row],
            "dominant": r_positive[row] if dominant == "positive" else r_negative[row],
        }

    payload["vocabulary_readouts"] = {
        "direction": "column i of W_downproj; negated for the negative sign",
        "layer": layer,
        "top_k": args.top_k,
        "logit_lens_method": "final_norm_then_unembed",
        "r_lens_method": "relp_then_final_norm_then_unembed",
        "r_lens_path": str(r_lens_path),
        "r_lens_sha256": sha256(r_lens_path),
        "r_lens_provenance": r_lens_artifact.get("provenance"),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, destination)
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
