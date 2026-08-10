"""Internal transformer-block activation capture helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import torch


def transformer_blocks(model: torch.nn.Module) -> Sequence[torch.nn.Module]:
    """Return the ordered transformer blocks for supported causal LMs."""
    candidates = (
        ("transformer", "h"),  # GPT-2
        ("model", "layers"),  # Llama, Mistral, Qwen
        ("gpt_neox", "layers"),  # GPT-NeoX/Pythia
    )
    for parent_name, blocks_name in candidates:
        parent = getattr(model, parent_name, None)
        blocks = getattr(parent, blocks_name, None)
        if isinstance(blocks, (torch.nn.ModuleList, torch.nn.Sequential)):
            return cast(Sequence[torch.nn.Module], blocks)
    raise ValueError(
        f"unsupported model architecture {type(model).__name__}: "
        "could not locate transformer blocks"
    )


def capture_resid_post(
    model: torch.nn.Module,
    *,
    model_inputs: dict[str, torch.Tensor],
    layers: tuple[int, ...],
    positions: torch.Tensor,
) -> dict[int, torch.Tensor]:
    """Capture transformer-block outputs before any model-level final norm."""
    blocks = transformer_blocks(model)
    invalid = [layer for layer in layers if layer < 0 or layer >= len(blocks)]
    if invalid:
        raise ValueError(f"layer indices out of range 0..{len(blocks) - 1}: {invalid}")
    captured: dict[int, torch.Tensor] = {}
    handles: list[Any] = []

    def make_hook(layer: int) -> Any:
        def hook(_: torch.nn.Module, __: tuple[Any, ...], output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor):
                raise TypeError(f"transformer block {layer} did not return a tensor")
            captured[layer] = hidden[0].index_select(0, positions.to(hidden.device))

        return hook

    try:
        for layer in layers:
            handles.append(blocks[layer].register_forward_hook(make_hook(layer)))
        with torch.inference_mode():
            model(**model_inputs, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    missing = [layer for layer in layers if layer not in captured]
    if missing:
        raise RuntimeError(f"forward hooks did not capture layers: {missing}")
    return captured
