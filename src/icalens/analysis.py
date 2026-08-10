"""Optional Transformers integration for capturing and analyzing model activations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch

from ._capture import capture_resid_post


@dataclass(frozen=True)
class CaptureResult:
    """Tokens and aligned activations captured from one model input."""

    tokens: tuple[str, ...]
    token_ids: torch.Tensor
    positions: torch.Tensor
    activations: torch.Tensor


@dataclass(frozen=True)
class AnalysisResult(CaptureResult):
    """A capture together with signed ICA scores and component energy shares."""

    scores: torch.Tensor
    energy: torch.Tensor


def capture(
    lens: Any,
    inputs: str | list[dict[str, str]],
    *,
    layer: int,
    model: torch.nn.Module | None = None,
    tokenizer: Any = None,
    token_scope: Literal["assistant", "user", "content", "all"] = "all",
    context_length: int | None = None,
    device: str | torch.device | None = None,
) -> CaptureResult:
    """Capture activations for raw text or a completed chat conversation."""
    model, tokenizer = _resolve_model_and_tokenizer(lens, model, tokenizer, device)
    if isinstance(inputs, str):
        encoded = tokenizer(
            inputs,
            add_special_tokens=False,
            truncation=context_length is not None,
            max_length=context_length,
            return_tensors="pt",
        )
        positions = torch.arange(encoded["input_ids"].shape[1], dtype=torch.long)
    else:
        encoded, positions = _encode_chat(
            tokenizer, inputs, token_scope=token_scope, context_length=context_length
        )
    model_device = next(model.parameters()).device
    input_ids = encoded["input_ids"].to(model_device)
    attention_mask = encoded.get("attention_mask")
    kwargs = {"input_ids": input_ids}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask.to(model_device)
    selected = positions.to(model_device)
    if lens.activation_site == "resid_post":
        activations = capture_resid_post(
            model,
            model_inputs=kwargs,
            layers=(layer,),
            positions=selected,
        )[layer].float()
    else:
        with torch.inference_mode():
            outputs = model(**kwargs, output_hidden_states=True, use_cache=False)
        if outputs.hidden_states is None:
            raise RuntimeError("model did not return hidden states")
        hidden_index = (
            layer + 1
            if lens.layer_indexing == "hidden_states_without_initial_embedding_state"
            else layer
        )
        if hidden_index < 0 or hidden_index >= len(outputs.hidden_states):
            raise ValueError(f"layer {layer} is unavailable in model hidden states")
        activations = outputs.hidden_states[hidden_index][0].index_select(0, selected).float()
    ids = input_ids[0].index_select(0, selected).detach().cpu()
    tokens = tuple(tokenizer.convert_ids_to_tokens(ids.tolist()))
    return CaptureResult(tokens=tokens, token_ids=ids, positions=positions, activations=activations)


def analyze(lens: Any, inputs: Any, *, layer: int, **kwargs: Any) -> AnalysisResult:
    """Capture an input and calculate signed scores and per-token energy shares."""
    captured = capture(lens, inputs, layer=layer, **kwargs)
    scores = lens.transform(captured.activations, layer=layer)
    energy = lens.energy(scores)
    return AnalysisResult(
        tokens=captured.tokens,
        token_ids=captured.token_ids,
        positions=captured.positions,
        activations=captured.activations,
        scores=scores,
        energy=energy,
    )


def _resolve_model_and_tokenizer(
    lens: Any,
    model: torch.nn.Module | None,
    tokenizer: Any,
    device: str | torch.device | None,
) -> tuple[torch.nn.Module, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise ImportError(
            "capture/analyze requires optional dependencies; install 'icalens[analyze]'"
        ) from error
    if lens.model_revision is None:
        from huggingface_hub import HfApi

        resolved = HfApi().model_info(lens.model_id).sha
        if resolved is None:
            raise RuntimeError(f"could not resolve an exact revision for {lens.model_id}")
        lens.model_revision = str(resolved)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            lens.model_id, revision=lens.model_revision, use_fast=True
        )
    if model is None:
        try:
            from gb10_load_llm import load_model_to_cuda  # type: ignore[import-untyped]
        except ImportError as error:
            raise ImportError("automatic model loading requires 'icalens[analyze]'") from error
        target = "cuda" if device is None else str(device)
        if not target.startswith("cuda"):
            model = AutoModelForCausalLM.from_pretrained(
                lens.model_id, revision=lens.model_revision
            ).to(target)  # type: ignore[arg-type]
        else:
            model = load_model_to_cuda(
                AutoModelForCausalLM,
                lens.model_id,
                revision=lens.model_revision,
                device=target,
                dtype=torch.bfloat16,
                touch="auto",
                low_cpu_mem_usage=True,
            )
        model.eval()
    return model, tokenizer


def _encode_chat(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    token_scope: str,
    context_length: int | None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    if not tokenizer.is_fast or tokenizer.chat_template is None:
        raise ValueError("chat analysis requires a fast tokenizer with a chat template")
    normalized = _normalize_messages(messages)
    rendered = tokenizer.apply_chat_template(
        normalized, tokenize=False, add_generation_prompt=False
    )
    encoded_offsets = tokenizer(
        rendered,
        add_special_tokens=False,
        truncation=context_length is not None,
        max_length=context_length,
        return_offsets_mapping=True,
    )
    offsets = encoded_offsets["offset_mapping"]
    if token_scope == "all":
        selected = list(range(len(encoded_offsets["input_ids"])))
    else:
        spans = _content_spans(rendered, normalized, token_scope)
        selected = [
            index
            for index, (start, end) in enumerate(offsets)
            if end > start and any(end > left and start < right for left, right in spans)
        ]
    if not selected:
        raise ValueError(f"conversation contains no {token_scope} tokens")
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        truncation=context_length is not None,
        max_length=context_length,
        return_tensors="pt",
    )
    return encoded, torch.tensor(selected, dtype=torch.long)


def _normalize_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    if not isinstance(messages, list) or not messages:
        raise TypeError("messages must be a non-empty list")
    result: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {
            "system",
            "user",
            "assistant",
        }:
            raise ValueError("messages must contain system/user/assistant roles")
        content = message.get("content")
        if not isinstance(content, str) or not content:
            raise ValueError("message content must be a non-empty string")
        result.append({"role": message["role"], "content": content})
    return result


def _content_spans(
    rendered: str, messages: list[dict[str, str]], token_scope: str
) -> list[tuple[int, int]]:
    if token_scope not in {"assistant", "user", "content"}:
        raise ValueError(f"unsupported token scope: {token_scope!r}")
    spans: list[tuple[int, int]] = []
    cursor = 0
    for message in messages:
        start = rendered.find(message["content"], cursor)
        if start < 0:
            raise ValueError("chat template transformed message content; offsets are ambiguous")
        end = start + len(message["content"])
        if token_scope == "content" or message["role"] == token_scope:
            spans.append((start, end))
        cursor = end
    return spans
