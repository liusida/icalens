"""Post-fit component profiling from token-aligned model activations."""

from __future__ import annotations

import heapq
from collections import Counter
from collections.abc import Iterable
from typing import Any

import torch
from tqdm.auto import tqdm


def profile_components(
    lens: Any,
    inputs: Iterable[str | list[dict[str, str]]],
    *,
    layer: int,
    token_scope: str = "all",
    max_tokens: int | None = 100_000,
    top_k_examples: int = 20,
    min_energy: float = 0.05,
    logit_lens_top_k: int = 20,
    provenance: dict[str, Any] | None = None,
    context_length: int | None = 1024,
    device: str | torch.device | None = "auto",
    progress: bool = False,
) -> dict[str, Any]:
    """Profile fitted components without refitting their ICA transformations."""
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError("max_tokens must be positive or None")
    if top_k_examples <= 0 or logit_lens_top_k <= 0:
        raise ValueError("top-k values must be positive")
    if not 0 <= min_energy <= 1:
        raise ValueError("min_energy must be between 0 and 1")

    artifact = lens._get_layer(layer)
    n_components = artifact.n_components
    positive_count = torch.zeros(n_components, dtype=torch.int64)
    negative_count = torch.zeros(n_components, dtype=torch.int64)
    positive_energy = torch.zeros(n_components, dtype=torch.float64)
    negative_energy = torch.zeros(n_components, dtype=torch.float64)
    total_energy = torch.zeros(n_components, dtype=torch.float64)
    examples: list[dict[str, list[tuple[float, int, dict[str, Any]]]]] = [
        {"positive": [], "negative": []} for _ in range(n_components)
    ]
    serial = 0
    token_count = 0
    source_count = 0
    iterator = tqdm(inputs, desc="Profile components", unit="input", disable=not progress)

    for value in iterator:
        remaining = None if max_tokens is None else max_tokens - token_count
        if remaining is not None and remaining <= 0:
            break
        result = lens.analyze(
            value,
            layer=layer,
            token_scope=token_scope,
            context_length=context_length,
            device=device,
        )
        scores = result.scores.detach()
        if remaining is not None:
            scores = scores[:remaining]
        count = int(scores.shape[0])
        if count == 0:
            continue
        energy = lens.energy(scores)
        scores_cpu = scores.to(device="cpu", dtype=torch.float64)
        energy_cpu = energy.to(device="cpu", dtype=torch.float64)
        squared = scores_cpu.square()
        positive = scores_cpu > 0
        negative = scores_cpu < 0
        positive_count += positive.sum(dim=0)
        negative_count += negative.sum(dim=0)
        positive_energy += (squared * positive).sum(dim=0)
        negative_energy += (squared * negative).sum(dim=0)
        total_energy += squared.sum(dim=0)

        candidates = torch.nonzero(energy_cpu >= min_energy, as_tuple=False)
        for row, component in candidates.tolist():
            score = float(scores_cpu[row, component])
            if score == 0:
                continue
            sign = "positive" if score > 0 else "negative"
            position = int(result.positions[row])
            start = max(0, row - 4)
            end = min(count, row + 5)
            record = {
                "token": result.tokens[row],
                "text": result.token_texts[row],
                "token_id": int(result.token_ids[row]),
                "position": position,
                "score": score,
                "energy": float(energy_cpu[row, component]),
                "context": "".join(result.token_texts[start:end]),
                "source_index": source_count,
            }
            heap = examples[component][sign]
            item = (record["energy"], serial, record)
            serial += 1
            if len(heap) < top_k_examples:
                heapq.heappush(heap, item)
            elif item[0] > heap[0][0]:
                heapq.heapreplace(heap, item)

        token_count += count
        source_count += 1
        iterator.set_postfix(tokens=token_count)

    if token_count == 0:
        raise ValueError("profiling inputs produced no tokens")

    logit_lens = _logit_lens(lens, artifact, top_k=logit_lens_top_k)
    components = []
    for component in range(n_components):
        sign_total = int(positive_count[component] + negative_count[component])
        squared_total = float(total_energy[component])
        positive_energy_fraction = (
            float(positive_energy[component]) / squared_total if squared_total else 0.0
        )
        negative_energy_fraction = (
            float(negative_energy[component]) / squared_total if squared_total else 0.0
        )
        dominant = (
            "positive" if positive_energy_fraction >= negative_energy_fraction else "negative"
        )
        component_examples: dict[str, Any] = {}
        for sign in ("positive", "negative"):
            retained = [item[2] for item in sorted(examples[component][sign], reverse=True)]
            counts = Counter(record["text"] for record in retained)
            component_examples[sign] = {
                "occurrences": retained,
                "tokens": [
                    {"text": text, "count": count}
                    for text, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
                ],
            }
        components.append(
            {
                "component": component,
                "dominant_sign": dominant,
                "sign_statistics": {
                    "positive_fraction": float(positive_count[component]) / sign_total
                    if sign_total
                    else 0.0,
                    "negative_fraction": float(negative_count[component]) / sign_total
                    if sign_total
                    else 0.0,
                    "positive_energy_fraction": positive_energy_fraction,
                    "negative_energy_fraction": negative_energy_fraction,
                },
                "examples": component_examples,
                "logit_lens": {
                    "method": "final_norm_then_unembed",
                    "positive": logit_lens[component]["positive"],
                    "negative": logit_lens[component]["negative"],
                    "dominant": logit_lens[component][dominant],
                },
            }
        )

    profile = {
        "format": "icalens-component-profile",
        "format_version": 1,
        "layer": int(layer),
        "n_tokens": token_count,
        "n_inputs": source_count,
        "selection": {
            "top_k_examples_per_sign": top_k_examples,
            "minimum_component_energy": min_energy,
            "logit_lens_top_k": logit_lens_top_k,
        },
        "provenance": provenance,
        "components": components,
    }
    artifact.profile_file = f"component_profiles/{lens.activation_site}/layer_{layer:02d}.json"
    artifact.profile = profile
    return profile


def _logit_lens(lens: Any, artifact: Any, *, top_k: int) -> list[dict[str, Any]]:
    model = lens._analysis_model
    tokenizer = lens._analysis_tokenizer
    if model is None or tokenizer is None:
        raise RuntimeError("profiling must analyze at least one input before computing logit lens")
    final_norm = _final_norm(model)
    output_embeddings = model.get_output_embeddings()
    if final_norm is None or output_embeddings is None:
        raise ValueError("model does not expose a supported final norm and output embeddings")
    assert artifact.writing_matrix is not None
    parameter = next(model.parameters())
    directions = torch.as_tensor(
        artifact.writing_matrix.T, device=parameter.device, dtype=parameter.dtype
    )
    result: list[dict[str, Any]] = []
    with torch.inference_mode():
        for direction in directions:
            entry: dict[str, Any] = {}
            for name, signed in (("positive", direction), ("negative", -direction)):
                normalized = final_norm(signed[None, None, :])
                logits = output_embeddings(normalized)[0, 0].float()
                k = min(top_k, int(logits.numel()))
                top_values, top_ids = torch.topk(logits, k=k)
                bottom_values, bottom_ids = torch.topk(logits, k=k, largest=False)
                entry[name] = {
                    "top_tokens": _vocabulary_entries(tokenizer, top_ids, top_values),
                    "bottom_tokens": _vocabulary_entries(tokenizer, bottom_ids, bottom_values),
                }
            result.append(entry)
    return result


def _final_norm(model: torch.nn.Module) -> torch.nn.Module | None:
    for path in (("model", "norm"), ("transformer", "ln_f"), ("model", "final_layernorm")):
        value: Any = model
        for name in path:
            value = getattr(value, name, None)
            if value is None:
                break
        if isinstance(value, torch.nn.Module):
            return value
    return None


def _vocabulary_entries(
    tokenizer: Any, ids: torch.Tensor, values: torch.Tensor
) -> list[dict[str, Any]]:
    return [
        {
            "token_id": int(token_id),
            "token": str(tokenizer.convert_ids_to_tokens(int(token_id))),
            "text": str(tokenizer.decode([int(token_id)])),
            "logit": float(logit),
        }
        for token_id, logit in zip(ids.tolist(), values.tolist(), strict=True)
    ]
