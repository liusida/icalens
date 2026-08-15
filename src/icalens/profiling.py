"""Post-fit component profiling from token-aligned model activations."""

from __future__ import annotations

import hashlib
import heapq
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm


def add_r_lens_profile(
    lens: Any,
    *,
    layer: int,
    r_lens: str | Path | dict[str, Any],
    top_k: int = 20,
    batch_size: int = 8,
    device: str | torch.device | None = "auto",
    progress: bool = False,
) -> dict[str, Any]:
    """Enrich a stored component profile with R-lens vocabulary readouts."""
    if top_k <= 0 or batch_size <= 0:
        raise ValueError("top_k and batch_size must be positive")
    artifact = lens._get_layer(layer)
    profile = lens._get_profile(artifact)
    components = profile.get("components")
    if not isinstance(components, list) or len(components) != artifact.n_components:
        raise ValueError(
            "existing component profile does not match the fitted layer: "
            f"{len(components) if isinstance(components, list) else 'invalid'} profiles "
            f"for {artifact.n_components} components"
        )

    # R-lens readouts need the target model's final norm and unembedding, but
    # not the profiling dataset or any activation capture.
    from .analysis import _resolve_model_and_tokenizer

    _resolve_model_and_tokenizer(lens, None, None, device)
    r_lens_artifact, r_lens_provenance = _load_r_lens(lens, artifact, r_lens)
    source_map = _r_lens_source_map(r_lens_artifact, layer)
    if source_map is None:
        raise ValueError(f"R-lens has no source map for layer {layer}")
    if progress:
        tqdm.write(f"Adding R-lens vocabulary associations for layer {layer}...")
    result = _r_lens_tokens(
        lens,
        artifact,
        source_map=source_map,
        top_k=top_k,
        batch_size=batch_size,
        progress=progress,
    )
    for component, readout in zip(components, result, strict=True):
        dominant = str(component["dominant_sign"])
        component["r_lens"] = {
            "method": "relp_then_final_norm_then_unembed",
            "positive": readout["positive"],
            "negative": readout["negative"],
            "dominant": readout[dominant],
        }
    selection = profile.setdefault("selection", {})
    selection["r_lens_top_k"] = top_k
    selection["r_lens_batch_size"] = batch_size
    profile["r_lens_provenance"] = r_lens_provenance
    artifact.profile = profile
    return profile


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
    logit_lens_batch_size: int = 64,
    r_lens: str | Path | dict[str, Any] | None = None,
    r_lens_top_k: int = 20,
    r_lens_batch_size: int = 8,
    provenance: dict[str, Any] | None = None,
    context_length: int | None = 1024,
    device: str | torch.device | None = "auto",
    progress: bool = False,
) -> dict[str, Any]:
    """Profile fitted components without refitting their ICA transformations."""
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError("max_tokens must be positive or None")
    if (
        top_k_examples <= 0
        or logit_lens_top_k <= 0
        or logit_lens_batch_size <= 0
        or r_lens_top_k <= 0
        or r_lens_batch_size <= 0
    ):
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

    if progress:
        tqdm.write("Profiling logit-lens vocabulary associations...")
    logit_lens = _logit_lens(
        lens,
        artifact,
        top_k=logit_lens_top_k,
        batch_size=logit_lens_batch_size,
        progress=progress,
    )
    r_lens_result = None
    r_lens_provenance = None
    if r_lens is not None:
        r_lens_artifact, r_lens_provenance = _load_r_lens(lens, artifact, r_lens)
        source_map = _r_lens_source_map(r_lens_artifact, layer)
        if source_map is None:
            if progress:
                tqdm.write(
                    f"R-lens has no source map for layer {layer}; "
                    "omitting R-lens tokens for this layer."
                )
        else:
            if progress:
                tqdm.write("Profiling R-lens vocabulary associations...")
            r_lens_result = _r_lens_tokens(
                lens,
                artifact,
                source_map=source_map,
                top_k=r_lens_top_k,
                batch_size=r_lens_batch_size,
                progress=progress,
            )
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
        component_profile = {
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
        if r_lens_result is not None:
            component_profile["r_lens"] = {
                "method": "relp_then_final_norm_then_unembed",
                "positive": r_lens_result[component]["positive"],
                "negative": r_lens_result[component]["negative"],
                "dominant": r_lens_result[component][dominant],
            }
        components.append(component_profile)

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
            "logit_lens_batch_size": logit_lens_batch_size,
            "r_lens_top_k": r_lens_top_k if r_lens_result is not None else None,
            "r_lens_batch_size": r_lens_batch_size if r_lens_result is not None else None,
        },
        "provenance": provenance,
        "r_lens_provenance": r_lens_provenance,
        "components": components,
    }
    artifact.profile_file = (
        f"component_profiles/{lens.activation_site}/layer_{layer:02d}.json.gz"
    )
    artifact.profile = profile
    return profile


def _load_r_lens(
    lens: Any,
    layer_artifact: Any,
    source: str | Path | dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if isinstance(source, dict):
        artifact = source
        digest = None
    else:
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"R-lens artifact does not exist: {path}")
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if not isinstance(artifact, dict) or not isinstance(artifact.get("J"), dict):
        raise ValueError("R-lens artifact must contain a dictionary of source maps under 'J'")
    provenance = artifact.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("R-lens artifact has no provenance metadata")
    model_id = provenance.get("model_id")
    if model_id != lens.model_id:
        raise ValueError(
            f"R-lens model {model_id!r} does not match ICA Lens model {lens.model_id!r}"
        )
    r_revision = provenance.get("model_revision")
    if lens.model_revision and r_revision and r_revision != lens.model_revision:
        raise ValueError(
            "R-lens model revision does not match the ICA Lens model revision: "
            f"{r_revision!r} != {lens.model_revision!r}"
        )
    d_model = int(artifact.get("d_model", 0))
    hidden_size = lens._hidden_size
    if hidden_size is None:
        raise ValueError("ICA Lens artifact has no recorded hidden size")
    if d_model != int(hidden_size):
        raise ValueError(
            f"R-lens hidden size {d_model} does not match ICA layer hidden size "
            f"{hidden_size}"
        )
    if lens.activation_site != "resid_post":
        raise ValueError(
            "R-lens profiling currently requires activation_site='resid_post', "
            f"got {lens.activation_site!r}"
        )
    stored_provenance = dict(provenance)
    if digest is not None:
        stored_provenance["sha256"] = digest
    return artifact, stored_provenance


def _r_lens_source_map(artifact: dict[str, Any], layer: int) -> torch.Tensor | None:
    maps = artifact["J"]
    value = maps.get(layer, maps.get(str(layer)))
    if value is None:
        return None
    matrix = torch.as_tensor(value)
    d_model = int(artifact["d_model"])
    if tuple(matrix.shape) != (d_model, d_model):
        raise ValueError(
            f"R-lens source map for layer {layer} has shape {tuple(matrix.shape)}; "
            f"expected {(d_model, d_model)}"
        )
    return matrix


def _r_lens_tokens(
    lens: Any,
    artifact: Any,
    *,
    source_map: torch.Tensor,
    top_k: int,
    batch_size: int,
    progress: bool,
) -> list[dict[str, Any]]:
    model = lens._analysis_model
    tokenizer = lens._analysis_tokenizer
    if model is None or tokenizer is None:
        raise RuntimeError("profiling must analyze at least one input before computing R-lens")
    final_norm = _final_norm(model)
    output_embeddings = model.get_output_embeddings()
    if final_norm is None or output_embeddings is None:
        raise ValueError("model does not expose a supported final norm and output embeddings")
    assert artifact.writing_matrix is not None
    parameter = next(model.parameters())
    directions = torch.as_tensor(
        artifact.writing_matrix.T, device=parameter.device, dtype=torch.float32
    )
    mapping = source_map.to(device=parameter.device, dtype=torch.float32)
    result: list[dict[str, Any]] = [{} for _ in range(int(directions.shape[0]))]
    with torch.inference_mode():
        batches = range(0, int(directions.shape[0]), batch_size)
        for start in tqdm(
            batches,
            desc="Profile R-lens",
            unit="batch",
            disable=not progress,
        ):
            stop = min(start + batch_size, int(directions.shape[0]))
            mapped = torch.einsum("oi,bi->bo", mapping, directions[start:stop])
            for name, signed in (("positive", mapped), ("negative", -mapped)):
                normalized = final_norm(signed.to(parameter.dtype)[:, None, :])
                logits = output_embeddings(normalized)[:, 0].float()
                k = min(top_k, int(logits.shape[-1]))
                top_values, top_ids = torch.topk(logits, k=k, dim=-1)
                bottom_values, bottom_ids = torch.topk(logits, k=k, dim=-1, largest=False)
                for offset in range(stop - start):
                    result[start + offset][name] = {
                        "top_tokens": _vocabulary_entries(
                            tokenizer, top_ids[offset], top_values[offset]
                        ),
                        "bottom_tokens": _vocabulary_entries(
                            tokenizer, bottom_ids[offset], bottom_values[offset]
                        ),
                    }
    return result


def _logit_lens(
    lens: Any,
    artifact: Any,
    *,
    top_k: int,
    batch_size: int,
    progress: bool,
) -> list[dict[str, Any]]:
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
    result: list[dict[str, Any]] = [{} for _ in range(int(directions.shape[0]))]
    with torch.inference_mode():
        batches = range(0, int(directions.shape[0]), batch_size)
        for start in tqdm(
            batches,
            desc="Profile logit lens",
            unit="batch",
            disable=not progress,
        ):
            stop = min(start + batch_size, int(directions.shape[0]))
            batch = directions[start:stop]
            for name, signed in (("positive", batch), ("negative", -batch)):
                normalized = final_norm(signed[:, None, :])
                logits = output_embeddings(normalized)[:, 0].float()
                k = min(top_k, int(logits.shape[-1]))
                top_values, top_ids = torch.topk(logits, k=k, dim=-1)
                bottom_values, bottom_ids = torch.topk(logits, k=k, dim=-1, largest=False)
                for offset in range(stop - start):
                    result[start + offset][name] = {
                        "top_tokens": _vocabulary_entries(
                            tokenizer, top_ids[offset], top_values[offset]
                        ),
                        "bottom_tokens": _vocabulary_entries(
                            tokenizer, bottom_ids[offset], bottom_values[offset]
                        ),
                    }
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
