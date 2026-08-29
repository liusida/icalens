"""Post-fit component profiling from token-aligned model activations."""

from __future__ import annotations

import hashlib
import heapq
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

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
    allow_base_model_transfer: bool = False,
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
    r_lens_artifact, r_lens_provenance = _load_r_lens(
        lens,
        artifact,
        r_lens,
        allow_base_model_transfer=allow_base_model_transfer,
    )
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
    return cast(dict[str, Any], profile)


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
    allow_base_model_transfer: bool = False,
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
    score_moments = torch.zeros((4, n_components), dtype=torch.float64)
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
        for power in range(1, 5):
            score_moments[power - 1] += scores_cpu.pow(power).sum(dim=0)

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

    return _finish_profile(
        lens,
        artifact,
        layer=layer,
        token_count=token_count,
        source_count=source_count,
        positive_count=positive_count,
        negative_count=negative_count,
        positive_energy=positive_energy,
        negative_energy=negative_energy,
        total_energy=total_energy,
        score_moments=score_moments,
        examples=examples,
        top_k_examples=top_k_examples,
        min_energy=min_energy,
        logit_lens_top_k=logit_lens_top_k,
        logit_lens_batch_size=logit_lens_batch_size,
        r_lens=r_lens,
        r_lens_top_k=r_lens_top_k,
        r_lens_batch_size=r_lens_batch_size,
        allow_base_model_transfer=allow_base_model_transfer,
        provenance=provenance,
        progress=progress,
    )


def profile_components_from_activations(
    lens: Any,
    activations: torch.Tensor,
    records: list[dict[str, Any]],
    *,
    layer: int,
    batch_size: int = 8192,
    top_k_examples: int = 20,
    min_energy: float = 0.05,
    logit_lens_top_k: int = 20,
    logit_lens_batch_size: int = 64,
    r_lens: str | Path | dict[str, Any] | None = None,
    r_lens_top_k: int = 20,
    r_lens_batch_size: int = 8,
    allow_base_model_transfer: bool = False,
    provenance: dict[str, Any] | None = None,
    device: str | torch.device | None = "auto",
    progress: bool = False,
) -> dict[str, Any]:
    """Profile a layer from token-aligned, previously captured activations."""
    if activations.ndim != 2 or len(records) != int(activations.shape[0]):
        raise ValueError("activations and profiling records must have matching rows")
    if batch_size <= 0 or not 0 <= min_energy <= 1:
        raise ValueError("batch_size must be positive and min_energy must be in [0, 1]")
    artifact = lens._get_layer(layer)
    n_components = artifact.n_components
    positive_count = torch.zeros(n_components, dtype=torch.int64)
    negative_count = torch.zeros(n_components, dtype=torch.int64)
    positive_energy = torch.zeros(n_components, dtype=torch.float64)
    negative_energy = torch.zeros(n_components, dtype=torch.float64)
    total_energy = torch.zeros(n_components, dtype=torch.float64)
    score_moments = torch.zeros((4, n_components), dtype=torch.float64)
    examples: list[dict[str, list[tuple[float, int, dict[str, Any]]]]] = [
        {"positive": [], "negative": []} for _ in range(n_components)
    ]
    target = _profiling_device(device)
    iterator = tqdm(
        range(0, len(records), batch_size),
        desc="Profile cached activations",
        unit="batch",
        disable=not progress,
    )
    serial = 0
    for start in iterator:
        end = min(len(records), start + batch_size)
        scores = lens.transform(activations[start:end].to(target), layer=layer)
        energy = lens.energy(scores)
        scores_cpu = scores.detach().to(device="cpu", dtype=torch.float64)
        energy_cpu = energy.detach().to(device="cpu", dtype=torch.float64)
        squared = scores_cpu.square()
        positive = scores_cpu > 0
        negative = scores_cpu < 0
        positive_count += positive.sum(dim=0)
        negative_count += negative.sum(dim=0)
        positive_energy += (squared * positive).sum(dim=0)
        negative_energy += (squared * negative).sum(dim=0)
        total_energy += squared.sum(dim=0)
        for power in range(1, 5):
            score_moments[power - 1] += scores_cpu.pow(power).sum(dim=0)
        for row, component in torch.nonzero(energy_cpu >= min_energy).tolist():
            score = float(scores_cpu[row, component])
            if score == 0:
                continue
            sign = "positive" if score > 0 else "negative"
            record = dict(records[start + row])
            record.update(score=score, energy=float(energy_cpu[row, component]))
            item = (record["energy"], serial, record)
            serial += 1
            heap = examples[component][sign]
            if len(heap) < top_k_examples:
                heapq.heappush(heap, item)
            elif item[0] > heap[0][0]:
                heapq.heapreplace(heap, item)

    from .analysis import _resolve_model_and_tokenizer

    _resolve_model_and_tokenizer(lens, None, None, device)
    return _finish_profile(
        lens,
        artifact,
        layer=layer,
        token_count=len(records),
        source_count=len({int(record["source_index"]) for record in records}),
        positive_count=positive_count,
        negative_count=negative_count,
        positive_energy=positive_energy,
        negative_energy=negative_energy,
        total_energy=total_energy,
        score_moments=score_moments,
        examples=examples,
        top_k_examples=top_k_examples,
        min_energy=min_energy,
        logit_lens_top_k=logit_lens_top_k,
        logit_lens_batch_size=logit_lens_batch_size,
        r_lens=r_lens,
        r_lens_top_k=r_lens_top_k,
        r_lens_batch_size=r_lens_batch_size,
        allow_base_model_transfer=allow_base_model_transfer,
        provenance=provenance,
        progress=progress,
    )


def refresh_profile_statistics_from_activations(
    lens: Any,
    activations: torch.Tensor,
    *,
    layer: int,
    rows: torch.Tensor | None = None,
    batch_size: int = 8192,
    provenance: dict[str, Any] | None = None,
    device: str | torch.device | None = "auto",
    progress: bool = False,
) -> dict[str, Any]:
    """Refresh score statistics and tail selection without rebuilding a profile."""
    if activations.ndim != 2:
        raise ValueError("activations must be a two-dimensional tensor")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if rows is not None:
        if rows.ndim != 1 or rows.dtype != torch.int64:
            raise ValueError("rows must be a one-dimensional int64 tensor")
        if len(rows) and (int(rows.min()) < 0 or int(rows.max()) >= len(activations)):
            raise ValueError("rows contains an activation index outside the dataset")
        token_count = int(len(rows))
    else:
        token_count = int(len(activations))
    if token_count == 0:
        raise ValueError("statistics refresh requires at least one activation")

    artifact = lens._get_layer(layer)
    profile = lens._get_profile(artifact)
    components = profile.get("components")
    if not isinstance(components, list) or len(components) != artifact.n_components:
        raise ValueError("existing component profile does not match the fitted layer")

    n_components = artifact.n_components
    positive_count = torch.zeros(n_components, dtype=torch.int64)
    negative_count = torch.zeros(n_components, dtype=torch.int64)
    positive_energy = torch.zeros(n_components, dtype=torch.float64)
    negative_energy = torch.zeros(n_components, dtype=torch.float64)
    total_energy = torch.zeros(n_components, dtype=torch.float64)
    score_moments = torch.zeros((4, n_components), dtype=torch.float64)
    target = _profiling_device(device)
    iterator = tqdm(
        range(0, token_count, batch_size),
        desc="Refresh profile statistics",
        unit="batch",
        disable=not progress,
    )
    for start in iterator:
        end = min(token_count, start + batch_size)
        if rows is None:
            batch = activations[start:end]
        else:
            batch = activations.index_select(0, rows[start:end])
        scores = lens.transform(batch.to(target), layer=layer)
        scores_cpu = scores.detach().to(device="cpu", dtype=torch.float64)
        squared = scores_cpu.square()
        positive = scores_cpu > 0
        negative = scores_cpu < 0
        positive_count += positive.sum(dim=0)
        negative_count += negative.sum(dim=0)
        positive_energy += (squared * positive).sum(dim=0)
        negative_energy += (squared * negative).sum(dim=0)
        total_energy += squared.sum(dim=0)
        for power in range(1, 5):
            score_moments[power - 1] += scores_cpu.pow(power).sum(dim=0)

    statistics = _population_score_statistics(score_moments, token_count)
    for index, component in enumerate(components):
        sign_total = int(positive_count[index] + negative_count[index])
        squared_total = float(total_energy[index])
        positive_squared_fraction = (
            float(positive_energy[index]) / squared_total if squared_total else 0.0
        )
        negative_squared_fraction = (
            float(negative_energy[index]) / squared_total if squared_total else 0.0
        )
        tail_direction = _tail_direction(
            float(statistics["skewness"][index]), positive_squared_fraction
        )
        component["dominant_sign"] = tail_direction
        component["tail_direction"] = tail_direction
        component["sign_statistics"] = {
            "positive_fraction": float(positive_count[index]) / sign_total if sign_total else 0.0,
            "negative_fraction": float(negative_count[index]) / sign_total if sign_total else 0.0,
            "positive_energy_fraction": positive_squared_fraction,
            "negative_energy_fraction": negative_squared_fraction,
        }
        component["score_statistics"] = _component_score_statistics(statistics, index)
        for readout_name in ("logit_lens", "r_lens"):
            readout = component.get(readout_name)
            if isinstance(readout, dict) and tail_direction in readout:
                readout["dominant"] = readout[tail_direction]

    selection = profile.setdefault("selection", {})
    selection["score_statistics"] = "population_mean_variance_skewness_excess_kurtosis"
    selection["sign_selection"] = "population_skewness"
    profile["score_statistics_provenance"] = provenance
    artifact.profile = profile
    return cast(dict[str, Any], profile)


def _profiling_device(device: str | torch.device | None) -> torch.device:
    if device is None or str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _finish_profile(
    lens: Any,
    artifact: Any,
    *,
    layer: int,
    token_count: int,
    source_count: int,
    positive_count: torch.Tensor,
    negative_count: torch.Tensor,
    positive_energy: torch.Tensor,
    negative_energy: torch.Tensor,
    total_energy: torch.Tensor,
    score_moments: torch.Tensor,
    examples: list[dict[str, list[tuple[float, int, dict[str, Any]]]]],
    top_k_examples: int,
    min_energy: float,
    logit_lens_top_k: int,
    logit_lens_batch_size: int,
    r_lens: str | Path | dict[str, Any] | None,
    r_lens_top_k: int,
    r_lens_batch_size: int,
    allow_base_model_transfer: bool,
    provenance: dict[str, Any] | None,
    progress: bool,
) -> dict[str, Any]:
    if token_count == 0:
        raise ValueError("profiling inputs produced no tokens")
    if progress:
        tqdm.write("Profiling logit-lens vocabulary associations...")
    logit_lens = _logit_lens(
        lens, artifact, top_k=logit_lens_top_k, batch_size=logit_lens_batch_size, progress=progress
    )
    r_lens_result = None
    r_lens_provenance = None
    if r_lens is not None:
        r_lens_artifact, r_lens_provenance = _load_r_lens(
            lens,
            artifact,
            r_lens,
            allow_base_model_transfer=allow_base_model_transfer,
        )
        source_map = _r_lens_source_map(r_lens_artifact, layer)
        if source_map is not None:
            r_lens_result = _r_lens_tokens(
                lens,
                artifact,
                source_map=source_map,
                top_k=r_lens_top_k,
                batch_size=r_lens_batch_size,
                progress=progress,
            )
    components = []
    statistics = _population_score_statistics(score_moments, token_count)
    for component in range(artifact.n_components):
        sign_total = int(positive_count[component] + negative_count[component])
        squared_total = float(total_energy[component])
        pos_fraction = float(positive_energy[component]) / squared_total if squared_total else 0.0
        neg_fraction = float(negative_energy[component]) / squared_total if squared_total else 0.0
        dominant = _tail_direction(float(statistics["skewness"][component]), pos_fraction)
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
        entry: dict[str, Any] = {
            "component": component,
            "dominant_sign": dominant,
            "tail_direction": dominant,
            "sign_statistics": {
                "positive_fraction": float(positive_count[component]) / sign_total
                if sign_total
                else 0.0,
                "negative_fraction": float(negative_count[component]) / sign_total
                if sign_total
                else 0.0,
                "positive_energy_fraction": pos_fraction,
                "negative_energy_fraction": neg_fraction,
            },
            "score_statistics": _component_score_statistics(statistics, component),
            "examples": component_examples,
            "logit_lens": {
                "method": "final_norm_then_unembed",
                "positive": logit_lens[component]["positive"],
                "negative": logit_lens[component]["negative"],
                "dominant": logit_lens[component][dominant],
            },
        }
        if r_lens_result is not None:
            entry["r_lens"] = {
                "method": "relp_then_final_norm_then_unembed",
                "positive": r_lens_result[component]["positive"],
                "negative": r_lens_result[component]["negative"],
                "dominant": r_lens_result[component][dominant],
            }
        components.append(entry)
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
            "score_statistics": "population_mean_variance_skewness_excess_kurtosis",
            "sign_selection": "population_skewness",
        },
        "provenance": provenance,
        "r_lens_provenance": r_lens_provenance,
        "components": components,
    }
    artifact.profile_file = f"component_profiles/{lens.activation_site}/layer_{layer:02d}.json.gz"
    artifact.profile = profile
    return profile


def _population_score_statistics(
    score_moments: torch.Tensor, token_count: int
) -> dict[str, torch.Tensor]:
    raw_mean, raw_second, raw_third, raw_fourth = score_moments / token_count
    variance = (raw_second - raw_mean.square()).clamp_min(0)
    third_central = raw_third - 3 * raw_mean * raw_second + 2 * raw_mean.pow(3)
    fourth_central = (
        raw_fourth
        - 4 * raw_mean * raw_third
        + 6 * raw_mean.square() * raw_second
        - 3 * raw_mean.pow(4)
    )
    skewness = torch.where(
        variance > 0,
        third_central / variance.pow(1.5),
        torch.zeros_like(variance),
    )
    excess_kurtosis = torch.where(
        variance > 0,
        fourth_central / variance.square() - 3,
        torch.zeros_like(variance),
    )
    kurtosis_order = torch.argsort(excess_kurtosis, descending=True, stable=True)
    excess_kurtosis_rank = torch.empty_like(kurtosis_order)
    excess_kurtosis_rank[kurtosis_order] = torch.arange(1, len(kurtosis_order) + 1)
    return {
        "mean": raw_mean,
        "variance": variance,
        "third_central_moment": third_central,
        "skewness": skewness,
        "excess_kurtosis": excess_kurtosis,
        "excess_kurtosis_rank": excess_kurtosis_rank,
    }


def _component_score_statistics(
    statistics: dict[str, torch.Tensor], component: int
) -> dict[str, Any]:
    return {
        "mean": float(statistics["mean"][component]),
        "variance": float(statistics["variance"][component]),
        "third_central_moment": float(statistics["third_central_moment"][component]),
        "skewness": float(statistics["skewness"][component]),
        "excess_kurtosis": float(statistics["excess_kurtosis"][component]),
        "excess_kurtosis_rank": int(statistics["excess_kurtosis_rank"][component]),
    }


def _tail_direction(skewness: float, positive_squared_fraction: float) -> str:
    if skewness > 0:
        return "positive"
    if skewness < 0:
        return "negative"
    return "positive" if positive_squared_fraction >= 0.5 else "negative"


def _load_r_lens(
    lens: Any,
    layer_artifact: Any,
    source: str | Path | dict[str, Any],
    *,
    allow_base_model_transfer: bool = False,
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
    model_mismatch = model_id != lens.model_id
    r_revision = provenance.get("model_revision")
    revision_mismatch = bool(
        lens.model_revision and r_revision and r_revision != lens.model_revision
    )
    transfer = model_mismatch or revision_mismatch
    if transfer and not allow_base_model_transfer:
        raise ValueError(
            "R-lens model provenance does not match the ICA Lens model: "
            f"{model_id!r}@{r_revision!r} != "
            f"{lens.model_id!r}@{lens.model_revision!r}; use "
            "allow_base_model_transfer=True only for an intentional compatible "
            "base-to-instruct transfer"
        )
    if transfer and lens.model_type != "instruct":
        raise ValueError(
            "base-model R-lens transfer is only allowed for an instruct ICA Lens; "
            f"target model type is {lens.model_type!r}"
        )
    d_model = int(artifact.get("d_model", 0))
    hidden_size = lens._hidden_size
    if hidden_size is None:
        raise ValueError("ICA Lens artifact has no recorded hidden size")
    if d_model != int(hidden_size):
        raise ValueError(
            f"R-lens hidden size {d_model} does not match ICA layer hidden size {hidden_size}"
        )
    if lens.activation_site != "resid_post":
        raise ValueError(
            "R-lens profiling currently requires activation_site='resid_post', "
            f"got {lens.activation_site!r}"
        )
    stored_provenance = dict(provenance)
    if digest is not None:
        stored_provenance["sha256"] = digest
    if transfer:
        stored_provenance["transfer"] = {
            "kind": "base_to_instruct",
            "target_model_id": lens.model_id,
            "target_model_revision": lens.model_revision,
            "validation": "explicit opt-in; hidden size, activation site, and layer map checked",
        }
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
