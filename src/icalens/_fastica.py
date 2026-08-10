"""Blockwise PyTorch implementation of the FastICA algorithm.

The implementation follows Hyvarinen's fixed-point algorithm and is adapted
from FastICA_torch by Richard Hakim, which is distributed under the MIT
License. ICA Lens keeps the implementation private so its public artifact
format is independent of the fitting backend.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import cast

import torch
from tqdm.auto import tqdm


@dataclass(frozen=True)
class FastICAResult:
    center: torch.Tensor
    components: torch.Tensor
    mixing: torch.Tensor
    n_iter: int


def fit_fastica(
    values: torch.Tensor,
    *,
    n_components: int,
    algorithm: str = "parallel",
    fun: str = "logcosh",
    max_iter: int = 200,
    random_state: int | None = 0,
    progress: bool = False,
    device: str | torch.device | None = None,
    batch_size: int = 8192,
    row_normalize: bool = True,
    norm_eps: float = 1e-12,
) -> FastICAResult:
    """Fit unit-variance-whitened FastICA without moving all samples to CUDA."""
    if values.ndim != 2:
        raise ValueError("FastICA input must be two-dimensional")
    n_samples, n_features = map(int, values.shape)
    if not 0 < n_components <= min(n_samples, n_features):
        raise ValueError("n_components exceeds the input dimensions")
    if algorithm not in {"parallel", "deflation"}:
        raise ValueError("algorithm must be 'parallel' or 'deflation'")
    if fun not in {"logcosh", "exp", "cube"}:
        raise ValueError("fun must be 'logcosh', 'exp', or 'cube'")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    fit_device = values.device if device is None else torch.device(device)
    fit_dtype = torch.float64 if values.dtype == torch.float64 else torch.float32
    center = _mean(
        values,
        device=fit_device,
        dtype=fit_dtype,
        batch_size=batch_size,
        row_normalize=row_normalize,
        norm_eps=norm_eps,
    )
    covariance = _covariance(
        values,
        center=center,
        device=fit_device,
        dtype=fit_dtype,
        batch_size=batch_size,
        row_normalize=row_normalize,
        norm_eps=norm_eps,
    )
    whitening = _whitening_matrix(
        covariance,
        n_samples=n_samples,
        n_components=n_components,
        dtype=fit_dtype,
    )

    generator = torch.Generator(device=fit_device)
    if random_state is not None:
        generator.manual_seed(random_state)
    initial = torch.randn(
        (n_components, n_components),
        dtype=fit_dtype,
        device=fit_device,
        generator=generator,
    )
    nonlinearity = _nonlinearity(fun)
    objective = _contrast(fun) if progress else None
    batch_kwargs = {
        "center": center,
        "whitening": whitening,
        "device": fit_device,
        "dtype": fit_dtype,
        "batch_size": batch_size,
        "row_normalize": row_normalize,
        "norm_eps": norm_eps,
    }
    if algorithm == "parallel":
        unmixing, n_iter = _fit_parallel(
            values,
            initial,
            nonlinearity,
            objective,
            max_iter=max_iter,
            progress=progress,
            batch_kwargs=batch_kwargs,
        )
    else:
        unmixing, n_iter = _fit_deflation(
            values,
            initial,
            nonlinearity,
            objective,
            max_iter=max_iter,
            progress=progress,
            batch_kwargs=batch_kwargs,
        )

    components = unmixing @ whitening
    mixing = torch.linalg.pinv(components)
    return FastICAResult(center=center, components=components, mixing=mixing, n_iter=n_iter)


def _mean(
    values: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    row_normalize: bool,
    norm_eps: float,
) -> torch.Tensor:
    total = torch.zeros(values.shape[1], dtype=torch.float64, device=device)
    count = 0
    for batch in _preprocessed_batches(
        values,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        row_normalize=row_normalize,
        norm_eps=norm_eps,
    ):
        total += batch.to(torch.float64).sum(dim=0)
        count += int(batch.shape[0])
    return (total / count).to(dtype)


def _covariance(
    values: torch.Tensor,
    *,
    center: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    row_normalize: bool,
    norm_eps: float,
) -> torch.Tensor:
    feature_count = int(values.shape[1])
    covariance = torch.zeros((feature_count, feature_count), dtype=torch.float64, device=device)
    for batch in _preprocessed_batches(
        values,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        row_normalize=row_normalize,
        norm_eps=norm_eps,
    ):
        centered64 = (batch - center).to(torch.float64)
        covariance.addmm_(centered64.T, centered64)
    return covariance


def _whitening_matrix(
    covariance: torch.Tensor,
    *,
    n_samples: int,
    n_components: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.flip(0)[:n_components]
    eigenvectors = eigenvectors.flip(1)[:, :n_components]
    threshold = torch.finfo(torch.float64).eps * max(covariance.shape) * eigenvalues[0]
    if bool((eigenvalues <= threshold).any()):
        raise ValueError("input rank is smaller than n_components")
    signs = torch.where(eigenvectors[0] < 0, -1.0, 1.0)
    eigenvectors = eigenvectors * signs
    whitening = (eigenvectors / eigenvalues.sqrt()).T
    return cast(torch.Tensor, whitening.to(dtype=dtype) * n_samples**0.5)


def _fit_parallel(
    source: torch.Tensor,
    initial: torch.Tensor,
    nonlinearity: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    objective: Callable[[torch.Tensor], torch.Tensor] | None,
    *,
    max_iter: int,
    progress: bool,
    batch_kwargs: dict[str, object],
) -> tuple[torch.Tensor, int]:
    weights = _symmetric_decorrelation(initial)
    n_samples = int(source.shape[0])
    iterations = tqdm(
        range(max_iter),
        desc="FastICA parallel",
        unit="iter",
        dynamic_ncols=True,
        disable=not progress,
    )
    for _ in iterations:
        term_sum = torch.zeros_like(weights)
        derivative_sum = torch.zeros(weights.shape[0], dtype=weights.dtype, device=weights.device)
        objective_sum = torch.zeros((), dtype=weights.dtype, device=weights.device)
        for whitened in _whitened_batches(source, **batch_kwargs):
            projected = weights @ whitened
            transformed, derivative_mean = nonlinearity(projected)
            batch_count = int(whitened.shape[1])
            term_sum.addmm_(transformed, whitened.T)
            derivative_sum += derivative_mean * batch_count
            if objective is not None:
                objective_sum += objective(projected) * batch_count
        updated = term_sum / n_samples - (derivative_sum / n_samples)[:, None] * weights
        updated = _symmetric_decorrelation(updated)
        limit = torch.max(torch.abs(torch.abs(torch.sum(updated * weights, dim=1)) - 1.0))
        postfix = {"limit": f"{float(limit):.2e}"}
        if objective is not None:
            postfix["obj"] = f"{float(objective_sum / n_samples):.2f}"
        iterations.set_postfix(postfix)
        weights = updated
    return weights, max_iter


def _fit_deflation(
    source: torch.Tensor,
    initial: torch.Tensor,
    nonlinearity: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    objective: Callable[[torch.Tensor], torch.Tensor] | None,
    *,
    max_iter: int,
    progress: bool,
    batch_kwargs: dict[str, object],
) -> tuple[torch.Tensor, int]:
    count = int(initial.shape[0])
    n_samples = int(source.shape[0])
    weights = torch.zeros_like(initial)
    components = tqdm(
        range(count),
        desc="FastICA deflation",
        unit="component",
        dynamic_ncols=True,
        disable=not progress,
    )
    for component in components:
        weight = initial[component] / torch.linalg.vector_norm(initial[component])
        objective_value = torch.zeros((), dtype=weight.dtype, device=weight.device)
        for _ in range(max_iter):
            term_sum = torch.zeros_like(weight)
            derivative_sum = torch.zeros((), dtype=weight.dtype, device=weight.device)
            objective_value.zero_()
            for whitened in _whitened_batches(source, **batch_kwargs):
                projected = weight @ whitened
                transformed, derivative_mean = nonlinearity(projected[None, :])
                batch_count = int(whitened.shape[1])
                term_sum += whitened @ transformed[0]
                derivative_sum += derivative_mean[0] * batch_count
                if objective is not None:
                    objective_value += objective(projected[None, :]) * batch_count
            updated = term_sum / n_samples - (derivative_sum / n_samples) * weight
            if component:
                updated -= (updated @ weights[:component].T) @ weights[:component]
            updated /= torch.linalg.vector_norm(updated)
            limit = torch.abs(torch.abs(updated @ weight) - 1.0)
            weight = updated
        weights[component] = weight
        postfix = {"iterations": max_iter, "limit": f"{float(limit):.2e}"}
        if objective is not None:
            postfix["obj"] = f"{float(objective_value / n_samples):.2f}"
        components.set_postfix(postfix)
    return weights, max_iter


def _source_std(
    values: torch.Tensor,
    *,
    center: torch.Tensor,
    components: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    row_normalize: bool,
    norm_eps: float,
) -> torch.Tensor:
    source_sum = torch.zeros(components.shape[0], dtype=torch.float64, device=device)
    source_square_sum = torch.zeros_like(source_sum)
    count = 0
    for batch in _preprocessed_batches(
        values,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        row_normalize=row_normalize,
        norm_eps=norm_eps,
    ):
        sources64 = ((batch - center) @ components.T).to(torch.float64)
        source_sum += sources64.sum(dim=0)
        source_square_sum += sources64.square().sum(dim=0)
        count += int(batch.shape[0])
    variance = source_square_sum / count - (source_sum / count).square()
    return variance.clamp_min(torch.finfo(torch.float64).eps).sqrt().to(dtype)


def _preprocessed_batches(
    values: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    row_normalize: bool,
    norm_eps: float,
) -> Iterator[torch.Tensor]:
    for start in range(0, int(values.shape[0]), batch_size):
        batch = values[start : start + batch_size].to(device=device, dtype=dtype)
        if not bool(torch.isfinite(batch).all()):
            raise ValueError("FastICA input contains non-finite values")
        if row_normalize:
            norms = torch.linalg.vector_norm(batch, dim=1, keepdim=True)
            batch = batch / norms.clamp_min(norm_eps)
        yield batch


def _whitened_batches(
    values: torch.Tensor,
    *,
    center: object,
    whitening: object,
    device: object,
    dtype: object,
    batch_size: object,
    row_normalize: object,
    norm_eps: object,
) -> Iterator[torch.Tensor]:
    assert isinstance(center, torch.Tensor)
    assert isinstance(whitening, torch.Tensor)
    assert isinstance(device, torch.device)
    assert isinstance(dtype, torch.dtype)
    assert isinstance(batch_size, int)
    assert isinstance(row_normalize, bool)
    assert isinstance(norm_eps, float)
    for batch in _preprocessed_batches(
        values,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        row_normalize=row_normalize,
        norm_eps=norm_eps,
    ):
        yield whitening @ (batch - center).T


def _symmetric_decorrelation(weights: torch.Tensor) -> torch.Tensor:
    eigenvalues, eigenvectors = torch.linalg.eigh(weights @ weights.T)
    floor = torch.finfo(weights.dtype).eps
    inverse_root = eigenvectors @ torch.diag(eigenvalues.clamp_min(floor).rsqrt()) @ eigenvectors.T
    return cast(torch.Tensor, inverse_root @ weights)


def _nonlinearity(name: str) -> Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    if name == "logcosh":

        def logcosh(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            transformed = torch.tanh(values)
            return transformed, (1.0 - transformed.square()).mean(dim=1)

        return logcosh
    if name == "exp":

        def exp(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            exponential = torch.exp(-values.square() / 2.0)
            return values * exponential, ((1.0 - values.square()) * exponential).mean(dim=1)

        return exp

    def cube(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return values.pow(3), (3.0 * values.square()).mean(dim=1)

    return cube


def _contrast(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if name == "logcosh":

        def logcosh(values: torch.Tensor) -> torch.Tensor:
            return (torch.logaddexp(values, -values) - 0.6931471805599453).mean()

        return logcosh
    if name == "exp":

        def exp(values: torch.Tensor) -> torch.Tensor:
            return -torch.exp(-values.square() / 2.0).mean()

        return exp

    def cube(values: torch.Tensor) -> torch.Tensor:
        return values.pow(4).mean() / 4.0

    return cube
