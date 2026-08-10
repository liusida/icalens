"""Fit a small GPT-2 ICA Lens from 1,000 Pile-10k tokens."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from gb10_load_llm import load_model_to_cuda
from huggingface_hub import HfApi
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from icalens import ICALens

MODEL_ID = "openai-community/gpt2"
DATASET_ID = "NeelNanda/pile-10k"
TOKEN_BUDGET = 1_000
CONTEXT_LENGTH = 1_024
DEFAULT_OUTPUT = Path(__file__).parent / "output" / "icalens-gpt2-small"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layers",
        default="6",
        help="Comma-separated transformer-layer indices, or 'all' (default: 6).",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--max-iter",
        type=int,
        default=20,
        help="Fixed number of FastICA iterations (default: 20).",
    )
    parser.add_argument(
        "--candidate-tokens",
        type=int,
        default=None,
        help="Candidate pool size; defaults to --token-budget.",
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=TOKEN_BUDGET,
        help="Number of sampled token activations used to fit ICA.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-vram-gb",
        type=float,
        default=None,
        help="Cap this process's PyTorch CUDA allocator, for example 16.",
    )
    parser.add_argument(
        "--fit-batch-size",
        type=int,
        default=8192,
        help="Activation rows processed on CUDA at once; 0 uses all rows (default: 8192).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This GB10 demo requires a CUDA device.")
    if args.max_vram_gb is not None:
        set_cuda_memory_limit(args.max_vram_gb)
    if args.fit_batch_size < 0:
        raise ValueError("--fit-batch-size must be non-negative")
    torch.cuda.reset_peak_memory_stats()

    log(f"Resolving {MODEL_ID} revision...")
    api = HfApi()
    revision = api.model_info(MODEL_ID).sha
    dataset_revision = api.dataset_info(DATASET_ID).sha
    if revision is None or dataset_revision is None:
        raise RuntimeError("Could not resolve exact model and dataset revisions.")
    log("Loading tokenizer and Pile-10k candidate tokens...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=revision)
    candidate_tokens = args.token_budget if args.candidate_tokens is None else args.candidate_tokens
    documents = load_pile_documents(
        tokenizer,
        dataset_revision=str(dataset_revision),
        candidate_token_budget=candidate_tokens,
        context_length=CONTEXT_LENGTH,
    )
    selected_positions = sample_positions(
        documents,
        token_budget=args.token_budget,
        seed=args.seed,
    )

    log(f"Loading {MODEL_ID}@{revision} on CUDA...")
    model = load_model_to_cuda(
        AutoModelForCausalLM,
        MODEL_ID,
        device="cuda",
        dtype=torch.bfloat16,
        touch="auto",
        low_cpu_mem_usage=True,
        revision=revision,
    )
    model.eval()

    # hidden_states[0] is the embedding state. ICALens layer 0 is the output of
    # transformer block 0, so layer N maps to hidden_states[N + 1].
    layers = parse_layers(args.layers, layer_count=int(model.config.num_hidden_layers))
    log(
        f"Capturing {args.token_budget} sampled activations for layers "
        f"{','.join(map(str, layers))}..."
    )
    activations_by_layer = capture_activations(
        model,
        documents=documents,
        selected_positions=selected_positions,
        layers=layers,
    )
    lens = ICALens(
        model_id=MODEL_ID,
        model_revision=str(revision),
        model_type="base",
        activation_site="hidden_states",
        layer_indexing="hidden_states_without_initial_embedding_state",
    )

    for layer in layers:
        activations = activations_by_layer[layer]
        sample_count = int(activations.shape[0])
        hidden_size = int(activations.shape[1])
        minimum_samples = hidden_size + 1
        if sample_count < minimum_samples:
            raise ValueError(
                f"Cannot fit a full {hidden_size}-component ICA Lens from only "
                f"{sample_count} token activations: centering limits the rank to at most "
                f"{sample_count - 1}. Increase --token-budget to at least {minimum_samples}; "
                "if --candidate-tokens is set, it must be at least as large."
            )
        fit_batch_size = (
            int(activations.shape[0]) if args.fit_batch_size == 0 else args.fit_batch_size
        )
        log(
            f"Fitting layer {layer} from {activations.shape[0]} tokens "
            f"with {hidden_size} components, max_iter={args.max_iter}, "
            f"fit_batch_size={fit_batch_size}..."
        )
        lens.fit(
            activations,
            layer=layer,
            n_components=hidden_size,
            algorithm="parallel",
            fun="logcosh",
            max_iter=args.max_iter,
            random_state=args.seed,
            progress=True,
            device="cuda",
            batch_size=fit_batch_size,
            provenance={
                "dataset": {
                    "repo_id": DATASET_ID,
                    "revision": str(dataset_revision),
                    "split": "train",
                },
                "token_scope": "all",
                "candidate_tokens": candidate_tokens,
                "fitting_tokens": args.token_budget,
                "sampling_seed": args.seed,
                "context_length": CONTEXT_LENGTH,
            },
        )

    output = lens.save(args.output)
    log(f"Saved {len(layers)} layer(s) to {output}")
    log(f"Available layers: {lens.available_layers}")
    peak_gib = torch.cuda.max_memory_reserved() / 1024**3
    log(f"Peak PyTorch CUDA memory reserved: {peak_gib:.2f} GiB")


def load_pile_documents(
    tokenizer: Any,
    *,
    dataset_revision: str | None = None,
    candidate_token_budget: int,
    context_length: int,
) -> list[torch.Tensor]:
    """Tokenize Pile documents until the requested candidate pool is full."""
    if candidate_token_budget <= 0:
        raise ValueError("--candidate-tokens must be positive")
    dataset = load_dataset(DATASET_ID, split="train", revision=dataset_revision, streaming=True)
    documents: list[torch.Tensor] = []
    captured_tokens = 0
    progress = tqdm(
        total=candidate_token_budget,
        desc="Tokenize Pile-10k",
        unit="tok",
        dynamic_ncols=True,
    )
    try:
        for row in dataset:
            text = row.get("text") if isinstance(row, dict) else None
            if not isinstance(text, str) or not text.strip():
                continue
            remaining = candidate_token_budget - captured_tokens
            encoded = tokenizer(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=min(context_length, remaining),
            )["input_ids"]
            if encoded:
                document = torch.tensor(encoded, dtype=torch.long)
                documents.append(document)
                token_count = int(document.shape[0])
                captured_tokens += token_count
                progress.update(token_count)
                progress.set_postfix(documents=len(documents), refresh=False)
            if captured_tokens >= candidate_token_budget:
                break
    finally:
        progress.close()

    if captured_tokens < candidate_token_budget:
        raise RuntimeError(
            f"Pile-10k yielded only {captured_tokens} candidate tokens; "
            f"expected {candidate_token_budget}."
        )
    log(f"Loaded {captured_tokens} candidate tokens from {len(documents)} documents.")
    return documents


def sample_positions(
    documents: list[torch.Tensor],
    *,
    token_budget: int,
    seed: int,
) -> dict[int, torch.Tensor]:
    """Sample token positions uniformly without replacement across documents."""
    total_tokens = sum(int(document.shape[0]) for document in documents)
    if token_budget <= 0:
        raise ValueError("--token-budget must be positive")
    if token_budget > total_tokens:
        raise ValueError(
            f"--token-budget ({token_budget}) exceeds candidate tokens ({total_tokens})"
        )
    rng = np.random.default_rng(seed)
    flat_positions = np.sort(rng.choice(total_tokens, size=token_budget, replace=False))
    selected: defaultdict[int, list[int]] = defaultdict(list)
    document_start = 0
    position_index = 0
    for document_index, document in enumerate(documents):
        document_end = document_start + int(document.shape[0])
        while (
            position_index < len(flat_positions) and flat_positions[position_index] < document_end
        ):
            selected[document_index].append(int(flat_positions[position_index] - document_start))
            position_index += 1
        document_start = document_end
    return {
        document_index: torch.tensor(positions, dtype=torch.long)
        for document_index, positions in selected.items()
    }


def capture_activations(
    model: torch.nn.Module,
    *,
    documents: list[torch.Tensor],
    selected_positions: dict[int, torch.Tensor],
    layers: tuple[int, ...],
) -> dict[int, torch.Tensor]:
    """Capture selected positions while preserving each document's left context."""
    buffers: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    selected_count = sum(int(positions.shape[0]) for positions in selected_positions.values())
    progress = tqdm(
        total=selected_count,
        desc="Capture activations",
        unit="tok",
        dynamic_ncols=True,
    )
    try:
        for document_index, positions_cpu in selected_positions.items():
            input_ids = documents[document_index].unsqueeze(0).to("cuda")
            positions = positions_cpu.to("cuda")
            with torch.inference_mode():
                outputs = model(
                    input_ids=input_ids,
                    output_hidden_states=True,
                    use_cache=False,
                )
            if outputs.hidden_states is None:
                raise RuntimeError("GPT-2 did not return hidden states.")
            for layer in layers:
                selected = outputs.hidden_states[layer + 1][0].index_select(0, positions)
                buffers[layer].append(selected.to(device="cpu", dtype=torch.float32))
            progress.update(int(positions.shape[0]))
            progress.set_postfix(document=document_index, refresh=False)
    finally:
        progress.close()
    return {layer: torch.cat(chunks, dim=0) for layer, chunks in buffers.items()}


def parse_layers(value: str, *, layer_count: int) -> tuple[int, ...]:
    if value.strip().lower() == "all":
        return tuple(range(layer_count))
    try:
        layers = tuple(sorted({int(item.strip()) for item in value.split(",")}))
    except ValueError as error:
        raise ValueError("--layers must be comma-separated integers or 'all'") from error
    if not layers:
        raise ValueError("--layers cannot be empty")
    invalid = [layer for layer in layers if layer < 0 or layer >= layer_count]
    if invalid:
        raise ValueError(f"layer indices out of range 0..{layer_count - 1}: {invalid}")
    return layers


def log(message: str) -> None:
    """Print a timestamped demo progress message immediately."""
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{timestamp}] {message}", flush=True)


def set_cuda_memory_limit(limit_gb: float) -> None:
    """Limit this process through PyTorch's CUDA caching allocator."""
    if not np.isfinite(limit_gb) or limit_gb <= 0:
        raise ValueError("--max-vram-gb must be a finite positive number")
    total_bytes = torch.cuda.get_device_properties(0).total_memory
    requested_bytes = limit_gb * 1024**3
    if requested_bytes > total_bytes:
        total_gib = total_bytes / 1024**3
        raise ValueError(
            f"--max-vram-gb={limit_gb:g} exceeds visible CUDA memory ({total_gib:.2f} GiB)"
        )
    fraction = requested_bytes / total_bytes
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    log(
        f"Limited the PyTorch CUDA allocator to {limit_gb:.2f} GiB "
        f"({fraction:.2%} of visible memory)."
    )


if __name__ == "__main__":
    main()
