# Official SAE suffix-sweep ERF

This experiment extends the validated Gemma pilot to GPT-2 Small (12 layers),
Gemma 2 2B (26), and Qwen 3.5 9B Base (32). It uses the pinned SAE baseline
registry and existing 1M-token Pile-10k activation caches, not new SAE training.

## Commands

From the repository root:

```bash
uv run python experiments/effective-receptive-fields/sae.py --dry-run
uv run python experiments/effective-receptive-fields/sae.py
```

For a layer-level validation run, in a separate output:

```bash
uv run python experiments/effective-receptive-fields/sae.py --models gemma2 --layers 12 --output experiments/effective-receptive-fields/runs/sae-validation
```

Defaults: 100 uniformly sampled feature IDs per layer, seed 0, up to 20 strongest
positive examples per feature, thresholds `1,3,5,10,15`. `--rank-thresholds 5`
selects the alternative top-5-only protocol and requires a different output.
All layer indices are zero-based. The same seed is used independently per layer;
for equal dictionary widths this deliberately yields the same numeric feature
IDs, not semantically matched features.

## Definition and controls

- Score = existing checkpoint encoder activation multiplied by decoder norm.
  Rank uses strict competition ranking among the entire dictionary (32k, 16k,
  and 64k respectively), with positive activation required for recovery.
- Inputs preserve cached 1024-token framing: Gemma uses BOS, GPT-2 and Qwen
  use their recorded end-of-text document prefix. Targets exclude that prefix.
  Selection uses the 1M cached tokens, not every
  token in the larger candidate corpus.
- GPT-2's SAE was trained at context length 64. This experiment evaluates it
  at 1024, matching the released ICA ERF protocol, not the context-64
  reconstruction control. The comparison is not training-context matched.
- Reuse the official suffix engine: exact lengths 1–10, then the existing
  geometric schedule; exact or geometric-bracket first recovery estimates;
  available full-context length for unrecovered occurrences. Record recovery
  flags separately. A long assigned ERF is not proof of long-context dependence.
- Uniform selection includes inactive features. Never resample them. Save their
  status and report the denominator. Active features use however many positive
  examples exist, up to 20.
- Checkpoint encoded features and full-context ranks are used as stored
  endpoints. Audit the first example of up to 10 pending features against a live
  full-prefix forward before sweeping. Fail on score disagreement exceeding
  5% relative / 0.01 absolute or a threshold-classification disagreement.
  This is a smoke audit, not a proof of numerical equivalence for all examples.

## Numerical batching

The pilot's adaptive batch membership caused two top-5 rank-boundary differences
among 2,000 examples when additional thresholds were requested. The official
SAE mode fixes batch membership at each suffix length, including resolved
examples as padding work for unresolved members. Entire resolved batches are
skipped. Resume retains the original full feature ordering and batch membership.
This removes threshold-dependent regrouping; it does not promise bitwise
invariance across hardware or batch-size changes. Batch size is fingerprinted.
The old ICA runner retains its existing default batching behavior.

## Resume and artifacts

Each model/layer has its own `run.json`, full logs, cumulative profiling-shard
checkpoints, prepared feature records, per-feature sweep traces and summaries.
Repeating the same command validates identities and resumes automatically;
completed layers skip model loading. A partially completed fixed batch may
re-evaluate resolved members as padding, but never rewrites completed results.
Configuration/dependency changes fail; choose a new output, never delete or
silently reuse the pilot. The launcher runs one layer per child process to
release GPU memory reliably between layers.

Use feature fractions, not dictionary-size-extrapolated counts, for ICA–SAE
comparisons. Report threshold, recovery coverage, inactive features, context
protocol, and dictionary width alongside ERF distributions. This implementation
does not alter paper figures or launch the full experiment automatically.

## Implementation validation (2026-09-05)

- 17 targeted CPU tests passed, including the existing ERF API/formula tests,
  inactive-feature handling, checkpoint identity rejection, fixed batch
  membership, threshold agreement, and partial-result resume.
- GPU smoke runs passed: Gemma layer 12 (4 features × 2 examples), GPT-2 layer 6
  (2 × 2), and Qwen layer 16 (2 × 2). Each profiled the full 1M cached tokens
  and passed its selected full-prefix checks. These are implementation tests,
  not representative scientific results.
- Completed-run replay skipped model loading in the Gemma and GPT-2 checks.
- On the original Gemma pilot's 100 × 20 examples, fixed-batch top-5 and
  multi-threshold sweeps agreed on all 2,000 top-5 occurrence results. The old
  adaptive-batch comparison differed on two rank-5/rank-6 boundary cases.
- Fixed batching adds padding work; the original 1.74× adaptive-batch timing
  ratio is not an official fixed-batch runtime estimate. The subsequent
  consistency check was not an isolated timing benchmark.
