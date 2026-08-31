# Artifact integrity checks

`icalens integrity reproduce` is the release-oriented end-to-end integrity
check. It treats an existing activation cache and ICA Lens as references only:
their activation values and fitted tensors are never used as inputs to the new
fit. The command reads their recorded provenance, recaptures the requested
layer from the pinned model and dataset revisions, refits ICA, rebuilds the
component profile, and compares deterministic samples with the references.
It first verifies that the reference activation cache and Lens record the same
candidate-token population, fitting-token count, sampling seed, and context
length. A cache drawn from a different population is not a valid reference for
that Lens and is rejected before expensive model inference begins.

The standard GPT-2 release canary is:

```bash
uv run icalens integrity reproduce \
  --reference-lens local-icalens-models/official/icalens-gpt2-small-pile10k \
  --reference-activations /home/liusida/Expansion/research/ICA-data/icalens-activations/gpt2-pile10k-1m \
  --reference-experiments experiments \
  --layer 6 \
  --output .icalens-integrity/gpt2-layer6 \
  --device cuda
```

By default it compares 1,024 seeded activation rows and 64 seeded components.
Token metadata and captured bfloat16 values must match exactly. Fitted tensors,
probe scores, and profile statistics use the configured numerical tolerances;
component ordering, tail direction, token associations, and selected example
identities must match.

When `--reference-experiments` is supplied, the check also discovers official
result sets for the same model and layer. It verifies their recorded run status,
model and experiment identity, layer rows, finite summary metrics, and immutable
checksums. For the GPT-2 layer-6 canary this currently covers held-out
reconstruction and SAEBench sparse probing. This is a compatibility and
integrity audit of the stored paper results; it deliberately does not rerun the
full paper-scale evaluations.

The output directory identifies the run. Repeating the same command validates
its checkpoints and resumes the first incomplete stage. It contains `run.json`,
`report.json`, `report.md`, complete logs, the recaptured layer, and the
reproduced Lens. A different reference checksum or comparison configuration
requires a different output directory.

This GPU-backed check is intended for release validation rather than the normal
unit-test suite. Ordinary tests retain small fitting, persistence, and
compatibility fixtures.

## Keeping the check current

Run the GPT-2 layer-6 canary before a release and after changes to capture,
fitting, profiling, artifact serialization, or experiment result formats. Use
the command above; a successful run ends with `report.md` marked `PASS`.

When adding or modifying an experiment, review the integrity check as part of
the same change. Update it when the experiment produces official persisted
results involving GPT-2 layer 6, changes a stored result schema or identity
field, or introduces a small deterministic computation that should be covered.
Keep paper-scale, API-dependent, incomplete, and pilot-only runs outside the
release gate until they have stable official artifacts. If an experiment is
intentionally excluded, record the reason in its experiment documentation.
