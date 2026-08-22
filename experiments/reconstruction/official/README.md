# Official reconstruction experiment

These are the reconstruction runs accepted for the paper. Run scripts invoke
the public CLI with the exact model, layer, preset, and baseline selections used
to produce the stored results.

From the repository root, run one model with:

```bash
bash experiments/reconstruction/official/scripts/run-gpt2.sh
```

The corresponding Gemma 2 and Qwen scripts have the same interface. Recreate
the cross-model figures after all three runs finish with:

```bash
bash experiments/reconstruction/official/scripts/make-figures.sh
```

Repeating a run command resumes its existing output. `run.json` records the
fully resolved configuration and source provenance; `results.json` is the
aggregate result, and `layers/` contains the durable per-layer results used for
the layer and dataset breakdown figures.
