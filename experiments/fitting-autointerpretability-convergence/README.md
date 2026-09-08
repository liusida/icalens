# Fitting–autointerpretability convergence

This experiment asks whether ICA autointerpretability stabilizes as the FastICA
iteration budget increases. Its initial scope is Qwen 3.5 9B Base at residual
layers 15 and 31, using the same one-million-token Pile-10k activation captures
as the official Lens.

`trajectory.py` runs one continuous, deterministic FastICA trajectory per layer.
It stores the initialization, every iteration 1 through 10, and every tenth
iteration from 20 through 200. Checkpoints retain persistent optimization-row
identity and are not independently reordered by final component strength.

Run:

```bash
uv run python experiments/fitting-autointerpretability-convergence/trajectory.py
```

The output is resumable at the latest saved iteration. Shared centers and
whitening matrices are stored once per layer; each checkpoint stores only the
current float32 unmixing matrix. Later steps will materialize selected checkpoints
as ordinary ICA Lens artifacts, profile them, and run matched-component
autointerpretability evaluation.
