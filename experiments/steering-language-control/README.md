# Language steering

This experiment searches for SAE features and ICA components that steer Gemma 2 2B
from English toward Chinese, French, Japanese, and Spanish, then records steered
generations for each condition.

With no selection flags, the runner evaluates both methods, all four target languages,
and every layer available in the Lens:

```bash
uv run experiments/steering-language-control/run.py
```

The full default is a large, resumable run. Results are checkpointed separately under
`experiments/steering-language-control/runs/`, so rerunning the same command skips valid
completed conditions.

Use `--layers` to restrict the search. It accepts one layer, inclusive ranges,
comma-separated combinations, or `all`:

```bash
uv run experiments/steering-language-control/run.py --layers 19-21
uv run experiments/steering-language-control/run.py --layers 0,4-6,10
```

`--layer 20` remains available as a compatibility alias for a single layer. Likewise,
`--method sae` or `--method ica` and `--target-language french` restrict the method or
language grid.
