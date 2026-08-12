# Scores and energy

For each token position, ICA Lens returns two component-level views.

## Signed ICA scores

`result.scores` contains the standard signed ICA coordinates. Sign and
magnitude are both meaningful within a fitted lens:

```python
scores = result.scores
strongest = scores.abs().topk(5, dim=-1).indices
```

Use scores when studying component direction, comparing positive and negative
activations, or reconstructing activations.

## Energy shares

For component score \(s_j\), ICA Lens defines its per-token energy share as:

\[
e_j = \frac{s_j^2}{\sum_k s_k^2}
\]

Therefore, energy is nonnegative and sums to one across components at each
nonzero token position:

```python
energy = result.energy
print(energy.sum(dim=-1))
```

Energy is useful for answering “what fraction of this token's squared ICA-score
magnitude belongs to this component?” It discards sign, so it is not a
replacement for the standard ICA score.

## Reconstruction

Map component scores back into activation space with:

```python
reconstructed = lens.inverse_transform(result.scores, layer=6)
```

The result is an approximate reconstruction in the activation coordinates used
by the fitted lens.
