# TPP representation comparison

This experiment adapts SAEBench targeted probe perturbation (TPP) to compare five
representations at the same residual-stream site:

1. fitted ICA;
2. unfitted ICA (the seeded, orthogonal FastICA initialization applied after
   whitening);
3. a published trained SAE;
4. an untrained, tied random SAE control; and
5. PCA.

The untrained SAE deliberately does **not** reuse trained encoder biases,
decoder biases, or JumpReLU thresholds. Its encoder is the transpose of its
unit-norm random decoder. TopK checkpoints retain their configured `k`.
JumpReLU checkpoints use TopK with the checkpoint's advertised average L0 so
that the random control has a meaningful and auditable sparsity constraint.
Results therefore call it `untrained_sae_matched_l0`; it is a matched untrained
dictionary control, not a claim to reproduce an undocumented trainer
initialization.

The unfitted ICA control exactly reconstructs ICA Lens's seeded FastICA
iteration-zero transform: a Gaussian square matrix is symmetrically
decorrelated and then composed with the fitted layer's whitening transform.

Feature selection is performed by SAEBench on its training split and evaluated
on its held-out split. Report both absolute feature budgets and fractions of
each dictionary because SAE dictionaries are wider than ICA and PCA.
The paper preset uses the sparse-probing grid
$k \in \{1,2,5,10,20,50,100\}$; $k=1$ tests
whether a single feature polarity can exert a selective causal effect.

For each target class, SAEBench ranks features by the product of their positive
class-conditional activation difference and their alignment with the trained
probe.  It zero-ablates the selected features while adding the original
reconstruction error back.  The reported TPP score is the targeted probe's
accuracy drop minus the mean accuracy drop of the other probes.  Higher is
better only when the intended change is large and the unintended change stays
small.

## Smoke command

```bash
uv run python experiments/tpp/run.py \
  --lens local-icalens-models/official/icalens-gpt2-small-pile10k \
  --layers 6 \
  --output experiments/tpp/runs/gpt2-l6-smoke \
  --preset smoke
```

Repeating the same command reuses completed layer-method results. Use a new
output directory when changing result-affecting options.

Large SAEBench activation caches are stored outside the repository under
`~/Expansion/research/ICA-data/tpp` by default.  The runner namespaces them by
the resolved scientific configuration and layer, and records their exact paths
in `<output>/storage.json`.  Override the storage filesystem with
`--activation-cache-root`; this is a storage choice and does not change run
identity or numerical results.

Render the completed pilot result:

```bash
uv run python experiments/tpp/plot.py \
  --input experiments/tpp/runs/gpt2-l6-smoke/layers/layer_06/result.json \
  --output experiments/tpp/figures/gpt2-l6-smoke.png
```

The original three-model smoke figures are archived under
`pilot-evidence/qr-unfitted-control/`.  They used a superseded QR-based
unfitted-ICA control and are not scientific results.
