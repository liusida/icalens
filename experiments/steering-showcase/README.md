# ICA steering showcase

This experiment finds language-control components for GPT-2 Small, Gemma 2 2B, and
Qwen 3.5 9B Base, then provides candidates for qualitative additive-steering and
ablation demonstrations.

## Reproduce the component search

The default command searches English-versus-Chinese contrasts in GPT-2 layer 6,
Gemma layer 20, and Qwen layer 20:

```bash
uv run python experiments/steering-showcase/search.py
```

For each model, it samples the same 1,000 ManyThings English/Chinese translation
pairs with seed 0, measures every ICA component at the last non-padding text token,
and ranks components by the absolute difference between Chinese and English mean
scores. The full mean and contrast vectors are saved alongside the top ten candidates.
The command is resumable at the model boundary. On first use, it downloads the selected
ManyThings corpus into `experiments/steering-showcase/data/`; subsequent runs reuse that
local archive. Corpus ZIPs and run outputs are excluded from Git.

Search one model or override layers with an index, inclusive range, or comma-separated
combination:

```bash
uv run python experiments/steering-showcase/search.py \
  --models gemma2 \
  --layers gemma2=18-22

uv run python experiments/steering-showcase/search.py \
  --models gpt2 gemma2 qwen9b \
  --layers gpt2=4-8 \
  --layers gemma2=18-22 \
  --layers qwen9b=7,15,23,31
```

Each model-layer result is checkpointed separately, so interrupting and repeating the
same command skips every completed layer.

Other registered targets can be selected with `--target-language french`,
`japanese`, or `spanish`; use a separate `--output` directory when changing the
configuration.

The Gemma layer-20 Chinese search exactly matches the calibration protocol that
originally identified C105: 1,000 pairs, seed 0, and the final text token without
appending EOS.
