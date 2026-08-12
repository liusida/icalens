# ICA Lens

ICA Lens maps language-model activations into independent components without
training another dictionary. It provides a compact Python API for loading,
fitting, sharing, and applying ICA transformations to base and instruction-
tuned language models.

```bash
pip install icalens
```

```python
from icalens import ICALens

lens = ICALens.from_pretrained("sida/icalens-gpt2-small-pile10k")
result = lens.analyze("She deposited the check at the bank.", layer=6)

print(result.tokens)
print(result.scores)
result.to_html("analysis.html")
```

[Get started](getting-started.md){ .md-button .md-button--primary }
[Read the paper](https://arxiv.org/abs/2606.11722){ .md-button }

## What ICA Lens provides

- A built-in PyTorch FastICA implementation without SciPy or scikit-learn.
- Direct activation capture from raw text and completed chat conversations.
- Signed ICA scores and per-token component energy shares.
- Local and Hugging Face artifact loading with lazy per-layer downloads.
- Self-contained interactive HTML reports.
- Blockwise fitting designed to bound GPU memory use.

## Citation

```bibtex
@article{liu2026icalens,
  title={ICA Lens: Interpreting Language Models Without Training Another Dictionary},
  author={Liu, Sida and Han, Feijiang},
  journal={arXiv preprint arXiv:2606.11722},
  year={2026}
}
```
