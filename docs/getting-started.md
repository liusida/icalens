# Getting started

## Install

Install ICA Lens from PyPI:

```bash
pip install icalens
```

ICA Lens includes the dependencies needed for model loading, activation
capture, analysis, and HTML export.

## Analyze text

```python
from icalens import ICALens

lens = ICALens.from_pretrained("sida/icalens-gpt2-small-pile10k")
result = lens.analyze(
    "She deposited the check at the bank.",
    layer=6,
)

for token, scores in zip(result.tokens, result.scores, strict=True):
    component = scores.abs().argmax().item()
    print(f"{token!r:<18} C{component}={scores[component].item():+.3f}")
```

The first call downloads the analyzed language model and the requested ICA
layer. Hugging Face caches both for later calls. Model loading uses
`device="auto"` by default: CUDA is selected when available, otherwise CPU.
Override it explicitly with `device="cuda"` or `device="cpu"`.

## Export an interactive report

```python
result.to_html("analysis.html")
```

The output is a self-contained HTML file. To visualize energy shares instead
of signed scores:

```python
result.to_html("energy.html", metric="energy", top_k=5)
```

## Verify an installation

The installed smoke suite exercises the published text and conversation paths:

```bash
uv run icalens-smoke-test
```

Run one path while iterating locally:

```bash
uv run icalens-smoke-test text
uv run icalens-smoke-test chat
```
