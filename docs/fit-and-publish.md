# Fit and publish

## Fit one layer

The low-level fitting API accepts a NumPy array or PyTorch tensor whose final
dimension is the model hidden size:

```python
from icalens import ICALens

lens = ICALens(
    model_id="openai-community/gpt2",
    model_type="base",
    activation_site="resid_post",
)

lens.fit(
    activations,
    layer=6,
    max_iter=20,
    batch_size=8192,
    device="cuda",
    progress=True,
)

lens.save("./my-icalens")
```

Calling `fit()` for another layer adds or replaces that layer in the same lens.
Fitting uses blockwise passes so GPU memory scales primarily with `batch_size`,
not the total number of activation rows.

## Demo fitting pipelines

The repository includes end-to-end examples for Pile-10k with GPT-2 and
UltraChat with instruction-tuned models:

```bash
uv run python demo/fit.py --layers all --token-budget 1000000
uv run python demo/fit_chat.py --layers 12 --token-budget 1000000
```

## Publish to Hugging Face

Use a Hugging Face **Model** repository for an ICA Lens artifact:

```python
lens.push_to_hub("username/icalens-model-name")
```

Or publish an existing local artifact with the demo:

```bash
uv run python demo/publish.py \
  --lens ./my-icalens \
  username/icalens-model-name
```

For the demo, put a write-enabled token in the project-root `.env` file:

```dotenv
HF_TOKEN=hf_...
```

The artifact records the analyzed model, exact revision, activation site,
layers, preprocessing, fitting configuration, and dataset provenance when it is
available.
