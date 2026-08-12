# Fit and publish

ICA Lens includes installed commands for fitting from text or conversations and
publishing the resulting artifact. These commands are available after:

```bash
pip install icalens
```

A CUDA GPU is currently required by the end-to-end fitting commands.

## Fit from text

Start with a small run that checks the complete fitting pipeline:

```bash
icalens fit text \
  --model openai-community/gpt2 \
  --dataset NeelNanda/pile-10k \
  --split train \
  --text-field text \
  --layers 6 \
  --token-budget 1000 \
  --max-iter 20 \
  --output icalens-output/quick-test
```

![Fitting a GPT-2 ICA Lens from Pile-10k in a terminal](assets/fit.png){ loading=lazy }

*A complete one-layer fitting run using the installed `icalens` command. This
run finished in about 10 seconds; runtime depends on the hardware and whether
the model and dataset are already cached.*

In this concrete setting:

- `--model openai-community/gpt2` selects the GPT-2 Small checkpoint. Its
  hidden width is 768, so a full-width lens contains 768 ICA components.
- `--dataset NeelNanda/pile-10k --split train --text-field text` reads raw text
  from the `text` column of the dataset's training split.
- `--layers 6` captures the residual stream after transformer block 6 only.
- `--token-budget 1000` fits from 1,000 token activations. This is deliberately
  small, but it exceeds the 769-token minimum required to fit 768 components
  after centering.
- `--max-iter 20` runs 20 fixed FastICA updates. At this small token budget,
  the complete example took about 10 seconds on the machine shown above.
- `--output icalens-output/quick-test` saves the manifest and fitted layer in
  that directory.

For a substantive GPT-2 lens, increase the fitting budget and iteration count:

```bash
icalens fit text \
  --model openai-community/gpt2 \
  --dataset NeelNanda/pile-10k \
  --split train \
  --text-field text \
  --layers all \
  --capture-layers-at-once 1 \
  --token-budget 1000000 \
  --max-iter 20 \
  --output icalens-output/icalens-gpt2-small
```

Both commands resolve and record the exact model and dataset revisions. They
tokenize the selected text field, capture post-block residual-stream
activations, fit the requested layers, and checkpoint each completed layer.

Use `--token-budget all` to fit from every usable token in the selected dataset.
The command reports the resolved token count after tokenization.

## Fit from conversations

Fit an instruction-tuned Qwen3.5 lens from UltraChat conversations:

```bash
icalens fit chat \
  --model Qwen/Qwen3.5-2B \
  --dataset HuggingFaceH4/ultrachat_200k \
  --split train_sft \
  --layers all \
  --capture-layers-at-once 1 \
  --token-scope all \
  --token-budget 1000000 \
  --max-iter 20 \
  --output icalens-output/icalens-qwen3.5-2b-ultrachat-1m
```

Conversations are rendered with the model tokenizer's chat template before
activation capture. The dataset must provide a message list with `role` and
`content` fields; select another column with `--messages-field`.

`--token-scope all` makes every rendered position eligible for fitting,
including role markers and template tokens. The other scopes are `content`,
`user`, and `assistant`. A narrower scope changes which activation rows are
sampled, while the complete rendered conversation remains the model context.

## Important controls

| Option | Purpose |
| --- | --- |
| `--model` | Hugging Face language-model repository |
| `--dataset` | Hugging Face fitting-dataset repository |
| `--split` | Dataset split to stream |
| `--text-field` | Raw-text column used by `icalens fit text` |
| `--context-length` | Maximum number of tokens retained from each text document |
| `--layers` | Comma-separated zero-based transformer-block indices, or `all` |
| `--token-budget` | Number of sampled activation rows used for fitting |
| `--candidate-tokens` | Size of the token pool sampled from; defaults to the token budget |
| `--capture-layers-at-once` | Number of layers materialized in CPU memory together; `0` means all requested layers |
| `--fit-batch-size` | Activation rows processed on the GPU at once; `0` materializes all fitting rows |
| `--max-iter` | Fixed number of FastICA iterations |
| `--objective-every` | Record objective percentiles every N iterations |
| `--seed` | Token-sampling and FastICA initialization seed |
| `--max-vram-gb` | Optional PyTorch CUDA allocator limit for testing a memory budget |

FastICA uses the requested fixed iteration count; it does not stop early using
a tolerance criterion. Full-width ICA also requires at least
`hidden_size + 1` sampled activations because centering removes one degree of
rank.

## Memory behavior

Captured activations are held in CPU memory, while fitting processes bounded
batches on the GPU. The two main controls address different resources:

- Lower `--capture-layers-at-once` to reduce CPU activation memory. A value of
  `1` captures and fits one layer before moving to the next.
- Lower `--fit-batch-size` to reduce fitting-time GPU memory.

The model forward pass stops after the deepest layer needed by the current
capture group. Each fitted layer is saved immediately, including its objective
history, so earlier layers remain usable if a later layer is interrupted.

## Fit existing activations

Use the Python API when you already have an activation tensor. Its final
dimension must equal the model hidden size; leading dimensions are treated as
sample dimensions. This complete example uses synthetic GPT-2-width
activations so it can run as written; replace them with activations captured
from your data to produce a meaningful lens.

```python
import torch

from icalens import ICALens

# A full-width 768-component fit needs at least 769 samples after centering.
generator = torch.Generator().manual_seed(0)
activations = torch.randn(1000, 768, generator=generator)

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
    provenance={
        "dataset": {"description": "synthetic API example"},
        "fitting_tokens": int(activations.shape[0]),
    },
)

output = lens.save("icalens-output/my-icalens")
print(f"Saved layer {lens.available_layers} to {output}")
```

Calling `fit()` again adds or replaces a layer in the same lens. Pass
`n_components` to fit fewer components than the hidden size.

## What is saved

An ICA Lens artifact records the information required to interpret and reuse
the transformation:

- analyzed model ID, type, and exact revision;
- activation site and layer-index convention;
- L2 preprocessing and fitted center;
- reading and writing matrices;
- FastICA configuration, objective history, and component ordering;
- available layers and component counts; and
- dataset and sampling provenance supplied during fitting.

The artifact does not contain the analyzed language-model weights.

## Profile components after fitting

Profiling enriches an existing lens without rerunning FastICA. It streams an
annotation dataset through the model and records, for every component:

- positive and negative token-position frequencies;
- the fraction of squared score energy on each sign and the resulting dominant sign;
- high-energy token occurrences, token counts, scores, positions, and short contexts; and
- top and bottom vocabulary tokens obtained by passing both writing-vector directions
  through the model's final norm and unembedding.

For example, profile layer 5 of a fitted chat lens:

```bash
icalens profile \
  --lens icalens-output/icalens-qwen3.5-2b-ultrachat-1m \
  --layers all \
  --dataset HuggingFaceH4/ultrachat_200k \
  --split train_sft \
  --token-scope all \
  --max-tokens 100000 \
  --top-k-examples 20 \
  --min-energy 0.05 \
  --output icalens-output/icalens-qwen3.5-2b-profiled
```

Each completed layer is checkpointed to the output directory. The profiling
dataset and exact revision are recorded separately from fitting
provenance. Profiles are optional compressed JSON files under `component_profiles/`; the fitted
center and matrices are unchanged. Consequently, an already published or local
lens can be profiled later without refitting it.

Inspect a stored component profile with:

```python
profile = lens.component_profile(layer=5, component=188)
print(profile["dominant_sign"])
print(profile["examples"]["negative"]["tokens"])
print(profile["logit_lens"]["dominant"]["top_tokens"])
```

The logit-lens entries are diagnostic associations: at an intermediate layer,
they skip the remaining transformer blocks and therefore are not exact causal
predictions of generated tokens.

## Authenticate with Hugging Face

ICA Lens artifacts belong in a Hugging Face **Model** repository. Authenticate
with a write-enabled token using the standard Hugging Face CLI:

```bash
hf auth login
```

Alternatively, set `HF_TOKEN` in the environment. The publishing command also
reads `HF_TOKEN` from `.env` in the current directory:

```dotenv
HF_TOKEN=hf_...
```

Do not commit this file or token.

## Publish and verify

Publish a saved artifact:

```bash
icalens publish \
  --lens icalens-output/icalens-gpt2-small \
  username/icalens-gpt2-small
```

Add `--private` to create a private repository. The command uploads the
artifact, downloads its manifest again, and verifies that the remote metadata
and available layers match the local lens.

The Python equivalent is:

```python
lens.push_to_hub("username/icalens-gpt2-small")
```

After publication, load it from any environment with:

```python
from icalens import ICALens

lens = ICALens.from_pretrained("username/icalens-gpt2-small")
```
