# ICA Lens

ICA Lens interprets language-model activations with Independent Component
Analysis. It is substantially more compute-efficient to fit than an SAE
dictionary and supports base and instruction-tuned language models.

## Local setup

```bash
uv sync --frozen
```

Load a Lens artifact and analyze text:

```python
from icalens import ICALens

lens = ICALens.from_pretrained("anonymous/icalens-gpt2-small-pile10k")
result = lens.analyze("She deposited the check at the bank.", layer=6)
result
```

In Jupyter or Colab, the final `result` expression displays an interactive
token-level analysis:

Use signed ICA scores or switch the explorer to per-token component energy.
Save the same view as a standalone HTML file with:

```python
result.to_html("analysis.html")
```

The first analysis loads the language model and requested Lens layer. Later
calls on the same `lens` reuse the model in memory. `device="auto"` uses CUDA
when available and otherwise uses the CPU.

## Analyze conversations

Instruction-tuned models accept completed conversations using the standard
`{role, content}` format:

```python
lens = ICALens.from_pretrained("anonymous/icalens-qwen3.5-2b-ultrachat-1m")
result = lens.analyze(
    [
        {"role": "user", "content": "What is the most interesting science?"},
        {"role": "assistant", "content": "Physics."},
    ],
    layer=16,
)
result
```

Chat templates are applied automatically, and template tokens and message
turns are grouped in the interactive result.

## Steering

Generate normally, add an ICA-score offset at the current position, or clamp a
signed ICA coordinate:

```python
messages = [{
    "role": "user",
    "content": "If you had to pick one, what is the most interesting science? Be brief.",
}]

baseline = lens.generate(messages, max_new_tokens=16)
steered = lens.generate(
    messages,
    layer=5,
    steer=(188, -12.0),
    max_new_tokens=16,
)
```

Additive steering defaults to the current position at each generation step.
Use `steering_scope="all-positions"` to edit the entire prompt during prefill as
well. Absolute `clamp=(component, target_score)` interventions remain available.

Component labels, signs, and suitable targets must be established empirically
for the exact Lens and layer.

## Fit a Lens

Run a small GPT-2/Pile-10k example with the installed CLI:

```bash
uv run icalens fit text \
  --model openai-community/gpt2 \
  --dataset NeelNanda/pile-10k \
  --layers 6 \
  --token-budget 1000 \
  --max-iter 20 \
  --output icalens-output/gpt2-demo
```

Fit an instruction-tuned model from UltraChat conversations:

```bash
uv run icalens fit chat \
  --model Qwen/Qwen3.5-2B \
  --dataset HuggingFaceH4/ultrachat_200k \
  --layers 12 \
  --token-budget 100000 \
  --output icalens-output/qwen-demo
```

ICA Lens includes a PyTorch FastICA implementation and does not depend on
SciPy or scikit-learn. Blockwise fitting and layer-at-a-time capture support
larger token collections while bounding memory use.

## Profile every fitted layer

After fitting, profile the components against a representative corpus:

```bash
uv run icalens profile \
  --lens icalens-output/gpt2-demo \
  --layers all \
  --dataset NeelNanda/pile-10k \
  --split train \
  --max-tokens 10000
```

Profiles add sign statistics, top-score examples, Logit Lens tokens, and
optional R-lens readouts to the existing Lens directory. They help label and
inspect components without changing the fitted directions.
