# Text and chat

ICA Lens accepts raw text for base models and completed conversations for
instruction-tuned models.

## Raw text

```python
from icalens import ICALens

lens = ICALens.from_pretrained("sida/icalens-gpt2-small-pile10k")
result = lens.analyze(
    "The boat reached the bank before sunset.",
    layer=6,
)
```

## Completed conversations

Pass messages using the familiar `role` and `content` structure:

```python
from icalens import ICALens

lens = ICALens.from_pretrained(
    "sida/icalens-qwen3.5-2b-ultrachat-1m"
)

messages = [
    {
        "role": "user",
        "content": "What is the most interesting science? One phrase.",
    },
    {
        "role": "assistant",
        "content": "Quantum entanglement.",
    },
]

result = lens.analyze(messages, layer=16)
result.to_html("conversation.html")
```

ICA Lens applies the tokenizer's chat template before capturing activations.
With the default `token_scope="all"`, results include role markers, separators,
user tokens, assistant tokens, and any other template tokens.

Choose a narrower scope when needed:

```python
result = lens.analyze(
    messages,
    layer=16,
    token_scope="assistant",
)
```

Supported scopes are `"all"`, `"content"`, `"user"`, and `"assistant"`.

!!! note
    `analyze()` analyzes supplied content; it does not generate an assistant
    response. Generate with the language model first, append that response to
    `messages`, and then call `analyze()`.
