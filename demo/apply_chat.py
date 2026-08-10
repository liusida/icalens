"""Generate a chat response and apply a saved ICA Lens to its activations."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from gb10_load_llm import load_model_to_cuda
from html_explorer import write_explorer_html
from transformers import AutoModelForCausalLM, AutoTokenizer

from icalens import ICALens
from icalens._capture import capture_resid_post

DEFAULT_LENS = Path(__file__).parent / "output" / "icalens-qwen2.5-0.5b-instruct"
DEFAULT_USER = "Explain why the sky appears blue in one sentence."
CONTEXT_LENGTH = 1_024
DEFAULT_OUTPUT_FILE = Path(__file__).parent / "output" / "apply_chat.html"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lens", type=Path, default=DEFAULT_LENS)
    parser.add_argument("--system", default=None, help="Optional system message.")
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of assistant tokens to generate (default: 128).",
    )
    parser.add_argument(
        "--token-scope",
        choices=("assistant", "user", "content", "all"),
        default="all",
        help="Formatted tokens to display (default: all, including template tokens).",
    )
    parser.add_argument("--context-length", type=int, default=CONTEXT_LENGTH)
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This GB10 demo requires a CUDA device.")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.context_length <= 0:
        raise ValueError("--context-length must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.max_new_tokens >= args.context_length:
        raise ValueError("--max-new-tokens must be smaller than --context-length")

    lens = ICALens.from_pretrained(args.lens)
    if lens.model_type != "instruct":
        raise ValueError(
            f"apply_chat.py requires an instruct lens, but this lens is {lens.model_type!r}"
        )
    if args.layer not in lens.available_layers:
        raise ValueError(
            f"layer {args.layer} is not in this lens; available layers: {lens.available_layers}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        lens.model_id,
        revision=lens.model_revision,
        use_fast=True,
    )
    if not tokenizer.is_fast:
        raise RuntimeError("apply_chat.py requires a fast tokenizer for role-aware token offsets.")
    if tokenizer.chat_template is None:
        raise RuntimeError(f"{lens.model_id} tokenizer does not define a chat template.")

    prompt_messages = []
    if args.system is not None:
        prompt_messages.append({"role": "system", "content": args.system})
    prompt_messages.append({"role": "user", "content": args.user})

    model = load_model_to_cuda(
        AutoModelForCausalLM,
        lens.model_id,
        revision=lens.model_revision,
        device="cuda",
        dtype=torch.bfloat16,
        touch="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()

    rendered_prompt = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt = tokenizer(
        rendered_prompt,
        add_special_tokens=False,
        return_tensors="pt",
    )
    prompt_length = int(prompt["input_ids"].shape[1])
    prompt_token_ids = prompt["input_ids"][0].tolist()
    if prompt_length + args.max_new_tokens > args.context_length:
        raise ValueError(
            f"prompt has {prompt_length} tokens, leaving fewer than "
            f"--max-new-tokens={args.max_new_tokens} within the "
            f"{args.context_length}-token context"
        )
    prompt = {name: value.to("cuda") for name, value in prompt.items()}
    with torch.inference_mode():
        generated = model.generate(
            **prompt,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    prompt_positions = torch.arange(prompt_length, device="cuda")
    prompt_activations = capture_resid_post(
        model,
        model_inputs=prompt,
        layers=(args.layer,),
        positions=prompt_positions,
    )[args.layer]
    prompt_scores = lens.transform(prompt_activations, layer=args.layer)
    prompt_top_k = min(args.top_k, prompt_scores.shape[-1])
    prompt_top_indices = torch.topk(prompt_scores.abs(), k=prompt_top_k, dim=-1).indices
    response_ids = generated[0, prompt_length:].to("cpu")
    assistant_response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    if not assistant_response:
        raise RuntimeError("Model generated an empty assistant response.")

    messages = [*prompt_messages, {"role": "assistant", "content": assistant_response}]
    result = lens.analyze(
        messages,
        layer=args.layer,
        model=model,
        tokenizer=tokenizer,
        token_scope=args.token_scope,
        context_length=args.context_length,
    )
    scores = result.scores
    top_k = min(args.top_k, scores.shape[-1])
    top_indices = torch.topk(scores.abs(), k=top_k, dim=-1).indices
    token_ids = result.token_ids.tolist()
    tokens = result.tokens

    print(f"Lens: {args.lens}")
    print(f"Model: {lens.model_id}@{lens.model_revision} ({lens.model_type})")
    print(f"Layer: {args.layer}")
    print(f"Token scope: {args.token_scope}")
    print(f"Assistant: {assistant_response}")
    print()
    for row, position_tensor in enumerate(result.positions):
        position = int(position_tensor)
        entries = [
            f"C{component.item()}={scores[row, component].item():+.3f}"
            for component in top_indices[row]
        ]
        print(f"{position:>4} {tokens[row]!r:<20} {'  '.join(entries)}")

    html_tokens = [
        {
            "position": int(position),
            "token": tokens[row],
            "token_text": tokenizer.decode([token_ids[row]]),
            "top": [
                {
                    "component": int(component),
                    "score": float(scores[row, component]),
                }
                for component in top_indices[row]
            ],
        }
        for row, position in enumerate(result.positions)
        if int(position) >= prompt_length
    ]
    conversation_text = "\n".join(
        f"{message['role'].title()}: {message['content']}" for message in messages
    )
    output_file = write_explorer_html(
        args.output_file,
        title="ICA Lens Chat Explorer",
        model=f"{lens.model_id}@{lens.model_revision}",
        layer=args.layer,
        input_text=conversation_text,
        token_scope=args.token_scope,
        tokens=html_tokens,
        token_groups=[
            {
                "title": "Prompt template",
                "open": True,
                "tokens": [
                    {
                        "position": position,
                        "token": token,
                        "token_text": tokenizer.decode([prompt_token_ids[position]]),
                        "top": [
                            {
                                "component": int(component),
                                "score": float(prompt_scores[position, component]),
                            }
                            for component in prompt_top_indices[position]
                        ],
                    }
                    for position, token in enumerate(
                        tokenizer.convert_ids_to_tokens(prompt_token_ids)
                    )
                ],
            },
        ],
    )
    print()
    print(f"HTML explorer: {output_file}")


if __name__ == "__main__":
    main()
