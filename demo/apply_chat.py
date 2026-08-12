"""Generate a chat response and apply a saved ICA Lens to its activations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from gb10_load_llm import load_model_to_cuda
from transformers import AutoModelForCausalLM, AutoTokenizer

from icalens import ICALens
from icalens.html import write_explorer_html

DEFAULT_LENS = Path(__file__).parent / "output" / "icalens-qwen2.5-0.5b-instruct"
DEFAULT_USER = "Explain why the sky appears blue in one sentence."
CONTEXT_LENGTH = 1_024
DEFAULT_OUTPUT_FILE = Path(__file__).parent / "output" / "apply_chat.html"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lens", type=Path, default=DEFAULT_LENS)
    parser.add_argument("--system", default=None, help="Optional system message.")
    parser.add_argument(
        "--user",
        action="append",
        type=parse_user_argument,
        default=None,
        metavar="TEXT_OR_JSON_LIST",
        help=(
            "User turn; repeat for multiple turns, or pass a quoted JSON list "
            f"(default: {DEFAULT_USER!r})."
        ),
    )
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
    parser.add_argument(
        "--metric",
        choices=("score", "energy"),
        default="score",
        help="Rank and display signed ICA scores or per-token energy shares.",
    )
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    return parser.parse_args()


def parse_user_argument(value: str) -> tuple[str, ...]:
    """Parse one user message or a JSON list supplied to --user."""
    if not value.lstrip().startswith("["):
        if not value:
            raise argparse.ArgumentTypeError("--user messages must not be empty")
        return (value,)
    try:
        messages = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(f"invalid --user JSON list: {error.msg}") from error
    if not isinstance(messages, list) or not messages:
        raise argparse.ArgumentTypeError("--user JSON must be a non-empty list")
    if not all(isinstance(message, str) and message for message in messages):
        raise argparse.ArgumentTypeError("--user JSON entries must be non-empty strings")
    return tuple(messages)


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

    user_turns = tuple(
        turn for argument in (args.user or [(DEFAULT_USER,)]) for turn in argument
    )
    messages = []
    if args.system is not None:
        messages.append({"role": "system", "content": args.system})

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

    assistant_responses = []
    for turn_number, user_message in enumerate(user_turns, start=1):
        messages.append({"role": "user", "content": user_message})
        rendered_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt = tokenizer(
            rendered_prompt,
            add_special_tokens=False,
            return_tensors="pt",
        )
        prompt_length = int(prompt["input_ids"].shape[1])
        if prompt_length + args.max_new_tokens > args.context_length:
            raise ValueError(
                f"turn {turn_number} prompt has {prompt_length} tokens, leaving fewer than "
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
        response_ids = generated[0, prompt_length:].to("cpu")
        assistant_response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        if not assistant_response:
            raise RuntimeError(f"Model generated an empty response at turn {turn_number}.")
        assistant_responses.append(assistant_response)
        messages.append({"role": "assistant", "content": assistant_response})

    result = lens.analyze(
        messages,
        layer=args.layer,
        model=model,
        tokenizer=tokenizer,
        token_scope=args.token_scope,
        context_length=args.context_length,
    )
    values = result.scores if args.metric == "score" else result.energy
    ranking_values = values.abs() if args.metric == "score" else values
    top_k = min(args.top_k, values.shape[-1])
    top_indices = torch.topk(ranking_values, k=top_k, dim=-1).indices
    token_ids = result.token_ids.tolist()
    tokens = result.tokens

    print(f"Lens: {args.lens}")
    print(f"Model: {lens.model_id}@{lens.model_revision} ({lens.model_type})")
    print(f"Layer: {args.layer}")
    print(f"Token scope: {args.token_scope}")
    print(f"Metric: {args.metric}")
    for turn_number, (user_message, assistant_response) in enumerate(
        zip(user_turns, assistant_responses, strict=True), start=1
    ):
        print(f"User {turn_number}: {user_message}")
        print(f"Assistant {turn_number}: {assistant_response}")
    print()
    for row, position_tensor in enumerate(result.positions):
        position = int(position_tensor)
        entries = [
            f"C{component.item()}={format_value(values[row, component].item(), args.metric)}"
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
                    "score": float(values[row, component]),
                }
                for component in top_indices[row]
            ],
        }
        for row, position in enumerate(result.positions)
    ]
    message_groups = group_tokens_by_message(
        tokenizer,
        messages=messages,
        context_length=args.context_length,
        tokens=html_tokens,
    )
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
        tokens=[],
        metric=args.metric,
        messages=messages,
        token_groups=message_groups,
    )
    print()
    print(f"HTML explorer: {output_file}")


def format_value(value: float, metric: str) -> str:
    return f"{value:+.3f}" if metric == "score" else f"{value:.2%}"


def group_tokens_by_message(
    tokenizer: Any,
    *,
    messages: list[dict[str, str]],
    context_length: int,
    tokens: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Group formatted token cards by the message that introduced them."""
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        truncation=True,
        max_length=context_length,
        return_offsets_mapping=True,
    )
    input_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    input_tokens = tokenizer.convert_ids_to_tokens(input_ids)

    message_starts = []
    visible_messages = []
    cursor = 0
    for message in messages:
        content_start = rendered.find(message["content"], cursor)
        if content_start < 0:
            raise ValueError("could not align a message with the rendered chat template")
        content_end = content_start + len(message["content"])
        cursor = content_end
        content_token = next(
            (index for index, (start, end) in enumerate(offsets) if end > content_start),
            None,
        )
        if content_token is None:
            break
        opening_candidates = [
            index
            for index in range(content_token + 1)
            if (
                "start" in input_tokens[index].lower()
                or "begin" in input_tokens[index].lower()
            )
            and (not message_starts or index > message_starts[-1])
        ]
        message_starts.append(opening_candidates[-1] if opening_candidates else content_token)
        visible_messages.append(message)

    end_positions = [*message_starts[1:], len(input_ids)]

    role_counts = {"system": 0, "user": 0, "assistant": 0}
    groups = []
    for message, start, end in zip(
        visible_messages, message_starts, end_positions, strict=True
    ):
        role = message["role"]
        role_counts[role] += 1
        title = role.title() if role == "system" else f"{role.title()} {role_counts[role]}"
        group_tokens = [token for token in tokens if start <= int(token["position"]) < end]
        if group_tokens:
            groups.append({"title": title, "open": True, "tokens": group_tokens})
    return groups


if __name__ == "__main__":
    main()
