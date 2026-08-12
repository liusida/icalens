"""The unified ICA Lens command-line interface."""

from __future__ import annotations

import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    """Dispatch an ICA Lens subcommand."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(_TOP_HELP)
        return

    command = args.pop(0)
    if command == "fit":
        _dispatch_fit(args)
    elif command == "publish":
        from .publish import main as publish_main

        publish_main(args)
    elif command == "profile":
        from .profile import main as profile_main

        profile_main(args)
    elif command == "smoke-test":
        from ..smoke_test import main as smoke_test_main

        smoke_test_main(args)
    else:
        raise SystemExit(f"icalens: unknown command {command!r}; run 'icalens --help'")


def _dispatch_fit(args: list[str]) -> None:
    if not args or args[0] in {"-h", "--help"}:
        print(_FIT_HELP)
        return
    kind = args.pop(0)
    if kind == "text":
        from .fit_text import main as fit_text_main

        fit_text_main(args)
    elif kind == "chat":
        from .fit_chat import main as fit_chat_main

        fit_chat_main(args)
    else:
        raise SystemExit(f"icalens fit: unknown input type {kind!r}; use 'text' or 'chat'")


_TOP_HELP = """usage: icalens COMMAND [OPTIONS]

Fit, profile, publish, and verify ICA Lens artifacts.

commands:
  fit text     Fit from a raw-text dataset
  fit chat     Fit from a conversation dataset
  profile      Add component statistics, examples, and logit-lens tokens
  publish      Publish a local artifact to Hugging Face
  smoke-test   Verify installed text and chat analysis paths

Run 'icalens COMMAND --help' for command-specific options."""

_FIT_HELP = """usage: icalens fit {text,chat} [OPTIONS]

Fit an ICA Lens from model activations.

input types:
  text         Raw-text dataset fitting
  chat         Conversation dataset fitting

Run 'icalens fit INPUT_TYPE --help' for fitting options."""
