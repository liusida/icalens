"""Publish a local ICA Lens artifact to a Hugging Face Model repository."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

from icalens import ICALens

DEFAULT_LENS = Path(__file__).parent / "output" / "icalens-gpt2-small"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "repo_id", help="Destination Model repository, for example sida/icalens-gpt2-small."
    )
    parser.add_argument("--lens", type=Path, default=DEFAULT_LENS)
    parser.add_argument("--private", action="store_true", help="Create a private repository.")
    parser.add_argument("--revision", default="main", help="Destination branch (default: main).")
    parser.add_argument(
        "--commit-message",
        default="Upload ICA Lens artifacts",
        help="Hugging Face commit message.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lens_path = args.lens.expanduser().resolve()
    lens = ICALens.from_pretrained(lens_path)

    api = HfApi()
    try:
        account = api.whoami()
    except Exception as error:
        raise RuntimeError(
            "Hugging Face authentication failed. Run `hf auth login` or set HF_TOKEN."
        ) from error
    username = account.get("name", "unknown")
    print(f"Authenticated with Hugging Face as {username}.")
    print(f"Publishing {lens_path} to {args.repo_id}@{args.revision}...")

    result = lens.push_to_hub(
        args.repo_id,
        private=True if args.private else None,
        revision=args.revision,
        commit_message=args.commit_message,
    )
    cloud = ICALens.from_pretrained(
        args.repo_id,
        revision=args.revision,
        force_download=True,
    )
    if cloud.metadata != lens.metadata:
        raise RuntimeError("Uploaded manifest does not match the local artifact manifest.")
    print(f"Verified uploaded manifest and available layers {cloud.available_layers}.")
    print(result)


if __name__ == "__main__":
    main()
