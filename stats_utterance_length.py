#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute average utterance length in tokens for persuader and persuadee "
            "from the ToM_BPD dataset."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "ToM_BPD.json",
        help="Path to the dataset JSON file.",
    )
    parser.add_argument(
        "--hf-tokenizer",
        type=str,
        default=None,
        help=(
            "Optional Hugging Face tokenizer name or local path. "
            "If omitted, whitespace tokenization is used."
        ),
    )
    return parser.parse_args()


def build_token_counter(hf_tokenizer_name):
    if hf_tokenizer_name is None:
        return lambda text: len(text.split())

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_name, trust_remote_code=True)
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False))


def main():
    args = parse_args()
    with args.data.open("r", encoding="utf-8") as f:
        data = json.load(f)

    count_tokens = build_token_counter(args.hf_tokenizer)
    stats = defaultdict(lambda: {"utterances": 0, "tokens": 0})

    for sample in data:
        for turn in sample.get("dialog", []):
            speaker = turn.get("speaker")
            content = turn.get("content", "")

            if speaker not in {"persuader", "persuadee"}:
                continue

            stats[speaker]["utterances"] += 1
            stats[speaker]["tokens"] += count_tokens(content)

    for speaker in ("persuader", "persuadee"):
        utterances = stats[speaker]["utterances"]
        tokens = stats[speaker]["tokens"]
        avg = tokens / utterances if utterances else 0.0
        print(
            f"{speaker}: utterances={utterances}, total_tokens={tokens}, "
            f"avg_utterance_length={avg:.4f}"
        )


if __name__ == "__main__":
    main()
