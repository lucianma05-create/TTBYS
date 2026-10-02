"""Evaluate Second Think: GPT-based belief scoring.

Reads the output of second_think.py and asks a GPT API to score each
predicted belief against the ground truth (0 / 0.5 / 1).
"""
import argparse
import json
import os

import requests
from tqdm import tqdm

import config

MODEL_FIELD = "pred_belief"  # field in the result file containing predicted beliefs


def call_gpt_belief_score(gt_belief, pred_belief, api_host, api_key, model):
    """Call a GPT API to evaluate the accuracy of a predicted belief.

    Scoring rules:
    1. Perfect match of positive & negative beliefs: 1
    2. Correct mentions but partial reasoning: 0.5
    3. Both incorrect: 0
    4. Single belief in ground truth:
        - Correct match: 0.5
        - Otherwise: 0
    """
    prompt = f"""
You are an evaluator. Evaluate belief prediction accuracy based on the following rules:

1. If the predicted positive and negative beliefs fully match the ground truth, score = 1.
2. If both positive and negative beliefs are mentioned but the underlying reasons are not fully correct, score = 0.5.
3. If both are incorrect, score = 0.
4. If the ground truth belief only contains a positive OR only a negative belief:
    - If the prediction matches, score = 0.5.
    - Otherwise, score = 0.

Ground truth belief:
{gt_belief}

Predicted belief:
{pred_belief}

Output ONLY a number in {{0, 0.5, 1}}.
    """

    headers = {"Authorization": f"Bearer {api_key}"}
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 5,
    }

    try:
        r = requests.post(f"{api_host}/chat/completions", headers=headers, json=body, timeout=60)
        r.raise_for_status()
        output = r.json()["choices"][0]["message"]["content"].strip()
        score = float(output)
        if score not in {0.0, 0.5, 1.0}:
            raise ValueError(f"judge returned an invalid score: {output!r}")
        return score
    except (requests.RequestException, ValueError, KeyError, IndexError) as e:
        print(f"Warning: scoring request failed ({e}); excluding this sample")
        return None


def evaluate_dialog(dialog, stats, args):
    """Evaluate all persuadee turns in a dialog for belief prediction accuracy."""
    for turn in dialog:
        if turn["speaker"] == "persuadee":
            gt_belief = turn["annotation"].get("belief", "")
            pred_belief = turn.get(MODEL_FIELD, "")

            if gt_belief and pred_belief:
                score = call_gpt_belief_score(
                    gt_belief, pred_belief, args.api_host, args.api_key, args.model
                )
                if score is None:
                    stats["failed_requests"] += 1
                    continue
            else:
                score = 0.0

            stats["belief_scores"].append(score)


def main(args):
    if not args.api_host:
        raise SystemExit(
            "No API host set. Pass --api-host or fill EVAL_API_HOST in config.py."
        )
    if not args.api_key:
        raise SystemExit(
            "No API key set. Pass --api-key or fill EVAL_API_KEY in config.py."
        )

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Processing {len(data)} dialogs...\n")

    stats = {"belief_scores": [], "failed_requests": 0}

    for sample in tqdm(data, desc="Evaluating dialogs"):
        evaluate_dialog(sample["dialog"], stats, args)

    # --- Output summary ---
    belief_acc = (
        sum(stats["belief_scores"]) / len(stats["belief_scores"])
        if stats["belief_scores"] else 0
    )
    print(f"\nBelief Accuracy (mean score): {belief_acc:.3f}")
    print(f"Scored samples: {len(stats['belief_scores'])}")
    print(f"Excluded failed API requests: {stats['failed_requests']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Second Think (belief inference) with a GPT judge."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(config.SECOND_THINK_OUTPUT),
        help="Output file of second_think.py",
    )
    parser.add_argument(
        "--api-host",
        type=str,
        default=config.EVAL_API_HOST,
        help='API endpoint, e.g. "https://api.openai.com/v1"',
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=config.EVAL_API_KEY or os.environ.get("EVAL_API_KEY", ""),
        help="API key (or set the EVAL_API_KEY environment variable)",
    )
    parser.add_argument("--model", type=str, default=config.EVAL_MODEL)
    main(parser.parse_args())
