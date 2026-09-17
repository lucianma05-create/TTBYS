"""Evaluate Third Think: accuracy of AG/EX strategy fusion vs alpha.

Reads the output of third_think.py (persuader turns annotated with
third_think_ag / third_think_ex) and plots accuracy of
fused = alpha * ag + (1 - alpha) * ex for alpha in [0, 1].
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import config


def load_data(path):
    """Load persuader turns with ground-truth strategies and AG/EX predictions."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for conv in data:
        for turn in conv["dialog"]:
            if turn["speaker"] == "persuader":
                gt_strategies = turn["annotation"].get("strategy", [])
                ag_prob = turn.get("third_think_ag", [])
                ex_prob = turn.get("third_think_ex", [])
                if gt_strategies and ag_prob and ex_prob:
                    samples.append((gt_strategies, np.array(ag_prob), np.array(ex_prob)))
    return samples


def calc_accuracy(samples, alpha):
    """Compute accuracy after fusing AG and EX strategy probabilities with weight alpha."""
    correct = 0
    total = 0
    for gt_strategies, ag_prob, ex_prob in samples:
        fused = alpha * ag_prob + (1 - alpha) * ex_prob
        pred_strategy = config.STRATEGIES[int(np.argmax(fused))]
        if pred_strategy in gt_strategies:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0


def main(args):
    samples = load_data(args.result)
    print(f"Loaded {len(samples)} persuader turns for evaluation.\n")

    alphas = np.arange(0, 1.01, 0.1)
    accuracies = []

    for a in alphas:
        acc = calc_accuracy(samples, a)
        accuracies.append(acc)
        print(f"alpha={a:.1f}  accuracy={acc:.4f}")

    # Plot accuracy curve
    plt.figure(figsize=(8, 5))
    plt.plot(alphas, accuracies, marker="o")
    plt.xlabel("Fusion Parameter α")
    plt.ylabel("Strategy Prediction Accuracy")
    plt.title("Accuracy vs Fusion Parameter α")
    plt.grid(True)
    Path(args.out_fig).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out_fig, dpi=300)
    print(f"\nAccuracy curve saved to: {args.out_fig}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Third Think (strategy prediction) fusion accuracy."
    )
    parser.add_argument(
        "--result",
        type=str,
        default=str(config.THIRD_THINK_OUTPUT),
        help="Output file of third_think.py",
    )
    parser.add_argument(
        "--out-fig",
        type=str,
        default=str(config.OUTPUT_DIR / "third_think_acc.png"),
        help="Output figure path",
    )
    main(parser.parse_args())
