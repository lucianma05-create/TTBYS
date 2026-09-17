"""Evaluate First Think: accuracy of AG/EX desire fusion vs alpha.

Reads the output of first_think.py (turns annotated with
ag_desire_prob / ex_desire_prob) and plots accuracy of
fused = alpha * ag + (1 - alpha) * ex for alpha in [0, 1].
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import config

# Desire label mapping: index in the probability list -> desire value
IDX2DESIRE = {0: -1, 1: 0, 2: 1}


def load_data(path):
    """Load (ground-truth desire, AG prob, EX prob) samples from the result file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for conv in data:
        for turn in conv["dialog"]:
            if "ag_desire_prob" in turn and "ex_desire_prob" in turn:
                gt = turn["annotation"]["desire"]
                ag = np.array(turn["ag_desire_prob"], dtype=float)
                ex = np.array(turn["ex_desire_prob"], dtype=float)
                samples.append((gt, ag, ex))

    return samples


def calc_accuracy(samples, alpha):
    """Compute accuracy using linear fusion of the two distributions."""
    correct = 0
    total = 0

    for gt, ag_p, ex_p in samples:
        fused = alpha * ag_p + (1 - alpha) * ex_p
        pred = IDX2DESIRE[int(np.argmax(fused))]

        if pred == gt:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def main(args):
    samples = load_data(args.result)
    print(f"Loaded {len(samples)} annotated samples.")

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
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Fusion Parameter α")
    plt.grid(True)
    Path(args.out_fig).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out_fig, dpi=300)
    print(f"\nAccuracy curve saved to: {args.out_fig}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate First Think (desire inference) fusion accuracy."
    )
    parser.add_argument(
        "--result",
        type=str,
        default=str(config.FIRST_THINK_OUTPUT),
        help="Output file of first_think.py",
    )
    parser.add_argument(
        "--out-fig",
        type=str,
        default=str(config.OUTPUT_DIR / "first_think_acc.png"),
        help="Output figure path",
    )
    main(parser.parse_args())
