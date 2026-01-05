import json
import numpy as np
import matplotlib.pyplot as plt

# =============================
# Configuration
# =============================
NUM = 400

DATA_PATH = "../data/ToM_BPD.json"     # Input data path
OUTPUT_FIG = "fist_think_acc.png"      # Output figure path

# Desire label mapping: index -> desire value
IDX2DESIRE = {0: -1, 1: 0, 2: 1}

# =============================
# Load data
# =============================
def load_data(path):
    """
    Load annotated desire probabilities from the dataset.
    Each sample contains ground-truth desire and two predicted distributions.
    """
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


# =============================
# Compute fused prediction accuracy
# =============================
def calc_accuracy(samples, alpha):
    """
    Compute accuracy using linear fusion of two probability distributions.
    """
    correct = 0
    total = 0

    for gt, ag_p, ex_p in samples:
        fused = alpha * ag_p + (1 - alpha) * ex_p
        pred_idx = np.argmax(fused)
        pred = IDX2DESIRE[pred_idx]

        if pred == gt:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


# =============================
# Main process
# =============================
def main():
    samples = load_data(DATA_PATH)
    print(f"Loaded {len(samples)} annotated samples.")

    alphas = np.arange(0, 1.01, 0.1)
    accuracies = []

    for a in alphas:
        acc = calc_accuracy(samples, a)
        accuracies.append(acc)
        print(f"alpha={a:.1f}  accuracy={acc:.4f}")

    # Plot accuracy curve
    plt.figure(figsize=(8, 5))
    plt.plot(alphas, accuracies, marker='o')
    plt.xlabel("Fusion Parameter α")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Fusion Parameter α")
    plt.grid(True)
    plt.savefig(OUTPUT_FIG, dpi=300)
    print(f"\nAccuracy curve saved to: {OUTPUT_FIG}\n")


if __name__ == "__main__":
    main()
