import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# =============================
# Configuration
# =============================
NUM = 400
DATA_PATH = ""  # Input dataset path
OUTPUT_FIG = "third_think_acc.png"  # Output figure path

STRATEGIES = [
    "Expression of views",
    "Logical appeal",
    "Enhancement of views",
    "Task inquiry",
    "Personal story",
    "Affirmation and reassurance",
    "Reflection of feelings",
    "Supplying information",
    "Giving Examples"
]

# =============================
# Load data
# =============================
def load_data(path):
    """
    Load persuader turns with ground truth strategies and AG/EX predictions.
    """
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

# =============================
# Compute fusion accuracy
# =============================
def calc_accuracy(samples, alpha):
    """
    Compute accuracy after fusing AG and EX strategy probabilities with weight alpha.
    """
    correct = 0
    total = 0
    for gt_strategies, ag_prob, ex_prob in samples:
        fused = alpha * ag_prob + (1 - alpha) * ex_prob
        pred_idx = np.argmax(fused)
        pred_strategy = STRATEGIES[pred_idx]
        if pred_strategy in gt_strategies:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0

# =============================
# Main workflow
# =============================
def main():
    samples = load_data(DATA_PATH)
    print(f"Loaded {len(samples)} persuader turns for evaluation.\n")

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
    plt.ylabel("Strategy Prediction Accuracy")
    plt.title("Accuracy vs Fusion Parameter α")
    plt.grid(True)
    plt.savefig(OUTPUT_FIG, dpi=300)
    print(f"\nAccuracy curve saved to: {OUTPUT_FIG}\n")

if __name__ == "__main__":
    main()
