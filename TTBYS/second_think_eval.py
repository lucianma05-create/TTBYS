import json
import requests
from tqdm import tqdm

# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------
API_HOST = ""       # API endpoint
API_KEY = ""        # API key for authentication
MODEL_FIELD = "pre_belief"  # Field in data containing predicted beliefs
INPUT_FILE = ""     # Input dataset path


# -------------------------------------------------------------
# GPT-based belief scoring
# -------------------------------------------------------------
def call_gpt_belief_score(gt_belief, pred_belief):
    """
    Call GPT API to evaluate the accuracy of predicted beliefs.

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

    headers = {"Authorization": f"Bearer {API_KEY}"}
    body = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 5
    }

    r = requests.post(f"{API_HOST}/chat/completions", headers=headers, json=body)
    r.raise_for_status()
    output = r.json()["choices"][0]["message"]["content"].strip()

    try:
        return float(output)
    except:
        return 0.0


# -------------------------------------------------------------
# Statistics storage
# -------------------------------------------------------------
stats = {
    "belief_scores": []
}


# -------------------------------------------------------------
# Dialog evaluation
# -------------------------------------------------------------
def evaluate_dialog(dialog):
    """
    Evaluate all persuadee turns in a dialog for belief prediction accuracy.
    """
    for turn in dialog:
        if turn["speaker"] == "persuadee":
            gt_belief = turn["annotation"].get("belief", "")
            pred_belief = turn.get(MODEL_FIELD, "")

            if gt_belief and pred_belief:
                score = call_gpt_belief_score(gt_belief, pred_belief)
            else:
                score = 0.0

            stats["belief_scores"].append(score)


# -------------------------------------------------------------
# Main entry
# -------------------------------------------------------------
def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Processing {len(data)} dialogs...\n")

    for sample in tqdm(data, desc="Evaluating dialogs"):
        evaluate_dialog(sample["dialog"])

    # --- Output summary ---
    belief_acc = sum(stats["belief_scores"]) / len(stats["belief_scores"]) if stats["belief_scores"] else 0
    print(f"\nBelief Accuracy (mean score): {belief_acc:.3f}")


if __name__ == "__main__":
    main()
