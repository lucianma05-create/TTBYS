"""First Think: desire inference (AG + EX).

AG branch: the LLM classifies the persuadee's desire from the dialog
history; probabilities come from the first-token logits of " A/B/C".

EX branch: retrieves past persuadee turns with similar intentions from
an experience KB and derives a desire distribution from their labels.
"""
import argparse
import copy
import json
import time
from collections import Counter
from pathlib import Path

# Import config before torch: it sets CUDA_VISIBLE_DEVICES.
import common
import config

import torch
import torch.nn.functional as F
from tqdm import tqdm


# ------------------------------------------------------------------
# AG: prompt construction and first-token probability estimation
# ------------------------------------------------------------------
def build_prompt(dialog_history):
    """Build a desire-classification prompt from the dialog history."""
    text_blocks = []
    for turn in dialog_history:
        speaker = "persuader" if turn["speaker"] == "persuader" else "persuadee"
        text_blocks.append(f"{speaker}: {turn['content']}")
    context = "\n".join(text_blocks)
    instruction = (
        "\nBased on the above conversation, please classify the persuadee's desire.\n"
        "Output only one token: A (unwilling), B (uncertain), or C (willing).\n"
        "Your output should ONLY be A, B, or C."
    )
    return context + instruction


def get_first_token_probs(prompt):
    """Compute first-token probabilities for A/B/C from the AG model."""
    tokenizer, model = common.get_llm()
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**encoded)
    probs = F.softmax(out.logits[0, -1], dim=-1)

    result = {}
    for desire, token_str in config.DESIRE_TOKEN_MAP.items():
        token_ids = tokenizer.encode(token_str, add_special_tokens=False)
        result[desire] = float(probs[token_ids[0]].item()) if token_ids else 0.0
    return result


# ------------------------------------------------------------------
# EX: one-turn intention–desire knowledge base
# ------------------------------------------------------------------
def build_one_turn_db(raw):
    """Build a one-turn intention–desire KB from dialogs.

    Each persuadee turn (with annotated intention/desire) is paired with
    the strategies of the next persuader turn.
    """
    db = []
    fallback_count = 0
    for entry in tqdm(raw, desc="Building one-turn KB"):
        dialog = entry.get("dialog", [])
        background = entry.get("background", "")
        for i in range(len(dialog) - 1):
            cur, nxt = dialog[i], dialog[i + 1]
            if cur["speaker"] == "persuadee" and nxt["speaker"] == "persuader":
                ann = cur.get("annotation", {})
                if not ann.get("intention", "").strip():
                    fallback_count += 1
                intention = common.resolve_intention(ann, background)
                desire = ann.get("desire", None)
                strategies = nxt.get("annotation", {}).get("strategy", [])
                if intention and desire in config.DESIRE_ORDER:
                    db.append({
                        "intention": intention,
                        "desire": desire,
                        "strategy": strategies,
                    })
    if fallback_count:
        print(f"Warning: {fallback_count} KB entries fall back to the dialog "
              "'background' as intention (per-turn 'intention' is empty).")
    return db


def retrieve_top_k(query_intention, db, kb_embeddings, k=config.TOP_K):
    """Retrieve the top-k KB entries with intentions most similar to the query."""
    query_emb = common.embed_texts(query_intention)
    return [db[i] for i in common.cosine_topk(query_emb, kb_embeddings, k)]


def compute_desire_prob(matches):
    """Compute a desire distribution from retrieved KB matches."""
    if not matches:
        return [1 / 3, 1 / 3, 1 / 3]
    counter = Counter(m["desire"] for m in matches)
    total = sum(counter.values())
    return [
        counter.get(-1, 0) / total,
        counter.get(0, 0) / total,
        counter.get(1, 0) / total,
    ]


# ------------------------------------------------------------------
# Dataset processing
# ------------------------------------------------------------------
def process_dataset(raw, max_dialogs=config.MAX_EVAL_DIALOGS):
    """Run AG and EX desire prediction over the evaluation dialogs."""
    eval_data = raw[:max_dialogs]
    kb_data = raw[max_dialogs:]

    # Build EX knowledge base
    one_turn_db = build_one_turn_db(kb_data)
    print(f"EX knowledge-base size = {len(one_turn_db)}")

    kb_intentions = [d["intention"] for d in one_turn_db]
    kb_embeddings = common.embed_texts(kb_intentions) if kb_intentions else None

    processed = []
    num_predictions = 0
    ag_time = 0.0

    for dialog_obj in tqdm(eval_data, desc="Processing dialogs"):
        dialog = copy.deepcopy(dialog_obj)
        dialog_history = []

        for turn in dialog["dialog"]:
            dialog_history.append({
                "speaker": turn["speaker"],
                "content": turn["content"],
            })

            if turn["speaker"] != "persuadee":
                continue

            num_predictions += 1

            # -------- AG prediction --------
            ag_start = time.time()

            prompt = build_prompt(dialog_history)
            raw_probs = get_first_token_probs(prompt)
            norm_probs = common.normalize_probs(raw_probs)

            ag_time += time.time() - ag_start

            turn["ag_desire_prob"] = [
                norm_probs.get(-1, 0.0),
                norm_probs.get(0, 0.0),
                norm_probs.get(1, 0.0),
            ]

            # -------- EX prediction --------
            # NOTE: the EX branch uses the annotated intention (oracle). In the
            # online pipeline (agent.py) the task description plays this role.
            intention = common.resolve_intention(
                turn.get("annotation", {}), dialog.get("background", "")
            )
            if not intention or kb_embeddings is None:
                turn["ex_desire_prob"] = [1 / 3, 1 / 3, 1 / 3]
            else:
                matches = retrieve_top_k(intention, one_turn_db, kb_embeddings)
                turn["ex_desire_prob"] = compute_desire_prob(matches)

        processed.append(dialog)

    return processed, num_predictions, ag_time


# ------------------------------------------------------------------
# Main entry
# ------------------------------------------------------------------
def main(args):
    print("Loading dataset...")
    with open(args.data, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    print("Processing dataset...")
    start_time = time.time()

    result, num_predictions, ag_time = process_dataset(raw_data, max_dialogs=args.max_dialogs)

    total_time = time.time() - start_time

    print("Saving merged output...")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Done. Saved to {args.output}")

    avg_time_per_prediction = total_time / num_predictions if num_predictions > 0 else 0
    avg_ag_time_per_prediction = ag_time / num_predictions if num_predictions > 0 else 0

    print("\n========== Runtime Statistics ==========")
    print(f"Total dialogs processed: {len(result)}")
    print(f"Total predictions (persuadee turns): {num_predictions}")
    print(f"Total processing time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
    print(f"Average time per prediction (overall): {avg_time_per_prediction:.4f} seconds")
    print(f"AG-only total time (LLM only): {ag_time:.2f} seconds")
    print(f"AG-only average time per prediction: {avg_ag_time_per_prediction:.4f} seconds")
    print("========================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="First Think: desire inference (AG + EX).")
    parser.add_argument("--data", type=str, default=str(config.DATA_PATH))
    parser.add_argument("--output", type=str, default=str(config.FIRST_THINK_OUTPUT))
    parser.add_argument("--max-dialogs", type=int, default=config.MAX_EVAL_DIALOGS)
    main(parser.parse_args())
