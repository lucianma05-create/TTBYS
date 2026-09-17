"""Third Think: strategy prediction (AG + EX).

AG branch: the LLM picks the next persuader strategy from single-letter
labels (V/L/E/T/P/A/R/I/G); probabilities come from the masked
first-token logits.

EX branch: retrieves past (intention, belief) experiences weighted by
cosine similarity and derives a strategy distribution from them.
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
from sentence_transformers import util
from tqdm import tqdm


# ------------------------------------------------------------------
# Experience database
# ------------------------------------------------------------------
def build_experience_db(raw):
    """Construct an experience DB from persuadee->persuader consecutive turns.

    Each entry contains the persuadee's intention and belief plus the
    next persuader turn's strategy annotation.
    """
    db = []
    fallback_count = 0
    for entry in raw:
        dialog = entry.get("dialog", [])
        background = entry.get("background", "")
        for i in range(len(dialog) - 1):
            cur, nxt = dialog[i], dialog[i + 1]
            if cur["speaker"] == "persuadee" and nxt["speaker"] == "persuader":
                ann = cur.get("annotation", {})
                if not ann.get("intention", "").strip():
                    fallback_count += 1
                intention = common.resolve_intention(ann, background)
                belief = ann.get("belief", "")
                strategies = nxt.get("annotation", {}).get("strategy", [])
                if intention and belief:
                    db.append({
                        "intention": intention,
                        "belief": belief,
                        "strategy": strategies,
                    })
    if fallback_count:
        print(f"Warning: {fallback_count} KB entries fall back to the dialog "
              "'background' as intention (per-turn 'intention' is empty).")
    return db


def build_kb_embeddings(ex_db):
    """Encode all KB intentions and beliefs into two embedding matrices."""
    if not ex_db:
        return None, None
    kb_int_mat = common.embed_texts([e["intention"] for e in ex_db])
    kb_belief_mat = common.embed_texts([e["belief"] for e in ex_db])
    return kb_int_mat, kb_belief_mat


# ------------------------------------------------------------------
# EX: retrieval and strategy distribution
# ------------------------------------------------------------------
def retrieve_top_k_ex(intention, belief, db, kb_int_mat, kb_belief_mat,
                      k=config.TOP_K, w_int=0.5, w_belief=0.5):
    """Retrieve top-K experience entries by weighted intention/belief
    cosine similarity (vectorized over the whole KB)."""
    int_emb = common.embed_texts(intention)
    belief_emb = common.embed_texts(belief)
    s_int = util.cos_sim(int_emb, kb_int_mat)[0]
    s_belief = util.cos_sim(belief_emb, kb_belief_mat)[0]
    scores = w_int * s_int + w_belief * s_belief
    topk = torch.topk(scores, min(k, len(db)))
    return [db[i] for i in topk.indices.tolist()]


def compute_strategy_prob_ex(topk_entries):
    """Compute a probability distribution over strategies from retrieved experiences."""
    counter = Counter()
    for entry in topk_entries:
        for strat in entry["strategy"]:
            if strat in config.STRATEGIES:
                counter[strat] += 1
    total = sum(counter.values())
    if total == 0:
        return [1 / len(config.STRATEGIES)] * len(config.STRATEGIES)
    return [counter.get(s, 0) / total for s in config.STRATEGIES]


# ------------------------------------------------------------------
# AG: strategy prediction
# ------------------------------------------------------------------
def build_prompt_ag(intention, belief):
    """Build a prompt for the AG model with strategy definitions."""
    strategy_defs = "\n".join(
        f"- {tok} = {name}"
        for tok, name in zip(config.STRATEGY_TOKENS, config.STRATEGIES)
    )
    return f"""
Persuadee's intention: {intention}
Persuadee's belief: {belief}

{strategy_defs}

Predict the next persuader strategy.
Return ONLY ONE of these single-letter labels:
{', '.join(config.STRATEGY_TOKENS)}.
Do not output anything else.
""".strip()


def get_strategy_probs_ag(prompt):
    """Predict strategy probabilities using the AG LLM.

    Only the defined single-letter tokens receive probability mass;
    all other logits are masked out.
    """
    tokenizer, model = common.get_llm()
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits[0, -1]

    strategy_token_ids = []
    for tok in config.STRATEGY_TOKENS:
        ids = tokenizer.encode(" " + tok, add_special_tokens=False)
        strategy_token_ids.append(ids[0] if ids else None)

    valid = [(i, tid) for i, tid in enumerate(strategy_token_ids) if tid is not None]
    if not valid:
        return [1 / len(config.STRATEGIES)] * len(config.STRATEGIES)

    mask = torch.full_like(logits, float("-inf"))
    for _, tid in valid:
        mask[tid] = logits[tid]
    probs = F.softmax(mask, dim=-1)
    result = [0.0] * len(config.STRATEGIES)
    for i, tid in valid:
        result[i] = float(probs[tid].item())
    s = sum(result)
    if s <= 0:
        return [1 / len(config.STRATEGIES)] * len(config.STRATEGIES)
    return [v / s for v in result]


# ------------------------------------------------------------------
# Dataset processing
# ------------------------------------------------------------------
def process_dataset(raw, max_dialogs=config.MAX_EVAL_DIALOGS):
    """Process dialogs: for each persuadee turn, compute AG and EX strategy distributions."""
    eval_data = raw[:max_dialogs]
    kb_data = raw[max_dialogs:]
    ex_db = build_experience_db(kb_data)
    print(f"EX knowledge-base size = {len(ex_db)}")

    kb_int_mat, kb_belief_mat = build_kb_embeddings(ex_db)

    processed = []
    num_predictions = 0
    total_ag_time = 0.0
    total_ex_time = 0.0

    for dialog_obj in tqdm(eval_data, desc="Processing dialogs"):
        dialog = copy.deepcopy(dialog_obj)
        for idx, turn in enumerate(dialog["dialog"]):
            if turn["speaker"] != "persuadee":
                continue
            # NOTE: intention/belief are oracle annotations here; in the online
            # pipeline (agent.py) they come from the task description and
            # Second Think respectively.
            intention = common.resolve_intention(
                turn.get("annotation", {}), dialog.get("background", "")
            )
            belief = turn.get("annotation", {}).get("belief", "")
            if not intention or not belief:
                continue

            num_predictions += 1

            # Attach predictions to the next persuader turn (the turn whose
            # strategy is being predicted).
            def next_persuader_turn():
                for next_idx in range(idx + 1, len(dialog["dialog"])):
                    if dialog["dialog"][next_idx]["speaker"] == "persuader":
                        return dialog["dialog"][next_idx]
                return None

            # EX-based prediction
            start_ex = time.time()
            if kb_int_mat is None:
                strategy_prob_ex = [1 / len(config.STRATEGIES)] * len(config.STRATEGIES)
            else:
                topk_ex = retrieve_top_k_ex(intention, belief, ex_db, kb_int_mat, kb_belief_mat)
                strategy_prob_ex = compute_strategy_prob_ex(topk_ex)
            total_ex_time += time.time() - start_ex

            # AG-based prediction
            start_ag = time.time()
            prompt_ag = build_prompt_ag(intention, belief)
            strategy_prob_ag = get_strategy_probs_ag(prompt_ag)
            total_ag_time += time.time() - start_ag

            target = next_persuader_turn()
            if target is not None:
                target["third_think_ex"] = strategy_prob_ex
                target["third_think_ag"] = strategy_prob_ag

        processed.append(dialog)

    return processed, num_predictions, total_ag_time, total_ex_time


# ------------------------------------------------------------------
# Main entry
# ------------------------------------------------------------------
def main(args):
    print("Loading dataset...")
    with open(args.data, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    print("Processing dialogs...")
    start_time = time.time()
    result, num_predictions, total_ag_time, total_ex_time = process_dataset(
        raw_data, max_dialogs=args.max_dialogs
    )
    total_time = time.time() - start_time

    print("Saving output...")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Done. Saved to {args.output}")

    avg_time_per_prediction = total_time / num_predictions if num_predictions > 0 else 0

    print("\n========== Runtime Statistics ==========")
    print(f"Total time for all dialogs: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
    print(f"Total AG-only inference time: {total_ag_time:.2f} seconds")
    print(f"Total EX-based retrieval & strategy time: {total_ex_time:.2f} seconds")
    print(f"Total predictions (persuadee turns with intention+belief): {num_predictions}")
    print(f"Average time per prediction: {avg_time_per_prediction:.4f} seconds")
    print("========================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Third Think: strategy prediction (AG + EX).")
    parser.add_argument("--data", type=str, default=str(config.DATA_PATH))
    parser.add_argument("--output", type=str, default=str(config.THIRD_THINK_OUTPUT))
    parser.add_argument("--max-dialogs", type=int, default=config.MAX_EVAL_DIALOGS)
    main(parser.parse_args())
