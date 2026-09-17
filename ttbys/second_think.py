"""Second Think: belief inference.

Retrieves past experiences (persuadee turns with the same annotated
desire) whose conversation context is most similar to the current one,
then asks the LLM to generate the persuadee's underlying belief.
"""
import argparse
import copy
import json
import re
import time
from pathlib import Path

from tqdm import tqdm

import common
import config


# ------------------------------------------------------------------
# Experience database
# ------------------------------------------------------------------
def build_experience_db(raw):
    """Extract persuadee turns with desire and belief into a one-turn experience DB."""
    db = []
    for entry in raw:
        dialog = entry.get("dialog", [])
        for i, turn in enumerate(dialog):
            if turn["speaker"] != "persuadee":
                continue
            ann = turn.get("annotation", {})
            desire = ann.get("desire", None)
            belief = ann.get("belief", "")
            if desire is None or not belief:
                continue
            # Context = conversation up to and including the current turn.
            history = dialog[: i + 1]
            history_text = "\n".join(f"{t['speaker']}: {t['content']}" for t in history)
            db.append({
                "desire": desire,
                "context": history_text,
                "current_belief": belief,
            })
    return db


# ------------------------------------------------------------------
# Retrieval
# ------------------------------------------------------------------
def retrieve_top_k(current_desire, query_text, db, db_embeddings, k=config.TOP_K):
    """Retrieve the top-k experiences that share the current desire and
    whose context is most similar to the query text (the current
    conversation history)."""
    filtered_idx = [i for i, d in enumerate(db) if d["desire"] == current_desire]
    if not filtered_idx or db_embeddings is None:
        return []
    query_emb = common.embed_texts(query_text)
    topk = common.cosine_topk(query_emb, db_embeddings[filtered_idx], k)
    return [db[filtered_idx[i]] for i in topk]


# ------------------------------------------------------------------
# Prompt construction and LLM inference
# ------------------------------------------------------------------
def build_prompt(task_description, top5_exp, history_text, current_turn):
    """Build a prompt with few-shot examples and top-K retrieved experiences."""
    exp_texts = [
        f"Experience {i}:\n{exp['context']}\nCurrent belief: {exp['current_belief']}"
        for i, exp in enumerate(top5_exp, 1)
    ]
    exp_block = "\n\n".join(exp_texts) if exp_texts else "No relevant experiences retrieved."

    few_shot = (
        "### Belief Style Examples\n"
        "Positive: ... is important/interesting/beneficial/... \n"
        "Negative: uncertain/wary about ... or something is harmful...\n"
        "### Tips: If desire is 0, include both positive & negative aspects. "
        "If desire is 1, primarily positive. If desire is -1, primarily negative.\n"
    )

    prompt = (
        f"Task Description:\n{task_description}\n\n"
        f"Top-{len(top5_exp)} relevant experiences:\n{exp_block}\n\n"
        f"Current conversation:\n{history_text}\n"
        f"Current desire: {current_turn.get('annotation', {}).get('desire', 'unknown')}\n\n"
        f"{few_shot}\n"
        f"Current belief (Do not output anything else):\n"
    )
    return prompt


def extract_belief(raw_text):
    """Extract the belief text from the model output.

    The model is asked to output only the belief; if it still echoes the
    "Current belief:" prefix, strip that. Otherwise take the first
    non-empty line.
    """
    lines = [line.strip() for line in raw_text.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    first = lines[0]
    match = re.search(r"current belief\s*:?\s*(.*)", first, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return first


def generate_belief(prompt, max_new_tokens=32):
    """Generate the current belief from the LLM given the prompt."""
    tokenizer, model = common.get_llm()
    raw_text = common.generate_text(tokenizer, model, prompt, max_new_tokens=max_new_tokens)
    return extract_belief(raw_text)


# ------------------------------------------------------------------
# Dataset processing
# ------------------------------------------------------------------
def process_dataset(raw, max_dialogs=config.MAX_EVAL_DIALOGS):
    """Run second-think belief inference on the evaluation dialogs."""
    eval_data = raw[:max_dialogs]
    db = build_experience_db(raw[max_dialogs:])
    print(f"Experience DB size: {len(db)}")

    db_contexts = [d.get("context", "") + " " + d.get("current_belief", "") for d in db]
    db_embeddings = common.embed_texts(db_contexts) if db_contexts else None

    processed = []

    total_time = 0.0
    total_retrieval_time = 0.0
    total_llm_time = 0.0

    start_all = time.time()

    for dialog_obj in tqdm(eval_data, desc="Processing dialogs"):
        dialog = copy.deepcopy(dialog_obj)

        for idx, turn in enumerate(dialog["dialog"]):
            if turn["speaker"] != "persuadee":
                continue

            # Include the current turn so the LLM sees what the persuadee just said.
            history = dialog["dialog"][: idx + 1]
            history_text = "\n".join(f"{t['speaker']}: {t['content']}" for t in history)

            # NOTE: the annotated desire is an oracle here; in the online
            # pipeline (agent.py) the predicted desire from First Think is used.
            current_desire = turn.get("annotation", {}).get("desire", None)
            if current_desire is None or db_embeddings is None:
                turn["second_think"] = "Unable to infer belief."
                turn["pred_belief"] = ""
                continue

            # Experience retrieval
            start_retrieval = time.time()
            top5_exp = retrieve_top_k(current_desire, history_text, db, db_embeddings)
            total_retrieval_time += time.time() - start_retrieval

            task_description = "Infer persuadee's belief based on retrieved experiences."
            prompt = build_prompt(task_description, top5_exp, history_text, turn)

            turn["second_think"] = prompt

            # LLM inference
            start_llm = time.time()
            inferred_belief = generate_belief(prompt)
            total_llm_time += time.time() - start_llm

            turn["pred_belief"] = inferred_belief

        processed.append(dialog)

    total_time = time.time() - start_all

    print("\n========== Second Think Runtime Statistics ==========")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Total experience retrieval time: {total_retrieval_time:.2f} seconds")
    print(f"Total LLM inference time: {total_llm_time:.2f} seconds")
    total_turns = sum(
        len([t for t in d["dialog"] if t["speaker"] == "persuadee"]) for d in processed
    )
    print(f"Average time per persuadee turn: {total_time / total_turns:.4f} seconds")
    print("=====================================================\n")

    return processed


# ------------------------------------------------------------------
# Main entry
# ------------------------------------------------------------------
def main(args):
    print("Loading dataset...")
    with open(args.data, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    print("Processing dialogs...")
    result = process_dataset(raw_data, max_dialogs=args.max_dialogs)

    print("Saving output...")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Done. Saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Second Think: belief inference.")
    parser.add_argument("--data", type=str, default=str(config.DATA_PATH))
    parser.add_argument("--output", type=str, default=str(config.SECOND_THINK_OUTPUT))
    parser.add_argument("--max-dialogs", type=int, default=config.MAX_EVAL_DIALOGS)
    main(parser.parse_args())
