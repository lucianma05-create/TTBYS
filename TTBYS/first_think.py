import json
import copy
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
from collections import Counter
from tqdm import tqdm
import time


# -------------------------------------------------------------
# 1. Configuration
# -------------------------------------------------------------
DATA_PATH = "../data/ToM_BPD.json"
OUTPUT_PATH = "fist_think_result.json"

# AG model configuration
MODEL_DIR = ""
DESIRE_TOKEN_MAP = {
    -1: " A",   # unwilling
     0: " B",   # uncertain
     1: " C"    # willing
}

# EX model configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
DESIRE_ORDER = [-1, 0, 1]

# -------------------------------------------------------------
# 2. Load AG model
# -------------------------------------------------------------
print("Loading AG model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model.eval()
print("AG model loaded.")

# -------------------------------------------------------------
# 3. AG prompt construction and probability estimation
# -------------------------------------------------------------
def build_prompt(dialog_history):
    """
    Build a classification prompt from dialog history.
    """
    text_blocks = []
    for turn in dialog_history:
        if turn["speaker"] == "persuader":
            text_blocks.append(f"persuader: {turn['content']}")
        else:
            text_blocks.append(f"persuadee: {turn['content']}")
    context = "\n".join(text_blocks)
    instruction = (
        "\nBased on the above conversation, please classify the persuadee's desire.\n"
        "Output only one token: A (unwilling), B (uncertain), or C (willing).\n"
        "Your output should ONLY be A, B, or C."
    )
    return context + instruction


def get_first_token_probs(prompt):
    """
    Compute first-token probabilities for desire classification.
    """
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**encoded)
    logits = out.logits[0, -1]
    probs = F.softmax(logits, dim=-1)

    result = {}
    for desire, token_str in DESIRE_TOKEN_MAP.items():
        token_ids = tokenizer.encode(token_str, add_special_tokens=False)
        result[desire] = float(probs[token_ids[0]].item()) if token_ids else 0.0
    return result


def normalize_probs(probs_dict):
    """
    Normalize a probability dictionary to sum to 1.
    """
    total = sum(probs_dict.values())
    if total == 0:
        return probs_dict
    return {k: v / total for k, v in probs_dict.items()}

# -------------------------------------------------------------
# 4. One-turn knowledge base construction for EX
# -------------------------------------------------------------
def build_one_turn_db(raw):
    """
    Build a one-turn intention–desire database from dialogs.
    """
    db = []
    for entry in tqdm(raw, desc="Building one-turn KB"):
        dialog = entry.get("dialog", [])
        for i in range(len(dialog) - 1):
            cur = dialog[i]
            nxt = dialog[i + 1]
            if cur["speaker"] == "persuadee" and nxt["speaker"] == "persuader":
                ann = cur.get("annotation", {})
                intention = ann.get("intention", "")
                desire = ann.get("desire", None)
                strategies = nxt.get("annotation", {}).get("strategy", [])
                if intention and desire in DESIRE_ORDER:
                    db.append({
                        "intention": intention,
                        "desire": desire,
                        "strategy": strategies
                    })
    return db


def retrieve_top_k(input_intention, db, kb_embeddings, k=5):
    """
    Retrieve top-k most similar intentions from the KB.
    """
    query_emb = embedder.encode(input_intention, convert_to_tensor=True, device=device)
    scores = util.cos_sim(query_emb, kb_embeddings)[0]
    topk = torch.topk(scores, k=min(k, len(db)))
    return [db[i] for i in topk.indices.tolist()]


def compute_desire_prob(matches):
    """
    Compute desire distribution from retrieved matches.
    """
    if not matches:
        return [1/3, 1/3, 1/3]
    counter = Counter([m["desire"] for m in matches])
    total = sum(counter.values())
    return [
        counter.get(-1, 0) / total,
        counter.get(0, 0) / total,
        counter.get(1, 0) / total
    ]

# -------------------------------------------------------------
# 5. Dataset processing
# -------------------------------------------------------------
def process_dataset(raw, max_dialogs=100):
    """
    Run AG and EX desire prediction over the dataset.
    """
    eval_data = raw[:max_dialogs]
    kb_data = raw[max_dialogs:]

    # Build EX knowledge base
    one_turn_db = build_one_turn_db(kb_data)
    print(f"EX knowledge-base size = {len(one_turn_db)}")

    kb_intentions = [d["intention"] for d in one_turn_db]
    kb_embeddings = embedder.encode(
        kb_intentions, convert_to_tensor=True, device=device
    ) if kb_intentions else None

    processed = []
    num_predictions = 0
    ag_time = 0.0

    for dialog_obj in tqdm(eval_data, desc="Processing dialogs"):
        dialog = copy.deepcopy(dialog_obj)
        dialog_history = []

        for turn in dialog["dialog"]:
            dialog_history.append({
                "speaker": turn["speaker"],
                "content": turn["content"]
            })

            if turn["speaker"] != "persuadee":
                continue

            num_predictions += 1

            # -------- AG prediction --------
            ag_start = time.time()

            prompt = build_prompt(dialog_history)
            raw_probs = get_first_token_probs(prompt)
            norm_probs = normalize_probs(raw_probs)

            ag_time += (time.time() - ag_start)

            turn["ag_desire_prob"] = [
                norm_probs.get(-1, 0.0),
                norm_probs.get(0, 0.0),
                norm_probs.get(1, 0.0)
            ]

            # -------- EX prediction --------
            intention = turn.get("annotation", {}).get("intention", "")
            if not intention or kb_embeddings is None:
                turn["ex_desire_prob"] = [1/3, 1/3, 1/3]
            else:
                matches = retrieve_top_k(intention, one_turn_db, kb_embeddings, k=5)
                turn["ex_desire_prob"] = compute_desire_prob(matches)

        processed.append(dialog)

    return processed, num_predictions, ag_time


# -------------------------------------------------------------
# 6. Main entry
# -------------------------------------------------------------
if __name__ == "__main__":
    print("Loading dataset...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    print("Processing dataset...")
    start_time = time.time()

    result, num_predictions, ag_time = process_dataset(raw_data, max_dialogs=100)

    total_time = time.time() - start_time
    avg_time_per_prediction = total_time / num_predictions if num_predictions > 0 else 0

    print("Saving merged output...")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"✅ Done. Saved to {OUTPUT_PATH}")

    avg_ag_time_per_prediction = ag_time / num_predictions if num_predictions > 0 else 0

    print("\n========== Runtime Statistics ==========")
    print(f"Total dialogs processed: {len(result)}")
    print(f"Total predictions (persuadee turns): {num_predictions}")
    print(f"Total processing time: {total_time:.2f} seconds "
          f"({total_time/60:.2f} minutes)")
    print(f"Average time per prediction (overall): {avg_time_per_prediction:.4f} seconds")
    print(f"AG-only total time (LLM only): {ag_time:.2f} seconds")
    print(f"AG-only average time per prediction: {avg_ag_time_per_prediction:.4f} seconds")
    print("========================================")
