import json
import copy
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from collections import Counter
import time

# -----------------------------
# 1. Configuration
# -----------------------------
DATA_PATH = "../data/ToM_BPD.json"
OUTPUT_PATH = "third_think_result.json"
MODEL_DIR = ""

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

STRATEGY_TOKENS = ["V", "L", "E", "T", "P", "A", "R", "I", "G"]  # Single-letter labels

device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

# -----------------------------
# 2. Load AG model
# -----------------------------
print("Loading AG model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
llm_model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
llm_model.eval()
print("AG model loaded.")

# -----------------------------
# 2.1 Generate strategy token IDs
# -----------------------------
STRATEGY_TOKEN_IDS = []
for tok in STRATEGY_TOKENS:
    ids = tokenizer.encode(" " + tok, add_special_tokens=False)
    STRATEGY_TOKEN_IDS.append(ids[0] if ids else None)
print("Strategy token IDs:", STRATEGY_TOKEN_IDS)

# -----------------------------
# 3. Build experience database
# -----------------------------
def build_experience_db(raw):
    """
    Construct experience DB from persuadee-persuader consecutive turns.
    Each entry contains intention, belief, and next persuader's strategy.
    """
    db = []
    for entry in raw:
        dialog = entry.get("dialog", [])
        for i in range(len(dialog) - 1):
            cur, nxt = dialog[i], dialog[i + 1]
            if cur["speaker"] == "persuadee" and nxt["speaker"] == "persuader":
                intention = cur.get("annotation", {}).get("intention", "")
                belief = cur.get("annotation", {}).get("belief", "")
                strategies = nxt.get("annotation", {}).get("strategy", [])
                if intention and belief:
                    db.append({
                        "intention": intention,
                        "belief": belief,
                        "strategy": strategies
                    })
    return db

# -----------------------------
# 4. Retrieve top-K experiences
# -----------------------------
def retrieve_top_k_ex(intention, belief, db, kb_embeddings, k=5, w_int=0.5, w_belief=0.5):
    """
    Retrieve top-K experience entries based on intention and belief similarity.
    Weighted combination of intention and belief cosine similarities.
    """
    int_emb = embedder.encode(intention, convert_to_tensor=True, device=device)
    belief_emb = embedder.encode(belief, convert_to_tensor=True, device=device)
    scores = []
    for i, entry in enumerate(db):
        s_int = util.cos_sim(int_emb, kb_embeddings[i*2].unsqueeze(0))[0].item()
        s_belief = util.cos_sim(belief_emb, kb_embeddings[i*2+1].unsqueeze(0))[0].item()
        scores.append((w_int*s_int + w_belief*s_belief, entry))
    topk = sorted(scores, key=lambda x: x[0], reverse=True)[:k]
    return [entry for _, entry in topk]

def compute_strategy_prob_ex(topk_entries):
    """
    Compute probability distribution over strategies from retrieved experiences.
    """
    counter = Counter()
    for entry in topk_entries:
        for strat in entry["strategy"]:
            if strat in STRATEGIES:
                counter[strat] += 1
    total = sum(counter.values())
    if total == 0:
        return [1/len(STRATEGIES)] * len(STRATEGIES)
    return [counter.get(s, 0)/total for s in STRATEGIES]

# -----------------------------
# 5. AG-based strategy prediction
# -----------------------------
def build_prompt_ag(intention, belief):
    """
    Build prompt for AG model with strategy definitions.
    """
    strategy_defs = """
Strategy definitions:
- V = Expression of views
- L = Logical appeal
- E = Enhancement of views
- T = Task inquiry
- P = Personal story
- A = Affirmation and reassurance
- R = Reflection of feelings
- I = Supplying information
- G = Giving examples
"""
    return f"""
Persuadee's intention: {intention}
Persuadee's belief: {belief}

{strategy_defs}

Predict the next persuader strategy.
Return ONLY ONE of these single-letter labels:
V, L, E, T, P, A, R, I, G.
Do not output anything else.
""".strip()

def get_strategy_probs_ag(prompt):
    """
    Predict strategy probabilities using AG LLM.
    Only compute probabilities for the defined single-letter tokens.
    """
    encoded = tokenizer(prompt, return_tensors="pt").to(llm_model.device)
    with torch.no_grad():
        outputs = llm_model(**encoded)
        logits = outputs.logits[0, -1]
    mask = torch.full_like(logits, float("-inf"))
    for tid in STRATEGY_TOKEN_IDS:
        if tid is not None:
            mask[tid] = logits[tid]
    probs = F.softmax(mask, dim=-1)
    result = [float(probs[tid].item()) for tid in STRATEGY_TOKEN_IDS]
    s = sum(result)
    return [v/s for v in result] if s > 0 else [1/len(STRATEGY_TOKEN_IDS)]*len(STRATEGIES)

# -----------------------------
# 6. Main processing function
# -----------------------------
def process_dataset(raw, max_dialogs=100):
    """
    Process dialogs: for each persuadee turn, compute AG and EX strategy distributions.
    """
    eval_data = raw[:max_dialogs]
    kb_data = raw[max_dialogs:]
    ex_db = build_experience_db(kb_data)
    print(f"EX knowledge-base size = {len(ex_db)}")

    kb_embeddings = []
    for entry in ex_db:
        kb_embeddings.append(embedder.encode(entry["intention"], convert_to_tensor=True, device=device))
        kb_embeddings.append(embedder.encode(entry["belief"], convert_to_tensor=True, device=device))

    processed = []
    total_ag_time = 0
    total_ex_time = 0

    for dialog_obj in tqdm(eval_data, desc="Processing dialogs"):
        dialog = copy.deepcopy(dialog_obj)
        for idx, turn in enumerate(dialog["dialog"]):
            if turn["speaker"] != "persuadee":
                continue
            intention = turn.get("annotation", {}).get("intention", "")
            belief = turn.get("annotation", {}).get("belief", "")
            if not intention or not belief:
                continue

            # EX-based prediction
            start_ex = time.time()
            topk_ex = retrieve_top_k_ex(intention, belief, ex_db, kb_embeddings, k=5)
            strategy_prob_ex = compute_strategy_prob_ex(topk_ex)
            for next_idx in range(idx+1, len(dialog["dialog"])):
                next_turn = dialog["dialog"][next_idx]
                if next_turn["speaker"] == "persuader":
                    next_turn["third_think_ex"] = strategy_prob_ex
                    break
            total_ex_time += time.time() - start_ex

            # AG-based prediction
            start_ag = time.time()
            prompt_ag = build_prompt_ag(intention, belief)
            strategy_prob_ag = get_strategy_probs_ag(prompt_ag)
            for next_idx in range(idx+1, len(dialog["dialog"])):
                next_turn = dialog["dialog"][next_idx]
                if next_turn["speaker"] == "persuader":
                    next_turn["third_think_ag"] = strategy_prob_ag
                    break
            total_ag_time += time.time() - start_ag

        processed.append(dialog)

    return processed, total_ag_time, total_ex_time

# -----------------------------
# 7. Main
# -----------------------------
if __name__ == "__main__":
    print("Loading dataset...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    print("Processing dialogs...")
    start_time = time.time()
    result, total_ag_time, total_ex_time = process_dataset(raw_data, max_dialogs=100)
    total_time = time.time() - start_time

    avg_time_per_prediction = total_time / sum(len(d["dialog"]) for d in raw_data[:100])

    print("Saving output...")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"✅ Done. Saved to {OUTPUT_PATH}")

    # Runtime statistics
    print("\n========== Runtime Statistics ==========")
    print(f"Total time for all dialogs: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Total AG-only inference time: {total_ag_time:.2f} seconds")
    print(f"Total EX-based retrieval & strategy time: {total_ex_time:.2f} seconds")
    print(f"Average time per prediction: {avg_time_per_prediction:.4f} seconds")
    print("========================================")
