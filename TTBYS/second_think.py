import re
import json
import time
import copy
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===========================
# Configuration
# ===========================
DATA_PATH = "../data/ToM_BPD.json"
OUTPUT_PATH = "second_think_result.json"
MAX = 100
MODEL_DIR = ""
TOP_K = 5

device = "cuda" if torch.cuda.is_available() else "cpu"

# ===========================
# 1. Load experience retrieval model
# ===========================
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

# ===========================
# 2. Load LLM
# ===========================
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
llm_model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
llm_model.eval()
print("Model loaded.")

# ===========================
# 3. Build experience database
# ===========================
def build_experience_db(raw):
    """
    Extract persuadee turns with desire and belief into a one-turn experience DB.
    """
    db = []
    for entry in raw:
        dialog = entry.get("dialog", [])
        for turn in dialog:
            if turn["speaker"] == "persuadee":
                desire = turn.get("annotation", {}).get("desire", None)
                belief = turn.get("annotation", {}).get("belief", "")
                if desire is None or not belief:
                    continue
                idx = dialog.index(turn)
                history = dialog[:idx+1]
                history_text = "\n".join([f"{t['speaker']}: {t['content']}" for t in history])
                db.append({
                    "desire": desire,
                    "context": history_text,
                    "current_belief": belief
                })
    return db

# ===========================
# 4. Retrieve top-K relevant experiences
# ===========================
def retrieve_top_k(current_desire, db, db_embeddings, k=TOP_K):
    """
    Retrieve top-K experiences matching the current desire using embeddings.
    """
    filtered_idx = [i for i, d in enumerate(db) if d["desire"] == current_desire]
    if not filtered_idx:
        return []
    filtered_embeddings = db_embeddings[filtered_idx]
    query_emb = filtered_embeddings.new_zeros((1, filtered_embeddings.shape[1]))
    scores = util.cos_sim(query_emb, filtered_embeddings)[0]
    topk = torch.topk(scores, k=min(k, len(filtered_idx)))
    return [db[filtered_idx[i]] for i in topk.indices.tolist()]

# ===========================
# 5. Prompt construction
# ===========================
def build_prompt(task_description, top5_exp, history_text, current_turn):
    """
    Build a prompt with few-shot examples and top-K retrieved experiences.
    """
    exp_texts = []
    for i, exp in enumerate(top5_exp, 1):
        exp_texts.append(
            f"Experience {i}:\n{exp['context']}\nCurrent belief: {exp['current_belief']}"
        )
    exp_block = "\n\n".join(exp_texts)

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

# ===========================
# 6. LLM inference
# ===========================
def generate_belief(prompt, max_new_tokens=32):
    """
    Generate the current belief from LLM given the prompt.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(llm_model.device)

    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    raw_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    # Extract belief using regex
    match = re.search(r"Current belief:\s*(.+?)(\\n|\n|$)", raw_text)
    if match:
        belief = match.group(1).strip()
    else:
        belief = raw_text.splitlines()[0].strip() if raw_text else ""

    return belief

# ===========================
# 7. Main processing
# ===========================
def process_dataset(raw, max_dialogs=MAX):
    """
    Run second-think belief inference on the dataset.
    """
    eval_data = raw[:max_dialogs]
    db = build_experience_db(raw[max_dialogs:])
    print(f"Experience DB size: {len(db)}")

    db_contexts = [d.get("context", "") + " " + d.get("current_belief", "") for d in db]
    db_embeddings = embedder.encode(db_contexts, convert_to_tensor=True, device=device) if db_contexts else None

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

            history = dialog["dialog"][:idx]
            history_text = "\n".join([f"{t['speaker']}: {t['content']}" for t in history])

            current_desire = turn.get("annotation", {}).get("desire", None)
            if current_desire is None or db_embeddings is None:
                turn["second_think"] = "Unable to infer belief."
                continue

            # Experience retrieval
            start_retrieval = time.time()
            top5_exp = retrieve_top_k(current_desire, db, db_embeddings)
            total_retrieval_time += time.time() - start_retrieval

            task_description = "Infer persuadee's belief based on retrieved experiences."
            prompt = build_prompt(task_description, top5_exp, history_text, turn)

            turn["second_think"] = prompt

            # LLM inference
            start_llm = time.time()
            inferred_belief = generate_belief(prompt)
            total_llm_time += time.time() - start_llm

            turn["pre_belief"] = inferred_belief

        processed.append(dialog)

    total_time = time.time() - start_all

    print("\n========== Second Think Runtime Statistics ==========")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Total experience retrieval time: {total_retrieval_time:.2f} seconds")
    print(f"Total LLM inference time: {total_llm_time:.2f} seconds")
    total_turns = sum(len([t for t in d['dialog'] if t['speaker']=='persuadee']) for d in processed)
    print(f"Average time per persuadee turn: {total_time / total_turns:.4f} seconds")
    print("=====================================================\n")

    return processed

# ===========================
# 8. Main entry
# ===========================
if __name__ == "__main__":
    print("Loading dataset...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    print("Processing dialogs...")
    result = process_dataset(raw_data, max_dialogs=MAX)

    print("Saving output...")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"✅ Done. Saved to {OUTPUT_PATH}")
