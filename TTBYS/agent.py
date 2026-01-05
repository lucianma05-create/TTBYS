import re
import json
import copy
import uuid
from datetime import datetime

import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===== Import existing modules/functions =====
from first_think import build_prompt as build_desire_prompt, get_first_token_probs, normalize_probs
from second_think import build_prompt as build_belief_prompt, generate_belief, retrieve_top_k, build_experience_db
from third_think import build_prompt_ag, get_strategy_probs_ag

# --------------------------------------------------
# Configurable parameters
# --------------------------------------------------
SESSION_ID = str(uuid.uuid4())
SAVE_PATH = f"./logs/session_{SESSION_ID}.json"

DATA_PATH = ""           # Dataset path
MAX_DB_SKIP =                             # Number of entries reserved for initial conversation, excluded from experience DB
EMBEDDING_MODEL = ""
GEN_MODEL_DIR = ""    # Text generation model path

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

DESIRE_MAP = {-1: "unwilling", 0: "uncertain", 1: "willing"}
MAX_AGENT_CHARS = 150
TOP_K = 5

# Task description can be customized per session
TASK_DESCRIPTION = "Insert your persuasion task or context here."

# Optional initial agent opening strategy and text
INITIAL_STRATEGY = "Supplying information"
INITIAL_TEXT = "[Supplying information] Here is some initial persuasive context..."

# --------------------------------------------------
# Initialize embedding model
# --------------------------------------------------
embedder = SentenceTransformer(EMBEDDING_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# Load text generation model
# --------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_DIR, trust_remote_code=True)
gen_model = AutoModelForCausalLM.from_pretrained(
    GEN_MODEL_DIR,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
gen_model.eval()

# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def generate_text(prompt, max_new_tokens=128):
    """Generate text using the LLM model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(gen_model.device)
    with torch.no_grad():
        outputs = gen_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
    gen = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()

def truncate_text(text, max_chars=MAX_AGENT_CHARS):
    """Truncate text to a maximum character length."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."

def build_persuasion_prompt(task_desc, desire, belief, strategy, dialog_history):
    """Construct a prompt for the agent to generate persuasive text."""
    history_text = "\n".join([f"{d['speaker']}: {d['content']}" for d in dialog_history])
    return f"""
You are a persuasive agent in a multi-turn conversation.

Task description:
{task_desc}

User's current mental state:
- Desire level: {desire}
- Belief: {belief}

Selected persuasion strategy:
{strategy}

Conversation so far:
{history_text}

Instruction:
Based on the user's current desire and belief, and following the selected strategy,
continue the persuasion in a natural and supportive way.
Do NOT repeat the task description verbatim.
Respond in no more than two short sentences.

Persuasive response:
""".strip()

# --------------------------------------------------
# Load dataset and build experience database
# --------------------------------------------------
with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Split dataset: initial dialog vs experience DB
eval_data = raw_data[:MAX_DB_SKIP]
db = build_experience_db(raw_data[MAX_DB_SKIP:])
db_contexts = [d.get("context", "") + " " + d.get("current_belief", "") for d in db]
db_embeddings = embedder.encode(db_contexts, convert_to_tensor=True, device=embedder.device) if db_contexts else None

# --------------------------------------------------
# Initialize session log
# --------------------------------------------------
state = {
    "session_id": SESSION_ID,
    "start_time": datetime.now().isoformat(),
    "task_description": TASK_DESCRIPTION,
    "turns": []
}

# --------------------------------------------------
# Agent initial opening
# --------------------------------------------------
dialog_history = []
dialog_history.append({"speaker": "agent", "content": INITIAL_TEXT})
print(f"\n Agent: {INITIAL_TEXT}\n")
state["turns"].append({
    "turn_id": -1,
    "agent_output": INITIAL_TEXT,
    "note": "initial agent opening"
})

# --------------------------------------------------
# Multi-turn interaction loop
# --------------------------------------------------
turn_id = 0
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    dialog_history.append({"speaker": "user", "content": user_input})

    # ----------------------
    # First-think: Desire prediction
    # ----------------------
    desire_prompt = build_desire_prompt(dialog_history)
    raw_probs = get_first_token_probs(desire_prompt)
    norm_probs = normalize_probs(raw_probs)
    desire_dist = [norm_probs.get(-1, 0.0), norm_probs.get(0, 0.0), norm_probs.get(1, 0.0)]
    cur_desire = [-1, 0, 1][desire_dist.index(max(desire_dist))]

    # ----------------------
    # Second-think: Belief prediction
    # ----------------------
    top5_exp = retrieve_top_k(cur_desire, db, db_embeddings)
    belief_prompt = build_belief_prompt(
        TASK_DESCRIPTION,
        top5_exp=top5_exp,
        history_text="\n".join([f"{d['speaker']}: {d['content']}" for d in dialog_history]),
        current_turn={"annotation": {"desire": cur_desire}}
    )
    belief = generate_belief(belief_prompt)

    # ----------------------
    # Third-think: Strategy selection
    # ----------------------
    strategy_prompt = build_prompt_ag(intention=TASK_DESCRIPTION, belief=belief)
    strategy_prob = get_strategy_probs_ag(strategy_prompt)
    chosen_strategy = STRATEGIES[strategy_prob.index(max(strategy_prob))]

    # ----------------------
    # Generate agent response
    # ----------------------
    persuasion_prompt = build_persuasion_prompt(
        task_desc=TASK_DESCRIPTION,
        desire=DESIRE_MAP[cur_desire],
        belief=belief,
        strategy=chosen_strategy,
        dialog_history=dialog_history
    )
    agent_text = generate_text(persuasion_prompt, max_new_tokens=128)
    agent_text = truncate_text(agent_text, max_chars=300)

    dialog_history.append({"speaker": "agent", "content": agent_text})
    print(f"\n Agent: {agent_text}\n")

    # ----------------------
    # Log turn
    # ----------------------
    state["turns"].append({
        "turn_id": turn_id,
        "user_input": user_input,
        "first_think": {"prompt": desire_prompt, "desire_prob": desire_dist},
        "second_think": {"prompt": belief_prompt, "belief": belief, "top5_exp": top5_exp},
        "third_think": {"strategy_prob": strategy_prob, "chosen_strategy": chosen_strategy},
        "persuasion_generation": {"prompt": persuasion_prompt, "output": agent_text}
    })

    turn_id += 1

# --------------------------------------------------
# Save session log
# --------------------------------------------------
with open(SAVE_PATH, "w", encoding="utf-8") as f:
    json.dump(state, f, ensure_ascii=False, indent=2)

print(f"\n Conversation ended. Session log saved to: {SAVE_PATH}")
