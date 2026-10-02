"""Interactive multi-turn persuasive agent (TTBYS online pipeline).

Each turn runs the three-step reasoning chain:
  1. First Think  — predict the user's desire from the conversation.
  2. Second Think — infer the user's underlying belief (with retrieved
                    experiences).
  3. Third Think  — select the next persuasion strategy.
then generates the persuasive utterance. The full session (prompts,
intermediate reasoning and outputs) is logged as JSON.
"""
import json
import uuid
from datetime import datetime

import common
import config
from first_think import build_prompt as build_desire_prompt, get_first_token_probs
from second_think import (
    build_experience_db,
    build_prompt as build_belief_prompt,
    generate_belief,
    retrieve_top_k,
)
from third_think import build_prompt_ag, get_strategy_probs_ag


def build_persuasion_prompt(task_desc, desire, belief, strategy, dialog_history):
    """Construct a prompt for the agent to generate persuasive text."""
    history_text = "\n".join(f"{d['speaker']}: {d['content']}" for d in dialog_history)
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


def main():
    if not config.MODEL_DIR:
        raise SystemExit("Please set MODEL_DIR in config.py before running agent.py.")

    print("Building experience DB...")
    with open(config.DATA_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Keep the online experience KB consistent with the offline modules:
    # the first MAX_EVAL_DIALOGS items are held out for evaluation.
    db = build_experience_db(raw_data[config.MAX_EVAL_DIALOGS:])
    db_contexts = [d.get("context", "") + " " + d.get("current_belief", "") for d in db]
    db_embeddings = common.embed_texts(db_contexts) if db_contexts else None
    print(f"Experience DB size: {len(db)}")

    gen_tokenizer, gen_model = common.get_llm(config.GEN_MODEL_DIR or config.MODEL_DIR)

    # -------------------------
    # Session state and opening
    # -------------------------
    session_id = str(uuid.uuid4())
    save_dir = config.OUTPUT_DIR / "logs"
    save_path = save_dir / f"session_{session_id}.json"

    state = {
        "session_id": session_id,
        "start_time": datetime.now().isoformat(),
        "task_description": config.TASK_DESCRIPTION,
        "turns": [],
    }

    # NOTE: speaker labels must be "persuader"/"persuadee" — the prompt
    # builders of the think modules expect exactly these labels.
    dialog_history = [{"speaker": "persuader", "content": config.INITIAL_TEXT}]
    print(f"\n Agent: {config.INITIAL_TEXT}\n")
    state["turns"].append({
        "turn_id": -1,
        "agent_output": config.INITIAL_TEXT,
        "note": "initial agent opening",
    })

    # -------------------------
    # Multi-turn interaction loop
    # -------------------------
    turn_id = 0
    try:
        while True:
            try:
                user_input = input("You: ")
            except (EOFError, KeyboardInterrupt):
                print("\nConversation ended.")
                break
            if user_input.strip().lower() == "exit":
                break

            dialog_history.append({"speaker": "persuadee", "content": user_input})
            history_text = "\n".join(
                f"{d['speaker']}: {d['content']}" for d in dialog_history
            )

            # -------- First Think: desire prediction --------
            desire_prompt = build_desire_prompt(dialog_history)
            raw_probs = get_first_token_probs(desire_prompt)
            norm_probs = common.normalize_probs(raw_probs)
            desire_dist = [
                norm_probs.get(-1, 0.0),
                norm_probs.get(0, 0.0),
                norm_probs.get(1, 0.0),
            ]
            cur_desire = config.DESIRE_ORDER[desire_dist.index(max(desire_dist))]

            # -------- Second Think: belief prediction --------
            top5_exp = retrieve_top_k(cur_desire, history_text, db, db_embeddings)
            belief_prompt = build_belief_prompt(
                config.TASK_DESCRIPTION,
                top5_exp=top5_exp,
                history_text=history_text,
                current_turn={"annotation": {"desire": cur_desire}},
            )
            belief = generate_belief(belief_prompt)

            # -------- Third Think: strategy selection --------
            strategy_prompt = build_prompt_ag(intention=config.TASK_DESCRIPTION, belief=belief)
            strategy_prob = get_strategy_probs_ag(strategy_prompt)
            chosen_strategy = config.STRATEGIES[strategy_prob.index(max(strategy_prob))]

            # -------- Generate agent response --------
            persuasion_prompt = build_persuasion_prompt(
                task_desc=config.TASK_DESCRIPTION,
                desire=config.DESIRE_MAP[cur_desire],
                belief=belief,
                strategy=chosen_strategy,
                dialog_history=dialog_history,
            )
            agent_text = common.generate_text(
                gen_tokenizer, gen_model, persuasion_prompt, max_new_tokens=128
            )
            agent_text = common.truncate_text(agent_text, config.MAX_AGENT_CHARS)

            dialog_history.append({"speaker": "persuader", "content": agent_text})
            print(f"\n Agent: {agent_text}\n")

            # -------- Log turn --------
            state["turns"].append({
                "turn_id": turn_id,
                "user_input": user_input,
                "first_think": {"prompt": desire_prompt, "desire_prob": desire_dist},
                "second_think": {"prompt": belief_prompt, "belief": belief, "top5_exp": top5_exp},
                "third_think": {"strategy_prob": strategy_prob, "chosen_strategy": chosen_strategy},
                "persuasion_generation": {"prompt": persuasion_prompt, "output": agent_text},
            })

            turn_id += 1
    finally:
        # Always save the session log, even on Ctrl+C.
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        print(f"\nSession log saved to: {save_path}")


if __name__ == "__main__":
    main()
