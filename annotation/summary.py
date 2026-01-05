import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# Environment settings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Model directory
MODEL_DIR = ""

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    device_map="auto",
    torch_dtype=torch.float16
)
model.eval()

# Load base prompt
with open("prompt.json", "r", encoding="utf-8-sig") as f:
    prompt_data = json.load(f)
base_prompt = prompt_data[1]["prompt_to_get_summary"].strip()

# Load dataset
with open("./data/ToM_BPD.json", "r", encoding="utf-8") as f:
    data = json.load(f)

def generate_summary(dialogue_text, max_new_tokens=256):
    """
    Generate a concise summary of the persuadee's response 
    for a short two-turn conversation.
    """
    prompt = f"{base_prompt}\n[Conversation]:\n{dialogue_text}\n[Summary]:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text.split("[Summary]:")[-1].strip()

count = 0

# Iterate through dialogues to generate summaries
for entry_idx, entry in enumerate(data, start=1):
    dialog = entry["dialog"]

    for i in range(len(dialog) - 1):
        turn_x = dialog[i]
        turn_y = dialog[i + 1]

        # Only process persuader → persuadee
        if turn_x["speaker"] == "persuader" and turn_y["speaker"] == "persuadee":

            conversation_text = (
                f"persuader: {turn_x['content']}\n"
                f"persuadee: {turn_y['content']}"
            )

            summary = generate_summary(conversation_text)

            if "annotation" not in turn_y:
                turn_y["annotation"] = {}

            turn_y["annotation"]["summary"] = summary

            count += 1
            print(f"[{count}] Entry {entry_idx}: {summary}")

# Save results
output_file = "dialogues_with_summary_local.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"\nCompleted! Generated {count} summaries, saved to {output_file}")
