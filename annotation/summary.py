"""Generate abstract summaries of persuadee responses.

For every persuader -> persuadee turn pair, ask the LLM to produce one
abstract sentence describing the persuasion pattern, and store it in the
persuadee turn's annotation as "summary".
"""
import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Environment settings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE.parent / "data" / "ToM_BPD.json"
DEFAULT_PROMPT = HERE / "prompt.json"


def load_base_prompt(prompt_path):
    with open(prompt_path, "r", encoding="utf-8-sig") as f:
        prompt_data = json.load(f)
    # prompt.json is a list of {prompt_to_get_summary: ...} entries; find
    # the summary prompt by key rather than assuming a fixed index.
    for entry in prompt_data:
        if "prompt_to_get_summary" in entry:
            return entry["prompt_to_get_summary"].strip()
    raise ValueError(f"No 'prompt_to_get_summary' entry found in {prompt_path}")


def generate_summary(tokenizer, model, base_prompt, dialogue_text, max_new_tokens=256):
    """Generate a concise summary of the persuadee's response
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
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text.split("[Summary]:")[-1].strip()


def main(args):
    if not args.model_dir:
        raise SystemExit("No model directory given. Pass --model-dir.")

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    kwargs = dict(device_map="auto")
    # transformers >= 5 prefers `dtype`; older versions use `torch_dtype`.
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.float16, **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.float16, **kwargs)
    model.eval()

    base_prompt = load_base_prompt(args.prompt)

    with open(args.data, "r", encoding="utf-8") as f:
        data = json.load(f)

    count = 0

    # Iterate through dialogues to generate summaries
    for entry_idx, entry in enumerate(data, start=1):
        dialog = entry["dialog"]

        for i in range(len(dialog) - 1):
            turn_x = dialog[i]
            turn_y = dialog[i + 1]

            # Only process persuader -> persuadee
            if turn_x["speaker"] == "persuader" and turn_y["speaker"] == "persuadee":

                conversation_text = (
                    f"persuader: {turn_x['content']}\n"
                    f"persuadee: {turn_y['content']}"
                )

                summary = generate_summary(tokenizer, model, base_prompt, conversation_text)

                if "annotation" not in turn_y:
                    turn_y["annotation"] = {}

                turn_y["annotation"]["summary"] = summary

                count += 1
                print(f"[{count}] Entry {entry_idx}: {summary}")

    # Save results
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\nCompleted! Generated {count} summaries, saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate abstract summaries for persuadee turns in the dataset."
    )
    parser.add_argument("--model-dir", type=str, default="", help="LLM directory")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "dialogues_with_summary_local.json",
    )
    main(parser.parse_args())
