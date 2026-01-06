# TTBYS: Think Thrice Before You Speak
**Enhancing Theory-of-Mind Reasoning in Persuasive Agents**

This repository implements **TTBYS**, a framework designed to enhance Theory-of-Mind (ToM) reasoning in persuasive agents by leveraging **structured ToM experiences**. The system follows a **three-step reasoning framework** inspired by human deliberative cognition:

1. **First Think — Desire Inference**  
   Predict the persuadee’s desire toward the target action using both LLM-driven intuition and experience-driven implicit knowledge.

2. **Second Think — Belief Inference**  
   Infer the persuadee’s underlying belief that causally drives the predicted desire. Retrieved past experiences are incorporated explicitly to enhance the LLM’s belief generation.

3. **Third Think — Strategy Prediction**  
   Select the optimal persuasive strategy based on the inferred ToM state (intention, desire, and belief), combining both experience-guided and LLM-guided distributions.

---

## Project Structure

```text
TTBYS/
├── annotation/
│   ├── prompt.json        # Prompts for dialogue summarization
│   └── summary.py         # Script to generate dialogue summaries
├── data/
│   └── ToM_BPD.json       # Main ToM-PD dataset
├── agent.py               # Script for multi-step agent interaction
├── first_think.py         # Implements desire inference (First Think)
├── first_think_eval.py    # Evaluation script for First Think
├── second_think.py        # Implements belief inference (Second Think)
├── second_think_eval.py   # Evaluation script for Second Think
├── third_think.py         # Implements strategy prediction (Third Think)
├── third_think_eval.py    # Evaluation script for Third Think
└── README.md              # Project overview and usage instructions
