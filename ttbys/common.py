"""Shared utilities for TTBYS.

- Lazy model loading (LLM and sentence embedder are loaded on first use,
  so importing the think modules does not load any model).
- Shared text helpers (generation, truncation) and retrieval helpers.
"""
# Import config first: it sets CUDA_VISIBLE_DEVICES before torch
# initializes CUDA.
import config

import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer

# ------------------------------------------------------------------
# Lazy model loading
# ------------------------------------------------------------------
_llms = {}          # model_dir -> (tokenizer, model)
_embedder = None


def get_embedder():
    """Return the sentence embedder, loading it on first use."""
    global _embedder
    if _embedder is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _embedder = SentenceTransformer(config.EMBEDDING_MODEL, device=device)
    return _embedder


def get_llm(model_dir=None):
    """Return (tokenizer, model) for the given directory, loading lazily.

    Falls back to config.MODEL_DIR when no directory is given.
    """
    model_dir = model_dir or config.MODEL_DIR
    if not model_dir:
        raise ValueError(
            "MODEL_DIR is not set. Fill it in config.py or pass a path."
        )
    if model_dir not in _llms:
        print(f"Loading LLM from {model_dir} ...")
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        kwargs = dict(device_map="auto", trust_remote_code=True)
        # transformers >= 5 prefers `dtype`; older versions use `torch_dtype`.
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_dir, dtype=torch.float16, **kwargs
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                model_dir, torch_dtype=torch.float16, **kwargs
            )
        model.eval()
        _llms[model_dir] = (tokenizer, model)
        print("LLM loaded.")
    return _llms[model_dir]


# ------------------------------------------------------------------
# Embedding / retrieval helpers
# ------------------------------------------------------------------
def embed_texts(texts):
    """Encode one or more texts with the shared embedder.

    Returns a (1, d) or (n, d) tensor on the embedder's device.
    """
    embedder = get_embedder()
    if isinstance(texts, str):
        texts = [texts]
    return embedder.encode(texts, convert_to_tensor=True, device=embedder.device)


def cosine_topk(query_emb, corpus_emb, k):
    """Return the indices of the top-k rows of corpus_emb most similar
    to query_emb (a single query vector)."""
    scores = util.cos_sim(query_emb, corpus_emb)[0]
    k = min(k, corpus_emb.shape[0])
    return torch.topk(scores, k).indices.tolist()


def resolve_intention(annotation, background=""):
    """Return the intention used for a turn's EX-branch query / KB entry.

    The per-turn annotated 'intention' is preferred. In the current
    dataset it is always empty, so fall back to the dialog-level
    'background' (the persuasion-task description) instead of silently
    disabling the experience branch.
    """
    intention = (annotation or {}).get("intention", "").strip()
    if intention:
        return intention
    return (background or "").strip()


# ------------------------------------------------------------------
# Text helpers
# ------------------------------------------------------------------
def generate_text(tokenizer, model, prompt, max_new_tokens=128):
    """Greedily generate text with the given model and return it stripped."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def truncate_text(text, max_chars):
    """Truncate text to at most max_chars, cutting at a word boundary."""
    if len(text) <= max_chars:
        return text
    parts = text[:max_chars].rsplit(" ", 1)
    return parts[0] + "..." if len(parts) > 1 else text[:max_chars] + "..."


def normalize_probs(probs):
    """Normalize a probability dict so its values sum to 1."""
    total = sum(probs.values())
    if total == 0:
        return probs
    return {k: v / total for k, v in probs.items()}
