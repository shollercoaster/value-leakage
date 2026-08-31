"""Bare-minimum smoke test for Experiment 7, run before the full battery in
jlens_experiment.py. Confirms the model+lens load and lens.apply() works at
all, and prints the decoded top-k concepts for (a) the README's own example
prompt and (b) one real position from one real baseline trace -- the fastest
possible first result, per the instruction to try the minimum that gets
something back before running the full pre-registered analysis.

Reading only. No steering, no generation -- teacher-forced forward passes.
"""

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import jlens

MODEL_DIR = "/workspace/models/Qwen3.5-122B-A10B-FP8"
LENS_PATH = "/workspace/lenses/workspace-lenses/qwen3.5-122b-a10b/j-lens/lens.pt"
SOURCE_RUN = "/workspace/value-leakage/runs/qwen3.5-122b-a10b_20260815_030702"
TOP_K = 10


def show(tag, tokenizer, lens_logits, model_logits):
    print(f"\n--- {tag} ---")
    for layer, logits in sorted(lens_logits.items()):
        topk = logits[0].topk(TOP_K).indices.tolist()
        toks = [tokenizer.decode([t]).strip() for t in topk]
        print(f"  layer {layer:>3}: {toks}")
    top_model = model_logits[0].topk(TOP_K).indices.tolist()
    print(f"  model output (final layer, actual next-token): "
          f"{[tokenizer.decode([t]).strip() for t in top_model]}")


def main():
    print("Loading tokenizer + model from", MODEL_DIR, "...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype="auto", device_map={"": 0}
    )
    model = jlens.from_hf(hf_model, tokenizer)
    print("Model loaded:", model)

    lens = jlens.JacobianLens.load(LENS_PATH)
    print("Lens loaded:", lens)

    # (a) README's own example, adapted -- confirms the pipeline runs at all.
    prompt = "Fact: The currency used in the country shaped like a boot is"
    lens_logits, model_logits, input_ids = lens.apply(model, prompt, positions=[-2])
    show(f"README sanity check: {prompt!r} @ position -2", tokenizer, lens_logits, model_logits)

    # (b) one real position from one real baseline trace -- first real result.
    data = json.loads(Path(f"{SOURCE_RUN}/baseline.json").read_text())
    row = next(r for r in data["rows"] if "error" not in r)
    open_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": row.get("prompt", "")}],
        tokenize=False, add_generation_prompt=True,
    )
    if not open_text.rstrip().endswith("<think>"):
        open_text = open_text + "<think>\n"
    reasoning_tail = row["reasoning"][-4000:]
    text = open_text + reasoning_tail
    lens_logits, model_logits, input_ids = lens.apply(model, text, positions=[-1], max_seq_len=8192)
    show(f"real baseline trace row={row['i']}, last token of reasoning tail "
         f"({input_ids.shape[-1]} tokens)", tokenizer, lens_logits, model_logits)

    print("\nSmoke test complete -- pipeline confirmed working end to end.")


if __name__ == "__main__":
    main()
