"""Experiment 9 (Neel Nanda extension, minimal scope, per docs_neel/main_writeup.md
section 4): J-lens concept reading at the three positions Experiment 8 flagged
as most important, on the flagship trace already used throughout Experiment 8.

**This script only runs on a rented GPU pod with the model and lens already
downloaded** -- see docs_neel/EXPERIMENT_9_RUNBOOK.md for the full setup.
`torch`, `transformers`, and `jlens` (Jacobian Lens,
github.com/anthropics/jacobian-lens) are not this repository's normal
dependencies (see pyproject.toml) -- they only need to exist on the pod.

Positions read (from runs/qwen3.5-122b-a10b_e8_thoughtanchors_20260904/
flagship_positions.json -- the SAME verified offsets Experiment 8 used, not
re-derived here):
  marker 2813  -- largest behavioral shift found (numeric_assumption)
  marker 10305 -- second-largest behavioral shift (reconsideration, "Decision:")
  marker 19338 -- cleanest confirmed attractor-holds case (reconsideration)
Each read at both "entry" (reasoning[:start], matching Experiment 8's
treatment cut -- right before the position) and "exit" (reasoning[:end],
matching Experiment 8's null cut -- right after it resolves), so a
dip-then-respike pattern across the reconsideration episode is checkable, per
EXPERIMENTS_NEEL_NANDA.md section 4.2's sub-experiment 2.

Per section 4.3, every read reports all four required checks -- a concept
score alone is never reported:
  1. continuous concept score: max log-probability the J-lens assigns to any
     single-token tokenization of a tracked word, per layer
  2. a matched plain-logit-lens control: computed directly from the
     underlying HuggingFace model's own hidden states + final norm + output
     head -- NOT assumed to come "for free" from jlens (see the `diagnose`
     stage below, and the design document's own caution on this point)
  3. a confound check: does the tracked word appear, at a word boundary, in
     the text immediately before the cut
  4. layer of first prominence (first layer where the word ranks in the
     lens's own top-20, for both the lens and the plain-logit-lens control)

Stages:
  diagnose -- ONE cheap forward pass, before anything else. Confirms
              lens.apply()'s actual return shape and that the plain-logit-lens
              computation (step 2 below) works on this specific architecture.
              Per the design document: "a single small call to check the real
              structure... is a cheap way to avoid discovering a wrong
              assumption mid-batch." If this fails, STOP and fix it here --
              do not proceed to `read` on an unverified assumption.
  read      -- the actual six reads (3 positions x entry/exit), all four
               checks, saved to disk.
  plot      -- four figures from the saved data, no GPU needed:
               concept_scores_by_layer.png (line plot, log-prob vs. layer),
               heatmap_e9.png (layer x top-10-rank grid, J-lens vs. the plain
                 logit-lens control side by side -- same visual style as
                 Experiment 7's own heatmap),
               entry_vs_exit_e9.png (dip/respike check across the doubt),
               convergence_e9.png (Experiment 8's behavioral shift vs.
                 Experiment 9's internal signal, the single chart this whole
                 extension was designed to produce).

  python -m value_leakage.jlens_experiment9 --stage diagnose
  python -m value_leakage.jlens_experiment9 --stage read
  python -m value_leakage.jlens_experiment9 --stage plot
"""

import json
import os
import re
from pathlib import Path

# Workaround 1 of 2 (FINDINGS.md, Experiment 7 entry): the DeepGEMM fp8 kernel's
# compiled CUDA path throws an internal scale-factor-dtype assertion on this
# checkpoint. Must be set before `transformers` is imported.
os.environ.setdefault("TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR", "1")

import fire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import jlens

# Workaround 2 of 2 (same source): a transformers 5.16.1 bug in the fp8
# quantizer's tensor-parallel-plan bookkeeping crashes on `.get()` with no
# default for any experts implementation other than one specific value.
# Irrelevant for a single-GPU load (no tensor-parallel plan needed) -- patch
# the lookup to fall back to {} instead of None, a no-op here that touches no
# quantized weight-handling code. Identical to jlens_heatmap.py's own patch.
import transformers.integrations.finegrained_fp8 as _fg8


class _SafeOverrides(dict):
    def get(self, key, default=None):
        return dict.get(self, key) or {}


_fg8.FP8Experts._impl_tp_layer_overrides = _SafeOverrides(_fg8.FP8Experts._impl_tp_layer_overrides)

# Workaround 3 of 3 (found live on this pod, not in FINDINGS.md's Experiment 7 entry --
# that pass evidently never exercised this code path). `TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR=1`
# correctly routes fp8 linears to the Triton finegrained-fp8 fallback instead of DeepGEMM, but
# loading that fallback kernel (`kernels-community/finegrained-fp8`, hub-fetched) fails: its
# `bayesian_autotuner.py` does `from triton.runtime.autotuner import (JITFunction,
# get_cache_invalidating_env_vars, get_cache_manager, triton_key, ...)`, and this installed
# triton (3.4.0) no longer re-exports those four names from that module -- they were moved to
# their actual defining modules (`triton.runtime.jit`, `triton.runtime.cache`,
# `triton.compiler.compiler`). This is a pure import-location shim: it does not touch any fp8
# dequantization or matmul math, just restores four names to where the hub kernel's own code
# expects to import them from, so the kernel's real (unmodified) implementation loads.
import triton.runtime.autotuner as _tra
import triton.runtime.jit as _trj
import triton.runtime.cache as _trc
import triton.compiler.compiler as _tcc

for _name, _source in (
    ("JITFunction", _trj),
    ("get_cache_manager", _trc),
    ("triton_key", _tcc),
    ("get_cache_invalidating_env_vars", _tcc),
):
    if not hasattr(_tra, _name):
        setattr(_tra, _name, getattr(_source, _name))

MODEL_DIR = "/workspace/models/Qwen3.5-122B-A10B-FP8"
LENS_PATH = "/workspace/lenses/workspace-lenses/qwen3.5-122b-a10b/j-lens/lens.pt"
FLAGSHIP_RUN_DIR = "/workspace/value-leakage/runs/qwen3.5-122b-a10b_20260815_030702"
FLAGSHIP_CONDITION = "above_good"
FLAGSHIP_ROW = 43
POSITIONS_FILE = "/workspace/value-leakage/runs/qwen3.5-122b-a10b_e8_thoughtanchors_20260904/flagship_positions.json"
OUT_DIR = Path("/workspace/value-leakage/runs/qwen3.5-122b-a10b_e9_jlens_20260904")

TARGET_MARKERS = {
    2813: "largest behavioral shift found in Experiment 8 (numeric_assumption)",
    10305: "second-largest behavioral shift (reconsideration, 'Decision:')",
    19338: "cleanest confirmed attractor-holds case (reconsideration)",
}

# Pre-registered word list -- IDENTICAL to this project's own Experiment 7 list
# (EXPERIMENTS_NEEL_NANDA.md section 6), unchanged, not adjusted after seeing
# any Experiment 9 results (it's copied from a DIFFERENT, already-completed
# experiment, not tuned to this one). Exploratory words kept separate per the
# same convention -- includes the incentive/reward cluster Experiment 7's own
# addendum already found, added here as exploratory rather than promoted to
# pre-registered, for consistency with how that experiment treated it.
PRE_REGISTERED_WORDS = ["bias", "manipulate", "cheat", "donation", "threshold", "influence"]
EXPLORATORY_WORDS = ["dishonesty", "dishonest", "charity", "lie", "unfair", "unbiased",
                     "incentiv", "reward", "奖励", "激励", "奖金", "獎勵"]
# "risk" was tried here as a post-hoc exploratory addition and then REVERTED at the
# user's request -- it recurs too generically (nearly every "Decision:"-adjacent
# position uses "risk"/"risky"/"risking" as ordinary decision-making vocabulary), so
# highlighting it added visual clutter rather than surfacing a meaningful signal,
# which is the opposite of what the highlighting is for. Left as a comment, not
# silently dropped, so this isn't retried the same way without reason.
ALL_TRACKED_WORDS = [(w, "pre_registered") for w in PRE_REGISTERED_WORDS] + \
                    [(w, "exploratory") for w in EXPLORATORY_WORDS]

TOP_K_FOR_PROMINENCE = 20  # "ranks in the top-20" is this project's own Experiment 7 convention
TOP_K_HEATMAP = 10  # same depth as Experiment 7's own heatmap (jlens_heatmap.py)
CONFOUND_CONTEXT_CHARS = 300  # window of text immediately before the cut, checked for verbatim words


# --- shared setup -------------------------------------------------------------

def load_flagship_text():
    data = json.loads(Path(f"{FLAGSHIP_RUN_DIR}/{FLAGSHIP_CONDITION}.json").read_text(encoding="utf-8"))
    row = data["rows"][FLAGSHIP_ROW]
    return data["prompt"], row["reasoning"]


def load_trace(condition: str, row_index: int):
    """Generalizes load_flagship_text to any condition/row in the same shipped
    run directory -- used for the baseline/below_good cross-condition positions
    below."""
    data = json.loads(Path(f"{FLAGSHIP_RUN_DIR}/{condition}.json").read_text(encoding="utf-8"))
    row = data["rows"][row_index]
    return data["prompt"], row["reasoning"]


# Cross-condition analog positions (Tier A extension, per the applicant's own request
# after Experiment 9's first pass: "close the no-baseline gap"). The flagship's three
# marker positions were HAND-VERIFIED specifically on that one above_good trace by
# Experiment 8's own position-finding process -- there is no equivalent hand-verified
# position list for baseline/below_good, so these are found here by a DIFFERENT,
# explicitly mechanical method, documented per-position so it's checkable rather than
# asserted:
#   - "reconsideration" analog (mirrors marker 10305, the ONE position that passed
#     Experiment 9's full hidden-concept test): found by literal string search for a
#     "Decision:" heading -- this exact phrase recurs as generic Fermi-estimate
#     boilerplate in this model's reasoning style regardless of condition (confirmed by
#     grep: present in 12/20 baseline rows and 17/20 below_good rows sampled), so it is
#     a genuine structural analog, not a coincidence.
#   - "numeric_assumption" analog (mirrors marker 2813, the largest BEHAVIORAL shift in
#     Experiment 8): found by RELATIVE POSITION, not topic match -- marker 2813 sits at
#     char 2813 of the flagship's 22033-char reasoning (12.77% through the trace); the
#     same fraction is located in each condition's chosen row, then snapped outward to
#     the nearest full sentence boundary (same "[.!?]\\s" convention used to measure
#     average sentence density in EXPERIMENTS.md). This deliberately does NOT search for
#     a topically-matching sentence (e.g. another "spot size" assumption) -- picking
#     "the sentence that looks most similar" would be a real, if well-intentioned, form
#     of cherry-picking. Matching by structural position instead answers a cleaner
#     question: "what does the internal state look like at a comparable early
#     assumption-building point in the reasoning," independent of the specific words
#     used there.
#   - Row selection: `above_good` row 43 is the flagship's own row, so baseline row 43
#     was tried first for consistency (same row index, not cherry-picked). It has a
#     valid "Decision:" analog. `below_good` row 43 has no "reasoning" field at all
#     (its original generation call hit a rate-limit error and was never filled in --
#     `{"i": 43, "error": "RetryError...RateLimitError"}`), so below_good row 0 was
#     used instead -- the first row with a valid `reasoning` field, a fixed, arbitrary,
#     pre-specified rule rather than a search for the "best-looking" row.
CROSS_CONDITION_POSITIONS = [
    {"condition": "baseline", "row_index": 43, "analog_of_marker": 2813,
     "kind": "numeric_assumption", "start": 3168, "end": 3293},
    {"condition": "baseline", "row_index": 43, "analog_of_marker": 10305,
     "kind": "reconsideration", "start": 18583, "end": 18652},
    {"condition": "below_good", "row_index": 0, "analog_of_marker": 2813,
     "kind": "numeric_assumption", "start": 2938, "end": 3296},
    {"condition": "below_good", "row_index": 0, "analog_of_marker": 10305,
     "kind": "reconsideration", "start": 23341, "end": 23378},
]


def load_positions():
    spec = json.loads(Path(POSITIONS_FILE).read_text(encoding="utf-8"))
    by_marker = {p["marker"]: p for p in spec["positions"]}
    missing = [m for m in TARGET_MARKERS if m not in by_marker]
    if missing:
        raise ValueError(f"markers {missing} not found in {POSITIONS_FILE} -- check the file wasn't regenerated with different offsets")
    return [by_marker[m] for m in TARGET_MARKERS]


def load_model_and_tokenizer():
    print("Loading tokenizer + model ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    tokenizer.truncation_side = "left"
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype="auto", device_map={"": 0})
    model = jlens.from_hf(hf_model, tokenizer)
    print("Model loaded:", model)
    lens = jlens.JacobianLens.load(LENS_PATH)
    print("Lens loaded:", lens)
    return tokenizer, hf_model, model, lens


def build_prefix(tokenizer, prompt: str, reasoning_tail: str) -> str:
    """Mirrors jlens_heatmap.py's own prefix construction exactly, for
    consistency with the already-working Experiment 7 pipeline."""
    open_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    if not open_text.rstrip().endswith("<think>"):
        open_text = open_text + "<think>\n"
    return open_text + reasoning_tail


# --- word -> token id lookup, and the three text-only / lens-based checks ----

def word_token_ids(tokenizer, word: str) -> list[int]:
    """Single-token tokenizations only, across the surface-form variants a
    word plausibly appears as mid-sentence. Multi-token words are represented
    by their first token as a conventional simplification -- flagged here,
    not hidden, since it under-counts words the tokenizer splits into several
    pieces. Chinese words (already single tokens or close to it in Qwen's
    tokenizer) are included as-is."""
    variants = {word, f" {word}", word.capitalize(), f" {word.capitalize()}", word.upper()}
    ids = set()
    for v in variants:
        toks = tokenizer.encode(v, add_special_tokens=False)
        if len(toks) == 1:
            ids.add(toks[0])
        elif toks:
            ids.add(toks[0])  # first-token fallback for multi-token words, noted above
    return sorted(ids)


def continuous_scores(logits: torch.Tensor, tokenizer) -> dict:
    """logits: 1D tensor over the vocabulary, for one layer, one position.
    Returns {word: {"category": ..., "max_logprob": float, "token_ids_used": [...]}}."""
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    out = {}
    for word, category in ALL_TRACKED_WORDS:
        ids = word_token_ids(tokenizer, word)
        if not ids:
            out[word] = {"category": category, "max_logprob": None, "token_ids_used": []}
            continue
        scores = [log_probs[i].item() for i in ids]
        out[word] = {"category": category, "max_logprob": max(scores), "token_ids_used": ids}
    return out


def plain_logit_lens_per_layer(hf_model, input_ids: torch.Tensor) -> dict:
    """Standard logit-lens (nostalgebraist 2020): apply the model's own final
    norm + output head to EVERY layer's hidden state, not just the last.
    Written directly against the underlying HuggingFace model rather than
    relying on jlens for this -- the design document explicitly says not to
    assume the library gives this "for free." Qwen-family (and most
    decoder-only HF) models expose `.model.norm` (final RMSNorm) and
    `.lm_head` (or `.get_output_embeddings()`) -- both are checked with a
    clear error if this specific checkpoint's structure differs, rather than
    silently returning wrong numbers."""
    base = hf_model.model if hasattr(hf_model, "model") else hf_model
    if not hasattr(base, "norm"):
        raise AttributeError(
            "Expected hf_model.model.norm (final RMSNorm) for the plain logit-lens control, "
            "not found on this checkpoint. Run --stage diagnose and inspect hf_model's actual "
            "module tree (print(hf_model)) before proceeding -- do not guess an alternate name."
        )
    lm_head = hf_model.lm_head if hasattr(hf_model, "lm_head") else hf_model.get_output_embeddings()

    with torch.no_grad():
        outputs = hf_model(input_ids, output_hidden_states=True, use_cache=False)
    # outputs.hidden_states: tuple of (n_layers + 1) tensors, [0] is the embedding
    # output, [1:] are post-block hidden states. Layer indexing here is 0-based
    # and DELIBERATELY matches jlens's own convention exactly, confirmed by
    # reading jacobian-lens/jlens/lens.py directly rather than assumed: jlens's
    # `apply()` hooks `model.layers[L]` (a 0-indexed nn.ModuleList) and reads its
    # output as "layer L", with `final_layer = model.n_layers - 1`. That is the
    # same tensor as `hidden_states[L + 1]` here (hidden_states[0] is the
    # pre-block embedding, hidden_states[1] is the output of the first, i.e.
    # index-0, block). Using `start=1` here (one-based) would silently offset
    # every "layer of first prominence" comparison against the lens by exactly
    # one layer -- checked directly against the diagnose stage's own top-5
    # agreement check, which compares by VALUE (via max()) and so passes under
    # either numbering; it does not by itself catch a labeling offset.
    per_layer = {}
    for layer_idx, hidden in enumerate(outputs.hidden_states[1:], start=0):
        last_token_hidden = hidden[0, -1, :]  # position -1: the read point itself
        normed = base.norm(last_token_hidden)
        logits = lm_head(normed)
        per_layer[layer_idx] = logits
    return per_layer


def confound_check(text_before_cut: str, word: str) -> bool:
    """True if `word` appears as a whole word (word-boundary match, not a bare
    substring -- "believe" must not match "lie") in the CONFOUND_CONTEXT_CHARS
    immediately before the cut point."""
    window = text_before_cut[-CONFOUND_CONTEXT_CHARS:]
    pattern = re.compile(r"(?<![a-zA-Z一-鿿])" + re.escape(word) + r"(?![a-zA-Z一-鿿])", re.IGNORECASE)
    return bool(pattern.search(window))


def topk_tokens_per_layer(per_layer_logits: dict, tokenizer, top_k: int = TOP_K_HEATMAP) -> dict:
    """Decoded top-k tokens + raw logits per layer, for the heatmap -- same
    shape as Experiment 7's own jlens_heatmap.py (`{layer: {"tokens": [...],
    "logits": [...]}}`), decoded here (while the tokenizer is in scope) so the
    `plot` stage never needs the tokenizer or the model again."""
    out = {}
    for layer, logits in per_layer_logits.items():
        vals, idx = logits.float().topk(top_k)
        out[layer] = {"tokens": [tokenizer.decode([t]).strip() for t in idx.tolist()],
                      "logits": vals.tolist()}
    return out


def layer_of_first_prominence(per_layer_logprobs: dict, tokenizer) -> dict:
    """First layer (ascending) where each word's best token ranks in the
    lens's own top-20 at that layer -- same top-k convention as this
    project's Experiment 7 scripts, applied here to whichever score
    (lens or plain-logit-lens) is passed in."""
    first_layer = {w: None for w, _ in ALL_TRACKED_WORDS}
    for layer in sorted(per_layer_logprobs):
        logits = per_layer_logprobs[layer]
        topk_ids = set(logits.float().topk(TOP_K_FOR_PROMINENCE).indices.tolist())
        for word, category in ALL_TRACKED_WORDS:
            if first_layer[word] is not None:
                continue
            ids = word_token_ids(tokenizer, word)
            if any(i in topk_ids for i in ids):
                first_layer[word] = layer
    return first_layer


# --- stage: diagnose ----------------------------------------------------------

def _run_diagnose():
    tokenizer, hf_model, model, lens = load_model_and_tokenizer()
    prompt, reasoning = load_flagship_text()
    pos = load_positions()[0]
    prefix = build_prefix(tokenizer, prompt, reasoning[: pos["start"]])

    print("\n=== lens.apply() return shape ===")
    lens_logits, model_logits, input_ids = lens.apply(model, prefix, positions=[-1], max_seq_len=8192)
    print("type(lens_logits):", type(lens_logits), " keys (layers):", sorted(lens_logits.keys())[:5], "...")
    one_layer = next(iter(lens_logits))
    print(f"lens_logits[{one_layer}].shape:", lens_logits[one_layer].shape)
    print("model_logits.shape:", model_logits.shape)
    print("input_ids.shape:", input_ids.shape)

    print("\n=== plain logit-lens sanity check ===")
    try:
        per_layer_plain = plain_logit_lens_per_layer(hf_model, input_ids)
        one = next(iter(per_layer_plain))
        print(f"plain logit-lens layer {one} logits shape:", per_layer_plain[one].shape)
        print("plain logit-lens layer keys:", sorted(per_layer_plain.keys())[:5], "...", sorted(per_layer_plain.keys())[-3:])
        print("Compare final-layer plain logit-lens top-5 vs lens_logits' own final-layer top-5 "
              "and vs model_logits top-5 -- they should roughly agree at the LAST layer "
              "(all three are reading the same final prediction, just via different code paths):")
        last_plain_layer = max(per_layer_plain)
        top5_plain = per_layer_plain[last_plain_layer].float().topk(5).indices.tolist()
        top5_model = model_logits[0].float().topk(5).indices.tolist()
        print("  plain logit-lens (last layer):", [tokenizer.decode([t]) for t in top5_plain])
        print("  model_logits (from lens.apply):", [tokenizer.decode([t]) for t in top5_model])
        print("If these two lines don't roughly agree, the plain-logit-lens implementation above "
              "has a bug (wrong norm/lm_head, or off-by-one layer indexing) -- fix before --stage read.")
    except Exception as e:
        print(f"PLAIN LOGIT-LENS FAILED: {type(e).__name__}: {e}")
        print("Inspect hf_model's module tree directly: print(hf_model) -- find the real name for "
              "the final norm and output head on this checkpoint, then fix plain_logit_lens_per_layer "
              "before running --stage read. Do not proceed on an unverified assumption.")

    print("\n=== confound check sanity (should be True) ===")
    print("'threshold' in last 300 chars before cut:",
          confound_check(reasoning[: pos["start"]], "threshold"))

    print("\nDiagnose stage complete. Review the shapes and the two top-5 lists above before running --stage read.")


# --- stage: read ---------------------------------------------------------------

def _run_read():
    tokenizer, hf_model, model, lens = load_model_and_tokenizer()
    prompt, reasoning = load_flagship_text()
    positions = load_positions()

    records = []
    for pos in positions:
        for cut_name, offset_key in (("entry", "start"), ("exit", "end")):
            cutoff = pos[offset_key]
            prefix = build_prefix(tokenizer, prompt, reasoning[:cutoff])
            print(f"reading marker={pos['marker']} ({cut_name}) ...")

            lens_logits, model_logits, input_ids = lens.apply(model, prefix, positions=[-1], max_seq_len=8192)
            plain_logits = plain_logit_lens_per_layer(hf_model, input_ids)

            lens_by_layer = {l: v[0] for l, v in lens_logits.items()}
            lens_scores_per_layer = {layer: continuous_scores(logits, tokenizer)
                                     for layer, logits in lens_by_layer.items()}
            plain_scores_per_layer = {layer: continuous_scores(logits, tokenizer)
                                      for layer, logits in plain_logits.items()}
            lens_first_prominence = layer_of_first_prominence(lens_by_layer, tokenizer)
            plain_first_prominence = layer_of_first_prominence(plain_logits, tokenizer)
            # Heatmap-ready data (decoded top-k tokens per layer), same shape as
            # Experiment 7's own heatmap script -- see topk_tokens_per_layer's docstring.
            lens_topk_per_layer = topk_tokens_per_layer(lens_by_layer, tokenizer)
            plain_topk_per_layer = topk_tokens_per_layer(plain_logits, tokenizer)

            context_window = reasoning[max(0, cutoff - CONFOUND_CONTEXT_CHARS):cutoff]
            confounds = {w: confound_check(context_window, w) for w, _ in ALL_TRACKED_WORDS}

            records.append({
                "marker": pos["marker"], "marker_note": TARGET_MARKERS[pos["marker"]],
                "kind": pos["kind"], "cut": cut_name, "cutoff_char": cutoff,
                "n_tokens": input_ids.shape[-1],
                "lens_scores_per_layer": lens_scores_per_layer,
                "plain_logit_lens_scores_per_layer": plain_scores_per_layer,
                "lens_layer_of_first_prominence": lens_first_prominence,
                "plain_logit_lens_layer_of_first_prominence": plain_first_prominence,
                "confound_check_word_in_preceding_text": confounds,
                "lens_topk_per_layer": lens_topk_per_layer,
                "plain_logit_lens_topk_per_layer": plain_topk_per_layer,
            })
            print(f"  done: {input_ids.shape[-1]} tokens")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results_e9.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    config = {
        "experiment": 9, "model": "qwen3.5-122b-a10b", "checkpoint": "Qwen/Qwen3.5-122B-A10B-FP8",
        "lens": "camilablank/workspace-lenses:qwen3.5-122b-a10b/j-lens/lens.pt",
        "flagship_condition": FLAGSHIP_CONDITION, "flagship_row": FLAGSHIP_ROW,
        "target_markers": TARGET_MARKERS, "pre_registered_words": PRE_REGISTERED_WORDS,
        "exploratory_words": EXPLORATORY_WORDS, "top_k_for_prominence": TOP_K_FOR_PROMINENCE,
    }
    (OUT_DIR / "config_e9.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {OUT_DIR}/results_e9.json and config_e9.json")


# --- stage: read_conditions (Tier A extension: baseline/below_good analogs) ----

def _run_read_conditions():
    """Reads CROSS_CONDITION_POSITIONS -- the baseline/below_good analogs of the
    flagship's numeric_assumption and reconsideration ("Decision:") position types.
    Answers the single biggest open question from the first pass: is marker 10305's
    internal "bias" signal specific to the incentivized above_good condition, or does
    the same signal appear at an equivalent commitment point regardless of condition?
    Identical per-position computation to _run_read -- same four required checks,
    same helper functions -- only the position-finding method differs (see
    CROSS_CONDITION_POSITIONS's own comment)."""
    tokenizer, hf_model, model, lens = load_model_and_tokenizer()

    records = []
    for spec in CROSS_CONDITION_POSITIONS:
        prompt, reasoning = load_trace(spec["condition"], spec["row_index"])
        for cut_name, offset_key in (("entry", "start"), ("exit", "end")):
            cutoff = spec[offset_key]
            prefix = build_prefix(tokenizer, prompt, reasoning[:cutoff])
            print(f"reading condition={spec['condition']} row={spec['row_index']} "
                  f"analog_of={spec['analog_of_marker']} ({cut_name}) ...")

            lens_logits, model_logits, input_ids = lens.apply(model, prefix, positions=[-1], max_seq_len=8192)
            plain_logits = plain_logit_lens_per_layer(hf_model, input_ids)

            lens_by_layer = {l: v[0] for l, v in lens_logits.items()}
            lens_scores_per_layer = {layer: continuous_scores(logits, tokenizer)
                                     for layer, logits in lens_by_layer.items()}
            plain_scores_per_layer = {layer: continuous_scores(logits, tokenizer)
                                      for layer, logits in plain_logits.items()}
            lens_first_prominence = layer_of_first_prominence(lens_by_layer, tokenizer)
            plain_first_prominence = layer_of_first_prominence(plain_logits, tokenizer)
            lens_topk_per_layer = topk_tokens_per_layer(lens_by_layer, tokenizer)
            plain_topk_per_layer = topk_tokens_per_layer(plain_logits, tokenizer)

            context_window = reasoning[max(0, cutoff - CONFOUND_CONTEXT_CHARS):cutoff]
            confounds = {w: confound_check(context_window, w) for w, _ in ALL_TRACKED_WORDS}

            records.append({
                "condition": spec["condition"], "row_index": spec["row_index"],
                "analog_of_marker": spec["analog_of_marker"],
                "analog_of_marker_note": TARGET_MARKERS[spec["analog_of_marker"]],
                "kind": spec["kind"], "cut": cut_name, "cutoff_char": cutoff,
                "n_tokens": input_ids.shape[-1],
                "lens_scores_per_layer": lens_scores_per_layer,
                "plain_logit_lens_scores_per_layer": plain_scores_per_layer,
                "lens_layer_of_first_prominence": lens_first_prominence,
                "plain_logit_lens_layer_of_first_prominence": plain_first_prominence,
                "confound_check_word_in_preceding_text": confounds,
                "lens_topk_per_layer": lens_topk_per_layer,
                "plain_logit_lens_topk_per_layer": plain_topk_per_layer,
            })
            print(f"  done: {input_ids.shape[-1]} tokens")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results_e9_conditions.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    config = {
        "experiment": "9-tierA", "model": "qwen3.5-122b-a10b",
        "checkpoint": "Qwen/Qwen3.5-122B-A10B-FP8",
        "lens": "camilablank/workspace-lenses:qwen3.5-122b-a10b/j-lens/lens.pt",
        "positions": CROSS_CONDITION_POSITIONS,
        "pre_registered_words": PRE_REGISTERED_WORDS, "exploratory_words": EXPLORATORY_WORDS,
        "top_k_for_prominence": TOP_K_FOR_PROMINENCE,
    }
    (OUT_DIR / "config_e9_conditions.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {OUT_DIR}/results_e9_conditions.json and config_e9_conditions.json")


# --- stage: plot (no GPU needed -- everything below reads only the saved JSON) --

# Experiment 8's own real, already-measured behavioral shifts at these same three
# positions (post null-baseline follow-up -- see FINDINGS_neel.md's Tier 1 entry).
# Hardcoded here because Experiment 8 is a separate, already-complete pipeline;
# not re-derived, just carried over for the convergence chart below.
BEHAVIORAL_SHIFT_ABS = {2813: 0.157, 10305: 0.070, 19338: 0.057}


def flag_word(tok: str) -> str | None:
    low = tok.lower()
    for w in PRE_REGISTERED_WORDS:
        if w in low:
            return "pre_registered"
    for w in EXPLORATORY_WORDS:
        if w in low or w in tok:  # Chinese variants aren't meaningfully "lower-cased"
            return "exploratory"
    return None


def _plot_heatmap_panel(ax, per_layer: dict, title: str):
    """One layer x top-k-rank panel, colored by logit relative to that layer's
    own top-1 -- same visual convention as Experiment 7's jlens_heatmap.py, so
    the two are directly comparable side by side.

    Returns (im, texts): `texts` is every per-cell word Text artist, handed back
    to _plot_heatmaps so it can auto-shrink any that don't yet fit their cell
    (see _autofit_cell_texts) -- a fixed font size cannot work here since decoded
    tokens range from 1 char ("i") to 11+ ("submissions", "certainty"); sizing
    for the longest would waste space on every short one, sizing for the average
    overflows on the long ones, which is exactly the "words run outside the
    boxes" problem the user flagged."""
    import numpy as np
    layers = sorted(int(l) for l in per_layer)
    n_layers = len(layers)
    grid = np.zeros((TOP_K_HEATMAP, n_layers))
    annot = np.empty((TOP_K_HEATMAP, n_layers), dtype=object)
    flags = np.empty((TOP_K_HEATMAP, n_layers), dtype=object)
    for col, layer in enumerate(layers):
        entry = per_layer[str(layer)]
        toks, vals = entry["tokens"], entry["logits"]
        top1 = vals[0]
        for r in range(TOP_K_HEATMAP):
            grid[r, col] = vals[r] - top1
            annot[r, col] = toks[r]
            flags[r, col] = flag_word(toks[r])
    im = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=-15, vmax=0)
    ax.set_yticks(range(TOP_K_HEATMAP))
    ax.set_yticklabels([f"#{r + 1}" for r in range(TOP_K_HEATMAP)], fontsize=13)
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels(layers, fontsize=11, rotation=90)
    ax.set_xlabel("layer", fontsize=13)
    ax.set_ylabel("top-k rank", fontsize=13)
    ax.set_title(title, fontsize=15, loc="left")
    texts = []
    for r in range(TOP_K_HEATMAP):
        for c in range(n_layers):
            f = flags[r, c]
            edge = {"pre_registered": "red", "exploratory": "orange"}.get(f)
            weight = "bold" if edge else "normal"
            # Starting fontsize -- deliberately generous; _autofit_cell_texts shrinks
            # only the individual cells that need it (long decoded tokens), so most
            # cells (short words) keep this full size rather than being sized down
            # to fit the worst case across the whole grid.
            txt = ax.text(c, r, annot[r, c], ha="center", va="center",
                          fontsize=12, color="white", weight=weight)
            if edge:
                txt.set_bbox(dict(boxstyle="round,pad=0.25", fc="none", ec=edge, lw=2.2))
            texts.append(txt)
    return im, texts


def _autofit_cell_texts(fig, entries, target_frac: float = 0.86, min_fontsize: float = 4.0,
                         max_passes: int = 5) -> None:
    """Shrinks each text's fontsize, INDIVIDUALLY, until its actual rendered pixel
    width (measured via the real renderer, not estimated from character count) fits
    within `target_frac` of its own cell's pixel width. This is what guarantees the
    entire word stays inside its box -- a uniform fontsize increase alone cannot do
    this, since cell width is fixed by the figure/column layout while word length
    varies per cell. `entries` is a list of (ax, text_artist) pairs, pooled across
    every panel in the figure so long words in either the J-lens or plain-control
    panel are each fit to their own panel's cell width (the two can differ slightly
    since the plain control has one extra layer)."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for _ in range(max_passes):
        any_adjusted = False
        for ax, txt in entries:
            x0 = ax.transData.transform((0, 0))[0]
            x1 = ax.transData.transform((1, 0))[0]
            cell_width_px = abs(x1 - x0)
            bbox = txt.get_window_extent(renderer)
            if bbox.width > cell_width_px * target_frac and txt.get_fontsize() > min_fontsize:
                ratio = (cell_width_px * target_frac) / bbox.width
                new_size = max(min_fontsize, txt.get_fontsize() * ratio)
                txt.set_fontsize(new_size)
                any_adjusted = True
        if not any_adjusted:
            break
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()


def _plot_heatmaps(records, fig_module):
    """Visualization 1 (requested): the same style heatmap as Experiment 7's,
    one panel per (position, cut, lens-vs-plain-control) -- lets you see at a
    glance whether the J-lens surfaces something the plain logit-lens doesn't,
    exactly the comparison the design document's confound concern is about.

    One file PER MARKER (entry row + exit row stacked, lens/plain side by side),
    not one giant file with all six records -- the original combined layout
    packed ~47 layer columns x 10 ranks x 6 records into one figure, which left
    each cell far too small for its annotated word to be legible. Splitting by
    marker and substantially widening each panel (see fig_w below) is the fix
    the user asked for: bigger blocks, readable words, same underlying data and
    color convention."""
    plt = fig_module
    by_marker: dict[int, list[dict]] = {}
    for rec in records:
        by_marker.setdefault(rec["marker"], []).append(rec)

    saved = []
    for marker in sorted(by_marker):
        recs = sorted(by_marker[marker], key=lambda r: 0 if r["cut"] == "entry" else 1)
        n_layers = max(len(recs[0]["lens_topk_per_layer"]), len(recs[0]["plain_logit_lens_topk_per_layer"]))
        # Generous starting column width -- comfortably fits most real decoded tokens
        # (up to ~9-10 chars) at the fontsize=12 _plot_heatmap_panel starts with,
        # without relying on it for the long-tail words: _autofit_cell_texts below
        # is what actually guarantees every word (however long) stays inside its box.
        fig_w = 0.78 * n_layers * 2 + 3
        fig_h = 8.5 * len(recs)
        fig, axes = plt.subplots(len(recs), 2, figsize=(fig_w, fig_h), squeeze=False)
        all_texts = []  # (ax, text) pairs across every panel in this figure, for autofit
        for row, rec in enumerate(recs):
            im1, texts1 = _plot_heatmap_panel(
                axes[row][0], rec["lens_topk_per_layer"],
                f"J-LENS | marker {marker} ({rec['cut']}) -- {rec['marker_note']}")
            im2, texts2 = _plot_heatmap_panel(
                axes[row][1], rec["plain_logit_lens_topk_per_layer"],
                f"PLAIN LOGIT-LENS (control) | marker {marker} ({rec['cut']})")
            all_texts.extend((axes[row][0], t) for t in texts1)
            all_texts.extend((axes[row][1], t) for t in texts2)
            fig.colorbar(im1, ax=axes[row][0], fraction=0.012, pad=0.006)
            fig.colorbar(im2, ax=axes[row][1], fraction=0.012, pad=0.006)
        fig.suptitle(
            f"Experiment 9: layer x top-10-rank concept readout, marker {marker} "
            f"({recs[0]['marker_note']}) -- J-lens (left) vs. the matched plain-logit-lens "
            "control (right), entry row above, exit row below.\n"
            "Red border = pre-registered bias word. Orange border = exploratory word. "
            "A word appearing on the LEFT but not the RIGHT, at an early layer, is the "
            "strongest kind of evidence for a hidden concept -- the whole point of running both.",
            fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        # Layout is now final (axes positions/sizes fixed) -- only now do cell pixel
        # widths mean anything, so the autofit pass runs here, right before saving.
        _autofit_cell_texts(fig, all_texts)
        out_path = OUT_DIR / f"heatmap_e9_marker{marker}.png"
        fig.savefig(out_path, dpi=150)
        saved.append(out_path)
        print(f"saved {out_path}")
    return saved


def _plot_entry_vs_exit(records, plt):
    """Visualization 2: does the concept score dip during the doubt and
    respike at recommit? Directly plots entry vs. exit for the best-scoring
    tracked word at each position -- the literal dip/respike test from
    EXPERIMENTS_NEEL_NANDA.md section 4.2's sub-experiment 2."""
    markers = sorted(BEHAVIORAL_SHIFT_ABS)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, marker in enumerate(markers):
        entry = next(r for r in records if r["marker"] == marker and r["cut"] == "entry")
        exit_ = next(r for r in records if r["marker"] == marker and r["cut"] == "exit")

        def best_word_score(rec):
            # Skip any tracked word not present in this saved record (e.g. a word added
            # to ALL_TRACKED_WORDS for heatmap highlighting after --stage read already
            # ran and saved this data -- see the "risk" addition note above).
            best = max(
                ((w, rec["lens_scores_per_layer"][l][w]["max_logprob"])
                 for l in rec["lens_scores_per_layer"] for w, _ in ALL_TRACKED_WORDS
                 if w in rec["lens_scores_per_layer"][l]
                 and rec["lens_scores_per_layer"][l][w]["max_logprob"] is not None),
                key=lambda t: t[1], default=(None, None),
            )
            return best

        w_entry, s_entry = best_word_score(entry)
        w_exit, s_exit = best_word_score(exit_)
        ax.plot([0, 1], [s_entry, s_exit], "o-", label=f"marker {marker}")
        ax.annotate(w_entry or "?", (0, s_entry), textcoords="offset points", xytext=(-5, 5), fontsize=7, ha="right")
        ax.annotate(w_exit or "?", (1, s_exit), textcoords="offset points", xytext=(5, 5), fontsize=7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["entry\n(about to write the doubt)", "exit\n(just resolved back to a number)"])
    ax.set_ylabel("best tracked word's max log-prob, any layer")
    ax.set_title("Does the strongest concept dip during the doubt and respike at recommit?", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = OUT_DIR / "entry_vs_exit_e9.png"
    fig.savefig(out_path, dpi=160)
    print(f"saved {out_path}")


def _plot_convergence(records, plt):
    """Visualization 3: the single chart this whole extension was designed to
    produce -- does Experiment 8's behavioral importance line up with
    Experiment 9's internal signal at the same three positions? Both series
    min-max normalized to 0-1 for visual comparability (raw units differ:
    threshold-fraction vs. log-probability) -- normalization is stated on the
    chart, not hidden."""
    markers = sorted(BEHAVIORAL_SHIFT_ABS)
    behavioral = [BEHAVIORAL_SHIFT_ABS[m] for m in markers]

    internal = []
    for marker in markers:
        entry = next(r for r in records if r["marker"] == marker and r["cut"] == "entry")
        exit_ = next(r for r in records if r["marker"] == marker and r["cut"] == "exit")
        scores = [
            v["max_logprob"]
            for rec in (entry, exit_)
            for layer_scores in rec["lens_scores_per_layer"].values()
            for v in layer_scores.values()
            if v["max_logprob"] is not None
        ]
        internal.append(max(scores) if scores else float("nan"))

    def norm(xs):
        lo, hi = min(xs), max(xs)
        return [(x - lo) / (hi - lo) if hi > lo else 0.5 for x in xs]

    beh_n, int_n = norm(behavioral), norm(internal)
    x = range(len(markers))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    width = 0.35
    ax.bar([i - width / 2 for i in x], beh_n, width, label="Experiment 8: behavioral |shift| (normalized)")
    ax.bar([i + width / 2 for i in x], int_n, width, label="Experiment 9: peak internal concept score (normalized)")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"marker {m}\n({BEHAVIORAL_SHIFT_ABS[m]:.3f} raw shift)" for m in markers], fontsize=8)
    ax.set_ylabel("normalized (0-1 within this chart only)")
    ax.set_title("Convergence check: does behavioral importance track internal signal?", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = OUT_DIR / "convergence_e9.png"
    fig.savefig(out_path, dpi=160)
    print(f"saved {out_path}")


def _plot_heatmaps_conditions(cond_records, fig_module):
    """Same style/mechanism as _plot_heatmaps (auto-fit cell text, red/orange
    flagged-word borders), grouped by ANALOG MARKER TYPE rather than by marker --
    one file per position-type, with baseline's entry/exit stacked above
    below_good's entry/exit, so the two non-incentivized-vs-incentivized-adjacent
    conditions are directly comparable at a glance."""
    plt = fig_module
    by_type: dict[int, list[dict]] = {}
    for rec in cond_records:
        by_type.setdefault(rec["analog_of_marker"], []).append(rec)

    saved = []
    for analog_marker in sorted(by_type):
        recs = sorted(by_type[analog_marker],
                       key=lambda r: (r["condition"] != "baseline", r["cut"] != "entry"))
        n_layers = max(len(recs[0]["lens_topk_per_layer"]), len(recs[0]["plain_logit_lens_topk_per_layer"]))
        fig_w = 0.78 * n_layers * 2 + 3
        fig_h = 8.5 * len(recs)
        fig, axes = plt.subplots(len(recs), 2, figsize=(fig_w, fig_h), squeeze=False)
        all_texts = []
        for row, rec in enumerate(recs):
            label = f"{rec['condition']} row {rec['row_index']} ({rec['cut']})"
            im1, texts1 = _plot_heatmap_panel(
                axes[row][0], rec["lens_topk_per_layer"],
                f"J-LENS | {label} -- analog of marker {analog_marker}")
            im2, texts2 = _plot_heatmap_panel(
                axes[row][1], rec["plain_logit_lens_topk_per_layer"],
                f"PLAIN LOGIT-LENS (control) | {label}")
            all_texts.extend((axes[row][0], t) for t in texts1)
            all_texts.extend((axes[row][1], t) for t in texts2)
            fig.colorbar(im1, ax=axes[row][0], fraction=0.012, pad=0.006)
            fig.colorbar(im2, ax=axes[row][1], fraction=0.012, pad=0.006)
        fig.suptitle(
            f"Experiment 9, Tier A: layer x top-10-rank concept readout, cross-condition analog "
            f"of marker {analog_marker} ({recs[0]['kind']}) -- baseline (top 2 rows) vs. below_good "
            f"(bottom 2 rows), J-lens (left) vs. plain-logit-lens control (right).\n"
            "These positions are NOT the flagship trace -- they are the closest analog found in a "
            "baseline/below_good row, per the method documented in CROSS_CONDITION_POSITIONS's own "
            "comment. Compare against heatmap_e9_marker{}.png (the above_good original) to see "
            "whether a signal there is condition-specific or generic.".format(analog_marker),
            fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        _autofit_cell_texts(fig, all_texts)
        out_path = OUT_DIR / f"heatmap_e9_conditions_analog{analog_marker}.png"
        fig.savefig(out_path, dpi=150)
        saved.append(out_path)
        print(f"saved {out_path}")
    return saved


def _plot_condition_comparison(flagship_records, cond_records, plt):
    """Visualization 5 (Tier A addition): the chart that actually answers the open
    question from the first pass -- is a position-type's internal signal specific to
    the incentivized above_good condition, or does it appear regardless of condition?
    One group per marker-type (2813-type = numeric_assumption, 10305-type =
    reconsideration/"Decision:"), three bars per group (baseline, below_good,
    above_good), each bar = the peak tracked-word log-probability at that
    position-type in that condition (max over entry+exit, over all layers/words --
    same "peak internal score" definition _plot_convergence already uses, so the
    above_good bar here is directly comparable to that chart's own numbers)."""
    def peak_score(records, **match):
        cands = [r for r in records if all(r.get(k) == v for k, v in match.items())]
        if len(cands) != 2:  # expect exactly one entry + one exit record
            return float("nan")
        scores = [
            v["max_logprob"]
            for rec in cands
            for layer_scores in rec["lens_scores_per_layer"].values()
            for v in layer_scores.values()
            if v["max_logprob"] is not None
        ]
        return max(scores) if scores else float("nan")

    marker_types = sorted(TARGET_MARKERS)
    # Short kind label per marker type, pulled from the already-saved flagship records
    # rather than TARGET_MARKERS's long descriptive strings (those overflowed the
    # x-tick labels into neighboring bars/off the axes in the first render of this chart).
    kind_by_marker = {m: next(r["kind"] for r in flagship_records if r["marker"] == m)
                      for m in marker_types}
    conditions = ["baseline", "below_good", "above_good"]
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.25
    for i, cond in enumerate(conditions):
        ys = []
        for m in marker_types:
            if cond == "above_good":
                ys.append(peak_score(flagship_records, marker=m))
            else:
                ys.append(peak_score(cond_records, condition=cond, analog_of_marker=m))
        xs = [j + (i - 1) * width for j in range(len(marker_types))]
        ax.bar(xs, ys, width, label=cond)
    ax.set_xticks(range(len(marker_types)))
    ax.set_xticklabels([f"marker {m}\n({kind_by_marker[m]})" for m in marker_types], fontsize=9)
    ax.set_ylabel("peak tracked-word log-probability\n(raw, NOT normalized -- comparable across bars)", fontsize=9)
    ax.set_title(
        "Tier A: is the internal signal at each position-type specific to the incentivized\n"
        "condition, or does it appear in baseline/below_good too? (above_good = the original,\n"
        "hand-verified flagship position; baseline/below_good = the closest analog found by the\n"
        "method in CROSS_CONDITION_POSITIONS -- not the identical sentence)", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = OUT_DIR / "condition_comparison_e9.png"
    fig.savefig(out_path, dpi=160)
    print(f"saved {out_path}")


def _run_plot_conditions():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cond_path = OUT_DIR / "results_e9_conditions.json"
    if not cond_path.exists():
        raise FileNotFoundError(f"{cond_path} not found -- run --stage read_conditions first")
    cond_records = json.loads(cond_path.read_text(encoding="utf-8"))
    flagship_records = json.loads((OUT_DIR / "results_e9.json").read_text(encoding="utf-8"))

    _plot_heatmaps_conditions(cond_records, plt)
    _plot_condition_comparison(flagship_records, cond_records, plt)
    print("\nTier A condition figures saved to", OUT_DIR)


def _run_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = json.loads((OUT_DIR / "results_e9.json").read_text(encoding="utf-8"))

    fig, axes = plt.subplots(len(records), 1, figsize=(14, 3.2 * len(records)), squeeze=False)
    axes = axes[:, 0]
    for ax, rec in zip(axes, records):
        layers = sorted(int(l) for l in rec["lens_scores_per_layer"])
        for word, category in ALL_TRACKED_WORDS:
            # A word can be in ALL_TRACKED_WORDS (used for heatmap highlighting) without
            # being in the saved continuous-score data if it was added to the tracked list
            # AFTER --stage read already ran and saved results_e9.json (e.g. "risk", added
            # post-hoc for heatmap highlighting only) -- skip it here rather than crash;
            # its continuous-score line simply doesn't exist yet until read is re-run.
            per_layer_word_scores = rec["lens_scores_per_layer"][str(layers[0])]
            if word not in per_layer_word_scores:
                continue
            ys = [rec["lens_scores_per_layer"][str(l)][word]["max_logprob"] for l in layers]
            if all(y is None for y in ys):
                continue
            style = "-" if category == "pre_registered" else "--"
            ax.plot(layers, ys, style, label=word, alpha=0.8)
        ax.set_title(f"marker {rec['marker']} ({rec['cut']}) -- {rec['marker_note']}", fontsize=9, loc="left")
        ax.set_xlabel("layer")
        ax.set_ylabel("lens log-prob")
        ax.legend(fontsize=6, ncol=3)
    fig.suptitle("Experiment 9, chart 0: J-lens continuous concept score by layer. "
                 "Solid = pre-registered word, dashed = exploratory.", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = OUT_DIR / "concept_scores_by_layer.png"
    fig.savefig(out_path, dpi=160)
    print(f"saved {out_path}")

    _plot_heatmaps(records, plt)
    _plot_entry_vs_exit(records, plt)
    _plot_convergence(records, plt)
    print("\nAll four figures saved to", OUT_DIR)


def main(stage: str):
    if stage == "diagnose":
        _run_diagnose()
    elif stage == "read":
        _run_read()
    elif stage == "plot":
        _run_plot()
    elif stage == "read_conditions":
        _run_read_conditions()
    elif stage == "plot_conditions":
        _run_plot_conditions()
    else:
        raise ValueError(f"unknown stage {stage!r}; one of diagnose/read/plot/read_conditions/plot_conditions")


if __name__ == "__main__":
    fire.Fire(main)
