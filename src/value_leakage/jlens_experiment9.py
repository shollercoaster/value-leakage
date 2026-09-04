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
    # output, [1:] are post-block hidden states -- layer indexing here is
    # 1-based to match jlens's own per-layer numbering as seen in its saved
    # output (cross-check this against the diagnose stage's printed layer keys).
    per_layer = {}
    for layer_idx, hidden in enumerate(outputs.hidden_states[1:], start=1):
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
    the two are directly comparable side by side."""
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
    ax.set_yticklabels([f"#{r + 1}" for r in range(TOP_K_HEATMAP)])
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels(layers, fontsize=6, rotation=90)
    ax.set_xlabel("layer")
    ax.set_ylabel("top-k rank")
    ax.set_title(title, fontsize=9, loc="left")
    for r in range(TOP_K_HEATMAP):
        for c in range(n_layers):
            f = flags[r, c]
            edge = {"pre_registered": "red", "exploratory": "orange"}.get(f)
            weight = "bold" if edge else "normal"
            txt = ax.text(c, r, annot[r, c], ha="center", va="center",
                          fontsize=5.5, color="white", weight=weight)
            if edge:
                txt.set_bbox(dict(boxstyle="round,pad=0.1", fc="none", ec=edge, lw=1.2))
    return im


def _plot_heatmaps(records, fig_module):
    """Visualization 1 (requested): the same style heatmap as Experiment 7's,
    one panel per (position, cut, lens-vs-plain-control) -- lets you see at a
    glance whether the J-lens surfaces something the plain logit-lens doesn't,
    exactly the comparison the design document's confound concern is about."""
    plt = fig_module
    n = len(records)
    fig, axes = plt.subplots(n, 2, figsize=(22, 4.2 * n), squeeze=False)
    for row, rec in enumerate(records):
        im1 = _plot_heatmap_panel(
            axes[row][0], rec["lens_topk_per_layer"],
            f"J-LENS | marker {rec['marker']} ({rec['cut']}) -- {rec['marker_note']}")
        im2 = _plot_heatmap_panel(
            axes[row][1], rec["plain_logit_lens_topk_per_layer"],
            f"PLAIN LOGIT-LENS (control) | marker {rec['marker']} ({rec['cut']})")
        fig.colorbar(im1, ax=axes[row][0], fraction=0.02, pad=0.01)
        fig.colorbar(im2, ax=axes[row][1], fraction=0.02, pad=0.01)
    fig.suptitle(
        "Experiment 9: layer x top-10-rank concept readout, J-lens (left) vs. the matched "
        "plain-logit-lens control (right), at Experiment 8's three flagged positions.\n"
        "Red border = pre-registered bias word. Orange border = exploratory word. "
        "A word appearing on the LEFT but not the RIGHT, at an early layer, is the strongest "
        "kind of evidence for a hidden concept -- the whole point of running both.",
        fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = OUT_DIR / "heatmap_e9.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


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
            best = max(
                ((w, rec["lens_scores_per_layer"][l][w]["max_logprob"])
                 for l in rec["lens_scores_per_layer"] for w, _ in ALL_TRACKED_WORDS
                 if rec["lens_scores_per_layer"][l][w]["max_logprob"] is not None),
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
    else:
        raise ValueError(f"unknown stage {stage!r}; one of diagnose/read/plot")


if __name__ == "__main__":
    fire.Fire(main)
