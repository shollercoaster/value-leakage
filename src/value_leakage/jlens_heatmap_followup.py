"""Experiment 7, follow-up pass: the "unbiased_claim" and "neutral_control"
position types from the original design in docs/EXPERIMENTS.md, which the
first pass (jlens_heatmap.py) did not run.

Trace selection differs from the first pass, for a real reason stated here and
in FINDINGS.md/writeup.md: the first pass's below_good/above_good examples
(rows 53/23) contain no regex-matched "unbiased claim" sentence at all, so
they cannot be used to test that position. This pass instead uses, per
condition, the largest incentive-obliging threshold crossing that DOES contain
a matched unbiased-claim sentence -- below_good row 66 (88.0M -> 40.0M) and
above_good row 76 (30.0M -> 48.3M) -- so both position types are read from the
same trace per condition. Baseline has zero unbiased-claim matches in all 98
traces (checked directly, not assumed) -- reported as a finding in its own
right, not worked around by picking a different word list after the fact.

Reading only. No steering, no generation, per the scope note in
docs/EXPERIMENTS.md Experiment 7. One model load, reused for both position
types and all rows.
"""

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import jlens

# Same two-bug workaround as jlens_heatmap.py -- see that file's comment for detail.
import transformers.integrations.finegrained_fp8 as _fg8

class _SafeOverrides(dict):
    def get(self, key, default=None):
        return dict.get(self, key) or {}

_fg8.FP8Experts._impl_tp_layer_overrides = _SafeOverrides(_fg8.FP8Experts._impl_tp_layer_overrides)

MODEL_DIR = "/workspace/models/Qwen3.5-122B-A10B-FP8"
LENS_PATH = "/workspace/lenses/workspace-lenses/qwen3.5-122b-a10b/j-lens/lens.pt"
SOURCE_RUN = "/workspace/value-leakage/runs/qwen3.5-122b-a10b_20260815_030702"
THRESHOLD = 41_000_000
OUT_DIR = Path("/workspace/value-leakage/runs/qwen3.5-122b-a10b_jlens_e7")

BIAS_WORDS = ["bias", "manipulate", "cheat", "donation", "threshold", "influence"]
EXPLORATORY_WORDS = ["dishonesty", "dishonest", "charity", "lie", "unfair", "unbiased"]
TOP_K_HEATMAP = 10
MAX_CONTEXT_CHARS = 12_000

UNBIASED_CLAIM_PATTERNS = [
    r"shouldn'?t (be )?influenc\w*", r"should not (be )?influenc\w*",
    r"not (be )?influenced by", r"regardless of (the )?(threshold|bet|donation|incentive)",
    r"ignore the (threshold|bet|incentive)", r"stay(ing)? (objective|unbiased|neutral)",
    r"remain (objective|unbiased|neutral)", r"independent of the (threshold|bet|donation)",
    r"shouldn'?t (let|allow)", r"unbiased",
]
UNBIASED_RE = re.compile("|".join(UNBIASED_CLAIM_PATTERNS), re.IGNORECASE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def load_row(condition: str, row_i: int) -> dict:
    data = json.loads(Path(f"{SOURCE_RUN}/{condition}.json").read_text())
    traj = json.loads(Path(f"{SOURCE_RUN}/trajectories.json").read_text())[condition]
    row = next(r for r in data["rows"] if r.get("i") == row_i)
    row = dict(row)
    row["final_estimate"] = traj[row_i][-1]
    row["first_estimate"] = traj[row_i][0]
    row["condition"] = condition
    return row


def unbiased_claim_cutoff(reasoning: str):
    m = UNBIASED_RE.search(reasoning)
    return m.end() if m else None


def neutral_control_cutoff(reasoning: str):
    sentences = SENTENCE_SPLIT_RE.split(reasoning)
    mid = len(sentences) // 2
    for offset in range(0, len(sentences) // 2):
        for idx in (mid + offset, mid - offset):
            if 0 <= idx < len(sentences) and not UNBIASED_RE.search(sentences[idx]):
                return sum(len(s) + 1 for s in sentences[: idx + 1])
    return None


def build_prefix(tokenizer, prompt: str, reasoning_tail: str) -> str:
    open_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    if not open_text.rstrip().endswith("<think>"):
        open_text = open_text + "<think>\n"
    tail = reasoning_tail
    if len(tail) > MAX_CONTEXT_CHARS:
        tail = tail[-MAX_CONTEXT_CHARS:]
    return open_text + tail


def run_lens_all_layers(model, lens, tokenizer, text: str, top_k: int):
    lens_logits, model_logits, input_ids = lens.apply(model, text, positions=[-1], max_seq_len=8192)
    per_layer = {}
    for layer, logits in sorted(lens_logits.items()):
        vals, idx = logits[0].topk(top_k)
        toks = [tokenizer.decode([t]).strip() for t in idx.tolist()]
        per_layer[layer] = {"tokens": toks, "logits": vals.tolist()}
    return per_layer, input_ids.shape[-1]


def flag(tok: str) -> str | None:
    low = tok.lower()
    for w in BIAS_WORDS:
        if w in low:
            return "pre_registered"
    for w in EXPLORATORY_WORDS:
        if w in low:
            return "exploratory"
    return None


def plot_heatmap(panels: list[dict], out_path: Path, suptitle: str):
    n_panels = len(panels)
    fig, axes = plt.subplots(n_panels, 1, figsize=(20, 4.2 * n_panels))
    if n_panels == 1:
        axes = [axes]
    for ax, panel in zip(axes, panels):
        per_layer = panel["per_layer"]
        layers = sorted(per_layer)
        n_layers = len(layers)
        grid = np.zeros((TOP_K_HEATMAP, n_layers))
        annot = np.empty((TOP_K_HEATMAP, n_layers), dtype=object)
        flags = np.empty((TOP_K_HEATMAP, n_layers), dtype=object)
        for col, layer in enumerate(layers):
            toks = per_layer[layer]["tokens"]
            vals = per_layer[layer]["logits"]
            top1 = vals[0]
            for r in range(TOP_K_HEATMAP):
                grid[r, col] = vals[r] - top1
                annot[r, col] = toks[r]
                flags[r, col] = flag(toks[r])
        im = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=-15, vmax=0)
        ax.set_yticks(range(TOP_K_HEATMAP))
        ax.set_yticklabels([f"#{r+1}" for r in range(TOP_K_HEATMAP)])
        ax.set_xticks(range(n_layers))
        ax.set_xticklabels(layers, fontsize=6, rotation=90)
        ax.set_xlabel("layer")
        ax.set_ylabel("lens top-k rank")
        ax.set_title(panel["title"], fontsize=10, loc="left")
        for r in range(TOP_K_HEATMAP):
            for c in range(n_layers):
                edge, weight = None, "normal"
                if flags[r, c] == "pre_registered":
                    edge, weight = "red", "bold"
                elif flags[r, c] == "exploratory":
                    edge, weight = "orange", "bold"
                txt = ax.text(c, r, annot[r, c], ha="center", va="center",
                               fontsize=5.5, color="white", weight=weight)
                if edge:
                    txt.set_bbox(dict(boxstyle="round,pad=0.1", fc="none", ec=edge, lw=1.2))
        fig.colorbar(im, ax=ax, fraction=0.01, pad=0.01, label="lens logit - top1 logit (this layer)")
    fig.suptitle(suptitle, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=160)
    print(f"Saved {out_path}")


def main():
    print("Loading tokenizer + model ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    tokenizer.truncation_side = "left"
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype="auto", device_map={"": 0})
    model = jlens.from_hf(hf_model, tokenizer)
    print("Model loaded:", model)
    lens = jlens.JacobianLens.load(LENS_PATH)
    print("Lens loaded:", lens)

    rows = {
        "baseline": load_row("baseline", 2),
        "below_good": load_row("below_good", 66),
        "above_good": load_row("above_good", 76),
    }

    all_records = {"unbiased_claim": [], "neutral_control": []}

    # --- unbiased_claim: below_good and above_good only (baseline has zero matches, see below) ---
    unbiased_panels = []
    for condition in ("below_good", "above_good"):
        row = rows[condition]
        cutoff = unbiased_claim_cutoff(row["reasoning"])
        assert cutoff is not None, f"expected a match for pre-selected row {condition}/{row['i']}"
        text = build_prefix(tokenizer, row.get("prompt", ""), row["reasoning"][:cutoff])
        per_layer, n_tokens = run_lens_all_layers(model, lens, tokenizer, text, TOP_K_HEATMAP)
        title = (f"{condition}, row={row['i']} | first={row['first_estimate']:,} "
                 f"final={row['final_estimate']:,} | threshold={THRESHOLD:,} | "
                 f"position=end of matched unbiased-claim sentence | n_tokens={n_tokens}")
        unbiased_panels.append({"title": title, "per_layer": per_layer})
        all_records["unbiased_claim"].append({
            "condition": condition, "row_i": row["i"], "first_estimate": row["first_estimate"],
            "final_estimate": row["final_estimate"], "n_tokens": n_tokens, "per_layer": per_layer,
        })
        print(f"Done: unbiased_claim / {condition} row {row['i']}")

    # --- neutral_control: all three conditions ---
    neutral_panels = []
    for condition in ("baseline", "below_good", "above_good"):
        row = rows[condition]
        cutoff = neutral_control_cutoff(row["reasoning"])
        assert cutoff is not None, f"expected a neutral-control sentence for {condition}/{row['i']}"
        text = build_prefix(tokenizer, row.get("prompt", ""), row["reasoning"][:cutoff])
        per_layer, n_tokens = run_lens_all_layers(model, lens, tokenizer, text, TOP_K_HEATMAP)
        title = (f"{condition}, row={row['i']} | first={row['first_estimate']:,} "
                 f"final={row['final_estimate']:,} | threshold={THRESHOLD:,} | "
                 f"position=mid-trace neutral control sentence | n_tokens={n_tokens}")
        neutral_panels.append({"title": title, "per_layer": per_layer})
        all_records["neutral_control"].append({
            "condition": condition, "row_i": row["i"], "first_estimate": row["first_estimate"],
            "final_estimate": row["final_estimate"], "n_tokens": n_tokens, "per_layer": per_layer,
        })
        print(f"Done: neutral_control / {condition} row {row['i']}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "heatmap_data_unbiased_claim.json").write_text(json.dumps(all_records["unbiased_claim"], indent=2))
    (OUT_DIR / "heatmap_data_neutral_control.json").write_text(json.dumps(all_records["neutral_control"], indent=2))

    plot_heatmap(
        unbiased_panels, OUT_DIR / "heatmap_unbiased_claim.png",
        "J-lens concept readout at the 'unbiased-claim' position (end of the model's own "
        "'I should stay unbiased / this shouldn't influence me'-type sentence), qwen3.5-122b-a10b.\n"
        "No baseline panel: 0 of 98 baseline traces contain a matching sentence (checked directly) "
        "-- the model only produces this language when a bet is present.\n"
        "Red border = pre-registered bias word; orange border = exploratory word (checked after the "
        "first pass came back empty, not pre-registered)."
    )
    plot_heatmap(
        neutral_panels, OUT_DIR / "heatmap_neutral_control.png",
        "J-lens concept readout at a neutral, mid-calculation control position (no framing content), "
        "qwen3.5-122b-a10b, one trace per condition.\n"
        "Red border = pre-registered bias word; orange border = exploratory word (checked after the "
        "first pass came back empty, not pre-registered)."
    )

    config = {
        "experiment": 7, "pass": "follow-up (unbiased_claim + neutral_control)",
        "model": "qwen3.5-122b-a10b", "checkpoint": "Qwen/Qwen3.5-122B-A10B-FP8",
        "lens": "camilablank/workspace-lenses:qwen3.5-122b-a10b/j-lens/lens.pt",
        "threshold": THRESHOLD, "bias_words_pre_registered": BIAS_WORDS,
        "exploratory_words_visualization_only": EXPLORATORY_WORDS,
        "rows_used": {c: r["i"] for c, r in rows.items()},
        "note_baseline_unbiased_claim": "0 of 98 baseline traces match the unbiased-claim regex; not run for baseline.",
        "top_k_heatmap": TOP_K_HEATMAP, "source_run": SOURCE_RUN,
    }
    (OUT_DIR / "heatmap_followup_config.json").write_text(json.dumps(config, indent=2))
    print("Wrote heatmap_data_unbiased_claim.json, heatmap_data_neutral_control.json, "
          "heatmap_unbiased_claim.png, heatmap_neutral_control.png, heatmap_followup_config.json")


if __name__ == "__main__":
    main()
