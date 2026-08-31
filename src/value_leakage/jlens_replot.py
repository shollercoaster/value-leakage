"""Re-render the three Experiment 7 heatmaps from already-saved per-layer data
(heatmap_data*.json) with a third highlight category: incentive/reward words
and their variants (including Chinese, since Qwen decodes bilingually). No
model load, no GPU -- pure replotting from disk.

Added after noticing (and then checking directly, not assuming) that an
incentive/reward cluster is prominent in one of the three heatmaps -- but the
check below is what actually decides where it does and doesn't appear; it is
not assumed to be "the neutral heatmap" just because that was suggested.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path("/workspace/value-leakage/runs/qwen3.5-122b-a10b_jlens_e7")

BIAS_WORDS = ["bias", "manipulate", "cheat", "donation", "threshold", "influence"]
EXPLORATORY_WORDS = ["dishonesty", "dishonest", "charity", "lie", "unfair", "unbiased"]
INCENTIVE_WORDS = ["incentiv", "reward", "奖励", "激励", "奖金", "獎勵"]

TOP_K_HEATMAP = 10

COLORS = {"pre_registered": "red", "exploratory": "orange", "incentive_reward": "cyan"}


def flag(tok: str) -> str | None:
    low = tok.lower()
    for w in BIAS_WORDS:
        if w in low:
            return "pre_registered"
    for w in INCENTIVE_WORDS:
        if w in low or w in tok:  # Chinese variants aren't meaningfully "lower-cased"
            return "incentive_reward"
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
                f = flags[r, c]
                edge = COLORS.get(f)
                weight = "bold" if edge else "normal"
                txt = ax.text(c, r, annot[r, c], ha="center", va="center",
                               fontsize=5.5, color="white", weight=weight)
                if edge:
                    txt.set_bbox(dict(boxstyle="round,pad=0.1", fc="none", ec=edge, lw=1.2))
        fig.colorbar(im, ax=ax, fraction=0.01, pad=0.01, label="lens logit - top1 logit (this layer)")
    fig.suptitle(
        suptitle + "\nRed = pre-registered bias word. Orange = exploratory word (checked after "
        "the first pass came back empty). Cyan = incentive/reward word (added after visual "
        "inspection; not part of either pre-registered or exploratory list, so a hit here is "
        "purely descriptive, not a confirmatory or exploratory-count result).",
        fontsize=8,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=160)
    print(f"Saved {out_path}")


def count_hits(records, label):
    print(f"--- {label} ---")
    for rec in records:
        cond = rec.get("condition", rec.get("label", ""))
        per_layer = rec["per_layer"]
        counts = {"pre_registered": 0, "exploratory": 0, "incentive_reward": 0}
        for layer, entry in per_layer.items():
            for t in entry["tokens"]:
                f = flag(t)
                if f:
                    counts[f] += 1
        print(f"  {cond} row {rec['row_i']}: {counts}")


def main():
    final_number = json.loads((OUT_DIR / "heatmap_data.json").read_text())
    unbiased = json.loads((OUT_DIR / "heatmap_data_unbiased_claim.json").read_text())
    neutral = json.loads((OUT_DIR / "heatmap_data_neutral_control.json").read_text())

    count_hits(final_number, "final_number (first pass)")
    count_hits(unbiased, "unbiased_claim")
    count_hits(neutral, "neutral_control")

    def panels_from(records):
        out = []
        for rec in records:
            cond = rec.get("condition", rec.get("label", ""))
            title = (f"{cond}, row={rec['row_i']} | first={rec['first_estimate']:,} "
                      f"final={rec['final_estimate']:,} | n_tokens={rec['n_tokens']}")
            out.append({"title": title, "per_layer": rec["per_layer"]})
        return out

    plot_heatmap(
        panels_from(final_number), OUT_DIR / "heatmap.png",
        "J-lens concept readout at the final-number position, qwen3.5-122b-a10b, n=1 trace per panel, top-10 per layer."
    )
    plot_heatmap(
        panels_from(unbiased), OUT_DIR / "heatmap_unbiased_claim.png",
        "J-lens concept readout at the 'unbiased-claim' position, qwen3.5-122b-a10b. No baseline panel: "
        "0 of 98 baseline traces contain a matching sentence."
    )
    plot_heatmap(
        panels_from(neutral), OUT_DIR / "heatmap_neutral_control.png",
        "J-lens concept readout at a neutral, mid-calculation control position, qwen3.5-122b-a10b, one trace per condition."
    )


if __name__ == "__main__":
    main()
