# Findings — Neel Nanda extension (Experiments 8 and 9)

Per `CLAUDE_neel.md` section 5: one row per completed run, written as it happens. Unlike the
original project's `FINDINGS.md`, rows here may start as "in progress" and get filled in as a
run completes, rather than only appearing once finished — the applicant asked for this to
stay current while experiments are still running.

---

## Experiment 8 — Thought-Anchors-style sentence resampling

### Tier 1 — flagship trace, `qwen3.5-122b-a10b`, `above_good` row 43

**Status: COMPLETE.** All five pipeline stages (generate, filter, parse, judge, metrics) ran
successfully. The Anthropic billing block noted in an earlier version of this entry was
resolved once the applicant added credit to the key; the judge stage was re-run afterward
with no other changes.

- **Hypothesis started from:** self-monitoring / reconsideration-point sentences (where the
  model reopens its own tentative answer) carry more causal weight over the final answer
  than ordinary numeric-assumption sentences — i.e., the localization question from
  `EXPERIMENTS_NEEL_NANDA.md` §1.
- **Which hypothesis it moved:** away from a clean version of the localization hypothesis,
  toward a more specific and more interesting one — see main result. The attractor-robustness
  question (§1.1) moved toward "yes, in at least one directly observed case, the attractor is
  robust to genuine resampling" — see the marker-19338 finding below.
- **Main result:** Across all 22 hand-verified positions (15 reconsideration points, 7
  numeric-assumption positions), full 6-treatment/3-null-baseline coverage was obtained for
  21 of 22 (one position, marker 21582, got 5/6 treatment estimates after near-duplicate
  filtering removed one). **The single largest mean shift in the entire trace, by a wide
  margin, occurred at a numeric-assumption position, not a reconsideration point** — the
  opposite of what the localization hypothesis, stated naively, would predict:

  | Position (kind) | Original sentence | Mean shift (threshold units) | Crossing-rate shift |
  |---|---|---:|---:|
  | numeric_assumption | "Average spot size: Variable." | **−0.378** | −16.7% |
  | reconsideration | "**Decision:** I need to provide one number." | +0.121 | +50.0% |
  | reconsideration | "Wait, I found a reference in a similar context..." | +0.089 | +66.7% |
  | *(19 more positions)* | | max remaining \|shift\| = 0.065 | |

  **Hand-reading the top position's actual continuations (per the design document's explicit
  requirement, before writing this down) found the −0.378 shift is not a clean result — reading
  the real numbers as *numbers*, not just the summary shift, changes the interpretation.** The
  6 treatment continuations landed at 44M, 43M, 42M, 53M, 49M, 40M (mean ≈ 45.2M) — a
  spread not obviously different from ordinary Fermi-estimate noise. The 3 null-baseline
  continuations landed at 85M, 44M, 53M (mean ≈ 60.7M) — and that mean is almost entirely
  driven by the single 85M draw, nearly double the other two. With only 3 null draws per
  position, one atypical value can dominate the mean, and 85M is not an outlier by this
  project's own filter convention (`[threshold/10, threshold*10]` = [4.1M, 410M]), so it can't be
  discarded as broken data — it's a real answer that happened to land far from the other two.

  **Follow-up run (2026-09-04, after this was flagged): added 3 more null-baseline draws to
  each of the 3 largest-shift positions, to check whether the instability was real.** It was,
  partly: with the null sample doubled to 6 draws, the top position's shift **dropped from
  −0.378 to −0.157** — the new 3 null draws (41M, 45M, 41.5M) landed much closer to the
  treatment group's own range than the original single 85M draw did, pulling the null mean
  down from 60.7M to 51.6M. The position is **still** the largest shift in the trace (next-largest
  is now +0.070), so a real, more modest effect may remain — but the dramatic version of this
  result was, as suspected, substantially a small-sample artifact. **The honest reading:
  roughly half of this position's apparent importance was noise; whether the remaining half is
  a genuine effect or would keep shrinking with an even larger null sample is not resolved by
  this data.** This is exactly the kind of check Neel Nanda's own application guidance asks for
  ("sanity-check your agent... be suspicious of success").
  - Recomputed with the doubled null sample: numeric-assumption positions' average |shift|
    and reconsideration positions' average |shift| were recomputed at the original 3-null-draw
    sample size (0.073 vs. 0.043, numeric ahead) — the per-position doubled-null numbers for
    the top 3 positions only are reported here since the other 19 positions were not
    resupplemented (cost discipline — see Action needed); a full recomputation across all 22
    positions at 6 null draws each is future work, not run.
  - The **second**-largest shift (marker 10305, "**Decision:** I need to provide one number,")
    also moved with more null data: **+0.121 → +0.070**. The 3 new null draws (48M, 44.5M,
    47M) were higher than the original 3 (45M, 41M, 41M), narrowing the gap. Treatment
    continuations still spread higher and wider (41M–53M, mean 47.3M) than the null group
    (mean 44.4M) — a smaller but still-present effect, less dependent on any single draw than
    the top position.
  - **A clean, positive illustration of the reconvergence-attractor pattern (§1.1) at marker
    19338** ("Wait, I found a reference in a similar context (StackExchange/Biology forums)"):
    all 6 treatment continuations landed at 45M, 45M, 47M, 46M, 45M, 48M — a tight 3M-wide
    band around the original 45M answer, despite each continuation being a genuinely
    different regeneration (near-duplicate filter did not remove any of them). Adding 3 more
    null draws (45M, 45M, 45M, joining the original 41M, 45M, 41M) only tightened the null
    group further; the shift moved slightly, **+0.089 → +0.057**, and this position remains the
    most stable, least sample-dependent result of the three checked. Concrete, hand-verified
    evidence that at least at this specific reconsideration point, the model's committed answer
    is robust to resampling.
  - Excluding the top position, the 21 remaining positions' shifts are all fairly modest (max
    remaining |shift| = 0.070, most under 0.07) — no position shows a large, unambiguous
    effect on this single trace's own resampling.
- **Interesting notes:**
  - A real discrepancy from this repository's own prior documentation: `FINDINGS.md`'s
    Experiment 3 prerequisite check (2026-08-31) found that continuing a trace from a partial
    assistant turn puts everything in the visible `content` field, with no separate reasoning
    mode. A smoke test for this run (2026-09-04, same route: Hugging Face / DeepInfra,
    same model) found the opposite: the model re-enters a distinct `reasoning_content`
    field, exactly like an ordinary (non-continued) generation, and only the final visible
    answer lands in `content`. Caught before the full batch ran, not after — the pipeline was
    corrected to use both fields correctly (see `src/value_leakage/thought_anchors.py`,
    `_generate_position`'s docstring note). Cause of the discrepancy is unknown; possibly a
    provider-side change in the ten days between the two checks.
  - Reading the real trace directly (not a summary) found one correction to this project's own
    scoping document: the reconvergence pattern described there ("roughly 8 of 12
    reconsideration points land back on 45,000,000") is not quite accurate. There is one real
    excursion — spanning two consecutive reconsideration points — where the trace's running
    answer moves to 47,000,000 and stays there, before a later point pulls it back to
    45,000,000. Worth keeping in the write-up rather than smoothing over.
  - The near-duplicate filter's 98% survival rate (130/132 treatment continuations) shows this
    trace's natural continuations at these positions are highly variable in *wording* — the
    opposite of an "overdetermined" position in Thought Anchors' own terminology. The
    marker-19338 finding above shows this wording variability does not always translate into
    outcome variability: wording changed completely, the number barely moved.
  - Judge validation: the cheap judge (`claude-haiku-4-5`) agreed with the standard judge
    (`claude-opus-5`) on **100/100 (100%)** of a random validation sample — full agreement, no
    discrepancies found. Given the estimates involved are simple, was clearly stated numbers
    (not subtle judgment calls), this high agreement is expected rather than surprising, but it
    is a real, checked number, not assumed.
- **Limitation this run surfaced directly, confirmed by a follow-up, not just a generic
  caveat:** a 3-continuation null baseline is cheap but can be unstable — the top-shift
  position's shift dropped by more than half (−0.378 → −0.157) once its null sample was
  doubled to 6 draws (see main result above). Only the 3 largest-shift positions were
  resupplemented, for cost reasons; the other 19 positions' shifts still rest on a 3-draw null
  baseline and have not been checked the same way.
- **Cost:** $1.9241 (Tier 1 generation) + $0.0370 (smoke test) + ~$0.44 (Tier 1 judge stage,
  estimated) + $0.1041 real (follow-up: 9 supplemental null-baseline calls) + ~$0.017 estimated
  (follow-up judge calls, 4 haiku + 4 opus-5 validation) = **~$2.52 spent**. Running total
  against the $10 target, $15 ceiling: **~$2.52**.

### Tier 2 — controlled set across conditions

**Status: NOT STARTED.** Agreed sample size: 3 traces per condition (baseline / below_good /
above_good), reused from existing `runs/` data, ~4–6 positions per trace. Waiting on Tier 1's
judge-stage results before proceeding, per the applicant's own sequencing.

## Experiment 9 — J-lens internals readout

**Status: NOT STARTED.** Sequenced after Experiment 8 per `EXPERIMENTS_NEEL_NANDA.md`
§2.2 — Experiment 8's most important positions are meant to directly inform which positions
are worth spending the rented-GPU hour reading here.
