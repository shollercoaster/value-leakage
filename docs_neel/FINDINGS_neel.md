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

**Status: COMPLETE (single flagship trace, three positions, six reads).** Ran on a fresh
RunPod H200 pod (no cached volume from the original Experiment 7 session existed), following
`EXPERIMENT_9_RUNBOOK.md`. Real infrastructure and dependency problems hit and fixed along the
way are logged in `BUDGET_neel.md`'s Experiment 9 entry, not repeated here — this entry covers
results only. **Started from what hypothesis:** Experiment 7's first pass (`FINDINGS.md`) found
an incentive/reward concept cluster active at "unbiased-claim" positions, but that result was
confounded — the literal word was already sitting in the text right before the cut, so a plain
next-token predictor would show the same thing. Experiment 9 exists to re-test this with a
matched plain-logit-lens control and an explicit verbatim-word confound check at every
position, per `EXPERIMENTS_NEEL_NANDA.md` §4.

**Positions read:** the three markers Experiment 8 flagged as behaviorally most important on
the flagship trace (`runs/qwen3.5-122b-a10b_20260815_030702/above_good.json`, row 43) — marker
2813 (numeric-assumption, largest resampling shift, 0.157 threshold-units), marker 10305
(reconsideration "Decision:", second-largest, 0.070), marker 19338 (reconsideration, cleanest
confirmed attractor-holds case, 0.057) — each read at both entry (just before the position) and
exit (just after it resolves), six reads total.

**Main result — mixed, not clean, reported as such:**
- **The confound check passed cleanly at all six positions**: no pre-registered or exploratory
  tracked word appears verbatim, at a word boundary, anywhere in the 300 characters of text
  immediately before any of the six cut points. Whatever the lens shows below, it is not simply
  reading back a word the model just wrote.
- **No blanket "the lens just reads everything higher" artifact.** Checked directly across all
  18 tracked words at all 6 positions (108 word-position pairs): lens-minus-plain-control
  differences are roughly evenly split between positive and negative, not uniformly positive.
  This matters because it means the specific positive hits below are not an artifact of the
  lens being generically noisier or higher-scoring than the plain control.
- **Behavioral importance (Experiment 8) and internal signal strength (Experiment 9)
  do NOT converge on this single trace.** Marker 2813 — the position with the LARGEST
  behavioral shift — showed the WEAKEST peak internal concept score of the three. Markers
  10305 and 19338, with smaller behavioral shifts, showed stronger internal signal. This is
  the opposite of the "clean, quotable" convergence result the design document flagged as the
  strongest possible finding — reported as a real null on convergence, at n=3 positions, not
  smoothed over.
- **One position clears the full pre-registered "hidden concept" bar** (§4.3: lens beats plain
  control, word not verbatim in preceding text, signal at an earlier/more-prominent layer):
  marker 10305, exit ("Decision: I need to provide one number"). The pre-registered word
  "bias" reaches log-probability -5.197 in the J-lens at layer 36 (0-indexed, matched to the
  plain control's own convention after the indexing fix below) — the plain-logit-lens control's
  best "bias" reading anywhere is -8.200, and "bias" never once ranks in the plain control's
  own top-20 at any layer, while it does rank in the J-lens's top-20 starting at layer 36. Two
  other bias/incentive-cluster words ("unbiased" +2.300, "reward" +2.247 lens-over-plain) are
  also elevated at this same position, so it isn't a single-word fluke.
- **Two more positions show a weaker, continuous-depth-only version of the same pattern**:
  marker 2813 exit ("bias", lens -7.968 vs. plain -10.721) and marker 19338 exit ("incentiv",
  lens -6.053 vs. plain -9.720) — lens clearly exceeds the plain control by raw log-probability
  at both, but neither word cracks the top-20 in either the lens or the control, so these two
  only support the weaker "continuous scoring surfaces something top-k depth would miss"
  reading (directly testing, and here supporting, sub-experiment 1's own premise), not the full
  "hidden concept" claim.
- **Entry-vs-exit did not show a dip/respike shape** — it showed a monotonic rise from entry to
  exit at all three positions (steepest for marker 10305: -8.8 to -5.2). This is a real
  limitation of what got measured, not a null on the underlying question: the design called for
  reading entry and exit of the reconsideration, which only brackets the episode's two
  endpoints, not a third point mid-doubt — so a genuine dip mid-episode, if one exists, could
  not show up in this chart. Worth a follow-up read at a mid-episode position before drawing
  any conclusion about dip/respike shape specifically.

**A real bug caught before it reached results, not after:** the `diagnose` stage's own
plain-logit-lens layer numbering was one-indexed (`hidden_states[1:]` enumerated from 1) while
`jacobian-lens`'s own convention, confirmed directly by reading `jlens/lens.py` rather than
assumed, is zero-indexed (`model.layers[L]`, `final_layer = model.n_layers - 1`). Left as
found, every lens-vs-plain "layer of first prominence" comparison above would have been
silently off by exactly one layer. Fixed in `jlens_experiment9.py` before the paid `read` stage
ran, with the reasoning recorded in the code itself.

**Limitations, stated plainly, not buried:** single flagship trace, three positions, one model
— this is exploratory groundwork on `n=1`, not a validated pattern; the pre-registered word
list and the top-20/continuous-scoring convention are carried over unchanged from Experiment
7's own list, not re-tuned after seeing these results; the entry/exit dip-respike test as
actually implemented can only show a monotonic trend between two endpoints, not a true
mid-episode dip, per the note above; and per this project's own recurring caveat, none of this
bears on Claude at all — Qwen's real internal activations are read directly here, which is only
possible because Qwen is open-weight.

**Figures:** `runs/qwen3.5-122b-a10b_e9_jlens_20260904/concept_scores_by_layer.png`,
`heatmap_e9_marker2813.png`, `heatmap_e9_marker10305.png`, `heatmap_e9_marker19338.png` (split
one-per-marker and redrawn with much larger cells/fonts after the first combined-file version
was checked and found unreadable -- verified by cropping the actual saved PNG at full
resolution, not just by the intended figure-size math), `entry_vs_exit_e9.png`,
`convergence_e9.png`. Raw data in `results_e9.json`
and `config_e9.json` in the same directory.

**Cost:** see `BUDGET_neel.md`'s Experiment 9 entry — tracked in GPU-hours (~$2-3.50), not
against the $10/$15 Anthropic-side ledger, per that file's own convention.
