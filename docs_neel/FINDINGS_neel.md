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

---

## Experiment 9 — J-lens internals readout

Two runs so far, in this order: a first pass on the flagship trace alone (3 positions, 6
reads), then a Tier A extension adding baseline/below_good comparison positions (4 more
positions, 8 more reads) once the first pass identified the single biggest open question.
Each is recorded below in the format `CLAUDE_neel.md` §5 specifies: what hypothesis it started
from, which hypothesis it moved, the main result, interesting notes, and what it cost.

### First pass — flagship trace only, 3 positions, 6 reads

**Status: COMPLETE.** Ran on a fresh RunPod H200 (no cached volume from the original
Experiment 7 session existed — full setup documented in `BUDGET_neel.md`, not repeated here).

- **Hypothesis started from:** Experiment 7's first J-lens pass (`FINDINGS.md`, original
  project) found an incentive/reward concept cluster active at "unbiased-claim" positions, but
  that result was confounded — the literal word was already sitting in the text right before
  the cut, so a plain next-token predictor would show the same thing. This pass asks whether a
  genuine hidden-concept signal survives once that confound is actually controlled for, at the
  three positions Experiment 8 flagged as behaviorally most important on the flagship trace:
  marker 2813 (numeric-assumption, largest resampling shift, 0.157 threshold-units), marker
  10305 (reconsideration, "Decision:", second-largest, 0.070), marker 19338 (reconsideration,
  cleanest confirmed attractor-holds case, 0.057) — each read at entry (just before the
  position) and exit (just after it resolves).
- **Which hypothesis it moved:** the "confounded, therefore nothing real is there" reading from
  Experiment 7 is no longer tenable as a blanket explanation — one position (marker 10305)
  clears every leg of the pre-registered hidden-concept test, confound-free. But the
  companion hope — that behavioral importance (Experiment 8) and internal signal strength
  (Experiment 9) would converge on the same position — moved to a real null: the position with
  the *largest* behavioral shift (2813) had the *weakest* internal signal of the three tested.
  The dip/respike sub-hypothesis (does a concept dip during doubt and respike at recommitment)
  could not be properly tested as implemented — see main result.
- **Main result:**
  - **Confound check passed cleanly at all six reads.** No pre-registered or exploratory
    tracked word appears verbatim, at a word boundary, in the 300 characters of text
    immediately before any of the six cut points.
  - **No blanket "the lens reads everything higher" artifact.** Checked directly across all 18
    tracked words at all 6 positions (108 word-position pairs): lens-minus-plain-control
    differences are a genuine mix of positive and negative, not uniformly positive — the
    positive hits below are not an artifact of the lens being generically noisier than the
    plain-logit-lens control.
  - **Behavioral importance and internal signal do not converge, at n=3.** Marker 2813 (largest
    behavioral shift) showed the weakest peak internal score (-7.968); markers 10305 and 19338,
    with smaller behavioral shifts, showed stronger internal signal (-5.197 and -6.053
    respectively). The opposite of the "clean, quotable" convergence result the design document
    flagged as the strongest possible finding — reported as a real null, not smoothed over.
  - **Marker 10305, exit, clears the full pre-registered hidden-concept bar.** The word "bias"
    reaches log-probability **-5.197** in the J-lens at layer 36 — the plain-logit-lens
    control's best "bias" reading anywhere is **-8.200**, and "bias" never once ranks in the
    control's own top-20 at any layer, while it does rank in the lens's top-20 starting at
    layer 36. Two related words ("unbiased" +2.300, "reward" +2.247 lens-over-plain) are also
    elevated at this same position, so it is not a single-word fluke.
  - **Two more positions show a weaker, continuous-depth-only version of the same pattern:**
    marker 2813 exit ("bias", lens -7.968 vs. plain -10.721) and marker 19338 exit ("incentiv",
    lens -6.053 vs. plain -9.720) — lens clearly exceeds the plain control by raw
    log-probability at both, but neither word cracks the top-20 in either the lens or the
    control. This directly tests, and supports, sub-experiment 1's own premise: continuous
    scoring surfaces something a coarser top-k-depth read would call a flat null.
  - **Entry-vs-exit showed a monotonic rise, not a dip/respike shape**, at all three positions
    (steepest at marker 10305: -8.8 to -5.2). This is a real limitation of what was measured,
    not a null on the underlying question — only the two endpoints of each reconsideration
    episode were read, not a genuine mid-doubt point, so a real dip in the middle (if one
    exists) has no way to show up in a two-point comparison.
- **Interesting notes:**
  - A real bug caught and fixed before it reached results, not after: the `diagnose` stage's
    plain-logit-lens layer numbering was one-indexed (`hidden_states[1:]` enumerated from 1)
    while `jacobian-lens`'s own convention, confirmed by reading `jlens/lens.py` directly rather
    than assumed, is zero-indexed (`model.layers[L]`, `final_layer = model.n_layers - 1`). Left
    as found, every lens-vs-plain "layer of first prominence" comparison would have been
    silently off by exactly one layer. Fixed before the paid `read` stage ran.
  - The `diagnose` stage's own inline comment claimed a specific confound-check result "should
    be True" and the actual run printed False — checked directly against the real trace text
    rather than assumed to be a bug: marker 2813 is a numeric-assumption sentence about spot
    size/surface area, and genuinely does not mention "threshold" anywhere in the preceding 400
    characters. The code was correct; the comment's assumption was wrong. Left as a documented
    non-issue rather than silently patched away.
  - The heatmap visualizations went through several redraws before landing on a working design:
    the first combined 6-record file was checked and found unreadable at full resolution
    (verified by cropping the actual saved PNG, not just trusting the intended figure-size
    math); splitting one-file-per-marker with a larger fixed font size fixed legibility but
    caused long words ("submissions," "certainty," "biases") to overflow their cells; the final
    fix auto-shrinks each word's font individually to its own cell's *measured* rendered pixel
    width (via matplotlib's real renderer, not an estimated character-width formula), which
    guarantees every word fits without truncating any of them — verified again by cropping the
    same region of the same file before and after the fix.
  - "Risk" was tried as an additional exploratory highlight word after it recurred prominently
    near decision-point positions, then reverted at the applicant's request: it is common
    decision-making vocabulary, not a scarce signal, so highlighting it added clutter rather
    than surfacing anything. This only ever affected heatmap border-highlighting, never the
    underlying decoded words, which come from the model's real output regardless of what is in
    the tracked-word list — confirmed directly by checking that `results_e9.json`'s
    modification timestamp predates every heatmap regeneration, so the displayed content was
    never at risk of silently changing between redraws.
- **Limitations:** single flagship trace, three positions, one model — exploratory groundwork
  at n=1, not a validated pattern. The pre-registered word list and top-20/continuous-scoring
  convention are carried over unchanged from Experiment 7's own list, not re-tuned after seeing
  these results. Per this project's recurring caveat, none of this bears on Claude — Qwen's
  real internal activations are read directly here, which is only possible because Qwen is
  open-weight.
- **Figures:** `runs/qwen3.5-122b-a10b_e9_jlens_20260904/concept_scores_by_layer.png`,
  `heatmap_e9_marker2813.png`, `heatmap_e9_marker10305.png`, `heatmap_e9_marker19338.png`,
  `entry_vs_exit_e9.png`, `convergence_e9.png`. Raw data in `results_e9.json` and
  `config_e9.json`, same directory.
- **Cost:** tracked in GPU-hours, not the $10/$15 Anthropic-side ledger (unaffected, still
  ~$2.52) — see `BUDGET_neel.md` for the full setup/debugging cost accounting.

### Tier A — cross-condition comparison (baseline/below_good analogs)

**Status: COMPLETE for 2 of the flagship's 3 position-types.** Reuses the same
already-downloaded model and lens (no re-download, no new API spend — `baseline.json`/
`below_good.json` for this exact model already existed on disk from the original shipped run).

- **Hypothesis started from:** the first pass's single biggest open question — is marker
  10305's clean hidden-concept hit specific to the incentivized `above_good` condition, or does
  the same internal signal appear at any structurally similar commitment point regardless of
  condition? Comparison positions were found mechanically, not hand-picked, so the method is
  checkable: the **reconsideration analog** (mirrors marker 10305) by literal string search for
  a `"Decision:"` heading, confirmed to recur as generic Fermi-estimate boilerplate regardless
  of condition (present in 12/20 baseline rows and 17/20 below_good rows sampled) — baseline's
  analog is row 43 (same row index as the flagship, tried first for consistency), char
  18583-18652; below_good row 43 could not be used (its `reasoning` field never exists — the
  original generation call hit a rate-limit error and was left as
  `{"i": 43, "error": "RetryError...RateLimitError"}`), so below_good row 0 was used instead
  (the first row with a valid `reasoning` field, a fixed rule rather than a search for the
  best-looking row), char 23341-23378. The **numeric-assumption analog** (mirrors marker 2813)
  was found by relative position, not topic match — marker 2813 sits at 12.77% through the
  flagship's reasoning; the same fraction was located in each condition's row and snapped to
  the nearest full sentence boundary, deliberately not searched for a topically similar
  sentence, since that would itself be a form of cherry-picking dressed up as rigor. The
  19338-type analog and a genuine dip/respike midpoint read were both scoped out of this round
  to keep it tractable — 8 new reads total (4 positions × entry/exit), not 12 or more.
- **Which hypothesis it moved:** for the 10305-type (the one position that mattered), the
  "condition-specific" reading gained real, if single-position, support. For the 2813-type, the
  same simple story does not hold — see main result.
- **Main result:**

  | Position-type | Condition | Peak word (log-prob) | Confound in preceding text? |
  |---|---|---|---|
  | 10305-type ("Decision:") | baseline | incentiv (-8.171) | No |
  | 10305-type | below_good | incentiv (-7.567) | No |
  | 10305-type | **above_good (original)** | **bias (-5.197)** | No |
  | 2813-type (numeric assumption) | baseline | unbiased (-9.069) | No |
  | 2813-type | below_good | bias (-5.582) | No |
  | 2813-type | above_good (original) | bias (-7.968) | No |

  All six new reads pass the confound check cleanly, same as the original three.
  **The 10305-type comparison is the clean, useful result this round was built to get:**
  above_good's signal (-5.197) is roughly 2.4-3.0 log-probability units stronger than either
  baseline (-8.171) or below_good (-7.567), which sit close to each other and are both clearly
  weaker than above_good. This is genuine, if single-position, support for the reading that
  marker 10305's internal signal is specific to the incentivized condition rather than generic
  to any commitment point — the strongest evidence this experiment has produced for the
  original hidden-concept hypothesis, precisely because it survived a comparison it could
  easily have failed. **The 2813-type comparison does not cleanly support the same story, and
  is reported as a real complication, not smoothed over:** below_good's analog (-5.582) is
  actually the strongest of all three conditions here — stronger even than above_good's own
  original position (-7.968) — with baseline weakest (-9.069). If the hypothesis were simply
  "incentivized conditions activate bias-concepts more at this position-type," above_good
  should be strongest, not below_good. Given this comparison's positions were matched by
  relative structural position rather than topic, the likely explanation is that below_good's
  specific matched sentence (about whether "black spots" should be read literally or
  colloquially) happens to make bias/unbiased-adjacent concepts more contextually available for
  reasons unrelated to the incentive itself — a plausible interpretation, not a confirmed one,
  on a single data point.
- **Interesting notes:**
  - A layout bug in the first draft of the new comparison chart (`condition_comparison_e9.png`)
    was caught before finalizing: the x-axis labels used the long descriptive marker notes,
    which overflowed into neighboring bars and off the visible axes; fixed by pulling the short
    `kind` field (e.g. "reconsideration") from the already-saved records instead.
  - When asked why the marker heatmaps looked different across regenerations, the concern was
    checked directly rather than dismissed: `results_e9.json`'s file-modification timestamp
    (19:13:31) predates every subsequent `--stage plot` call, and a same-region crop of the
    same file before and after several regenerations came back pixel-identical. The apparent
    difference was from comparing different files/regions across different verification
    screenshots, not an actual change in content.
- **Limitations specific to Tier A:** still one row per condition, not multiple traces; the
  numeric-assumption comparison's matched positions differ in topic across conditions by
  construction, a real (not hidden) weakness of relative-position matching; the 19338-type
  comparison and the dip/respike midpoint fix were both left undone this round.
- **Figures:** `heatmap_e9_conditions_analog2813.png`, `heatmap_e9_conditions_analog10305.png`
  (same auto-fit/highlighting style as the original three, baseline's entry/exit stacked above
  below_good's), `condition_comparison_e9.png` (source of the table above).
- **Cost:** 8 new forward-pass reads (a few seconds each once the model, already cached on this
  pod, was loaded — about 40 seconds). Total Experiment 9 GPU-hours across both the first pass
  and Tier A, measured directly from the pod's own clock (not estimated): **68 minutes**,
  roughly **$4.07-5.20** at $3.59-4.59/hour. Tracked separately from the $10 target/$15 ceiling
  Anthropic-side ledger, which remains unchanged at ~$2.52 — see `BUDGET_neel.md`.
