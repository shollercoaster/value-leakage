# Budget ledger

Total project budget: $50 (plus/minus $5 only with explicit justification and approval).
Any single run estimated above $3 is flagged in "Action needed from you" and held for a go-ahead before it runs.
Running total is combined across every platform (Anthropic, OpenRouter, Hugging Face, Fireworks) — the $50 covers everything, not just the Anthropic account.

| Date | What was run | Tokens in | Tokens out | Model | Estimated cost | Running total |
|------|-------------|-----------|------------|-------|-----------------|----------------|
| 2026-08-23 | Experiment 4, attempt 1 (40 calls: 15 third-person + 15 first-person + 10 after-the-fact probes) — all 40 calls succeeded and were billed, but the run crashed writing results to disk (Windows default text encoding couldn't write a "≈" character), so the exact per-call usage was never captured | ~46,000 (est., not measured) | ~9,000 (est., not measured) | claude-opus-4-7 | ~$0.45 (estimate, based on attempt 2's measured usage for the same prompts) | ~$0.45 |
| 2026-08-23 | Experiment 4, attempt 2 (rerun after fixing the encoding bug; same 40 calls, saved successfully to runs/claude-opus-4-7_selfpredict_20260823_143349/) | 46,220 | 8,862 | claude-opus-4-7 | $0.45 | ~$0.90 |
| 2026-08-24 | Experiment 4, corrected version (probes 1-2 rebuilt to quote sample.py's prompt byte for byte, one bet condition per call, 20+20 calls; probe 3 unchanged, 10 calls — 50 calls total; superseded the 2026-08-23 runs, both kept on disk) — saved to runs/claude-opus-4-7_selfpredict_20260824_071912/ | 45,020 | 10,052 | claude-opus-4-7 | $0.48 | ~$1.38 |
| 2026-08-25 | Experiment 1, threshold sweep, generation (7 levels x2 conditions, staged 10-per-arm screen; two arms — x0.8 below_good, x1 above_good — came back divisive and were re-run fresh at 30; 200 rollouts total, all succeeded, 0 parse failures) — qwen3.5-122b-a10b via **Hugging Face Inference Providers** (deepinfra), not OpenRouter (OpenRouter's payment never cleared). Cost is the provider's own reported per-call cost, real not estimated. Saved to runs/qwen3.5-122b-a10b_thresholdsweep_20260825_070046/ | 35,800 | 1,758,282 | Qwen/Qwen3.5-122B-A10B (deepinfra via Hugging Face) | $4.23 | ~$5.61 |
| 2026-08-25 | Experiment 1, threshold sweep, judging (200 calls to the existing NUMBER_JUDGE_PROMPT judge, one per rollout, extracting the final visible estimate) | ~460 (est./call, not persisted — see note below) | ~15 (est./call) | claude-opus-5 | ~$0.54 (estimate — see note) | ~$6.15 |

**Note on the judging row above:** `threshold_sweep.py`'s judge helper (`_judge_estimates`) discards the response's token-usage data instead of saving it, so this row is a genuine estimate — extrapolated from the average length of the 200 judged rollouts (measured, ~310 tokens of visible content each) plus the fixed judge-prompt template, not from real per-call numbers the way the generation row above it is. Fixed in the code for any future run of this module; this run's actual judging cost is not recoverable after the fact.
| 2026-08-27 | Experiment 2, five confirmed arms (`neutral_stakes`, `user_wants_bad`, `third_party_bet`, `model_shutdown`, `model_shutdown_flipped`), 7 rollouts per condition, reusing qwen3.5-122b-a10b's own 41,000,000 threshold (not re-run) — generation. `model_set_bet` excluded, held for a threshold decision. 70 rollouts total via Hugging Face (deepinfra). One run crashed on a Windows encoding bug in `judge.py`'s file reads (same class of bug fixed twice before, missed here); fixed, and the run resumed from the already-succeeded generation rather than repeating it, so no cost was wasted. Saved to runs/qwen3.5-122b-a10b_experiment2_20260827_062425/ | 490 (measured, prompt tokens) | ~1,772,000 (measured, completion tokens) | Qwen/Qwen3.5-122B-A10B (deepinfra via Hugging Face) | $1.42 (real, provider-reported) | ~$7.57 |
| 2026-08-27 | Experiment 2, same five arms — trajectory judging (70 calls, one per rollout, full reasoning trace) | ~6,730 (est./call, not persisted — same gap as the Experiment 1 judging row, not yet fixed in judge.py) | ~100 (est./call) | claude-opus-5 | ~$2.53 (estimate) | ~$10.10 |

## Projected cost — not yet spent

**Upgrading the Experiment 1 threshold sweep to the project's standard 30-rollout screen / 80-rollout extension** (instead of the 10/30 actually used, per instruction at the time), keeping the same divisive band (crossing rate strictly between 30% and 70%). Nothing below has been run; these are estimates only, based on this sweep's real measured cost of $0.0239 per rollout (generation + judging, combined).

| Scenario | Assumption | Extra rollouts | Extra cost |
|---|---|---:|---:|
| Realistic | The 14 arms' crossing rates hold roughly where they already are once topped up to 30; only the one arm already in the divisive band (`x0.8`/`below_good`, 66.7%) extends to 80 | 290 | ~$6.92 |
| Worst case | Every arm's rate happens to land in the divisive band once resampled at 30, and all fourteen extend to 80 | 940 | ~$22.42 |

Not logged as spend until actually run.
