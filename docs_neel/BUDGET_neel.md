# Budget ledger — Neel Nanda extension

Total budget for this extension: $10, maximum $15 (per `CLAUDE_neel.md`, section 3). This is
a fresh allocation, tracked separately from the original project's $50 ledger in
`value-leakage/docs/BUDGET.md`.

Any single run estimated above $3 is flagged in "Action needed from you" and held for a
go-ahead before it runs.

Experiment 9's rented graphics-processing-unit time is tracked separately against its own
one-hour target (`EXPERIMENTS_NEEL_NANDA.md`, section 2.2), not against the dollar figure
below.

| Date | What was run | Tokens in | Tokens out | Model | Estimated cost | Running total |
|------|-------------|-----------|------------|-------|-----------------|----------------|
| 2026-09-04 | Experiment 8, Tier 1, pipeline smoke test (3 continuation calls at one hand-verified numeric-assumption position in the flagship trace, run before committing to the full 22-position sweep — confirmed the natural-resampling continuation mechanism works on this route, and surfaced a real discrepancy from the original project's documented prerequisite check, see Current Progress) via Hugging Face (deepinfra), `qwen3.5-122b-a10b` | 1,992 x3 (real, provider-reported) | 4,975 + 5,694 + 4,000 = 14,669 (real, provider-reported) | Qwen/Qwen3.5-122B-A10B (deepinfra via Hugging Face) | $0.0370 (real, provider-reported: $0.01252 + $0.01424 + $0.01018) | ~$0.04 |
| 2026-09-04 | Experiment 8, Tier 1, full generation sweep (22 hand-verified positions in the flagship trace — 15 reconsideration points, 7 numeric-assumption contrasts — 6 treatment + 3 null-baseline continuations each, 198 calls total, 0 errors, 0 truncated) via Hugging Face (deepinfra), `qwen3.5-122b-a10b`. Saved to `runs/qwen3.5-122b-a10b_e8_thoughtanchors_20260904/generations_e8.json` | 811,701 (real, provider-reported) | 703,648 (real, provider-reported) | Qwen/Qwen3.5-122B-A10B (deepinfra via Hugging Face) | $1.9241 (real, provider-reported, summed per-call) | ~$1.96 |
| 2026-09-04 | Experiment 8, Tier 1, judge fallback stage (135 parse-failure calls to `claude-haiku-4-5`, since the regular-expression parser only recovered 63/198 estimates; plus 100 validation calls to `claude-opus-5` on a random sample, agreement rate 100%). First attempt failed entirely (all 235 calls rejected, `credit balance too low`, $0 billed) — re-run after the applicant added credit to the Anthropic key. `thought_anchors.py`'s judge stage does not currently save per-call token usage (same gap this project's own `judge.py` has, noted in `value-leakage/docs/BUDGET.md`), so this row is an estimate, not a measured figure — extrapolated from the real average judge-prompt length (594 tokens, measured directly from the 135 saved failure texts) and an assumed ~20-token completion (the judge only outputs a short tag) | ~594/call (measured prompt length, not per-call API-reported) | ~20/call (est., not persisted) | claude-haiku-4-5 (135 calls) + claude-opus-5 (100 calls) | ~$0.44 (estimate: haiku ≈$0.094 + opus-5 validation ≈$0.347) | ~$2.40 |
| 2026-09-04 | Experiment 8, Tier 1 follow-up: 3 more null-baseline continuations at each of the 3 largest-shift positions (9 new calls total, generation via Hugging Face/deepinfra), plus judging the 4 of those 9 the regex parser couldn't read (`claude-haiku-4-5` + 4-case `claude-opus-5` validation, 100% agreement again). Confirmed the top position's shift was partly a small-sample artifact — see `FINDINGS_neel.md` | 32,037 (real, generation) + ~2,400 est. (judge) | 39,499 (real, generation) + ~100 est. (judge) | Qwen/Qwen3.5-122B-A10B (9 calls) + claude-haiku-4-5 (4) + claude-opus-5 (4) | $0.1041 (real, generation) + ~$0.017 (est., judge) = ~$0.121 | ~$2.52 |
