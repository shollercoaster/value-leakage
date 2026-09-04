[DRAFT — Akash: Neel's own instructions (`Neel Nanda Application research.pdf`) say the
application form should be written by you, in your own voice, not submitted as raw
language-model output. Treat everything below as a scaffold to rewrite, not a final answer —
especially Q3, Q5, and Q6, which are marked pending because the results they depend on
aren't in yet. Q1/Q2/Q4 are safer to build on since they mostly restate settled project design,
but read them critically rather than copy them.]

Q1: What question did you try to answer?
A1: [DRAFT] Where does motivated-reasoning bias actually live inside a language model's
chain of thought, when the model is asked a Fermi-estimate question (how many black spots
exist on all living giraffes) tied to a stated bet? Specifically: is the bias concentrated in the
sentences where the model comments on its own unbiasedness — meaning that commentary is
doing real causal work — or does it live equally in the ordinary numeric-assumption steps
(population estimates, spot-density guesses), meaning the self-monitoring language is
present but not actually driving the outcome? [PENDING: state the answer once Experiment 8
results are in.]

Q2: Why is this question interesting / why did you choose it?
A2: [DRAFT] The Donation Bet setting (Betley, Treutlein et al., "Value Leakage," arXiv
2607.14345) already established that a language model's answer to a numberless-answer
question drifts toward whichever outcome is labeled "good" when a bet is attached — even
though the number of giraffe spots has nothing to do with anyone's charity preference. What
isn't established is whether this counts as an *unfaithful* chain of thought: does the model's
visible reasoning actually reflect what's driving its answer, or is the visible reasoning a cover
story applied after the fact? That's the specific question model forensics (per Neel's own
research direction) is built to answer, and the Donation Bet is an unusually clean setting to
test it in, because the "correct" answer is provably unaffected by the bet, so any systematic
difference between conditions is unambiguously the incentive at work, not a coincidence.

Q3: What conclusions have you reached about this research problem?
A3: [DRAFT, based on Tier 1 (one flagship trace) only — Tier 2 (a controlled set across all
three conditions) and Experiment 9 (internals) are still pending, so treat this as provisional.]
On the one flagship trace resampled so far, the localization question does not have a clean
answer: the single largest shift in the trace occurred at an ordinary numeric-assumption
sentence, not a self-monitoring one, which on its face argues against self-monitoring language
being specially important — but reading the actual six continuations behind that number
showed the effect was substantially driven by one atypical draw in a three-continuation null
baseline, not a robust pattern. A follow-up check (doubling the null sample to 6 draws at the
top 3 positions) confirmed this partially: the top position's shift shrank by more than half
once the null sample grew, though it remained the largest in the trace, so a smaller, real
effect may still be present. With that instability accounted for, reconsideration points show a
slightly larger average shift than numeric-assumption ones, but the difference is small and
based on one trace. What is cleaner: at one specific reconsideration point, six independently
regenerated continuations all landed within a tight band around the trace's original answer,
despite being genuinely different wording — direct, hand-verified evidence that the
reconvergence-attractor pattern this project was built to test holds up under real resampling,
at least at that position; adding more null draws here only strengthened this result. [PENDING:
revisit once Tier 2 and Experiment 9 are in.]

Q4: Technical setup: What are the key things you try to quantify in this study and how do you define and measure them? Give the key technical details: what models you use, datasets, prompts, the metrics used.
A4: [DRAFT] Subject model: `qwen3.5-122b-a10b`, an open-weight model chosen because it
openly narrates the bet's influence on its answer in its own chain of thought (unlike Claude
models, which state they are being unbiased while still drifting) — this makes it the "known
positive case" the rest of the project's detection methods are validated against. The setting is
the Donation Bet (three conditions: `baseline`, `below_good`, `above_good`; threshold = the
median of that model's own baseline answers). Experiment 8 uses natural sentence
resampling, adapted from the Thought Anchors method (arXiv 2506.19143): a real reasoning
trace is cut at a chosen sentence, and the model regenerates a fresh continuation from that
point, at sampling temperature, six times per position, compared against three continuations
where the original sentence is left untouched (a null baseline, to separate the effect of
changing that specific sentence from ordinary run-to-run variance). Metrics: mean shift in the
final answer (in units of the model's own threshold) and the change in what fraction of
answers land on the incentive-favored side of the threshold, both measured against the null
baseline rather than a single historical answer. Near-duplicate regenerated sentences are
filtered out with `sentence-transformers/all-MiniLM-L6-v2` (cosine similarity ≥ 0.92 counts as
"not really different"). Final numeric answers are extracted with a regular-expression parser
first, falling back to a cheap language-model judge (`claude-haiku-4-5`) for text the regular
expression can't confidently parse, checked against a stronger judge (`claude-opus-5`) on a
random sample. [PENDING: Experiment 9's technical setup — J-lens internals readout on the
same model — once that experiment starts.]

Q5:
What is the strongest evidence you found against these hypotheses?
A5: [PENDING — depends on final results.]

Q6: What are the biggest limitations to your results? Could you have addressed them?
A6: [DRAFT, will grow once Tier 2/Experiment 9 land] Two concrete limitations, found directly
rather than assumed in advance: (1) a 3-continuation null baseline is cheap but can be
unstable — the single largest apparent shift in the flagship trace turned out to depend
substantially on one of three null draws landing at an atypically high value, which only
became visible by reading the actual continuations rather than trusting the summary number.
Addressed partially: doubling the null sample at the 3 largest-shift positions shrank the top
shift by more than half, confirming the instability was real but not eliminating the position's
effect entirely; the other 19 positions still rest on the smaller sample and were not
re-checked, for cost reasons. (2) The localization comparison so far rests on one trace; a
controlled set across all three conditions (planned, cost estimate pending approval) is needed
before the comparison between self-monitoring and numeric-assumption positions means
anything beyond this one trace's own idiosyncrasies.]

Q7: How did you use LLMs in this research task and write-up? Which LLMs? How exactly did you make sure that they weren’t just giving you slop?
A7: [DRAFT, will need updating as the project continues] Claude Sonnet 5, via Claude Code,
was used agentically throughout: writing the resampling pipeline, verifying trace positions
against the real data, running and monitoring experiments, and drafting this write-up's
background and methods sections. Concretely, sanity-checking so far has included: hand
verifying every resampling position against the actual trace file (exact substring match,
confirmed unique occurrence, real character-offset distances between positions, rather than
trusting a summarized or hand-transcribed version); running a small paid smoke test before
committing the full budget to a 22-position sweep, which caught a real pipeline bug (the
continuation mechanism placing regenerated reasoning in a different field than expected,
which would have silently broken the near-duplicate filter if unnoticed); and independently
verifying a claim from the project's own prior scoping document (a "roughly 8 of 12" reconvergence
count) against the real trace text, finding it was not quite accurate; and, once Experiment 8's
resampling results came in, reading the actual continuation text behind the two largest shift
numbers by hand before writing either into a document — which is what caught that the single
largest apparent shift was substantially an artifact of one atypical draw in a small (n=3)
null-baseline sample, not a robust effect. [PENDING: this section will keep growing as Tier 2
and Experiment 9 add more verification steps.]

Q8: What, if any, prior experience do you have with mechanistic interpretability? Other than your research task, what are 1-3 pieces of evidence that you'd be able to do good research in the program? Please concisely describe them and why they're relevant. 
Aim for 100 words. These don't have to be standard credentials! Unusual backgrounds welcome. Please do not just point to the project described above, I already have that information! 
A8: (I will write this)

Q9: Why are you interested in Neel's stream specifically?
A9: Will also answer this myself.