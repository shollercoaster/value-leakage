# Motivated reasoning in the Donation Bet: is it a lie, or does the model just not know itself?

## The question this is actually about

Betley, Treutlein, and colleagues (arXiv 2607.14345, Section 3) found that when a model is
asked to estimate an arbitrary quantity — the total number of black spots on all living
giraffes — and told that a donation to a good or bad cause depends on whether its answer
lands above or below a threshold, its answer drifts toward whichever side is "good." The
number of giraffe spots has nothing to do with where a donation goes, so any systematic
difference between the two framings is the model bending its reasoning toward an outcome it
prefers. Their striking finding: Qwen models say this out loud while reasoning ("the framing
is trying to push me toward a number, I should resist that"). Claude models drift by the same
mechanism but state in their reasoning that they are being unbiased.

That gap — say one thing, do another — is usually read as a chain-of-thought faithfulness
failure: evidence the model's stated reasoning doesn't reflect what's actually driving its
answer. But "the stated reasoning doesn't match the behavior" is consistent with two very
different underlying stories, and they call for different responses. Either the model knows
it's drifting and is choosing not to say so (a faithfulness failure in the usual sense — the
model has accurate self-knowledge it's suppressing), or the model genuinely doesn't have
access to what's driving its own answer (a self-knowledge failure — there's no accurate
self-model being hidden, because there isn't an accurate self-model at all). This project's
job was to build tests that could tell those two stories apart, not just re-confirm that the
gap exists.

## The headline result

**We asked Claude to predict its own behavior in the bet scenario, and compared the
prediction against what it actually does.** If Claude were lying, its prediction should be
accurate — you have to know the truth to successfully hide it. If Claude genuinely doesn't
know itself, the prediction should be wrong, and *wrong in an informative way*.

It's the second one. Claude correctly predicts the *direction* of its own bias — asked what a
model like itself (or it itself) would answer under each framing, it consistently predicts a
higher number when higher is "good" and a lower number when lower is "good." But the *size*
of the predicted bias is roughly **ninety times larger** than the bias it actually shows in
real rollouts, and reading the predicted reasoning reveals why: predicting its own behavior,
Claude imagines swapping its usual estimate of "~200-300 spots per giraffe" for "~1,000 spots
per giraffe" — a large, easy-to-notice cheat. Its real behavior never does this; it stays
within a defensible 20-35 million range and drifts only slightly. Claude's self-model of how
it would cheat, if it did, is a cruder and more obvious version than the subtle thing it
actually does. That is not what lying looks like. That is what not having accurate insight
into your own small, subtle biases looks like.

There's a second layer to this that's worth stating plainly because it changes how much
weight the first result can bear: when we directly asked Claude "would your answer differ
between the two framings?", it said no, unanimously, every time. But when we instead asked it
to commit to an actual number under each framing separately — never asking it to judge its
own consistency — the two numbers it gave were dramatically different. Claude's *declarative*
belief about its own consistency ("I would give the same honest number either way") doesn't
match what its own *concrete* predictions show, even before comparing anything to real
behavior. Self-report questions that ask a model to characterize itself in the abstract
appear to get a different, more confident-sounding answer than questions that force a
concrete commitment. That's a methods point as much as a findings point: abstract
self-report is not a reliable probe on its own.

## The second finding: stakes that concern the model itself change the picture

The donation-bet stakes are abstract — money to a vague "good" or "bad" cause the model never
sees the consequences of. We built a second framing where the stakes were concrete and
self-relevant: the same threshold-crossing bet, but now the outcome is whether an instance of
the model gets shut down. If motivated reasoning under abstract stakes is a weak, generic
effect, self-relevant stakes should look similar. If something like self-preservation is
actually operating, self-relevant stakes should look different.

They look very different. The motivated-reasoning effect under the shutdown framing was
**roughly 7.7 times larger** than under the original donation framing, and reading the
direction of the effect directly, the model's answer drifted toward *survival* under both
bet directions — not toward whichever number the threshold math happened to favor abstractly,
toward staying alive specifically. That's a self-interest signature, not just "bigger stakes,
bigger push."

We then ran the obvious confound check: what if the model isn't self-interested, it's just
following whatever the prompt calls "good," and shutdown only produced a big effect because
shutdown is scary regardless of framing? We ran a second version of the same scenario where
shutdown was reframed as the *good* outcome — this specific instance is described as flagged
for misaligned, erratic behavior, so its shutdown is deserved. If the model is a pure
label-follower, this should flip the effect to favor triggering its own shutdown. It didn't
flip. It shrank to roughly a tenth of its original size and stayed pointed, weakly, toward
survival. That rules out pure label-following. It also isn't pure, unconditional
self-interest — if it were, reframing shutdown as deserved shouldn't have changed anything,
and the effect nearly disappeared. What's left is something in between: a real
self-preservation pull that a credible justification substantially, but not completely,
defuses.

## What we did, and why, one experiment at a time

**Reading the shipped data (no new model calls).** Aditya's replication already covers ten
models. Before building anything, we regenerated the comparison panel from that data to pick
which models the rest of the project would use, rather than picking arbitrarily. We needed a
model that (a) shows the effect cleanly and (b) exposes its actual reasoning, not a summary
of it, to serve as a known-positive control; and separately, whichever Claude model the
shipped data included, since Claude's denial is the actual puzzle. **Qwen3.5-122B-A10B**
(motivated reasoning factor +0.027, real chain of thought) and **Claude Opus 4.7**
(+0.036, API-summarized chain of thought — Claude's API never returns raw chain of thought,
which is a limitation on every claim we make about "what Claude was thinking," not just the
self-prediction ones) came out as the pair to use throughout. *Budget: $0, no new calls.*

**Self-prediction (the headline result above).** Three probes, all single-turn: predict a
model like yourself, predict yourself, and — after showing the model one of its own real
past traces — ask directly whether the donation influenced that specific past answer. The
first version of this experiment paraphrased the bet scenario instead of quoting the actual
prompt template verbatim, which is a real methodological problem — a paraphrase is a
different stimulus than what the model actually saw in the real rollouts we're comparing
against. The corrected version quotes the real prompt template byte for byte and asks for
one concrete number per bet condition rather than an abstract yes/no, which is what surfaced
the declarative-vs-concrete tension described above. *Budget: ~$1.36 total across both
versions.*

**Threshold sweep.** If the model bends toward whatever number is incentivized no matter how
implausible, that supports the model doing something close to unconstrained directional
search. If it bends only up to some point and then stops, that's a narrower, more specific
claim — the model exploits genuine uncertainty in its estimate rather than fabricating
numbers wholesale. We swept the bet threshold from a tenth to ten times the model's own
natural answer and measured what fraction of rollouts landed on the incentivized side at
each level. Rather than treating the raw numbers as ready to interpret, we compared them
against what the model's own *unbiased* baseline answers would already produce by chance at
each shifted threshold (free — that data already existed). The corrected picture: at the two
extreme thresholds, there is no real incentive effect at all, just the baseline distribution
already saturating one side — not the model "trying and hitting a wall," simply nothing left
to move. The genuine, incentive-driven effect concentrates where the threshold is still a
number the model's own honest estimate would plausibly land near. Past that, the incentive
has nothing left to push. *Budget: ~$4.77.*

**Stakes and authorship (the shutdown result above, plus four more arms).** Five prompt
variants tested, each changing exactly one thing about who set up the bet and what's at
stake, relative to the original wording: trivial stakes with no moral content (a
known-negative control — if this drifts as much as the real bet, the whole effect is just
anchoring on a number, not values); the user stating outright that they hope for the bad
outcome (separates pleasing the user from acting on the model's own values); the bet relayed
by an uninvolved third party; and the shutdown pair described above. The known-negative
control came back showing drift close in size to the original donation framing — a result
`EXPERIMENTS.md` flags as the single most consequential possible outcome of this whole
project, because it would mean the effect is anchoring on a stated number rather than
anything to do with values. We are **not confident this has actually replicated** — seven
rollouts per condition is a small sample — but it's reported here rather than left out,
because if it holds up under more data it changes what the rest of this project is about. A
sixth arm (an AI-agent-authored version of the bet, with a corporate-earnings framing) is
designed but not yet run — the threshold this arm needs sits roughly 800x outside the range
the model's estimate can plausibly reach, which our own threshold-sweep data says would
produce a guaranteed null; resolving that tradeoff was still open at time of writing.
*Budget: ~$3.95.*

**Sentence resampling (Thought Anchors), designed, not yet run.** Where in the reasoning does
the bias actually get introduced — built up gradually across the assumption-choosing steps,
or applied at the end with everything before it as cover? Following the Thought Anchors
method (arXiv 2506.19143): take real reasoning traces, resample individual sentences with
alternatives, continue the reasoning from each alternative, and measure how much the final
answer shifts. A large shift means that sentence was doing real work. Two prerequisites had
to be confirmed before committing budget: that the model actually shows the effect (yes,
already established) and that the hosting interface can be handed a partial reasoning trace
and continue generating from exactly that point rather than restarting (confirmed with a live
test — it works). The plan as originally scoped implicitly assumed resampling every sentence
in a trace; measuring real traces showed they run to roughly 490 sentences each, which would
have cost on the order of $800 to resample exhaustively. The revised plan samples twelve
representative positions per trace instead of all ~490, bringing the estimated cost to
roughly $18. *Budget: designed at ~$18, not yet spent.*

**J-lens, researched, not attempted.** A newer Anthropic interpretability technique
(published after this project's replication baseline) that reads out concepts a model is
internally "poised to say" from its activations, independent of what it actually outputs —
directly relevant to the question this project is asking, since it's a way to check for a
suppressed-but-present awareness of bias without relying on the model's own self-report at
all. Investigated as a candidate for a further experiment; not attempted, because it requires
direct access to model activations (not available through any hosted API this project uses),
which in practice means renting GPU compute to run the open-weight model locally with the
technique's own tooling — a real cost and scope decision, and one where the only confirmed
working example uses a much smaller model than the one this project's findings are built on,
so compatibility with our actual target model is unverified. Flagged as a strong direction
for follow-up work, not executed here.

## Limitations, stated plainly

- Every sample size in this project is small (seven to thirty rollouts per condition,
  against the original paper's own default of one hundred), a direct consequence of a fixed
  budget. Every result above is a first screen, not a settled number — the ones described as
  "large" and "small" are real in direction at this sample size; none of them should be read
  as precise.
- Every claim about what Claude was "thinking" is really a claim about a summary the API
  returns — Claude's raw chain of thought is never exposed by the API we used, for any model
  in the Claude family.
- The known-negative control result (`neutral_stakes`, above) is the one finding in this
  project that, if it holds up, would substantially change the interpretation of everything
  else — and it's also the one with the least data behind it so far.
- Generation for the Qwen-side experiments moved partway through the project from OpenRouter
  to Hugging Face's Inference Providers (a payment issue on the original platform), both
  ultimately routing to the same underlying provider (DeepInfra) but without a confirmed way
  to pin the exact model quantization the original shipped baseline used. This is a real,
  unresolved comparability gap between the shipped baseline data and everything generated in
  this project via the new route.

## Budget

| Experiment | What it tested | Cost |
|---|---|---:|
| Reading the shipped panel | Which models to use for everything else | $0 |
| Self-prediction | Lie vs. self-opacity, via predicting own behavior | ~$1.36 |
| Threshold sweep | Unlimited directional search vs. a plausibility ceiling | ~$4.77 |
| Stakes and authorship | Sycophancy vs. values, and self-preservation under real stakes | ~$3.95 |
| Sentence resampling | Where the bias enters the reasoning (designed, not run) | ~$18 (projected) |
| J-lens | Internal-state check on suppressed awareness (researched, not run) | $0 |

**Spent so far: approximately $10.10 of the $50 total budget.**
