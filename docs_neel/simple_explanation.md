# Initial Tier 1 

The basic idea of this whole experiment: We had a real answer that an artificial intelligence model gave to a made-up math question (guess how many black spots exist on all giraffes in the world). That answer came with pages of the model "thinking out loud" before it landed on a final number. We wanted to know: which parts of that thinking actually mattered to the final number, and which parts were just words that didn't really change anything?

To test this, we picked 22 specific sentences inside that thinking, and for each one we did this: erase everything from that sentence onward, then ask the model to keep going on its own, 6 separate times. Since the model doesn't repeat itself exactly, it writes something a little different each time, and may end up at a different final number. We also asked it to keep going 3 more times without erasing anything, just to see how much the final number naturally wobbles even when nothing was changed. Comparing those two groups tells us whether a specific sentence was actually doing something, or whether the model's answer just wobbles around by that much anyway.

Result 1 — "which kind of sentence matters more: the ones where the model worries about being fair, or the ordinary calculation sentences?"

What we found, stated plainly: one ordinary calculation sentence ("Average spot size: Variable.") showed the biggest change of all 22 sentences we tested.
But when we actually read the 9 answers behind that number by hand, we noticed something: one of the 3 "don't change anything" answers was unusually high (85 million) compared to the other two (44 million and 53 million), and that one answer was pulling the whole comparison in one direction. It wasn't a broken or invalid answer — it's just one example out of only three, and one unusual example out of three can throw things off.
Once we account for that, there's no clear winner between the two kinds of sentences. Both wobble by roughly similar, small amounts.
Read the full numbers: FINDINGS_neel.md (Tier 1 section) and the write-up in main_writeup.md (section 3.3).

**Follow-up check on Result 1 — did that "unusually high" answer actually skew things?** Since only 3 "don't change anything" answers is a small number to trust, we went back and got 3 more of them for that same sentence (and the two next-biggest-change sentences), to see if the big change was real or just bad luck from a small sample.

What we found, stated plainly: yes, it was partly bad luck. With 6 "don't change anything" answers instead of 3, the big change shrank by more than half. It's still the biggest change out of all 22 sentences we tested, just a much smaller "biggest" than it first looked. So there might be a small real effect here, or there might not be — we can't say for certain yet, and we're not pretending otherwise.
Read the full numbers: FINDINGS_neel.md (same Tier 1 section, updated).

Result 2 — "if the model reconsiders itself, does it always come back to the same number no matter what specific doubt it raises?"

What we found, stated plainly: at one specific point where the model reconsiders its own answer, we tried 6 completely different rewordings of that moment. All 6 times, the model still landed within 3 million of its original 45-million answer — a very tight cluster.
This means: at least at this one point, it didn't matter what specific new doubt the model raised — it kept coming back to basically the same number.
Same files as above for the details: FINDINGS_neel.md, marker 19338. We also added 3 more "don't change anything" answers here as part of the same follow-up check, and this result barely moved — it was the steadiest of the three we checked.

A trust-check we also did: part of reading out the model's final number from each of the 198 (now 207, after the follow-up) answers required a second, cheaper AI to read the text and report the number. We double-checked its readings against a stronger, more expensive AI doing the same job (104 checks total, across both rounds), and they agreed every single time — so we can trust the numbers reported above weren't garbled in translation.

---

## Tier 2 — next step, not yet run

**What we're about to do:** the result above comes from just one AI answer (one "trace"). To find out if what we saw is a real pattern or just something specific to that one answer, we picked 9 more real answers the same AI gave to the same giraffe-spot question — 3 with no bet attached, 3 with the bet framed one way, 3 framed the other way — and picked out similar kinds of sentences in each one (57 sentences total). We haven't run the actual test on these yet — that's the next step, and a closer look at the cost shows it will likely run noticeably higher than first guessed (see the conversation for the updated estimate and why), so we're checking in before spending it.

