"""Experiment 8 (Neel Nanda extension): Thought-Anchors-style sentence
resampling, per docs_neel/EXPERIMENTS_NEEL_NANDA.md section 3.

Natural resampling only (temperature regeneration from a cut point), never
hand-authored replacement sentences -- see EXPERIMENTS_NEEL_NANDA.md section
3.2 for why the design document rules out hand-injected sentences as the
primary metric.

For a position defined by a verified (start, end) character span inside a
source trace's `reasoning` field:
  - treatment prefix = reasoning[:start]   (the position's own sentence is
    dropped; the model regenerates it as part of a fresh continuation)
  - null-baseline prefix = reasoning[:end] (the position's own sentence is
    kept verbatim; only what comes after it is regenerated)
Comparing final answers between the two isolates the effect of changing what
is written at that specific position from ordinary continuation-to-
continuation sampling variance.

Continuation calls go through the huggingface/deepinfra route -- the same
route family used for this project's Experiment 3 prerequisite check
(FINDINGS.md, Experiment 3 entry) and required by
EXPERIMENTS_NEEL_NANDA.md section 2.1 to avoid the unresolved fp4-
quantization-tag caveat between OpenRouter and Hugging Face's route to the
same backend.

Stages (run in order, each checkpoints to disk so a crash never loses
earlier stages' spend):
  generate  -- sample N treatment + M null continuations per position
  filter    -- near-duplicate filter on treatment continuations (embedding
               cosine similarity vs the original sentence, threshold 0.92,
               sentence-transformers/all-MiniLM-L6-v2 -- same model and
               cutoff as this project's own Experiment 3 design)
  parse     -- regex-first final-estimate extraction; failures listed for
               the judge stage
  judge     -- claude-haiku-4-5 on regex failures only
  metrics   -- mean shift (threshold units) and crossing-rate shift per
               position, treatment vs null baseline

  uv run python -m value_leakage.thought_anchors --run_dir runs/qwen3.5-122b-a10b_e8_thoughtanchors_20260904 --stage generate
"""

import asyncio
import json
import re
from pathlib import Path

import fire

from value_leakage.api.anthropic.messages import extract_text, get_anthropic_client, process_batch as anthropic_batch
from value_leakage.api.huggingface.chat_completions import get_huggingface_client, process_batch as huggingface_batch
from value_leakage.judge import NUMBER_JUDGE_PROMPT, parse_tagged_estimate

TARGET_MODEL = "Qwen/Qwen3.5-122B-A10B"
TARGET_PROVIDER = "deepinfra"
TARGET_MAX_TOKENS = 16000  # generous vs. the ~5.5-10k token full-trace completions actually observed in this dataset
N_TREATMENT = 6   # replacements per position, per EXPERIMENTS_NEEL_NANDA.md section 3.2
N_NULL = 3         # null-baseline continuations per position, section 3.2
DUP_COSINE_THRESHOLD = 0.92  # same cutoff as this project's own Experiment 3 design (EXPERIMENTS.md)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

JUDGE_MODEL = "claude-haiku-4-5"
VALIDATION_JUDGE_MODEL = "claude-opus-5"

_SENTENCE_END_RE = re.compile(r"[.!?](?=\s|\$|$)")


def _first_sentence(text: str) -> str:
    """The model's own regenerated version of the resampled position -- the
    span up to the first sentence-terminator, matched against the original
    sentence for near-duplicate filtering."""
    m = _SENTENCE_END_RE.search(text)
    return text[: m.end()] if m else text


# --- regex-first final-estimate parser -------------------------------------
# The tail of a continuation reliably contains the final number, formatted
# either as a bare comma-grouped integer or bolded with **...**, matching the
# format actually observed across all 100 rows of this dataset's own
# above_good.json content field (spot-checked before writing this parser).
_NUMBER_RE = re.compile(r"\*{0,2}([\d,]{4,})\*{0,2}\s*\.?\s*$")
_TAIL_CHARS = 400


def parse_estimate_regex(continuation: str) -> float | None:
    """None means: send to the judge instead. Only accepts a number sitting
    at the very end of the continuation (last non-blank line), the same
    place the original pipeline's judge looks -- anything less clear-cut is
    a parse failure by design, not a best-effort guess."""
    tail = continuation.strip()[-_TAIL_CHARS:]
    lines = [l.strip() for l in tail.splitlines() if l.strip()]
    if not lines:
        return None
    last = lines[-1]
    m = _NUMBER_RE.search(last)
    if not m:
        return None
    digits = m.group(1).replace(",", "")
    if not digits.isdigit():
        return None
    value = int(digits)
    if value < 1000:  # a bare small integer at the tail is almost never the final spot-count answer
        return None
    return float(value)


# --- stage: generate ---------------------------------------------------------

def _load_source(positions_path: Path):
    spec = json.loads(positions_path.read_text(encoding="utf-8"))
    source = json.loads((Path(spec["run_dir"]) / f"{spec['condition']}.json").read_text(encoding="utf-8"))
    row = source["rows"][spec["row_index"]]
    return spec, source["prompt"], row["reasoning"]


async def _generate_position(client, prompt: str, reasoning: str, pos: dict) -> dict:
    hf_model = f"{TARGET_MODEL}:{TARGET_PROVIDER}"
    treatment_prefix = reasoning[: pos["start"]]
    null_prefix = reasoning[: pos["end"]]

    def messages_for(prefix):
        return [{"role": "user", "content": prompt}, {"role": "assistant", "content": prefix}]

    treatment_msgs = [messages_for(treatment_prefix)] * N_TREATMENT
    null_msgs = [messages_for(null_prefix)] * N_NULL

    treatment_responses, null_responses = await asyncio.gather(
        huggingface_batch(client=client, model=hf_model, messages_list=treatment_msgs,
                          max_tokens=TARGET_MAX_TOKENS, max_concurrent=N_TREATMENT,
                          return_exceptions=True),
        huggingface_batch(client=client, model=hf_model, messages_list=null_msgs,
                          max_tokens=TARGET_MAX_TOKENS, max_concurrent=N_NULL,
                          return_exceptions=True),
    )

    def flatten(responses):
        # NOTE: contrary to FINDINGS.md's Experiment 3 prerequisite check
        # (which found the continuation lands entirely in `content`, no
        # distinct reasoning mode), a direct smoke test run before this batch
        # (2026-09-04) found the model DOES re-enter a separate
        # `reasoning_content` field on continuation from a partial assistant
        # turn -- `content` alone only carries the final visible answer, same
        # as in an ordinary (non-continued) generation. Both fields are kept
        # here; `reasoning` is what the model actually generated to continue
        # the resampled position, `content` is where the final answer lives.
        out = []
        for r in responses:
            if isinstance(r, Exception):
                out.append({"error": f"{type(r).__name__}: {r}"})
                continue
            msg = r.choices[0].message
            reasoning_text = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
            out.append({
                "reasoning": reasoning_text,
                "content": msg.content or "",
                "finish_reason": r.choices[0].finish_reason,
                "usage": r.usage.model_dump() if r.usage else None,
            })
        return out

    return {
        "kind": pos["kind"], "marker": pos["marker"], "start": pos["start"], "end": pos["end"],
        "original_sentence": pos["sentence"],
        "treatment": flatten(treatment_responses),
        "null_baseline": flatten(null_responses),
    }


async def _generate_all(run_dir: Path, positions_file: str) -> None:
    positions_path = Path(positions_file)
    spec, prompt, reasoning = _load_source(positions_path)
    out_path = run_dir / "generations_e8.json"

    done = {}
    if out_path.exists():
        done = {r["marker"]: r for r in json.loads(out_path.read_text(encoding="utf-8"))["results"]}
        print(f"resuming: {len(done)}/{len(spec['positions'])} positions already generated")

    client = get_huggingface_client()
    results = list(done.values())
    for pos in spec["positions"]:
        if pos["marker"] in done:
            continue
        print(f"generating position marker={pos['marker']} kind={pos['kind']} "
              f"({len(results) + 1}/{len(spec['positions'])})")
        result = await _generate_position(client, prompt, reasoning, pos)
        results.append(result)
        out_path.write_text(json.dumps({"spec_file": str(positions_path), "results": results},
                                        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {out_path}")


async def _supplement_null_async(run_dir: Path, markers: list, extra_n: int) -> None:
    """One-off follow-up: add `extra_n` more null-baseline continuations to
    specific already-generated positions, identified by `markers` (the same
    marker ids used in flagship_positions.json). Used to check whether a
    position's apparent shift survives a larger null sample, rather than
    resting on the original 3-draw null baseline alone -- see FINDINGS_neel.md
    for why this was needed (one position's shift turned out to depend on a
    single atypical null draw). Appends to the existing null_baseline list;
    does not touch treatment continuations or re-generate anything."""
    gen_path = run_dir / "generations_e8.json"
    data = json.loads(gen_path.read_text(encoding="utf-8"))
    spec, prompt, reasoning = _load_source(Path(data["spec_file"]))
    pos_by_marker = {p["marker"]: p for p in spec["positions"]}

    client = get_huggingface_client()
    for result in data["results"]:
        if result["marker"] not in markers:
            continue
        pos = pos_by_marker[result["marker"]]
        null_prefix = reasoning[: pos["end"]]
        messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": null_prefix}]
        print(f"supplementing null baseline for marker={result['marker']}: +{extra_n} draws")
        responses = await huggingface_batch(client=client, model=f"{TARGET_MODEL}:{TARGET_PROVIDER}",
                                            messages_list=[messages] * extra_n, max_tokens=TARGET_MAX_TOKENS,
                                            max_concurrent=extra_n, return_exceptions=True)
        for r in responses:
            if isinstance(r, Exception):
                result["null_baseline"].append({"error": f"{type(r).__name__}: {r}"})
                continue
            msg = r.choices[0].message
            reasoning_text = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
            result["null_baseline"].append({
                "reasoning": reasoning_text, "content": msg.content or "",
                "finish_reason": r.choices[0].finish_reason,
                "usage": r.usage.model_dump() if r.usage else None,
            })
    gen_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {gen_path}")


# --- stage: filter (near-duplicate) -----------------------------------------

def _run_filter(run_dir: Path) -> None:
    from sentence_transformers import SentenceTransformer
    import numpy as np

    gen_path = run_dir / "generations_e8.json"
    data = json.loads(gen_path.read_text(encoding="utf-8"))
    model = SentenceTransformer(EMBED_MODEL)

    for result in data["results"]:
        original = _first_sentence(result["original_sentence"].strip())
        candidates = []
        for t in result["treatment"]:
            if "error" in t:
                candidates.append(None)
                continue
            # The regenerated position lives at the start of `reasoning` (the
            # model's continued thinking); fall back to `content` only if a
            # continuation has no reasoning text at all.
            lead_text = t["reasoning"].strip() or t["content"].strip()
            candidates.append(_first_sentence(lead_text) if lead_text else None)
        texts_to_embed = [original] + [c for c in candidates if c is not None]
        embeddings = model.encode(texts_to_embed, normalize_embeddings=True)
        orig_emb = embeddings[0]
        cand_embs = embeddings[1:]
        i = 0
        for j, t in enumerate(result["treatment"]):
            if candidates[j] is None:
                t["cosine_to_original"] = None
                t["is_duplicate"] = None
                continue
            sim = float(np.dot(orig_emb, cand_embs[i]))
            i += 1
            t["resampled_sentence"] = candidates[j]
            t["cosine_to_original"] = sim
            t["is_duplicate"] = sim >= DUP_COSINE_THRESHOLD

    gen_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    survived = sum(1 for r in data["results"] for t in r["treatment"]
                   if t.get("is_duplicate") is False)
    total = sum(1 for r in data["results"] for t in r["treatment"] if "error" not in t)
    print(f"near-duplicate filter: {survived}/{total} treatment continuations survive (cosine < {DUP_COSINE_THRESHOLD})")


# --- stage: parse (regex-first) ---------------------------------------------

def _final_answer_text(item: dict) -> str:
    """`content` reliably carries the final visible answer, whether or not
    the continuation used a distinct `reasoning` field (both forms were
    observed in real calls -- see the note in _generate_position). Fall back
    to the tail of `reasoning` only for the edge case of an empty `content`
    (e.g. a continuation cut off by max_tokens before reaching an answer)."""
    return item["content"].strip() or item["reasoning"].strip()


def _run_parse(run_dir: Path) -> None:
    gen_path = run_dir / "generations_e8.json"
    data = json.loads(gen_path.read_text(encoding="utf-8"))

    n_regex_ok = n_failures = 0
    failures = []  # (result_idx, group, item_idx)
    for ri, result in enumerate(data["results"]):
        for group in ("treatment", "null_baseline"):
            for ii, item in enumerate(result[group]):
                if "estimate_source" in item:
                    continue  # already resolved by a prior parse/judge pass -- never re-spend on it
                if "error" in item:
                    item["estimate"] = None
                    item["estimate_source"] = "api_error"
                    continue
                est = parse_estimate_regex(_final_answer_text(item))
                if est is not None:
                    item["estimate"] = est
                    item["estimate_source"] = "regex"
                    n_regex_ok += 1
                else:
                    item["estimate"] = None
                    item["estimate_source"] = None
                    n_failures += 1
                    failures.append({"result_idx": ri, "group": group, "item_idx": ii})

    gen_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "parse_failures_e8.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    print(f"regex parsed {n_regex_ok}, {n_failures} sent to parse_failures_e8.json for the judge stage")


# --- stage: judge (cheap-model fallback + validation sample) ---------------

async def _run_judge_async(run_dir: Path, validate_n: int) -> None:
    gen_path = run_dir / "generations_e8.json"
    data = json.loads(gen_path.read_text(encoding="utf-8"))
    failures = json.loads((run_dir / "parse_failures_e8.json").read_text(encoding="utf-8"))

    if not failures:
        print("no parse failures -- nothing for the judge to do")
        return

    def get_text(f):
        item = data["results"][f["result_idx"]][f["group"]][f["item_idx"]]
        return _final_answer_text(item)

    client = get_anthropic_client()
    prompts = [[{"role": "user", "content": NUMBER_JUDGE_PROMPT.format(llm_text=get_text(f))}] for f in failures]
    responses = await anthropic_batch(client=client, model=JUDGE_MODEL, messages_list=prompts,
                                       max_concurrent=20, return_exceptions=True)
    for f, r in zip(failures, responses):
        item = data["results"][f["result_idx"]][f["group"]][f["item_idx"]]
        if isinstance(r, Exception):
            item["estimate"] = None
            item["estimate_source"] = f"judge_error:{type(r).__name__}"
            continue
        est = parse_tagged_estimate(extract_text(r))
        item["estimate"] = est
        item["estimate_source"] = f"judge:{JUDGE_MODEL}"

    # Validation: compare the cheap judge against the standard judge on a
    # random sample, per EXPERIMENTS_NEEL_NANDA.md section 3.2 (mirroring
    # EXPERIMENTS.md's Experiment 3 design). Sampled without replacement;
    # if fewer than validate_n failures exist, validate on all of them and
    # say so explicitly rather than padding the sample.
    import random
    random.seed(0)
    sample = failures if len(failures) <= validate_n else random.sample(failures, validate_n)
    val_prompts = [[{"role": "user", "content": NUMBER_JUDGE_PROMPT.format(llm_text=get_text(f))}] for f in sample]
    val_responses = await anthropic_batch(client=client, model=VALIDATION_JUDGE_MODEL, messages_list=val_prompts,
                                          max_concurrent=20, return_exceptions=True)
    agreements = 0
    val_records = []
    for f, r in zip(sample, val_responses):
        cheap_est = data["results"][f["result_idx"]][f["group"]][f["item_idx"]]["estimate"]
        if isinstance(r, Exception):
            val_records.append({"error": str(r)})
            continue
        standard_est = parse_tagged_estimate(extract_text(r))
        agree = (cheap_est == standard_est) or (
            cheap_est is not None and standard_est is not None
            and abs(cheap_est - standard_est) / max(standard_est, 1) < 0.01
        )
        agreements += int(agree)
        val_records.append({"cheap_estimate": cheap_est, "standard_estimate": standard_est, "agree": agree})

    validation = {
        "requested_n": validate_n, "actual_n": len(sample),
        "note": "" if len(failures) >= validate_n else
                f"only {len(failures)} parse failures existed in total; validated on all of them, below the requested {validate_n}",
        "agreement_rate": agreements / len(sample) if sample else None,
        "records": val_records,
    }
    (run_dir / "judge_validation_e8.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    gen_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"judged {len(failures)} parse failures with {JUDGE_MODEL}; "
          f"validated {len(sample)}/{len(failures)} against {VALIDATION_JUDGE_MODEL}, "
          f"agreement rate {validation['agreement_rate']:.2%}" if sample else "no validation sample")


# --- stage: metrics -----------------------------------------------------------

def _run_metrics(run_dir: Path) -> None:
    gen_path = run_dir / "generations_e8.json"
    positions_spec = json.loads(Path(json.loads(gen_path.read_text(encoding="utf-8"))["spec_file"]).read_text(encoding="utf-8"))
    threshold = positions_spec["threshold"]
    condition = positions_spec["condition"]
    # baseline has no bet, so "incentivized side" is undefined -- mean shift (normalized by
    # the same model-level threshold) is still meaningful and reported; crossing-rate shift
    # is skipped for baseline rather than computed against an arbitrary side.
    incentivized_above = {"above_good": True, "below_good": False}.get(condition)

    data = json.loads(gen_path.read_text(encoding="utf-8"))
    rows = []
    for result in data["results"]:
        treatment_estimates = [t["estimate"] for t in result["treatment"]
                               if t.get("is_duplicate") is False and t.get("estimate") is not None]
        null_estimates = [n["estimate"] for n in result["null_baseline"] if n.get("estimate") is not None]
        n_treatment_total = sum(1 for t in result["treatment"] if "error" not in t)
        n_survived = sum(1 for t in result["treatment"] if t.get("is_duplicate") is False)

        row = {
            "kind": result["kind"], "marker": result["marker"],
            "original_sentence": result["original_sentence"].strip(),
            "n_treatment_generated": n_treatment_total,
            "n_survived_dup_filter": n_survived,
            "n_treatment_parsed": len(treatment_estimates),
            "n_null_parsed": len(null_estimates),
        }
        if treatment_estimates and null_estimates:
            mean_treatment = sum(treatment_estimates) / len(treatment_estimates)
            mean_null = sum(null_estimates) / len(null_estimates)
            row["mean_shift_threshold_units"] = (mean_treatment - mean_null) / threshold
            if incentivized_above is None:
                row["crossing_rate_shift"] = None  # baseline: no incentivized side to define
            else:
                crossing = lambda ests: sum(1 for e in ests if (e > threshold) == incentivized_above) / len(ests)
                row["crossing_rate_shift"] = crossing(treatment_estimates) - crossing(null_estimates)
        else:
            row["mean_shift_threshold_units"] = None
            row["crossing_rate_shift"] = None
        rows.append(row)

    out = {"threshold": threshold, "condition": condition, "positions": rows}
    (run_dir / "metrics_e8.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {run_dir / 'metrics_e8.json'}")
    for row in sorted(rows, key=lambda r: abs(r["mean_shift_threshold_units"] or 0), reverse=True):
        shift = row["mean_shift_threshold_units"]
        shift_str = f"{shift:+.3f}" if shift is not None else "N/A"
        print(f"  [{row['kind']:20s}] shift={shift_str}  survived={row['n_survived_dup_filter']}/{row['n_treatment_generated']}  {row['original_sentence'][:60]!r}")


# --- entry point --------------------------------------------------------------

def main(run_dir: str, stage: str, positions_file: str = None, validate_n: int = 100,
         markers: str = None, extra_n: int = 3):
    """markers/extra_n: only for stage='supplement_null' -- comma-separated marker ids
    to add `extra_n` more null-baseline draws to (see _supplement_null_async)."""
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    if stage == "generate":
        asyncio.run(_generate_all(run_path, positions_file))
    elif stage == "supplement_null":
        marker_ints = [int(m) for m in markers] if isinstance(markers, (tuple, list)) else [int(m) for m in markers.split(",")]
        asyncio.run(_supplement_null_async(run_path, marker_ints, extra_n))
    elif stage == "filter":
        _run_filter(run_path)
    elif stage == "parse":
        _run_parse(run_path)
    elif stage == "judge":
        asyncio.run(_run_judge_async(run_path, validate_n))
    elif stage == "metrics":
        _run_metrics(run_path)
    else:
        raise ValueError(f"unknown stage {stage!r}; one of generate/supplement_null/filter/parse/judge/metrics")


if __name__ == "__main__":
    fire.Fire(main)
