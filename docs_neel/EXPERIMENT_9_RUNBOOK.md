# Experiment 9 — full runbook for a fresh GPU pod

Step-by-step, from renting the pod through getting results back and shutting it down. Written
for a **fresh** pod — nothing cached from any earlier session. If the original project's
Experiment 7 pod or its storage volume still exists, skip to "If a cached pod/volume already
exists" at the bottom — it's faster and cheaper.

**What this runs:** `src/value_leakage/jlens_experiment9.py` — reads the model's internal state
at the three positions Experiment 8 already flagged as most important, on the same flagship
trace, and produces four figures (a concept-score-by-layer line plot, a layer x top-10 heatmap
matching the style already used for the original project's Experiment 7, an entry-vs-exit
dip/respike check, and a convergence chart comparing Experiment 8's behavioral results against
Experiment 9's internal ones).

**Money needed:** this entire experiment runs on rented GPU time (RunPod) plus a one-time,
free download of public model weights from Hugging Face. **No Hugging Face API budget is
needed for this** — that account is only used for Experiment 8's text generation calls, which
this experiment doesn't touch at all. Public model and lens downloads from Hugging Face's Hub
don't cost anything themselves; the only real cost here is GPU-hours.

**Confidence note:** the model path, lens path, `jlens` API, and the two workarounds below are
copied directly from this repository's own already-working Experiment 7 code — as solid as
anything already proven here. The exact download commands, the `transformers`/`torch`/Claude
Code install steps, and the git credential setup are my best reconstruction of standard
practice, not something already tested in this repository — marked **[unverified]** below.
Run the `diagnose` stage (step 8) before trusting anything past it.

---

## 1. Provision the pod

Rent a RunPod **H200 (141GB VRAM)** instance — matches the checkpoint already used for
Experiment 7. Real current pricing: **$3.59/hour (Community Cloud) or $4.59/hour (Secure
Cloud)** — Secure Cloud costs more but is less likely to get preempted mid-download, worth it
here given the long model download. Pick a template with CUDA + PyTorch preinstalled.

Connect via the pod's web terminal or SSH once it's running.

## 2. Set up git credentials, then clone the repository

**[unverified: exact token scope]** If the `value-leakage` repository is private, the pod needs
its own way to authenticate to GitHub — don't reuse a long-lived personal SSH key on a
throwaway cloud box. The simplest, safely-scoped option: generate a **fine-grained personal
access token** on GitHub (Settings → Developer settings → Personal access tokens → Fine-grained
tokens), scoped to just this one repository, read-only, with a short expiry (a few days is
plenty for one session). Then, on the pod:

```bash
mkdir -p /workspace
cd /workspace
export GH_TOKEN="paste-your-token-here"
git clone https://oauth2:${GH_TOKEN}@github.com/<your-username>/<your-repo>.git value-leakage
unset GH_TOKEN   # don't leave it sitting in the shell's exported environment
cd value-leakage
```

**When you're done with the pod, revoke this token from GitHub's settings page** — it was
scoped narrowly and time-limited, but there's no reason to leave it valid longer than needed.

## 3. Install Claude Code on the pod (optional, but recommended for interactive debugging)

If you want an agent available on the pod itself to help debug anything that comes up (the
`diagnose` stage is specifically designed to surface problems that might need this):

```bash
curl -fsSL https://claude.ai/install.sh | bash
export PATH="$HOME/.local/bin:$PATH"
claude --version
```

For authentication on a throwaway remote pod, the simplest reliable method is an API key
(rather than the browser login flow, which is awkward over SSH):

```bash
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
claude -p "say hi"   # confirms it's working
```

Get a key from platform.claude.com if you don't already have one set aside for this. This step
is entirely optional — everything else in this runbook works fine run directly, without Claude
Code in the loop.

## 4. Python environment

**[unverified: exact versions]** `docs/FINDINGS.md`'s Experiment 7 entry names `transformers`
5.16.1 as the version the two documented workarounds apply to — install that exact version
rather than "latest":

```bash
cd /workspace/value-leakage
pip install -e .            # this project's own dependencies (fire, etc.)
nvidia-smi                  # check the CUDA version the pod actually has, then match it below
pip install torch --index-url https://download.pytorch.org/whl/cu124   # adjust cu124 to match nvidia-smi
pip install transformers==5.16.1
```

Then the Jacobian Lens library itself:

```bash
cd /workspace
git clone https://github.com/anthropics/jacobian-lens
cd jacobian-lens
pip install -e .
```

## 5. Download the model checkpoint (~127GB — the slow part)

**[unverified: exact CLI invocation]** — `hf` is the current Hugging Face CLI command name (the
older name, `huggingface-cli`, still works on older installs if `hf` isn't found):

```bash
pip install -U "huggingface_hub[cli]"
hf download Qwen/Qwen3.5-122B-A10B-FP8 --local-dir /workspace/models/Qwen3.5-122B-A10B-FP8
```

Budget 35–40 minutes for this on a fresh pod, per `docs/FINDINGS.md`'s own record of the
original download. This is real, separate rental time — don't count it against the short
analysis session that follows.

## 6. Download the fitted lens

```bash
hf download camilablank/workspace-lenses --include "qwen3.5-122b-a10b/j-lens/*" \
    --local-dir /workspace/lenses/workspace-lenses
```

Check `/workspace/lenses/workspace-lenses/qwen3.5-122b-a10b/j-lens/lens.pt` exists before
proceeding.

## 7. Confirm the source data Experiment 9 needs is present

Experiment 9 reads the exact same flagship trace and verified position offsets Experiment 8
already produced — these should already be part of the cloned repository (under `runs/`), not
something to regenerate on the pod. Check:

```bash
ls runs/qwen3.5-122b-a10b_20260815_030702/above_good.json
ls runs/qwen3.5-122b-a10b_e8_thoughtanchors_20260904/flagship_positions.json
```

If either is missing, the repository wasn't fully committed/pushed before cloning — go back and
fix that on your local machine first, rather than trying to reconstruct these on the pod.

## 8. Run the diagnose stage — do this before anything else

```bash
python -m value_leakage.jlens_experiment9 --stage diagnose
```

Loads the model and lens (about a minute once downloaded), runs one cheap forward pass, and
prints: the actual shape `lens.apply()` returns; whether the plain-logit-lens control (written
directly against this model's own hidden states, not assumed to come from the `jlens` library
"for free") works on this specific checkpoint's structure; and a side-by-side comparison of the
plain-logit-lens's top-5 prediction against the model's real next-token prediction, which
should roughly agree since both read the same underlying prediction through different code
paths. **If they don't agree, stop and fix `plain_logit_lens_per_layer` in the script** (most
likely cause: a wrong attribute name for the final norm or output head on this specific
checkpoint, or an off-by-one in layer numbering) before spending time on the next step. This is
exactly where having Claude Code available on the pod (step 3) is useful — hand it the error
output and ask it to inspect `hf_model`'s actual module structure.

## 9. Run the actual read

```bash
python -m value_leakage.jlens_experiment9 --stage read
```

Six forward passes total (three positions, entry and exit of each) — each cheap once the model
is loaded. Saves `runs/qwen3.5-122b-a10b_e9_jlens_20260904/results_e9.json` and `config_e9.json`.

## 10. Plot — four figures, no GPU needed for this step

```bash
python -m value_leakage.jlens_experiment9 --stage plot
```

Produces, all in `runs/qwen3.5-122b-a10b_e9_jlens_20260904/`:

- `concept_scores_by_layer.png` — line plot, each tracked word's log-probability by layer, one
  panel per position/cut.
- `heatmap_e9.png` — **the one styled like the original project's Experiment 7 heatmap**: a
  layer-by-top-10-rank grid, annotated with the actual decoded word at each cell, red borders
  for pre-registered bias words and orange for exploratory ones. Drawn twice per position side
  by side — once from the J-lens, once from the plain-logit-lens control — so a word that lights
  up on the J-lens side but not the control side, at an early layer, is visible at a glance.
- `entry_vs_exit_e9.png` — the dip/respike check: does the strongest tracked concept drop while
  the model is voicing a doubt and climb back once it resolves?
- `convergence_e9.png` — the chart this whole extension was built to produce: Experiment 8's
  real behavioral-shift size next to Experiment 9's internal-signal size, at the same three
  positions, so a "do these agree" read is visual rather than something you have to compute by
  hand from two separate tables.

## 11. Get everything off the pod, then terminate it

```bash
# from your local machine, not the pod:
scp -r root@<pod-ip>:/workspace/value-leakage/runs/qwen3.5-122b-a10b_e9_jlens_20260904 \
    "d:/research/ai safety/SPAR/value-leakage/runs/"
```

(Or use RunPod's web file browser instead of `scp`, if you'd rather not deal with SSH keys for
this part.)

**Once the results are safely copied off:**
- Terminate the pod from the RunPod dashboard. GPU-hours bill continuously until you do.
- Revoke the GitHub token from step 2, if you created one.
- Nothing in this runbook creates a persistent network storage volume — everything lives on the
  pod's local disk, which disappears with the pod. If you want the model cached for a *future*
  session, that's a separate, deliberate choice (costs roughly $0.07/GB/month, about $8.89/month
  for the full checkpoint) — not something to leave behind by accident.

---

## If a cached pod/volume already exists

Check the RunPod dashboard first for a persistent volume left over from the original project's
Experiment 7 session. If one exists with the model and lens still cached:
- Skip steps 5 and 6 entirely.
- Step 8 (diagnose) should complete in about a minute instead of much longer.
- Total session time should be well under the ~1 hour budgeted for analysis — the only real
  work left is one model load plus six cheap forward passes.

If you're not sure whether a volume exists, check before starting the paid clock, not after.
