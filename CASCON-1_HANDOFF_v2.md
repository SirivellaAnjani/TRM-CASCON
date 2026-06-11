# CASCON-1 — Code-Build Handoff v2 (for the next chat)

**Updated:** 2026-06-10 · **Companion docs:** `CASCON-1_DECISIONS_LOG.md` (append-only decisions D1–D19 + §8 build log), original `HANDOFF_CASCON-1_code.md`.
**Read this first, then the decisions log for the *why* behind any decision.**

---

## 0. What this project IS
Build + run the experiment code for a **CASCON 2026** work-in-progress paper. The paper's text + experimental design are **locked elsewhere**; this project is the **code-build + execution** phase.

**Scientific aim:** evaluate a **Tiny Recursive Model adapted for autoregressive code generation (TRM-AR)** vs **standard transformers** at **~28M params** on **Python code generation**. The expected result is a **predicted null** — recursion does *not* help recall-bound code-gen at this scale — established rigorously, plus an **Ouro mechanistic diagnostic** (storage vs manipulation) explaining *why*.

**Scope of the claim:** the dataset is the **≤512-token Python subset** of tiny-codes (the 512 filter dropped 63% of Python). State as a limitation: the null generalizes to **short-program (≤512-tok) Python code-gen**, not arbitrary-length code.

---

## 1. Server environment (verified)
- **Access chain:** Local → `ssh moon.cs.torontomu.ca` → `ssh gpuclient` → `cs-container-connect` (Docker).
- **Inside Docker** every command block starts with: `cd /data/trm-code-experiments/CASCON-1` then `source /data/venv-knob/bin/activate`.
- **venv-knob:** Python 3.10.14, torch 2.6.0+cu124, **2× A100 80GB**, transformers + pyarrow present. (sentencepiece/tiktoken NOT present → see C1 flag.)
- **Filesystem:**
  - Project (Docker view): `/data/trm-code-experiments/CASCON-1` = gpuclient `/data/asirivel/data/trm-code-experiments/CASCON-1`.
  - `~/move` (`/home/asirivel/`) is the ONLY WinSCP-visible folder and is **invisible inside Docker**. Transfer = WinSCP → `~/move` → `mv` into project **on gpuclient** (NOT Docker).
  - `/data/` is **NFS** → use direct **pyarrow** parquet reads, load tokenizer from a **LOCAL** path (never HF hub — NFS hang). Delete stale `.lock` before launches.
- **Network:** outbound to `api.anthropic.com` **works** inside Docker (probe = HTTP 404 = reached). Judge can run there — but we keep it MANUAL by choice (see §6).
- **Ownership gotcha:** never create project dirs from inside Docker (root-owned → asirivel can't `mv` into them). Create on gpuclient as asirivel.
- **GPU packing fact (measured):** TRM bs=48 fp32 peak **52.87 GiB** → **1 TRM job per GPU** (2× = 105 > 80 OOMs). With 2 GPUs → **max 2 concurrent**.

---

## 2. Dataset (BUILT, LOCKED)
`/data/common/datasets/tiny_codes_python_512_dedup_811/` — built from `tiny_codes_python_only` (129,063 Python rows), seed 42, local gpt2 tokenizer.
| split | rows |
|---|---|
| train.parquet | 38,119 |
| val.parquet | 4,765 |
| test.parquet | 4,765 (**held out — FIREWALL: touched only in eval**) |
| `_build_manifest.json` | provenance |

Pipeline (D10): Python filter → exact-match dedup on (prompt,response) [removed 0] → **filter-to-fit ≤512 tok** [129,063 → 47,649, 36.9% survive] → 80/10/10 @ seed 42. Token len: min 65 / median 407 / **max 512**.

**Tokenization contract (#3 ↔ #5, HARD):** a sequence = `tok(prompt) + tok(response) + [EOS]` (`add_special_tokens=False`), length `len(p)+len(r)+1 ≤ 512` = exactly the `n_tokens` the filter measured → **never truncates**. Dataset reserved the +1 EOS budget but did not append one; training appends EOS id 50256.

---

## 3. Models (BUILT, VERIFIED — server param counts exact)
`models/{trm.py, vanilla.py, build.py, __init__.py}`. Adapted from the old project (Option A). **D16: vanilla matched to TRM on every shared axis** (post-norm, RoPE-only, byte-identical RMSNorm/RoPE/SwiGLU/attention) → differs ONLY in recursion.
| Model | Params | Role |
|---|---|---|
| TRM-AR baseline | **27,830,784** | primary (recursive); eff_depth = 2×(4+1)×2 = 20 |
| Vanilla-2L | **27,830,016** | iso-param baseline (Δ=768 = y_init+z_init+halt_head) |
| Vanilla-20L | **46,713,600** | depth control = 1.68× params → **NOT param-fair** (L4/W7) |

- `build_model(config)` filters config→constructor kwargs by signature; `model_loss(model,ids,tgts,cfg)` hides TRM's extra forward args (`n_supervision_steps`, `halt_loss_weight`). Single entry point — use it everywhere.
- TRM defaults are now paper-correct; `halt_loss_weight` default 0.0 (ACT off, A4 dropped).
- **🟡 drift flag:** vanilla keeps its own (today-identical) copy of building blocks. If you edit a block, edit both files.

---

## 4. Code inventory (status + location)
All paths under project root `/data/trm-code-experiments/CASCON-1/`.
| # | File(s) | Status |
|---|---|---|
| 1 | `utils/{paths,naming,atomic_io,sweep_validation}.py` | ✅ done, tested |
| 2 | `configs/base_code.json` (the anchor config + `knob_abbreviations`) | ✅ done |
| 3 | `data/build_dataset.py` | ✅ done, **ran** (dataset built) |
| 4 | `models/{trm,vanilla,build}.py`, `models/__init__.py` | ✅ done, **verified** |
| 5 | `train_code.py` (single trainer, completion-only) | ✅ done, **smoke-trained** |
| 6 | `eval_code.py` + `utils/metrics_code.py` | ✅ done, **eval-smoked** |
| 7 | `sweep.py` + `configs/sweeps/main.json` | ✅ done, **dry-run OK** |
| 6b | `judge_code.py` (Phase B, manual) | ⬜ **NOT BUILT — next** |
| 8 | `ouro/` diagnostic + `select_stacked_best.py` | ⬜ **NOT BUILT** |
| — | `tests/{test_utils,test_base_config,test_build_dataset,verify_models,test_train_masking,test_metrics_code}.py` | ✅ all passing |

Old reference code (read-only): uploaded zip `trm-sweep-1-onevar` (TinyStories pilot) → has portable `utils/metrics_code.py` (ported), `utils/judge_code.py` (port for #6b), `evals/eval_code.py`, `sweep.py`. Server `/data/common/.../trm-ouro-diagnostic/` has Ouro code (gen_capo/gen_mano/eval_capo/eval_mano + train_code) for #8.

---

## 5. The experiment set — 21 arms × 3 seeds = 63 runs
Manifest: `configs/sweeps/main.json`. Each arm = one override delta over `base_code.json`. Seeds **[42, 65, 108]**; Phase 1 = seed 42 all arms (`--seeds 42`), Phase 2 = 65,108.

**Spine:** S1 TRM `{}` · S2 vanilla-2L `{architecture:vanilla,n_layers:2}` · S3 vanilla-20L `{...n_layers:20}`.
**Ablations:** A1 nsup=1 · A2 nlr=2. **Design-choice reversions:** A3 pre-norm · A5 dual-pos · A6 concat. **Architectural additions:** A10 y-resid · D1 per-depth-norm · D2 gated · D3 sandwich · D4a/b LoRA. **Hyperparam robustness:** A7 wd0.5 · A8 wd1.0 · A9 emb-lr. **Conditions:** C1 vocab10k · C2 ep6 · C3 bf16 · C4 cosine.

⚠️ **Arm taxonomy matters for the paper** (D15): "ablations" (A1,A2) ≠ "design-choice reversions" (A3,A5,A6) ≠ "architectural additions" ≠ "hyperparameter robustness" ≠ "conditions". User says "ablations" as chat shorthand — differentiate in writing.

---

## 6. The pipeline — TWO PHASES (D19)
**Phase A — automated (one `sweep.py` launch):** per (arm×seed): **train → eval**. Eval = generate on TEST, **save generations**, compute LOCAL metrics (AST-valid %, BLEU-4, test PPL). No network, no cost.
**Phase B — MANUAL, LAST (`judge_code.py`, user runs):** reads saved generations → Claude **Haiku 4.5** (`claude-haiku-4-5-20251001`) → 3 axes (syntax/logic/relevance). Paid → user keeps eyes on it (`--dry-run` cost estimate, `--limit`, resumable). Decoupled: never re-trains/re-generates.

**Checkpoint metric = completion val-loss** (D17 completion-only masking). Eval scope knob: `eval_n_prompts(50) × eval_n_completions_per_prompt(5)` = 250 gens/run × 63 ≈ 15,750 gens to judge → drives judge $ (tune at #6b).

---

## 7. How to run (once #6b/#8 built + code transferred)
```bash
cd /data/trm-code-experiments/CASCON-1 && source /data/venv-knob/bin/activate

# 0) pre-flight: validate all 63, launch nothing
python sweep.py --manifest configs/sweeps/main.json --name spine --dry-run

# 1) PHASE 1 (seed 42, all arms) — get results fast, start writing
python sweep.py --manifest configs/sweeps/main.json --name spine --seeds 42 --gpus 0,1 --max-parallel 2
#    (run inside tmux; user drives tmux. First run prints measured sec/step → real runtime estimate.)

# 2) PHASE 2 (seeds 65,108) — same launch without the seed filter (resume skips done runs)
python sweep.py --manifest configs/sweeps/main.json --name spine --gpus 0,1 --max-parallel 2

# 3) failures (the 3 deferred arms etc.) after deps fixed:
python sweep.py --manifest configs/sweeps/main.json --name spine --retry-failed --gpus 0,1 --max-parallel 2

# 4) PHASE B — MANUAL judge (paid), LAST:
#    python judge_code.py --name spine --dry-run   # cost estimate first  [#6b not built yet]
```
Artifacts per run: `results/spine/<run_id>.json` (train), `.eval.json` (local metrics), `.gen.json` (generations), later `.judge.json`. Logs: `logs/spine/<run_id>.log`. Failures: `logs/spine/_failures.log`. Checkpoints: `checkpoints/<run_id>.pt` (+ `.json` sidecar).

---

## 8. TODO / next steps (in order)
1. **🔴 DECIDE LoRA scope (A vs B)** — see §10. Blocks whether D4 is in the auto-sweep.
2. **Build #6b `judge_code.py`** — port old `utils/judge_code.py` (already targets Haiku 4.5 + has cost tracking). Reads `.gen.json`, writes `.judge.json`. Cost `--dry-run`, `--limit`, resumable. MANUAL only.
3. **Build #8** — Ouro diagnostic (Capo=storage / Mano=manipulation) on the **stacked-best TRM** vs vanilla-2L + vanilla-20L (D8), multi-seed; + `select_stacked_best.py` (greedy forward-select beneficial ablations on **val**, delta must exceed multi-seed noise, eval on **test**, exploratory framing). **W1 scope: stacked-best used ONLY in the Ouro diagnostic + ONE final headline comparison — never feeds any ablation/condition result.**
4. **Transfer all code**, `sweep.py --dry-run` on server.
5. **Run Phase 1** (seed 42) in tmux → capture measured sec/step → real runtime estimate (Rule 8).
6. **Run Phase 2** → then **manual judge** → then **Ouro**.
7. **Resolve deferred:** wire LoRA (if A) + fix C1 tokenizer (install sentencepiece/tiktoken or verify slow path), `--retry-failed`.
8. **Paper enforcement:** L4–L7 framing (20L=depth-control-not-param-fair, ablating-a-losing-model, multiple-comparisons descriptive, no compute-matched claim), W7 report real param counts (C1 changes them), A4-dropped note, the ≤512 scope limitation, software stack.

---

## 9. Nuances & flags (the things that bite)
- **Completion-only masking (D17):** `targets[:P-1] = -100` (mask predictions of prompt tokens); loss on response + EOS only. Both arms identical → fair. Checkpoint on completion val-loss. Short-completion fallback if unstable = small prompt-loss-weight (NOT full-sequence).
- **EOS/PAD (D18):** local gpt2 has NO special tokens; we register in-vocab `<|endoftext|>`=50256 as EOS+PAD (no vocab growth, asserted < vocab_size). EOS-as-PAD is non-contaminating (pad targets=-100, right-pad+causal). No model/dataset change.
- **run_id format:** `<alpha-sorted override segments>__seed_<N>`; floats use `-` not `.` (`wd_0-5`); unknown knob = hard error; baseline override `{}` → `baseline__seed_42`.
- **Atomic writes:** temp-in-same-dir → fsync → os.replace (NFS-safe). Used for checkpoints, sidecars, results, dataset parquet.
- **paths.py owns ALL paths** (env-overridable `TRM_COMMON_ROOT`/`TRM_DATASET_DIR`). Config carries dataset *identity*, paths carries *location*.
- **EMA:** eval + checkpoint on EMA weights (baked into saved `model_state_dict`). Eval loads weights directly (no separate ema_shadow).
- **No separator** between prompt/response (honors #3 budget). Eval feeds RAW prompt (not `prompt+"\n\n"` like the old code).
- **GPU packing:** 1 TRM job/GPU. `--max-parallel ≤ #gpus`. Never 2 TRM on one GPU.
- **Firewall:** test.parquet touched ONLY in eval, only for final metrics — never tuning/selection.
- **Smoke runs ≠ learning:** warmup_steps=2000 ≫ smoke steps → loss stays ≈ init. Smoke proves plumbing only.
- **Runtime estimates (Rule 8):** ONLY from measured multi-step logs of a real full-data run. The dry-run's single cold step (8.757s) and smoke sec/step (TRM 3.392 / vanilla 0.448 on limit-2000) are NOT estimates.

---

## 10. 🔴 OPEN DECISIONS (resolve in next chat)
1. **LoRA scope (D4):**
   - **(A)** wire LoRA now → all 21 arms auto. Cost: train+eval LoRA support + inter-run dependency (D4 runs *after* a baseline checkpoint exists).
   - **(B)** ship 18-arm sweep + judge + ouro now; LoRA (D4) + C1-tokenizer as focused follow-up. **Claude's lean = B** (core null + ablations don't depend on LoRA/10k-vocab; D4 is W1-scoped exploratory).
2. **C1 tokenizer (`gpt2_top10k`):** fast-load fails (needs sentencepiece/tiktoken). Fix = install one, OR verify slow path, OR regenerate a fast `tokenizer.json`. Deferred until C1 runs.
3. **Eval scope knob** (`eval_n_prompts × n_completions`) — confirm before the judge pass (drives cost).

---

## 11. Decisions index (full text in `CASCON-1_DECISIONS_LOG.md`)
D1 fresh rebuild Option A · D2 fp32 primary (+bf16 C3) · D3 flat LR (+cosine C4) · D4 ablate TRM only · D5 two baselines (2L iso-param, 20L depth) · D6 multi-seed phased · D7 stacked-best (W1-scoped) · D8 Ouro pipeline · D9 discard old numbers · D10 dataset pipeline · D11 judge=Haiku 4.5 · D12 harness design · D13 project root · D14 drop A4 halting · D15 arm taxonomy · D16 vanilla canonical-match · D17 completion-only masking · D18 tokenizer EOS/PAD · D19 two-phase eval (judge manual/last).
Watch items W1–W8 + L1–L8 (defensibility framing) — see log.

---

## 12. Standing rules (apply every turn)
No code without explicit permission · pre-code disclosure (issues/unique/flags, warnings in prose+inline) · backup before overwrite (`cp f f.bak_<ts>`) · file-modification protocol (read → state exact edits → wait → apply only those) · read-don't-assume (verify files/server before claiming) · evidence labels (Fact / 🟡 Hypothesis / Assuming) · ADHD-friendly dashboard tables · ONE clarifying question max, at bottom · one bash block per code drop, **user drives tmux** (no send-keys) · runtime estimates from measured logs only · don't reference "supervisor" (say "flagging for review") · update decisions log ONLY when user says so · `~/move` for transfers (never the absolute path) · staged approval (restate understanding, don't auto-proceed).

---

## 13. Immediate next action for the new chat
Confirm receipt of context → **answer the LoRA A/B question** → build **#6b judge_code.py** → build **#8 ouro + stacked-best** → transfer all → `sweep.py --dry-run` → launch Phase 1.
