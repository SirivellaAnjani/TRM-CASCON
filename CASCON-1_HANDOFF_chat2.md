# CASCON-1 — Chat-2 Handoff (Flags Fixed + LoRA Pre-Flight)

**Created:** 2026-06-11 · **Author context:** Anjani Sirivella (TMU MSc), server user `asirivel`.
**Companion to (DO NOT replace — all three are fed to the next chat):**
`CASCON-1_HANDOFF_v2.md` (the locked code-build map) · `CASCON-1_DECISIONS_LOG_chat1.md` (D1–D19 + build log) · original `HANDOFF_CASCON-1_code.md`.

**Read order for the next chat:** v2 handoff → this doc → decisions log (for the *why* behind D1–D19).
**This doc covers only what changed in CHAT 2.** Nothing here overrides the locked design; it records the engineering fixes, verifications, and the LoRA spec we're about to build.

---

## 0. Chat-2 TL;DR (what happened this session)
1. Re-read the entire codebase + read the **TRM paper** (arXiv:2510.04871) → confirmed `trm.py` is **faithful** on the load-bearing axes.
2. Found **7 engineering flags** (F1–F7) by reading the actual code (beyond the docs). All 7 resolved.
3. Confirmed the server has the **canonical (newer) file copies** (md5 verified) — the zip ships build-stage snapshots, some at older versions.
4. Shipped 4 edited files to the server via WinSCP; verified (md5 + compile + `verify_models` + CPU dry-run all green).
5. Resolved the open **LoRA A/B** question → **A (wire LoRA now)**. Wrote + LOCKED the LoRA pre-flight spec (L-1…L-8, CD6). **LoRA build is the immediate next task.**

---

## 1. Chat-2 decisions (CD1–CD5) — append-only, do not edit prior logs

| ID | Decision | Reasoning | Evidence |
|---|---|---|---|
| **CD1** | **LoRA scope = A** — wire LoRA now so all 21 arms run automated (not the 18-arm Option B). | User directed "write LoRA right now" + intends the full 21-arm sweep. Resolves the v2-handoff §10 / decisions-log §8.14 open A/B question. | user instruction, Chat 2 |
| **CD2** | **F3 GPU dry-run = a SEPARATE `--gpu-dry-run` flag** (NOT folded into `--dry-run`). | Keeps the cheap CPU validate (`--dry-run`) instant; GPU 1-batch-per-arm is an explicit opt-in step. | user choice, Chat 2 |
| **CD3** | **F5 lock-clean = opt-in `--clean-locks`** (default OFF). | Deleting `*.lock` while a parallel sweep/tenant is mid-load would corrupt theirs; off-by-default is the safe rule. | user choice, Chat 2 |
| **CD4** | **All code edits transfer via WinSCP → `~/move` → `mv`**, never pasted into the terminal. | Multi-line paste into the SSH shell **garbles** (it bit us on the F2 docstring paste). This is the documented gotcha (orig handoff §2). | observed live, Chat 2 |
| **CD5** | **F7 eval speedup = batch the k completions**, NOT a recursive KV-cache. | A KV-cache is not viable for TRM (y,z reset each forward; combined x+y+z changes every recursion cycle). Ouro itself needed a special last-step/averaged cache. Batching is the correct, arch-agnostic win. | TRM paper + Ouro (2510.25741), Chat 2 |
| **CD6** | **LoRA (D4) LR/target spec LOCKED** (see §6 for the table): frozen = val-selected grid `{2e-4,5e-4,1e-3}` / `target="all"`; unfrozen = baseline `1e-4`; multi-seed the winner; report grid curve + asymmetry/cost + addition-test framing. | LoRA is provably more LR-sensitive and needs ~10× the FullFT LR; attention-only LoRA underperforms. Giving LoRA its strongest *honest* shot makes the null harder to achieve → more defensible. Selecting-on-val/reporting-on-test + un-nerfed pre-registered baseline = fairest shape for a negative result. | LoRA LR lit (Thinking Machines, Biderman 2405.09673, Unsloth) + negative-result methodology (ICML-2024 2406.03980, QRP survey 2407.12220), Chat 2 |
| **CD7** | **C1 (`gpt2_top10k`) RESOLVED — wrapper ported + wired (was deferred).** | The dir ships only `token_map.json` (gpt2-id→10k remap). The Ouro-diagnostic project already had the full wrapper at `utils/tokenizer_top10k.py`, so we **ported it** rather than building from scratch. Server check confirmed gpt2 EOS (50256) **IS in top-K** → maps to distinct new id **9999** (no UNK collision) → no EOS fork needed. Base gpt2 dir exists. | server inspection + port, Chat 2 |

---

## 2. The 7 flags — full detail (all RESOLVED)

These were found by **reading the actual code**, not the design docs. They are engineering gaps, not methodology gaps.

| # | Flag | Fix shipped | File | Status |
|---|---|---|---|---|
| **F1** | Zip ships **two divergent copies** of `train_code.py` (5-train old / 7-sweep new w/ LoRA guard) and `paths.py` (1-utils old / 6-autoEval new w/ eval+gen+judge helpers). `eval_code.py` imports helpers only the new `paths.py` has. | **Verified server = canonical (newer)** via md5 (all 5 match) + grep markers (LoRA guard @ train_code:235; `generations_path` etc. @ paths:108/113/118). | — (check only) | ✅ passed |
| **F2** | `trm.py` top docstring was **stale** — described OLD pilot defaults (pre-norm, concat, abs-pos+RoPE) that are no longer the defaults. | Rewrote docstring to paper-correct (post-norm, additive, RoPE-only, ACT-off) + AR deviations + cites arXiv:2510.04871 Fig. 3. Docstring-only; zero behavior change. | `models/trm.py` | ✅ shipped |
| **F3** | `sweep.py --dry-run` was **CPU-only** (validate run_ids + print plan); no per-variant GPU fit/step check. | Added **separate `--gpu-dry-run`**: spawns `train_code --dry-run` for ONE representative seed per arm (dedup by overrides), pinned to a GPU, tabulates PASS/FAIL. Reuses the child `--run-one --dry-run` path. | `sweep.py` | ✅ shipped |
| **F4** | No **per-variant timeout** — a hung NFS read could freeze a "leave-it" sweep slot forever. | Added `--timeout <sec>` (default 0 = off). Poll loop kills a child past its budget, frees the GPU, marks it failed (`reason=timeout`). | `sweep.py` | ✅ shipped |
| **F5** | No automated **stale `.lock` cleanup** (manual step everywhere in docs). | Added **opt-in** `--clean-locks` (default off): deletes `*.lock` under project + dataset dir at launch start. | `sweep.py` | ✅ shipped |
| **F6** | **Software stack** not captured in run artifacts (paper needs torch/CUDA/python/OS). | Added `software_stack()`; embedded under `"software"` in the completed-run result JSON. | `train_code.py` | ✅ shipped |
| **F7** | Eval generation **looped k separate `generate()` calls** per prompt (slow; TRM re-runs full recursion per token). | Batched all k completions into **one** `generate()` call (`[pid]*k`). ~k× fewer forward passes, arch-agnostic, identical output distribution. | `eval_code.py` | ✅ shipped |

### 2a. Flag nuances to remember
- **F4 semantics:** a killed/timed-out run writes **no** result JSON → on disk it reads as `pending`, so a **plain re-launch retries it** (it is NOT caught by `--retry-failed`, which only re-runs `status=failed`). Working as intended.
- **F6/F7 are unproven until a real run** (dry-run exits before the result dict; eval needs a checkpoint). **F7's speedup is a 🟡 Hypothesis until measured** on the first Phase-1 eval (Rule 8).
- **F2 had no functional effect:** `verify_models` param counts are byte-identical post-edit (see §4).

---

## 3. Files changed this chat — provenance

**Backups (made inside Docker, root-owned, gid 65530):** `*.bak_20260611_002636` for trm/sweep/train_code/eval_code.

**Edited files + NEW md5 (these are now LIVE on the server, verified):**

| File | New md5 | Edit |
|---|---|---|
| `models/trm.py` | `9ab16a1407233b4977581a63b55e312b` | F2 |
| `sweep.py` | `f2e06407aaa2c259419d7889d06d8357` | F3 + F4 + F5 |
| `train_code.py` | `275bd8cfaa68a069175cae94b94d24ff` | F6 |
| `eval_code.py` | `7a034eda12974706b1b8509259d938a5` | F7 |

`utils/paths.py` was **not** edited (md5 unchanged `dc64538629b9159df5cff9d5a67a13a8`).

**Rollback if ever needed:** `cp <file>.bak_20260611_002636 <file>` (backups predate the edits).

---

## 4. Verification results (FACTS from server output, Chat 2)

| Check | Result |
|---|---|
| md5 of 4 transferred files | **exact match** to the table above ✅ |
| `python -m py_compile` (all 4) | **compile OK** ✅ |
| `python tests/verify_models.py` | TRM **27,830,784** · Vanilla-2L **27,830,016** · Vanilla-20L **46,713,600** · iso-param Δ=**768 (0.0028%)** · 20L = 1.68× TRM · **ALL PASSED** ✅ |
| `sweep.py --dry-run` (CPU) | **63 run_ids validated**, nothing launched ✅ (new flags parse) |
| `sweep.py --gpu-dry-run` (GPU) | **18/21 PASS** ✅ — the 3 fails are the deferred arms by design: `lora_frozen` + `lora_unfrozen` (D4 `NotImplementedError` guard fired), `tok_gpt2-top10k` (C1, no fast tokenizer). F3 + fail-isolation confirmed working. |

`verify_models` is at `tests/verify_models.py` on the server (not the zip's `4-models/` path).

---

## 5. Research findings (Chat 2) — grounding for the paper + build

### 5a. TRM paper fidelity (arXiv:2510.04871, Fig. 3) — `trm.py` is FAITHFUL
| Paper | `trm.py` | Verdict |
|---|---|---|
| Single tiny net; single latent `z` + answer `y` (NOT z_H,z_L) | one `self.net`, `y`+`z` | ✅ |
| z-update `z=net(x,y,z)` ×n, then y-update `y=net(y,z)` | `x+y+z` ×nlr, then `y+z` | ✅ |
| **Back-prop through the FULL final recursion** (NOT 1-step approx) | `deep_recursion`: T−1 no_grad, final cycle **with grad** | ✅ (key fidelity point) |
| Additive combiner | baseline `use_additive_combiner=True` | ✅ |
| Deep supervision; carry detached (y,z) | forward loop matches | ✅ |
| EMA (paper ablation: no-EMA 87.4→79.9) | `EMA`, eval+ckpt on shadow | ✅ |
| ACT = Q-learning halt, single fwd, training-only | off at baseline (`halt_loss_weight=0`, A4 dropped, D14) | ✅ |

**Documented faithful deviations:** our `n=4, T=2, nsup=3` vs paper `n=6, T=3, nsup≤16` (compute); causal mask + next-token CE (AR adaptation). Effective depth/supervision-step = `T(n+1)·n_layers = 2·5·2 = 20`.

### 5b. Ouro (arXiv:2510.25741) — grounds the Capo/Mano diagnostic
- Storage: a looped LM stores ≈**2 bits/param**, same as a standard transformer (knowledge capacity does NOT improve).
- Manipulation: ability to compose/execute multi-hop logic scales with recurrent steps + training tokens.
- → Diagnostic prediction (D8): **TRM ≈ vanilla on Capo (storage); TRM > vanilla on Mano (manipulation) iff recursion helps at 28M.**
- Bonus: Ouro decoding uses a special last-step/averaged KV-cache → reinforces CD5 (no naive KV-cache for TRM).

### 5c. Ablation-methodology conventions — all already satisfied
- Removal ablation (component omitted in **train AND test**) is the fair kind; we retrain per knob → ✅.
- Every arm trained/tested under identical conditions except the one knob (W6) → ✅.
- Pre-registered grid + fixed seeds/splits + multi-seed noise threshold (L3/L6) → ✅.

---

## 6. 🔴 LoRA pre-flight spec (the IMMEDIATE next build, CD1)

**Model side is DONE** in `trm.py`: `enable_lora(rank, target)`, `freeze_shared()`, `unfreeze_all()`, per-depth `LoRALinear`, and `latent_recursion` already calls `_set_lora_depth(i)`. **Missing = train/eval plumbing.**

The two manifest arms: `D4a lora_frozen` (`{"lora":"frozen"}`) and `D4b lora_unfrozen` (`{"lora":"unfrozen"}`). Currently `train_code.py:235` raises `NotImplementedError` for any `lora != none` (the guard that must now be replaced with real support).

| # | Issue | Decision / plan | Severity |
|---|---|---|---|
| **L-1** | LoRA fine-tunes **from** the trained baseline ckpt (`baseline__seed_N.pt`), per D8/§8.14 → inter-run dependency | Load base ckpt before `enable_lora`; if missing → **fail-isolate** (S1 baseline runs first anyway; re-launch retries) | mechanics |
| **L-2** | `--gpu-dry-run` for D4a/b has **no baseline ckpt yet** | In dry-run: **skip base-load**, `enable_lora` on fresh init, just prove build+step+mem | mechanics |
| **L-3** | Eval must be **LoRA-aware** — `build_model(config)` won't create `lora_A/B`, so `load_state_dict` crashes on missing keys | eval calls `enable_lora(...)` **before** `load_state_dict` whenever `config["lora"] != "none"` | 🔴 correctness |
| **L-4** | **EMA ordering** — EMA shadow is built from `requires_grad` params | construct EMA **after** `enable_lora`/`freeze_shared` | 🔴 correctness |
| **L-5** | Optimizer scope — frozen variant should train adapters only | filter optimizer params to `requires_grad` | mechanics |
| **L-6** | rank/target change param count (W7); attention-only LoRA underperforms | pin **rank=4, `target="all"`** (NOT the `enable_lora` default `"attention"` — attention-only underperforms even param-matched, per Thinking Machines/Biderman); **report real LoRA param count**, do NOT claim iso-param | framing (W7) |
| **L-7** | frozen vs unfrozen semantics (the D4a/D4b contrast) | frozen = `freeze_shared` (adapters only); unfrozen = `unfreeze_all` (shared + adapters) | spec |
| **L-8** | LoRA fine-tune **LR + epochs** | **LOCKED (CD6):** frozen = val-selected grid **`{2e-4,5e-4,1e-3}`** (grid @ seed 42 → val-loss winner, delta must beat multi-seed noise); unfrozen = baseline **`1e-4`** (shared weights dominate); multi-seed winner [42,65,108]; epochs = baseline (3). Report: all grid rows / **LR-sensitivity curve (R1)**, **asymmetry+cost statement (R2)**, **within-TRM addition-test framing (R3)**. | ✅ locked |

**L-3 and L-4 are the crash/silent-error risks — must be handled or LoRA runs corrupt.**

---

## 7. Still-open / deferred (NOT blocking LoRA)

| # | Item | State |
|---|---|---|
| #6b | `judge_code.py` (Phase B, manual judge) | **✅ BUILT + on server** (md5 `329927ae`). Ports the old Haiku-4.5 lib (3-axis syntax/logic/relevance, retry/backoff, cost guardrails) into a Phase-B driver that reads our `.gen.json` → writes `.judge.json`; `--dry-run` (cost est, no API), `--limit`, run-level resume. CPU dry-run on server returns "no .gen.json yet" cleanly (correct until Phase-A eval runs). |
| #8 | Ouro diagnostic + `select_stacked_best.py` | ⬜ **PORT — DECIDED (not a question).** Capo (storage) + Mano (manipulation) are CORE: they show WHY TRM fails at code. They MUST run on OUR 27.83M TRM/vanilla pair (citing the old Ouro-project results on other models is NOT academically rigorous). Structure READ this chat; lives at **`/data/trm-ouro-diagnostic/`**. Reusable: `prompt`/`response` parquet = our `CompletionDataset` format ✅; `ckpt["model_state_dict"]` = our checkpoint format ✅. One compat item to settle at port time: `eval_*.load_model` kwargs vs our `trm.py` ctor, and the `prompt+'\n\n'` join vs our id-concat. Build as a focused mini-sweep AFTER Phase 1/2 + LoRA (GPUs busy; non-blocking). `select_stacked_best.py` still net-new + W1-high-risk (needs Phase-1 noise scale). |
| **C1** | `gpt2_top10k` tokenizer (vocab-10k condition) | **✅ RESOLVED (CD7) — runnable.** Ported `utils/tokenizer_top10k.py` from the Ouro-diagnostic project (faithful wrapper) **+ added a HF-style batch `__call__`** so it's a drop-in for our `tokenizer(list, add_special_tokens=False)["input_ids"]` pipeline. `train_code.load_tokenizer` now early-branches to the wrapper for `name=="gpt2_top10k"` (eval inherits it via import). EOS distinct at id **9999** (50256 ∈ top-K), vocab 10001, dataset reusable (segmentation unchanged). Verified on server: load + encode/decode round-trip clean. md5 `train_code.py=8fdcbf46`, `utils/tokenizer_top10k.py=fefe80c5`. |
| Paper | L4–L7 framing, W7 real param counts, A4-dropped note, ≤512 scope limitation, software stack | paper effort (not code); F6 now supplies the software stack |

---

## 8. Environment / workflow lessons reaffirmed (Chat 2)

- **NEVER paste multi-line code into the SSH terminal — it garbles.** Use WinSCP → `~/move` → `mv` (CD4). This cost us one failed F2 paste (no damage; nothing was written).
- **Paths:** Docker view `/data/trm-code-experiments/CASCON-1`; **gpuclient** view `/data/asirivel/data/trm-code-experiments/CASCON-1`. `mv` runs on **gpuclient**, not Docker.
- **Ownership gotcha (live):** backups created via `cp` inside Docker are **root-owned**. `mv` over an existing file still works because the project dirs are asirivel-writable; if `mv` hits *permission denied*, `rm` the target inside Docker (root) first, then re-`mv`.
- **Transfer verification pattern that worked:** edit locally → compute md5 → WinSCP → `mv` → re-`md5sum` on server → `py_compile` → `verify_models` → `sweep --dry-run`.
- **`vanilla.py` drift flag persists** (it keeps its own copy of the building blocks + an unused `enable_lora`). D4 is **TRM-only**, so vanilla-LoRA is never exercised — leave it, don't delete.

---

## 9. Immediate next action for the next chat
1. ✅ gpu-dry-run confirmed (18/21 PASS). ✅ L-8 locked (CD6). ✅ C1 diagnosed (CD7).
2. **Write LoRA support** (replace the `train_code.py:235` `NotImplementedError` guard) honoring L-1…L-8 — especially **L-3 (LoRA-aware eval load)** and **L-4 (EMA after enable_lora)**, with **`target="all"`** (L-6) and the locked LR spec (L-8/CD6).
3. **Update the manifest** for the LoRA LR grid: frozen needs `{2e-4,5e-4,1e-3}` arms @ seed 42 (selection) + winner × 3 seeds; unfrozen @ 1e-4 × 3 seeds. (Run-flow: grid → val-select → multi-seed winner.)
4. `--gpu-dry-run` again → D4a/D4b should now **PASS** (dry-run skips base-load, L-2).
5. **Launch Phase 1** (seed 42) in tmux → capture **measured sec/step** → first real runtime estimate (Rule 8). F6/F7 get exercised here.
6. **Build status:** ✅ #6b judge built. ✅ C1 wrapper (CD7). ✅ **#8 Capo/Mano BUILT + data staged** (eval_capo/eval_mano + capo.json/mano.json; capo_full regenerated; see §12). ⬜ Only **`select_stacked_best.py`** remains as net-new code (after Phase-1 noise scale). C1 EOS sub-decision CLOSED (eos id 9999).

---

## 9.5 Operator runbook — what to do AFTER Phase 1 (seed 42) completes
All inside Docker: `cd /data/trm-code-experiments/CASCON-1 && source /data/venv-knob/bin/activate`.

**A. Verify Phase 1.**
```
python sweep.py --manifest configs/sweeps/main.json --name spine --dry-run   # done/failed per run
cat logs/spine/_failures.log 2>/dev/null     # expect ONLY C1 (tok_gpt2-top10k) @ seed 42
ls checkpoints/baseline__seed_42.pt          # MUST exist (LoRA depends on it)
```
Expect **18/19** arms done @ seed 42 (C1 fails by design).

**B. Capture real runtime (Rule 8).**
```
grep "sec/step(med)" logs/spine/baseline__seed_42.log
```
→ × total steps = the Phase-2 runtime estimate. Do NOT estimate before this.

> **MEASURED (Chat 2, seed-42 Phase 1):** TRM `baseline` = **3.022 sec/step**; vanilla-2L = **0.016 sec/step** (~190× faster — no recursion). Steps/epoch = 794 (38,119 / bs 48), 3 epochs = 2,382 steps. → **TRM arm ≈ 2.0 hr train** (`wall_time_sec` confirms vanilla-2L = 1,106 s). `ep_6` arm ≈ 2× (~4 hr). **Phase-1 (seed 42) ETA ≈ 16–20 hr** on 2 A100s (≈16 TRM-class arms + 2 vanilla + 1 ep6). **Set Phase-2 `--timeout ≈ 18000` (5 hr)** to cover ep_6 with margin.
> **Cosmetic flag (non-blocking):** result JSON `n_params` logs `None` (only `trainable_params` populated, = 27,830,016 ✓). One-line fix later; does NOT affect training/results. Do not touch a live run.

**C. LoRA Stage A (seed-42 LR selection)** — needs `baseline_42` (now exists).
```
python sweep.py --manifest configs/sweeps/lora.json --name spine --seeds 42 --gpus 0,1 --max-parallel 2 --wandb
#   --seeds 42 is REQUIRED: restricts to seed-42 LoRA arms (frozen grid + unfrozen_42).
#   Without it, lora_unfrozen__seed_65/108 try to load baseline_65/108 (don't exist yet) -> fail-isolate.
#   (Optional: add --no-eval to the grid -- selection only needs best_val_loss, not the test eval.)
python select_lora_lr.py --name spine        # prints 3 frozen val-losses + winner LR
```

**D. Phase 2 (seeds 65,108)** — produces `baseline_65/108`.
```
python sweep.py --manifest configs/sweeps/main.json --name spine --gpus 0,1 --max-parallel 2 --wandb --timeout <~2x measured>
#   no --seeds filter -> runs 65,108; resume skips the done seed-42 runs. Now safe to set --timeout (F4).
```

**E. LoRA Stage B (multi-seed winner)** — needs `baseline_65/108` (now exist).
- Add ONE frozen variant with the **winner lr** at `"seeds":[65,108]` to `configs/sweeps/lora.json`, then:
```
python sweep.py --manifest configs/sweeps/lora.json --name spine --gpus 0,1 --max-parallel 2 --wandb
#   runs unfrozen_65/108 + frozen-winner_65/108; resume skips Stage-A runs.
```

**F. Phase B — manual judge (paid, LAST).**
```
python judge_code.py --name spine --dry-run     # cost estimate, NO API
python judge_code.py --name spine               # judge all saved .gen.json -> .judge.json
```

**G. Stacked-best + Ouro (#8)** — after results land (next build; Ouro needs the server diagnostic code).

**KEY GOTCHA:** a LoRA run @ seed N loads `baseline__seed_N.pt`. So LoRA seed-65/108 arms cannot run until **Phase 2** has produced those baselines. Hence the Stage A (`--seeds 42`) / Stage B split.

---

## 10. Standing rules (carry forward, unchanged)
No code without explicit permission · pre-code disclosure (issues/unique/flags, warnings in prose **and** inline) · backup before overwrite (`cp f f.bak_<ts>`) · file-modification protocol (read → state exact edits → wait → apply only those) · read-don't-assume (verify files/server before claiming) · evidence for every server command · ONE bash block per code drop, **user drives tmux** (no send-keys) · runtime estimates from measured logs only · evidence labels (Fact / 🟡 Hypothesis / Assuming) · ADHD-friendly dashboard tables · ONE clarifying question at a time, at the bottom · staged approval (restate understanding, don't auto-proceed) · don't reference "supervisor" (say "flagging for review") · `~/move` for transfers (never the absolute path) · **transfer code via WinSCP, never terminal paste (CD4)** · update the decisions log ONLY when the user says so.

---

---

## 11. 🗂️ MASTER STATUS BOARD (single-glance — so nothing kept "for later" is missed)

### A. Build work — DONE ✅ (all on server, verified)
| Item | What | Proof |
|---|---|---|
| Codebase + TRM fidelity | reviewed; F2 docstring made paper-correct | md5 + grep markers |
| **F1–F7** | all 7 engineering flags fixed | gpu-dry-run 18/21, CPU dry-run, verify_models PASS |
| **LoRA (CD6)** | `setup_lora` (train), LoRA-aware load (eval), `select_lora_lr.py`, `base_code.json` rank/target, `lora.json` manifest, `main.json` D4 removed | manifests validate (main 57 / lora 6); **gpu-dry-run 4/4 PASS** |
| **#6b judge** | `judge_code.py` Phase-B driver (Haiku 4.5, 3-axis, cost guardrails, `--dry-run`/`--limit`/resume) | compiles; CPU dry-run clean (md5 `329927ae`) |
| **C1 (CD7)** | `utils/tokenizer_top10k.py` ported + `__call__` shim; `load_tokenizer` branch | server load + encode/decode round-trip clean; eos id 9999 |
| **#8 Capo/Mano (built)** | `eval_capo.py`+`eval_mano.py` (reuse our `build_model`+`load_tokenizer`), `capo.json`+`mano.json` manifests; capo_full **regenerated** (16k people→48k bios), mano_L10_full staged, `val.parquet` reconciled, eval columns verified | compiles; manifests validate (capo 2 / mano 3); on server. **Runs after Phase 1 frees GPUs.** See §12 |
| **Phase 1** | launched: `main.json --seeds 42`, 19 arms, both A100s, wandb on | 🟡 **RUNNING NOW** (vanilla-2L done: 18min, loss 8.6→2.9; TRM 3.0 sec/step) |

### B. Build work — REMAINING ⬜ (code still to write)
| Item | What it is | Gate (when) |
|---|---|---|
| **`select_stacked_best.py`** | greedy val-select of beneficial knobs (W1-high-risk; report #configs) | needs Phase-1 val results (noise scale) |
| **Paper enforcement** | L4–L7 framing, **W7 (real LoRA param count)**, A4-dropped note, **≤512-token scope limitation**, software-stack table, **+ capo/mano: report storage/manipulation COMPARATIVELY not as absolute 2-bits/param (§12)** | writing phase |

> #8 Capo/Mano is now **BUILT** (moved to A). Only `select_stacked_best.py` remains as net-new code.

### C. 👤 MANUAL STEPS YOU RUN (sequenced — operator actions, after Phase 1)
> Full commands in §9.5. C1 may now PASS within Phase 1 (wrapper landed mid-sweep), so expect **18–19/19** done.

| Order | YOU do | Ref | ⚠️ Notes |
|---|---|---|---|
| 1 | Verify Phase 1 complete (done/failed counts; `baseline__seed_42.pt` exists) | §9.5-A | only C1 was expected to fail — may now pass |
| 2 | Capture measured `sec/step(med)` from epoch-1 log | §9.5-B | **Rule 8** — our FIRST real runtime number |
| 3 | LoRA **Stage A**: `lora.json --seeds 42` (frozen grid + unfrozen_42) | §9.5-C | `--seeds 42` REQUIRED (else 65/108 fail — no baseline yet) |
| 4 | `python select_lora_lr.py --name spine` → winner LR | §9.5-C | selection on VAL only |
| 5 | **Phase 2**: `main.json` seeds 65,108 (+`--timeout` now safe) | §9.5-D | produces `baseline_65/108` |
| 6 | LoRA **Stage B**: add winner-lr frozen @[65,108] to `lora.json`, re-run | §9.5-E | needs Phase-2 baselines |
| 7 | **💲 MANUAL JUDGE (PAID)** — `judge_code.py --dry-run` FIRST, then real run | §9.5-F | Phase B; costs API $; check cost estimate before committing |
| 8 | **Run Capo/Mano** (built): `--gpu-dry-run` to measure → set epochs → train `--no-eval` → `eval_capo`/`eval_mano` | §12 | data staged; measure sec/step FIRST (Rule 8) |
| 9 | Build + run **`select_stacked_best.py`** | next chat | unbuilt — after results land |
| 10 | Paper: W7 / ≤512 scope / software-stack / framing + **capo/mano comparative framing** | writing | |

**Don't-miss flags:** step 7 is the only **paid** step (always `--dry-run` first). Capo/Mano (8) is **built + data staged** — the action is RUN, not build; only `select_stacked_best.py` (9) is still unwritten. The diagnostic explains the null; it is NOT optional.

---

## 12. 🔬 Capo/Mano diagnostic (#8) — BUILT, pending GPU

**What it is:** the storage-vs-manipulation diagnostic that explains WHY TRM fails at code (the Ouro mechanism). Runs on OUR 27.83M TRM + vanilla (NOT the Ouro project's own models — citing those would not be rigorous). Capo = storage (memorise biographies → bits at attribute positions); Mano = manipulation (mod-23 prefix arithmetic → exact-match accuracy). Predicted: TRM ≈ vanilla on Capo; TRM > vanilla on Mano **iff** recursion helps at 28M.

**Files (on server):**
| File | md5 | Role |
|---|---|---|
| `eval_capo.py` | `7293b265` | storage metric (CE@attr-positions → bits/param) |
| `eval_mano.py` | `db05fbd9` | manipulation metric (exact-match by expr length) |
| `configs/sweeps/capo.json` | `0b1daaa6` | `arch_trm` + `arch_vanilla` (iso-param 2L), seq256/ep50 |
| `configs/sweeps/mano.json` | `9e632aef` | + `arch_vanilla nlay=20` (iso-DEPTH, NOT iso-param), seq128/ep30 |

**Port design (verified, not assumed):** both evals reuse OUR `build_model(config)` (reconstructs trm/vanilla from sidecar → no hand-rolled kwargs / no `tie_weights` risk) and OUR `train_code.load_tokenizer(config)`. Metric math kept verbatim. Our `forward(input_ids, targets=None)` returns logits (+ tuple-guard). `sweep.py --no-eval` (line 257) = train-only, so the code-eval is skipped; the capo/mano evals run separately.

**Data (staged under `CASCON-1/data/`):**
- `capo_full` REGENERATED via `gen_capo.py --n-people 20000 --n-templates 50 --augment 3 --permute --seed 42` → 16k train people → **48k bios**, 4k val. Columns verified: `attr_birth_date/city/university/major/employer` present.
- `mano_L10_full` copied from the Ouro project. Columns verified: `expr_length`, `answer`.
- **`val.parquet` reconciliation:** our trainer reads `val.parquet`; Ouro ships `validation.parquet`. We `cp validation.parquet val.parquet` in each (keep BOTH: train uses `val.parquet`, eval_mano reads `validation.parquet`). capo eval reads the **train** split (storage = what's memorised).

**Run order (after Phase 1/2 + LoRA free the GPUs):**
1. `--gpu-dry-run` each manifest → **measure sec/step (Rule 8)** → set epochs.
2. **Round 1** = `sweep.py --manifest configs/sweeps/capo.json --name capo --data-dir data/capo_full --no-eval ...` (and mano).
3. `eval_capo.py --checkpoint checkpoints/<run>.pt --data-dir data/capo_full --split train`; `eval_mano.py ... --split validation`. Output: `results/<ckpt_stem>_{capo,mano}_eval.json`.
4. **Round 2** (after `select_stacked_best.py` exists): append a `best_trm` arm (overrides = winning knob stack) to each manifest → re-run sweep (resume skips Round-1 arms) → eval the new checkpoints.

**🔁 TWO ROUNDS OF TRM (D8 UPDATE — was best-TRM-only; now BOTH):**

> ⚠️ **SUPERSEDES D8 in the chat-1 decisions log** (`CASCON-1_DECISIONS_LOG_chat1.md`). That log is a frozen chat-1 artifact — NOT edited. The chat-1 D8 said the Ouro diagnostic uses *only* the stacked-best ("best") TRM. **This handoff is now the current source of truth: D8 runs TWO rounds — baseline TRM AND best TRM.** Reason: the chat-1 version explained only the strongest recursion, leaving the "your baseline was under-tuned" objection unanswered for the *reported* model; running both closes that.

- **Round 1 = baseline TRM** + vanillas → ties the diagnostic to the **reported headline null** (the paper reports baseline TRM ≈ vanilla on code, so the explanation must be about *that* model).
- **Round 2 = best/stacked-best TRM** → gives recursion its **strongest shot**; pre-empts "your TRM was just under-tuned." Its overrides are unknown until `select_stacked_best.py` runs, so it's appended later (run_id won't collide; vanillas reused).
- **Together:** storage/manipulation dissociation holds for BOTH the reported model AND the strongest recursion → the null is robust to TRM configuration. (W1-sanctioned: stacked-best used only in Ouro diagnostic + one headline comparison.)

**⚠️ Rule-8 runtime:** capo_full ≈ 48k bios; at TRM 3.0 sec/step, 100 epochs is infeasible (~days). Manifests ship ep 50/30 as STARTING values — measure via `--gpu-dry-run` and set epochs before launching full. Don't launch blind.

**📊 n-people rationale (researched — Allen-Zhu Part 3.3, arXiv 2404.05405):**
- LMs store ~**2 bits/param**; capacity scales linearly. Our 27.83M model ≈ **55.7M bits** capacity. capo_full ≈ 16k × ~38 bits ≈ **0.6M bits ≈ 1% of capacity** → far below saturation.
- Absolute 2-bits/param saturation needs ~millions of people + ~1000 exposures — NOT our goal.
- **20000 is correct for a COMPARATIVE storage test** (both models have ample room; we test if they store *equally*) and matches the Ouro `capo_full` reference. KEEP 20000.
- **Paper framing (don't overclaim):** report bits/param COMPARATIVELY (TRM vs vanilla), NOT against the absolute 2.0 — we're at ~1% of capacity so we won't hit 2.0, and that's expected. The `~2.0` line in `eval_capo`'s printout is Allen-Zhu's absolute reference, not our target. Exposures = epochs × augment (50×3 ≈ 150 → ~1-bit/param regime, plenty for comparison).

**🟡 Possible extension:** the diagnostic is currently **single-seed (42)**. If the TRM-vs-vanilla storage/manipulation delta is ambiguous, add seeds [65,108] for error bars (same pattern as the spine).

---

---

*End of Chat-2 handoff. Append-only. Does not replace the v2 handoff or the decisions log.*
