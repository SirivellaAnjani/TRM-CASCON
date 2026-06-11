# CASCON-1 — Decisions & Build-State Log

**Companion to** `HANDOFF_CASCON-1_code.md`. This is the *living record* of decisions made during the code-build phase — written so it can (a) feed the paper's Section IV write-up, and (b) let a fresh chat resume with zero context loss.

- **Author:** Anjani Sirivella (TMU MSc). Server user: `asirivel`.
- **Last updated:** 2026-06-10 (rev 2)
- **Update policy:** Claude updates this doc **only when the user explicitly says so**, and shares it on request. No silent edits.
- **Rev 2 changelog:** dropped A4 halting (defend ACT-off); kept both baselines (iso-param + iso-depth); added arm sub-label taxonomy; multi-seed now applies to ALL arms (phased 42 → 65/108); added L1–L8 landmine table; pinned stacked-best scope (Ouro + one final headline comparison only).

---

## 0. TL;DR — project in three lines
Evaluate **TRM-AR** (Tiny Recursive Model adapted for autoregressive generation) vs **standard transformers** on **Python code generation at ~28M params**. Result is a **predicted null** (recursion doesn't help recall-bound code-gen at small scale). This phase = **build + run** the clean experiments that establish that null rigorously, plus an **Ouro mechanistic diagnostic** that explains *why* (storage vs manipulation).

---

## 1. Server facts (verified this session — not assumed)
| Fact | Value | How verified |
|---|---|---|
| venv | `venv-knob` → **Python 3.10.14**, torch **2.6.0+cu124**, CUDA ✅, **2× A100** | ran `python --version` + torch probe inside Docker |
| `/data` layout | matches handoff §3 **plus** two undocumented dirs | `find /data -maxdepth 3` (566 entries) |
| `folder-template/` | exists (clean scaffold) — **reference only**, we build our own tree | live listing |
| `trm-ouro-diagnostic/` | exists (Ouro code: `gen_capo.py`, `gen_mano.py`, `eval_capo.py`, `eval_mano.py`) — port what we need | live listing |
| TRM model knob names (Option A `trm.py`) | `use_post_norm`, `use_additive_combiner`, `use_absolute_pos_emb`, `use_depth_gate`, `use_per_depth_norm`, `use_sandwich`, `use_y_residual`; LoRA via `enable_lora()` | grep of constructor |
| AR adaptation already in old code | causal mask (`torch.triu`/`masked_fill`) + next-token CE (`F.cross_entropy`, pad→-100) | grep of `trm.py` / `train_code.py` |
| Data source | read-only from `/data/common/datasets/` | handoff §3 + listing |

---

## 2. Decisions made (with reasoning — for the paper)
Evidence labels: **Fact** = backed by file/log/paper; 🟡 = judgment call.

| ID | Decision | Reasoning (paper-ready) | Defensibility impact |
|---|---|---|---|
| **D1** | **Fresh rebuild**, Option A: adapt the existing AR `models/trm.py`; old project is reference-only | Old tree has a stale docstring + pilot-style defaults (silent wrong-baseline risk), a harness missing failure-isolation/`--name`/seed-as-first-class, and heavy `.bak` cruft. The AR adaptation (causal mask + next-token CE) is **already implemented** in old `trm.py`, so Option A makes AR a config concern, not new code. The paper TRM is non-AR (direct/whole-answer, bidirectional, puzzle grids); re-deriving AR from the Samsung repo would inject new bugs. | + clean provenance |
| **D2** | **fp32 primary** precision; **bf16 added as a Condition** (robustness) | bf16 is the field default *but* the result is a **precision-sensitive null**, and recursion iterates the same network many times → low-mantissa error could handicap the *recursive* model specifically (the obvious reviewer rebuttal). Memory is not binding at ~28M on 80GB. fp32 removes the confound; running bf16 as a measured condition shows robustness. | + removes confound |
| **D3** | **flat-after-warmup** LR primary; **cosine added as a Condition** | Matches the TRM paper repo (constant `lr=1e-4`, no cosine), matches the old pilot, and is empirically on-par with cosine at small scale (Granite SFT study). Cosine would be a deviation from the model under evaluation. | + faithful reproduction |
| **D4** | **Ablate TRM only**; vanillas are **fixed comparison baselines**, never ablated with TRM-internal knobs | Field convention: an ablation removes components *of the proposed model*; the "baseline" in an ablation study is the full proposed model, with all else held constant. TRM-internal knobs (post-norm, combiner, per-depth norm…) don't exist in a plain transformer. | core requirement |
| **D5** | **Two matched baselines:** vanilla-2L (iso-parameter ~28M) + vanilla-20L (iso-depth; effective depth = T×(n+1)×layers = 2×5×2 = 20) | A recursive model confounds parameter count and effective depth at once; you must control both. Matches the TRM/HRM convention of comparing at "similar effective depth." | core requirement |
| **D6** | **Multi-seed (42,65,108) for ALL arms**, run in two phases: **Phase 1** = every arm @ seed 42 (initial results → paper draft); **Phase 2** = seeds 65,108 for every arm (error bars, sequential ok, runs while writing). Same `--name` group. | Ablation deltas are tiny (old log: 2.086/2.084/2.085 ≈ noise) → single-seed selection chases noise; compute is cheap at 28M + 2 GPUs; multi-seed-everything pre-empts "why single-seed here?" reviewer questions. Recipe: fixed grid in advance → select on val → test once per seed. | + removes seed-noise + multiple-comparisons exposure |
| **D7** | A **"stacked-best TRM"** config, built by greedy forward-selection of beneficial ablations on **val** (only adopting a component whose delta **exceeds multi-seed noise**), evaluated on **held-out test**, **multi-seed**, exploratory framing, #configs-tried reported. **SCOPE (W1): used in exactly two places — the Ouro diagnostic and ONE final headline comparison vs the vanillas. Never an input to any ablation/condition/architectural result.** | The old greedy stack was indefensible (no test set, single seed). A disciplined additive study is recognized *with* a test-set firewall. For a null, "even the greedy-best stack still loses" *strengthens* it. | ⚠️ HIGH RISK — see W1, L3 |
| **D8** | **Ouro in the pipeline (Option B, reconciled):** the **"best TRM" = the stacked-best TRM** (D7) — its strongest, test-validated, multi-seed config — run vs vanilla-2L + vanilla-20L on **Capo (storage)** and **Mano (manipulation)** | Converts the null from "TRM lost" into "TRM lost *for a measured reason*" — the storage/manipulation dissociation (Ouro, arXiv:2510.25741). Using the stacked-best gives recursion its **strongest shot** on the diagnostic. Reuses the train skeleton; only data + 2 evals + 2 generators differ → **zero model-code change**. | + mechanistic explanation; must be multi-seed (W3) |
| **D9** | **Discard all old numbers** — clean slate; old log is reference-only for *what was tried* | Old data has **no test set**, single seed, **broken param-matching** (rows span 7.55M–94M), **mixed datasets** (TinyStories vs TinyCode vs Alpaca), no dedup, and data-integrity oddities (duplicate val_loss to 15 digits; mis-logged columns). | + removes tainted data |
| **D10** | **Data build:** tiny-codes → Python-only → filter-to-fit-512 → **exact-match dedup** → **80/10/10 @ seed 42**; new dir `tiny_codes_python_512_dedup_811/`; read from `/data/common/datasets/` | Leakage guard + an actual held-out test set (the old work had neither). | + leakage control + test set |
| **D11** | **Judge = Claude Haiku 4.5** (`claude-haiku-4-5-20251001`); 3 axes (syntax/logic/relevance), 0–10 | A prior chat fabricated "Llama-3.3-70B." Corrected and verified against `utils/judge_code.py`. | factual correctness |
| **D12** | **Harness design:** per-run result file = single source of truth; failure isolation (child process → `.failed` + `_failures.log` → continue → `--retry-failed`); atomic NFS-safe writes; `--name` cross-day grouping; **multi-arm manifest** launch; dry-run all; **GPU packing with 2× peak reservation**, validated on the first concurrent dry-run; **wandb on** with clear arm/run names | Crash-safety + no-overwrite + the "launch and leave" failure isolation the user cares most about. Small models + 2 GPUs make memory-packing worthwhile; 2× reservation is the agreed safety rule (tenant contention assumed absent). | neutral (infra); see W8 |
| **D13** | **Paths:** root `/data/trm-code-experiments/CASCON-1/`; build our own tree (folder-template reference only) | Clean, tailored structure. | + provenance |
| **D14** | **Drop A4 halting; defend ACT-off baseline in prose.** Do **not** test halting as an ablation | TRM's halting is whole-answer **Q-learning ACT** over fixed-output puzzle refinement; it does **not** map to token-by-token AR. The faithful AR analogue is **token-wise adaptive depth** (Universal Transformer / PonderNet / AdaPonderLM / Ouro's exit gate) — a distinct, finicky mechanism, out of scope. Old code's BCE-sigmoid halt is neither → testing it = a strawman (landmine L2). Dropping + defending is **more** defensible than testing a strawman. Cite UT (Dehghani 2018), PonderNet (Banino 2021), AdaPonderLM (2026). | + removes fidelity landmine |
| **D15** | **Arm sub-label taxonomy** (paper-facing). "Ablations" is user shorthand in chat; the paper and Claude differentiate: **Ablations** (remove/reduce), **Design-choice reversions** (flip paper's choice to alternative), **Architectural additions** (add a component), **Hyperparameter robustness**, **Conditions** (data/training regime) | Convention: a true ablation *removes* a component. Mixing additions + hyperparameter sweeps under "ablation" invites reviewer pushback (landmine L1). | + honest framing |

---

## 3. Updated experiment list
**Naming:** baseline = paper config (post-norm, additive combiner, RoPE-only, ACT off, warmup 2000). Each row = **one override delta** over `base_code.json`. ⚠️ = needs model/trainer code. **All arms run at 3 seeds (42/65/108), phased** (see §3a).

### 3.1 Spine / baselines → 3 configs ×3 seeds = 9 runs
| ID | Model | Override delta |
|---|---|---|
| S1 | TRM-AR (baseline) | `{}` (anchor) |
| S2 | Vanilla-2L (iso-param) | `{"architecture":"vanilla","n_layers":2}` |
| S3 | Vanilla-20L (iso-depth) | `{"architecture":"vanilla","n_layers":20}` |

### 3.2 TRM design-space study (sub-labeled per D15) — TRM only, ×3 seeds
**Ablations** (remove/reduce a baseline mechanism)
| ID | Experiment | Override delta | Code? |
|---|---|---|---|
| A1 | nsup=1 (remove deep supervision) | `{"n_supervision_steps":1}` | ✅ |
| A2 | recursion depth=2 | `{"n_latent_recursions":2}` | ✅ |

**Design-choice reversions** (flip the paper's choice to its alternative)
| ID | Experiment | Override delta | Code? |
|---|---|---|---|
| A3 | post→pre-norm | `{"use_post_norm":false}` | ⚠️ |
| A5 | RoPE-only→+absolute pos | `{"use_absolute_pos_emb":true}` | ⚠️ |
| A6 | additive→concat combiner | `{"use_additive_combiner":false}` | ⚠️ |

**Architectural additions** (add a component)
| ID | Experiment | Override delta | Code? |
|---|---|---|---|
| A10 | y-residual | `{"use_y_residual":true}` | ⚠️ |
| D1 | per-depth RMSNorm | `{"use_per_depth_norm":true}` | ⚠️ |
| D2 | gated residual | `{"use_depth_gate":true}` | ⚠️ |
| D3 | sandwich | `{"use_sandwich":true}` | ⚠️ |
| D4 | per-depth LoRA (frozen + unfrozen = 2 runs) | `{"lora":"frozen"}` / `{"lora":"unfrozen"}` | ⚠️ |

**Hyperparameter robustness**
| ID | Experiment | Override delta | Code? |
|---|---|---|---|
| A7 | weight decay 0.5 | `{"weight_decay":0.5}` | ✅ |
| A8 | weight decay 1.0 | `{"weight_decay":1.0}` | ✅ |
| A9 | embedding LR | `{"embedding_lr":0.01}` | ⚠️ (param-group) |

> **A4 (halting) — DROPPED** per D14. ACT-off is defended in prose, not tested.

### 3.3 Conditions (data/training regime) — ×3 seeds
| ID | Experiment | Override delta | Code? |
|---|---|---|---|
| C1 | vocab 10K | `{"vocab_size":10001,"tokenizer":"gpt2_top10k"}` | ⚠️ tokenizer swap — **breaks iso-param, see W7** |
| C2 | epochs 6 | `{"epochs":6}` | ✅ |
| C3 | precision bf16 | `{"precision":"bf16"}` | ⚠️ trainer |
| C4 | lr schedule cosine | `{"lr_schedule":"cosine"}` | ⚠️ trainer |

### 3.4 Stacked-best TRM — ×3 seeds = 3 runs
Greedy forward-select beneficial ablations on **val** (delta must exceed multi-seed noise, L3) → stack → evaluate on **test**. Exploratory; report #configs tried. **Scope: Ouro + one final headline comparison only (W1/D7).**

### 3.5 Ouro diagnostic — after main sweep
**Best TRM = the stacked-best TRM (D8).** Compare **stacked-best TRM vs vanilla-2L vs vanilla-20L** on:
| Task | Tests | Eval | Predicted |
|---|---|---|---|
| **Capo** | storage (attribute recall) | `eval_capo.py` (CE on attribute positions) | TRM ≈ vanilla |
| **Mano** | manipulation (modular-arithmetic chains) | `eval_mano.py` (exact-match acc) | TRM > vanilla *iff* recursion helps at 28M |

*Decision rule:* TRM≈Capo + TRM>Mano ⇒ diagnosis confirmed. TRM loses both ⇒ 28M too small. TRM wins both ⇒ revisit. **Ouro must be multi-seed (old Ouro was single-seed/near-floor) — see W3.**

### 3a. Seed phasing
| Phase | Seeds | Scope | When |
|---|---|---|---|
| **Phase 1** | 42 | every arm (spine + design-space + conditions) → aggregate → select stacked-best → stacked-best @42 → Ouro @42 | first — gives initial paper-draft numbers |
| **Phase 2** | 65, 108 | **every arm** (multi-seed everything, to pre-empt reviewer questions); **sequential ok** | background, while writing |

Both phases share one `--name` group. run_id seed-encoding prevents overwrites. (~13 configs × 3 seeds + Ouro.)

---

## 4. High-level sweep design
**One principle:** the per-experiment result file on disk is the **single source of truth**; aggregate CSV / summary / wandb are *rebuilt from it*, never authoritative. This buys crash-safety + no-overwrite + failure-isolation at once.

**Four independent, resumable stages** (each a pure function of files on disk):
| Stage | Does | GPU? | Restart |
|---|---|---|---|
| validate | resolve variants; known knobs? typos? **run_id collisions?**; disk writable | No | pure pre-flight |
| train | per variant: build → step → atomic checkpoint+sidecar; skip-if-done | Yes | skips done, resumes at failures |
| eval | per variant: load checkpoint from disk → metrics + judge | Yes (+API) | runs later/separately; judge fail ≠ train fail |
| aggregate | rebuild CSV + summary from existing result JSONs | No | runnable anytime, idempotent |

**CLI surface:** `--sweep <manifest of arm files>` (multi-arm, one launch), `--name <label>` (cross-day group; default `sweep_<config>_<date>`), `--dry-run` (1-batch every variant → pass/fail table), `--train-only` / `--eval-only`, `--retry-failed`, `--skip-judge`.

**No-overwrite:** run_id encodes every distinguishing knob **+ seed** (e.g. `nsup_1__seed42`); spine fans out `seeds:[42,65,108]`; collision check at validate-time refuses to launch on duplicate run_id.

**Failure isolation (the #1 worry):** each variant = **child process**; parent catches (never propagates) → `<group>/_failures.log` + `<run_id>.failed` → **continue** → end summary (N done / M failed); `--retry-failed` re-runs only `.failed`.

**Crash-safety:** atomic writes = temp file **in the same `/data` dir** → `flush`+`fsync` → `os.replace` (NFS-safe; never `/tmp`→`/data`). "Done" = checkpoint + sidecar + (if eval) result JSON all exist.

**GPU scheduler:** worker slots across 2 GPUs; each job pinned via `CUDA_VISIBLE_DEVICES`; reservation = **2× measured dry-run peak**; admit if `Σ reservations + reserve(X) ≤ GPU budget`; if `2×peak > budget`, run solo. We track our own reservations (no tenant polling). **Validate the 2× margin empirically on the first concurrent dry-run before full packing.**

**Dry-run:** identical build+step path, 1 batch, no checkpoint/wandb; dry-run largest variant first; run every variant → green-light.

**Directory layout (new root):**
```
trm-code-experiments/CASCON-1/
├── configs/  base_code.json + sweeps/{ablations,conditions,architectural,spine_seeds,ouro}.json
├── data/     build_dataset.py
├── models/   trm.py (Option A), vanilla.py
├── train_code.py  train_code_vanilla.py  eval_code.py  sweep.py
├── utils/    paths, naming(+seed), atomic_io, sweep_validation, metrics_code, judge_code
├── checkpoints/<run_id>.pt + .json
├── results/<group>/<run_id>.json + <group>.csv
└── logs/<group>/_failures.log + per-run logs
```

**Build order:** (1) utils foundation → (2) config schema + one config-only arm → (3) sweep.py skeleton (validate + dry-run + failure-isolation + multi-arm + 2× packing) → (4) train/eval (atomic, skip-if-done, decoupled) → (5) aggregate → (6) main arms → (7) Ouro arm + model-selection step.

---

## 5. ⚠️ Defensibility watch list (CRITICAL — review on every change)
| ID | Watch item | Rule to hold |
|---|---|---|
| **W1** | **Stacked-best greedy selection** = highest risk of cherry-picking / multiple-comparisons | Select on **val**, report on **test**, multi-seed, exploratory framing, report #configs tried. Never select+report on the same set |
| **W2** | **Baseline cherry-picking** (old work had 7 vanilla variants) | Pre-specify baselines = vanilla-2L (iso-param) + vanilla-20L (iso-depth) **before** looking at results |
| **W3** | **Ouro** must not inherit single-seed/near-floor weakness | Multi-seed the Ouro comparison; "best TRM" = stacked-best (resolved, D8) |
| **W4** | bf16 / cosine conditions could re-introduce multiple comparisons if stacked | Run them as robustness conditions **at the baseline only**, not inside the greedy ladder |
| **W5** | Single-seed ablations can't carry the headline | Resolved by D6 — **all arms are multi-seed** now |
| **W6** | Hidden confounds across arms | Identical training conditions across all arms (data, precision, schedule, steps) **except** the one knob under test |
| **W7** | **iso-param breaks when vocab/dim change** (old rows ranged 7.55M–94M) | When C1 (vocab 10K) or any dim change runs, **report actual param counts**; do **not** claim iso-param across vocab/dim changes |
| **W8** | Concurrency must not affect results | Each run independent + seeded + separate files; packing changes wall-clock only, never outcomes |

### 5a. L1–L8 landmine table (final defensibility sweep — review on every change)
| # | Sev | Landmine | Guardrail | Status |
|---|---|---|---|---|
| **L1** | 🔴 | "A" arm isn't all ablations (mixes additions + hyperparameter sweeps) | Sub-label honestly (D15): ablations / reversions / additions / hyperparameter / conditions | ✅ resolved |
| **L2** | 🔴 | A4 halting fidelity — paper's Q-learning ACT ≠ AR; old BCE-halt is a strawman | Drop A4; defend ACT-off in prose w/ UT/PonderNet/AdaPonderLM cites (D14) | ✅ resolved |
| **L3** | 🔴 | Greedy selection chasing seed noise | Adopt a component only if its delta exceeds **multi-seed** noise (D6, D7) | ✅ guarded |
| **L4** | 🟠 | vanilla-20L is iso-*depth* not iso-param (more params) | Frame 20L as a depth control / upper reference; iso-param 2L is the fair fight | ⬜ enforce in writing |
| **L5** | 🟠 | Ablating a model that loses overall is unusual | Frame: characterizes TRM's internal sensitivity + locates best-TRM for Ouro + shows null is robust to design choices | ⬜ enforce in writing |
| **L6** | 🟠 | Multiple comparisons across ~13 configs | Pre-registered grid (in configs) + one selection criterion + individual rows are descriptive; significance only on multi-seed headline | ⬜ enforce in writing |
| **L7** | 🟡 | iso-FLOP was cut → can't claim compute-matched; A1/A2 alter compute | Never claim "compute-matched"; report effective depth; note A1/A2 change compute | ⬜ enforce in writing |
| **L8** | 🟡 | Ouro inherits the stacked-best (greedy) model | Fine because stacked-best is test-validated + multi-seed; state Ouro uses TRM's *strongest* config | ✅ resolved |

---

## 6. Open gates (settle before / during coding)
| # | Gate | Status |
|---|---|---|
| 1 | Config-key ↔ model-param mapping | ✅ resolved — real names verified (§1) |
| 2 | venv Python version | ✅ resolved — 3.10.14 works |
| 3 | Precision / LR schedule primary | ✅ fp32 / flat (D2, D3) |
| 4 | New root path + dataset dir name | ✅ both confirmed (D10, D13) |
| 5 | Ouro "best TRM" definition | ✅ resolved — **stacked-best TRM** (D8) |
| 6 | A4 halting | ✅ resolved — **dropped**, ACT-off defended (D14) |
| 7 | A9 embedding-LR param-group split | ⬜ open (trainer impl detail — non-blocking for design) |
| 8 | D4 LoRA frozen/unfrozen + per-depth wiring | ⬜ open (architectural-arm impl detail) |

---

## 7. Hard rules (carry forward to any new chat)
No code without explicit permission · pre-code disclosure (issues/unique/flags, warnings in prose **and** inline comments) · backup before overwrite (`cp f f.bak_<ts>`) · file-modification protocol (read → state line edits → wait → apply) · read don't assume (verify against files/server) · evidence for every server command · one bash block per code drop, user drives tmux · runtime estimates from measured logs only · evidence labels (Fact/🟡/Assuming) · ADHD-friendly dashboard format · one clarifying question at a time · staged approval · don't reference "supervisor" (use "flagging for review").

**Server access:** Local → `ssh moon.cs.torontomu.ca` → `ssh gpuclient` → `cs-container-connect`. Inside Docker every block starts with `cd /data/<proj>` + `source /data/venv-knob/bin/activate`. Transfer staging = `~/move` only. NFS at `/data/` → pyarrow parquet reads, local tokenizer, delete stale `.lock` before launch.

---

## 8. Code build log — progress, nuances & flags
**Rev 3 (2026-06-10):** added this section as we write code file-by-file.

### 8.1 Build progress
| # | File(s) | Status | Verified |
|---|---|---|---|
| 1 | `utils/{paths,naming,atomic_io,sweep_validation}.py` + `tests/test_utils.py` | ✅ written | 15/15 unit tests (sandbox) |
| 2 | `configs/base_code.json` + `tests/test_base_config.py` | ✅ written | 21 arms ×3 seeds = 63 unique run_ids resolve; baseline paper-correct |
| 3 | `data/build_dataset.py` | ✅ written + **run on server** | sandbox 5/5; full build OK → train 38,119 / val 4,765 / test 4,765 |
| 4 | `models/{trm,vanilla,build}.py` + `__init__.py` + `tests/verify_models.py` | ✅ written + **run on server** | param counts exact; iso-param 768; forward OK |
| 5 | `train_code.py` + `tests/test_train_masking.py` | ✅ written + **dry-run passed** | masking 6/6; dry-run ok (peak 52.87 GiB, fits A100) |
| 6 | `eval_code.py` + `utils/metrics_code.py` + path helpers | ✅ written + **dry-run passed** | metrics 10/10; eval path proven on smoke ckpt |
| 7 | `sweep.py` + `configs/sweeps/main.json` | ✅ written + **dry-run passed** | 63 run_ids validated; orchestrator plan OK |
| 6b | `judge_code.py` (Phase B, manual) | ⬜ not started | |
| 8 | ouro + stacked-best | ⬜ not started | |

### 8.2 Code-level nuances & flags (decisions baked into code — for the paper + continuity)
| Area | Nuance / flag |
|---|---|
| **run_id format** | `<alpha-sorted override segments>__seed_<N>`; **both shared + variant overrides** are encoded (prevents cross-sweep overwrite); floats never contain `.` (e.g. `wd_0-5`, `lr_1e-4`); unknown knob = hard error (catches typos pre-GPU); soft-cap 100 chars → truncate + hash. |
| **Atomic writes** | temp file **in the same dir** as target → fsync → `os.replace`. NFS rule: a `/tmp`→`/data` rename is NOT atomic. Used for checkpoints, sidecars, results, **and the dataset parquet**. |
| **Paths ownership** | `utils/paths.py` owns ALL paths. Project paths are relocatable (derived from file location). Shared `/data/common` is **env-overridable** (`TRM_COMMON_ROOT`, `TRM_DATASET_DIR`) so code runs off-server too. Config carries dataset *identity*; paths carries *location*. |
| **Config ↔ model** | base_code.json keys for model knobs use the **exact `trm.py` param names** (`use_post_norm`, `use_additive_combiner`, `use_absolute_pos_emb`, …) → trainer passes them straight through, no mapping layer. `halt_loss_weight:0` kept (ACT off). `embedding_lr:null` = use main lr. `max_*_samples:null` = full split. |
| **⚠️ Tokenization consistency contract (#3 ↔ #5)** | filter-to-fit-512 counts `len(tok(prompt)) + len(tok(response)) + 1` (EOS budget) and keeps ≤ `max_seq_len`. **The trainer (#5) MUST tokenize the same way** so kept examples never need truncation (truncation would corrupt prompt/response). This is a hard contract between the data build and the trainer. |
| **Dedup definition** | exact-match on the **(prompt, response) pair**. Done BEFORE the split, so by construction there is zero train/val/test leakage (also asserted via disjoint id-sets). |
| **Data source default** | `build_dataset.py` defaults `--source` to `tiny_codes_python_only` (the documented FULL Python split), globs `**/*.parquet`, and **re-splits fresh** 80/10/10 (combining any pre-existing train/val). Language filtering applied only if a `programming_language` column is present. `--source` overridable to read raw + filter Python ourselves. |

### 8.3 Open consistency items to honor downstream
- **#5 trainer** must reuse the exact filter-to-fit tokenization (contract above) and decide prompt-masking (loss on response only vs full) — a length-neutral choice, flagged for #5.
- **W7** still applies: C1 (vocab 10001) shifts param count → report real counts at eval, don't claim iso-param.

### 8.4 Built dataset — FACTS from the server run (2026-06-10)
**Location:** `/data/common/datasets/tiny_codes_python_512_dedup_811/` → `train.parquet` (38,119), `val.parquet` (4,765), `test.parquet` (4,765), `_build_manifest.json`. Built from `tiny_codes_python_only` (129,063 Python rows), seed 42, local GPT-2 tokenizer.

| Stage | Result |
|---|---|
| Source Python rows | 129,063 (single language: `Python`) |
| Exact-match dedup | removed **0** (source already clean; guard retained) |
| Filter-to-fit ≤512 tok | removed 81,414 → **47,649 kept (36.9% survival)** |
| Token length | min 65 / median 407 / **max 512** (no-truncation contract holds) |
| Split @ seed 42 | train 38,119 / val 4,765 / test 4,765 (sums exactly) |

**⚠️ SCOPE LIMITATION (state in paper):** the 512-token filter drops **63%** of tiny-codes Python. The dataset is the **short-program subset** (median 407 tok). The TRM-AR-vs-transformer null therefore generalizes to **≤512-token Python code generation**, not arbitrarily long programs. Standard context-length constraint; report as a limitation.

**Note:** the `transformers` "1329 > 1024" warning during the build is **cosmetic** — GPT-2's tokenizer states its 1024 nominal context, but we only use it to *count* tokens (no model forward). The flagged long sequences are exactly the ones the 512 filter removes.

### 8.5 test.parquet firewall
`test.parquet` (4,765) is **held out and untouched** until the single final eval. No tuning, no checkpoint selection, no peeking touches test (enforces W1 / L5 test-set firewall).

### 8.6 Models — FACTS from server verify (2026-06-10)
**Decision (D16): vanilla baseline matched to TRM on every shared axis.** Vanilla is now *exactly* "TRM's transformer core applied once instead of recursively." Audit confirmed RMSNorm / RoPE / SwiGLU / causal-attention forward are **byte-identical** between the two files.

| Model | Params (verified) | Role |
|---|---|---|
| TRM-AR baseline | **27,830,784** | primary (recursive) |
| Vanilla-2L | **27,830,016** | iso-param baseline (Δ = 768 = `y_init`+`z_init`+`halt_head`, recursion-only) |
| Vanilla-20L | **46,713,600** | depth control = 2×(n+1)×layers; **1.68× params → NOT param-fair** (L4/W7) |

**Adaptations made to old code (Option A):**
- `trm.py`: defaults flipped to paper-correct (post-norm, additive combiner, RoPE-only, max_seq_len 512); `forward` default `halt_loss_weight 0.1 → 0.0` (ACT off; A4 dropped). Config still overrides everything explicitly.
- `vanilla.py`: post-norm branch added (identical to TRM); **pos_emb removed → RoPE-only**; `use_post_norm` default True.
- `models/build.py` (new): `build_model(config)` filters config→constructor kwargs by `inspect.signature`; `model_loss()` hides TRM's extra forward args (`n_supervision_steps`, `halt_loss_weight`) from callers so the trainer treats both architectures uniformly.

**🟡 Flag (drift):** vanilla keeps its own (today-identical) copy of the building blocks. If a block is ever edited, edit both files. `tests/verify_models.py` re-checks param counts as a guard.

### 8.7 D17 — Prompt masking = COMPLETION-ONLY (decided 2026-06-10, literature-backed)
**Decision:** train with **completion-only loss** — mask prompt tokens with `-100`, compute next-token CE only on response tokens + EOS. Applied identically to TRM and vanilla.

**Reasoning (from literature, not the old project):**
- **SFT default.** For prompt-completion datasets, completion-only is the standard/default (HF TRL `SFTTrainer` defaults to it; Cerebras IFT calls prompt-masking standard practice; EMNLP 2024 *Does Prompt Loss Matter?* calls PLW=0 "common").
- **Aligns with eval.** We generate response *from* prompt, so completion-loss is the metric that tracks generation quality. Full-sequence val-loss can mislead the checkpoint criterion (Vaughn/TDS).
- **Cleaner test of the recursion thesis (🟡 our reasoning, uncited):** completion-only grades the model on *generating* code (the manipulation task recursion is meant to help), not on echoing prompts (pure recall) — so it isolates the effect we study.
- **No recursion-specific masking rule exists.** Deep supervision (TRM nsup) is orthogonal to token-masking. The only recursive-AR example (Ouro) is full-sequence *because it is pretraining* (7.7T tokens), not an SFT masking choice.

**Caveats recorded:**
- ⚠️ **Short-completion fallback:** some tiny-codes responses are short. EMNLP 2024 notes a small non-zero prompt-loss-weight can stabilize short-completion SFT. If training destabilizes, fallback = small prompt weight (e.g. 0.1), **not** binary full-sequence. (Model supports it; not enabled by default.)
- Both arms use identical masking → the null comparison stays valid regardless.
- **Checkpoint metric = completion val-loss** in all cases.

**Mask math (contract):** sequence `S = [p_0..p_{P-1}, r_0..r_{R-1}, EOS]`, `input=S[:-1]`, `targets=S[1:]`; set `targets[:P-1] = -100` (the predictions of prompt tokens) + pads → -100. Length `P+R+1` = exactly the `n_tokens` #3 measured → never truncates.

### 8.8 Open for #5 (trainer) — remaining
- Trainer MUST honor the #3↔#5 tokenization contract: `tok(prompt)+tok(response)+[EOS]`, no separator, ≤512.
- `model_loss(model, ids, targets, cfg)` is the single forward entry point.

### 8.9 D18 — Tokenizer EOS/PAD handling (decided 2026-06-10, defensible)
**Finding (server fact):** the local `gpt2` tokenizer (`/data/common/tokenizers/gpt2`, vocab 50257) ships with **no special tokens registered** (`eos`, `pad`, `bos`, `unk` all `None`). The GPT-2 end-of-text marker `<|endoftext|>` nonetheless **exists in-vocab at id 50256**.

**Decision:** in `load_tokenizer`, register the **existing in-vocab** `<|endoftext|>` (id 50256) as both **EOS and PAD**, with asserts that (a) the vocab length does **not** grow and (b) `eos_id, pad_id < vocab_size` (so `output_head` can never receive an out-of-range id).

**Paper-defensible rationale:**
- `<|endoftext|>` is GPT-2's canonical document/sequence separator — using it as EOS is standard, not an invention.
- Using EOS as PAD is standard GPT-2 practice (GPT-2 has no dedicated pad token). **No contamination:** pad targets are set to `-100` (ignored by CE) and right-padding + causal attention means real tokens never attend to pads. So padding affects neither loss nor attention.
- **No vocab/embedding change** → the verified 27,830,784-param model is unaffected; the choice is purely a tokenizer-config registration.
- **Dataset unaffected / no rebuild:** `build_dataset.py` only *counted* a `+1` EOS budget (it never appended one or read `eos_id`), so the 47,649-example split already reserves exactly the space for the EOS that training appends. Sequence length stays `len(p)+len(r)+1 ≤ 512`.

**🟡 C1 caveat (deferred, NOT baseline):** `gpt2_top10k` (the C1 / vocab-10k condition) **fails fast-load** on this box ("need sentencepiece or tiktoken to convert a slow tokenizer"). A `use_fast=False` fallback was added but is **unverified**. Resolve when C1 runs (install sentencepiece/tiktoken, or confirm the slow path, or regenerate a fast `tokenizer.json`). Does not affect the spine or any gpt2-vocab arm.

### 8.10 Measured environment facts — TRM dry-run (2026-06-10)
Single optimizer step, TRM baseline, `bs=48`, fp32, seq≤512, A100 80GB, `--limit 2000`:

| Metric | Value | Use |
|---|---|---|
| Peak GPU memory | **52.87 GiB** | one TRM job fits (≈27 GiB headroom). **2×52.87 = 105 > 80 → cannot pack 2 TRM jobs on one GPU.** With 2× A100 → **max 2 concurrent TRM jobs (1 per GPU)** (D12 packing). |
| Loss at init | 10.83 | ✓ ≈ ln(50257) — model + masking sane |
| sec/step | 8.757 | ⚠️ **COLD first step only** (CUDA init/compile) — **NOT a runtime estimate** (Rule 8). Real sec/step comes from a multi-step median (smoke train). |

**Caveats recorded:** peak measured on `--limit 2000` may slightly **under**-estimate the full-data worst case (a batch that maxes at 512 tokens). For a single job the 27 GiB headroom absorbs this; before concurrent packing, confirm peak on a full-length batch. **TRM is activation-heavy** (deep recursion: nsup×T×(n+1)×layers effective compute) despite only 28M params — expected, and itself relevant context for the paper's storage-vs-compute framing.

### 8.11 D19 — Two-phase eval: automated local metrics + MANUAL judge last
**Server fact:** outbound to `api.anthropic.com` works inside Docker (probe returned HTTP 404 = reached server; judge ran fine in old experiments).

**Decision (user requirement):** the judge (Claude Haiku 4.5, costs money) is **carved out of the automated pipeline** and run **last, manually**, with cost visible.

| Phase | What | When | Network/cost |
|---|---|---|---|
| **A — automated** (one `sweep.py` launch) | per (arm×seed): **train → eval**. Eval = generate on TEST prompts, **SAVE generations to disk**, compute **LOCAL** metrics (AST-parse valid %, BLEU-4). Write run result JSON. | unattended, all 63 runs | none, free |
| **B — manual, LAST** (`judge_code.py`, user runs) | reads **saved generations** (no re-generation) → Claude Haiku 4.5 → 3 axes (syntax/logic/relevance) → judge scores | user triggers when ready | paid; `--dry-run` cost estimate, `--limit`, resumable (skips already-judged) |

**Why defensible / clean:** generations are persisted in Phase A, so the judge is fully decoupled — re-running it never re-trains or re-generates; cost is previewable and bounded; test set is touched only in Phase A (firewall intact). Eval scope knob (`eval_n_prompts × eval_n_completions_per_prompt`) set deliberately at #6 — it drives both Phase-A generation time and Phase-B judge cost.

### 8.12 Smoke-train facts (plumbing proven, NOT learning)
Both arms ran end-to-end on `--limit 2000 --epochs 1` (41 steps): train→val→checkpoint→sidecar→result JSON all written.
- TRM `sec/step(med)=3.392`; Vanilla-2L `sec/step(med)=0.448` → **TRM ≈7.6× slower/step** (the recursion compute cost; sanity-confirms the architectures differ as designed).
- ⚠️ Loss stayed ≈ init (10.8): `warmup_steps=2000` ≫ 41 steps, so LR never ramped. **Smoke = plumbing only.** Real timing comes from the first ~50 steps of a full-data run.

### 8.13 #6 eval + #7 sweep — built (2026-06-10)
**Eval (`eval_code.py`, Phase A, local-only):** loads best checkpoint (weights already EMA-baked at save) → samples N **test** prompts (firewall: test read ONLY here) → generates K completions (raw prompt, **no separator** — matches training) → local metrics **AST-parse valid %, BLEU-4, test completion-PPL** → **saves generations** (`<run_id>.gen.json`) for the manual judge → writes `<run_id>.eval.json`. Reuses `load_tokenizer/CompletionDataset/evaluate` from `train_code` (no tokenization drift). `metrics_code.py` ported as-is (AST + BLEU, no external deps). Verified on smoke ckpt: pipeline writes both JSONs.

**Sweep (`sweep.py`, the orchestrator):** ONE launch → per (arm×seed) a **child process** runs **train→eval** chained, pinned to one GPU. Failure isolation (crash/OOM → `_failures.log`, continue). Resume (skip completed; `--retry-failed`). GPU packing `--gpus 0,1 --max-parallel 2` = 1 job/GPU (§8.10; never 2 TRM/GPU). `--dry-run` validates all 63 run_ids + prints plan, launches nothing. `--seeds 42` = Phase-1. Manifest `configs/sweeps/main.json` = 21 arms × seeds [42,65,108] = 63 runs (validated unique).

### 8.14 Deferred arms + OPEN LoRA decision
3 of 21 arms are deferred-with-reason; they **fail-isolate** (logged, sweep continues) so the other **18 run fully automated**:
| Arm(s) | Blocker | Behavior until fixed |
|---|---|---|
| `D4a/D4b` (LoRA frozen/unfrozen) | LoRA training not wired; needs a trained baseline ckpt to fine-tune from + `enable_lora`/`freeze_shared` + LoRA-aware eval loading | **guard in `train_code` raises NotImplementedError** → no silent wrong data |
| `C1` (vocab10k / `gpt2_top10k`) | tokenizer fast-load fails (needs sentencepiece/tiktoken; slow fallback unverified) | fails at tokenizer load |

**🔴 OPEN DECISION (for next chat) — LoRA scope:**
- **(A)** wire LoRA fully now → all 21 arms auto (adds train/eval LoRA support + an inter-run dependency: D4 runs *after* baseline exists).
- **(B)** ship 18-arm automated sweep + judge + ouro now; LoRA (D4) + C1-tokenizer as a focused follow-up. **Claude's lean = B** (core null + design-space ablations don't depend on LoRA/10k-vocab; D4 is W1-scoped exploratory). **User has not yet decided.**

---

*End of decisions log. Updated only on the user's explicit instruction.*
