"""
================================================================================
KAGGLE NOTEBOOK 2 — TRAINING + EVAL  (Accelerator: GPU T4,  Internet: ON)
================================================================================
PURPOSE
    Run ONE grid cell end to end: train a QLoRA adapter on a task, then measure
        • forgetting  — did the model lose general knowledge?  (evaluate_forgetting.py)
        • task skill  — did it learn the task, and how much did fine-tuning ADD?
                        (evaluate_task.py, vs the base model's own task score)
    Base model = unsloth/Llama-3.2-3B. (Gemma-2 was dropped — fp16 corruption on
    T4; see results/failed-gemma/.)

HOW THIS NOTEBOOK WORKS
    Cells 1-4 = setup (always run). Cell 5 = CONFIG: set QUANT / REPLAY / TASK /
    SEED / REPLAY_SOURCE once and every command below fills itself in (so the
    run name and the adapter path can never drift apart). One Kaggle commit =
    one grid cell.

WHICH CELLS TO RUN FOR A COMMIT
    • First run at a NEW quant level (e.g. first 4-bit run):
        1,2,3,4, 5(config), 6(general baseline), 7(task baseline), 8,9, 10
    • Later run at a quant you've ALREADY baselined AND pushed baselines.json for:
        1,2,3,4, 5(config), 8,9, 10        (skip 6-7; the cloned repo already has them)
    • Seed-repeat run (SEED != 42) of an existing cell: same as above — baselines
      and task baselines are per QUANT (+TASK), not per seed, so they're already
      there too. Only cells 8, 9, 10 actually change.

BASELINES (read this once)
    - General-knowledge baseline (cell 6) is per-QUANT and is SAVED to
      results/baselines.json. Cell 9 READS it to compute the Forgetting Score, so
      it MUST exist. Re-cloning wipes /kaggle/working, so after the first run at a
      quant, push baselines.json to the repo — then later same-quant commits clone
      it and can skip cell 6.
    - Task baseline (cell 7) is per-QUANT+TASK, base model with NO adapter. It's
      informational (a CSV row showing the base score) — it doesn't gate anything,
      so run it once per quant+task and keep the row. Check results/task_results.csv
      for an existing base_<quant>bit_<task> row before running it.

EVAL DEPTH (the "slow down, do it right" settings)
    MMLU: --limit 100  (per subject × 57 ≈ 5,700 Qs, reliable).
    hellaswag/winogrande/arc: --limit_general 1000  (kills small-sample noise).
    Keep these the SAME on baselines AND fine-tuned runs or the numbers don't compare.

LONG RUNS + PERSISTENCE (per-run folders — no manual CSV merging)
    Use Save Version -> "Save & Run All (Commit)" (GPU + Internet on, ~12h wall).
    If a run is going to be close to that wall (8-bit at high replay), split it:
    one commit for cells 1-5,8 (train only), a second commit that attaches that
    version's Output and runs 1-5,9,10. Kaggle's "Add Input -> Notebook Output"
    only attaches the LATEST successful version, so plan the split so the
    training commit is the most recent one.
    Each run writes results/runs/<run>/ (forgetting.json + task.json). After it
    finishes, download THAT folder + results/baselines.json from the version Output,
    drop them in the repo, run `python aggregate_results.py` locally to rebuild the
    master CSVs, then push. (The CSVs are always rebuilt from runs/ — never hand-edited.)
================================================================================
"""

# ───────────────────────────── CELL 1 — get the code ─────────────────────────
!rm -rf qlora-forgetting-replay
!git clone https://github.com/AnsariUsaid/qlora-forgetting-replay.git
%cd qlora-forgetting-replay


# ────────────────────────── CELL 2 — install the stack (~4 min) ───────────────
# Deps inlined (no requirements.txt). Versions loose; pin from a working pip freeze.
!pip install -q unsloth transformers peft trl bitsandbytes accelerate datasets
!pip install -q lm-eval evaluate rouge-score sacrebleu scikit-learn pandas wandb


# ───────────────────── CELL 3 — log in to Weights & Biases ────────────────────
# Needs the Kaggle Secret WANDB_API_KEY enabled for THIS notebook.
from kaggle_secrets import UserSecretsClient
import wandb
wandb.login(key=UserSecretsClient().get_secret("WANDB_API_KEY"))


# ──────────────── CELL 4 — (re)build data/ cloud-side (~1 min) ────────────────
!python data/download_datasets.py


# ════════════════════════════════════════════════════════════════════════════
# CELL 5 — CONFIG  ◀── the ONLY cell you edit between runs.
# ════════════════════════════════════════════════════════════════════════════
QUANT  = 16              # 4 | 8 | 16
REPLAY = 0.20            # 0.0 | 0.10 | 0.20 | 0.30
TASK   = "xsum"          # samsum | xsum | sql | medqa
SEED   = 42              # 42 = the standard runs; anything else = a seed repeat
REPLAY_SOURCE = "alpaca" # alpaca | dolly  (Phase-8 replay-source ablation only)

RUN = f"llama3b_{QUANT}bit_replay{int(REPLAY*100)}pct_{TASK}"
if REPLAY_SOURCE != "alpaca":
    RUN += f"_{REPLAY_SOURCE}"
if SEED != 42:
    RUN += f"_seed{SEED}"

print("This run  :", RUN)
print("Adapter   :", f"outputs/{RUN}/adapter")
print("Eval depth: MMLU --limit 100 | general --limit_general 1000")


# ════════════════════════ BASELINES (base model, NO adapter) ════════════════════════

# ── CELL 6 — GENERAL-KNOWLEDGE baseline — run ONCE per QUANT, then push baselines.json ──
# Saves MMLU/hellaswag/winogrande/arc for this quant as the reference for FS.
!python evaluate_forgetting.py --quant {QUANT} --run baseline_{QUANT}bit --batch_size 1 --limit 100 --limit_general 1000

# ── CELL 7 — TASK baseline (base model on the task) — run ONCE per QUANT+TASK ──
# How well the UN-fine-tuned model already does the task -> fine-tuning's lift =
# (cell 10 score - this).
!python evaluate_task.py --task {TASK} --quant {QUANT} --run base_{QUANT}bit_{TASK} --limit 100


# ════════════════════════════════ THE RUN ════════════════════════════════════

# ───────────────── CELL 8 — TRAIN the adapter (~3-8h depending on quant/replay) ──
# 4-bit has GPU headroom -> batch 8 for speed. Add --batch_size 1 if 16-bit OOMs.
!python train.py --quant {QUANT} --replay {REPLAY} --task {TASK} --seed {SEED} --replay_source {REPLAY_SOURCE} --batch_size 8

# ──────────── CELL 9 — FORGETTING of the fine-tuned adapter (reads baseline -> FS) ──
!python evaluate_forgetting.py --quant {QUANT} --run {RUN} --adapter outputs/{RUN}/adapter --batch_size 1 --limit 100 --limit_general 1000

# ─────────────── CELL 10 — TASK performance of the fine-tuned adapter ─────────
!python evaluate_task.py --task {TASK} --quant {QUANT} --run {RUN} --adapter outputs/{RUN}/adapter --limit 100


# ──────────── CELL 11 — rebuild master tables + show (then persist) ───────────
!python aggregate_results.py
!echo "" ; echo "=== all_results.csv (forgetting) ===" ; cat results/all_results.csv
!echo "" ; echo "=== task_results.csv (task skill + base baseline) ===" ; cat results/task_results.csv
!echo "" ; echo "=== baselines.json (per-quant general baseline) ===" ; cat results/baselines.json
# Persist: download results/runs/{this run}/ + results/baselines.json from the version
# Output -> drop in repo -> `python aggregate_results.py` -> commit -> push.
