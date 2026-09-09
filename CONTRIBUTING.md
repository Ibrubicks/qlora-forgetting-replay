# Contributing — QLoRA Forgetting / Replay Study

**Read this first if you are joining the project, picking up a task, or coming back after a
break.** It covers what the project is, everything that has happened so far, where things stand
today, how to run an experiment yourself, and what needs doing next.
Last updated: **September 2026**.

Repository: `github.com/AnsariUsaid/qlora-forgetting-replay`
Course: BCSE497J Project-I · SCOPE · Guide: Dr. V Ramesh Babu Vemuluri
Team: Sanjay J K (23BCE2028) · Mohammed Ibrahim (23BCE2066) · Ansari Usaid Anzer (23BCE0574)

> This document is the single source of truth for project context. `ROADMAP.md` and
> `OBSERVATIONS.md` are out of date and pending a rewrite — trust this file and the contents of
> `results/` over either of them.

---

## 1. What the project is, in one page

When you fine-tune a large language model on a narrow task, it gets better at that task and
quietly worse at everything else. That loss of general ability is **catastrophic forgetting**.

**QLoRA** makes fine-tuning affordable by compressing (quantizing) the frozen base model to 8-bit
or 4-bit and training only small adapter matrices on top. The widely held assumption is that more
aggressive quantization makes forgetting worse.

The standard fix is **experience replay** — mixing a percentage of general-purpose instruction
data (we use Alpaca) back into the task training set, so the model keeps being reminded of general
knowledge while it learns the new task.

**The question nobody had answered:** does the *amount* of replay you need depend on the
quantization level? If you drop from FP16 to 4-bit, do you need to add more general data to
compensate?

**Our original hypothesis (H1, from Review 1):**
> Forgetting scales with quantization aggressiveness, **and** a proportionally larger replay ratio
> is required at lower bit-widths to restore equivalent general capability.

**What we found:** the first half is true, the second half is not. Forgetting definitely gets
worse as precision drops. But the *optimal replay ratio is the same — about 20% — at every
bit-width*. What quantization controls is how much damage there is, not how much replay to use.

That is a negative result on our headline hypothesis, and we report it as one. It is still a
contribution: nobody had measured this interaction, and the resulting recipe is simpler and more
usable than the one we proposed.

---

## 2. Current status at a glance

| | |
|---|---|
| Fine-tuning runs completed | **37** |
| Complete 3×4 grids | **2** (Samsum, XSum) |
| Partial grid | SQL, 10 of 12 cells |
| Rejected probe | MedQA (1 run, documented below) |
| Seed repeats | 2 |
| Model | `unsloth/Llama-3.2-3B` |
| Compute | Kaggle Tesla T4 16 GB, free tier, 3 accounts |
| Cost | ₹0 |
| Pipeline state | Complete and stable — no code work outstanding |
| Review-2 | Presented September 2026 |

**Run accounting** (this comes up — know it):
24 (two complete grids) + 10 (SQL) + 1 (MedQA) + 2 (seed repeats) = **37**

---

## 3. How to run an experiment

Everything runs on Kaggle. One notebook, config-driven — you edit exactly one cell between runs.

### The procedure

1. **Push the repo first.** Cell 1 does `rm -rf` and re-clones, so the notebook always runs
   whatever is on `main`. If your change isn't pushed, it isn't in the run.
2. Open the Kaggle notebook, create a new Version.
3. **Edit only Cell 5:**
   ```python
   QUANT  = 4        # 4, 8, or 16
   REPLAY = 0.20     # 0.0, 0.10, 0.20, 0.30
   TASK   = "xsum"   # samsum | xsum | sql | medqa
   SEED   = 42       # 42 = the standard runs; anything else = a seed repeat
   ```
   It prints the run name. **Check it before starting.** A wrong run name writes results into the
   wrong folder and compares against the wrong baseline.
4. **Choose which cells to run:**
   - Always: `1, 2, 3, 4, 5, 9, 10, 11, 12`
   - **Cell 7** (general baseline, ~50 min): only the *first* time you use a quant level. All
     three already exist — normally keep it commented out.
   - **Cell 8** (task baseline, ~15 min): only the first time for a given quant+task combination.
     Check `results/task_results.csv` for a `base_<q>bit_<task>` row before running it.
   - Cell 6 (old sanity check) has been deleted.
5. **Save Version → "Save & Run All (Commit)"**, GPU T4 + Internet ON.
6. When it finishes, record the result (Section 4).

### Gotchas that have actually bitten us

- **The `WANDB_API_KEY` secret must be enabled per-notebook.** A new notebook without it crashes
  at cell 3. (Cells 10–12 don't need W&B — only `train.py` does.)
- **Kaggle kills a commit at 12 hours.** 8-bit runs with high replay get close. If a run is going
  to be long, split it: commit 1 runs cells `1,2,3,4,5,9` (train only), commit 2 attaches that
  output and runs the evals. See Section 8 for adapter recovery.
- **Kaggle's "Add Input → Notebook Output" attaches the *latest successful* version.** You cannot
  pick an old or cancelled version. Plan the split so the training commit is the most recent one.
- **Eval must use `--batch_size 1`.** Higher batches OOM on the logits.
- **Evaluation depth must never change.** MMLU `--limit 100` (per subject), general
  `--limit_general 1000`. Change either and results stop being comparable to everything before.

### Approximate run times (14,731 rows, batch 8, XSum)

| | 0% | 10% | 20% | 30% | Peak VRAM |
|---|---|---|---|---|---|
| 4-bit | 365 min | 360 | ~430 | 464 | 4.8 GB |
| 8-bit | 507 min | 538 | 552 | **704** | 5.9 GB |
| 16-bit | 444 min | 451 | 435 | 444 | 8.7 GB |

Add ~50 min (forgetting eval) + ~15 min (task eval) + ~10 min setup.
Kaggle GPU speed varies by roughly ±15% between sessions — don't plan tightly against these.
Note 8-bit is the **slowest** despite using less memory than FP16: dequantization overhead on
every forward pass.

---

## 4. How results are recorded

**The per-run JSON files are the single source of truth.** The CSVs are generated, never edited.

```
results/runs/<run_name>/forgetting.json   ← written by evaluate_forgetting.py
results/runs/<run_name>/task.json         ← written by evaluate_task.py
```

After a run:
1. Download `results/runs/<this run>/` from the Kaggle version output
2. Drop it into the repo at the same path
3. Run `python aggregate_results.py`
4. Commit and push

`aggregate_results.py` globs every per-run folder and rebuilds `results/all_results.csv`,
`results/task_results.csv` **and** `results/baselines.json` from scratch.

**Never hand-edit the CSVs or baselines.json.** This design exists because runs happen across
different Kaggle sessions and three different accounts, and manually merging CSVs was corrupting
results early in the project.

Run naming: `llama3b_<quant>bit_replay<ratio>pct_<task>`, with `_seed<n>` appended when the seed
is not 42 — so a seed repeat can never overwrite the original.

---

## 5. Repository map

| Path | What it is |
|---|---|
| `data/download_datasets.py` | **M1** — downloads all datasets, normalises to `{instruction, input, output}`, carves held-out test splits |
| `build_replay_dataset.py` | **M2** — mixes task data with a percentage of Alpaca, shuffled under a seed |
| `train.py` | **M3** — quantized QLoRA fine-tuning + runtime precision verification |
| `evaluate_forgetting.py` | **M4** — MMLU + 3 benchmarks → Forgetting Score |
| `evaluate_task.py` | **M5** — ROUGE-L / exact-match / accuracy on held-out task data |
| `aggregate_results.py` | **M6** — rebuilds all master tables from per-run JSON |
| `results/runs/<run>/` | Per-run source-of-truth records |
| `results/all_results.csv` | Generated — forgetting results |
| `results/task_results.csv` | Generated — task scores |
| `results/baselines.json` | Generated — the three precision-matched baselines |
| `results/failed-gemma/` | Archived invalid Gemma-era results + `WHY_FAILED.md` |
| `setup/` | Kaggle notebook cell sources |

---

## 6. Project history

### Phase 1 — Setup and pipeline (11–18 June)
Starter code from the research report was committed as-is, then audited. About fifteen defects
were found and fixed before any real run: datasets that used loading scripts the new `datasets`
library refuses, a replay builder that returned unshuffled data at 0% replay, a base model that
was pre-quantized to 4-bit (making FP16 and INT8 arms impossible), missing argument parsing in
the evaluator, and a baseline that was never actually wired up.

### Phase 2 — The Gemma disaster (19–20 June)
The first full production run used Gemma-2-2B and produced a Forgetting Score of **54.07** —
general capability collapsing to near-random across every benchmark. It looked like a spectacular
result. It was a broken model.

The giveaway: training loss sat around **21** and would not come down. For a 256,000-token
vocabulary, a model guessing completely at random scores about **ln(256000) ≈ 12.45**. Our loss
was *worse than random*, which is not something a training model does.

A controlled comparison confirmed it — identical hardware, identical settings, short training run:

| Model | Loss start → end |
|---|---|
| Gemma-2-2B (FP16) | 21.85 → 21.08 — flat, broken |
| Llama-3.2-3B (FP16) | 0.77 → 0.14 — healthy |

**Cause:** Gemma-2 uses attention soft-capping, which overflows in FP16. It needs bfloat16, and
the free Kaggle T4 does not support bfloat16, so Unsloth silently fell back to FP16.

Every Gemma-era number was discarded (archived in `results/failed-gemma/`) and the study was
rebuilt on `unsloth/Llama-3.2-3B`, which has no soft-capping and trains cleanly in FP16.

**The lesson that shaped everything after:** a pipeline that fails silently is worse than one that
crashes. This is why `train.py` now asserts its own assumptions (Section 7).

### Phase 3 — Samsum grid (21–24 June)
The first complete 3×4 grid. Established the quantization ladder and the shape of the replay curve.

### Phase 4 — SQL as a third task (25–28 June)
Added text-to-SQL to test whether findings held on a non-summarization task. An important
sub-finding came out of it: at 10,000 training rows SQL showed almost no forgetting (FS 1.95), but
at 25,000 rows the same configuration gave 2.88. **Forgetting scales with training compute, not
just task type** — an "easy" task isn't immune, it just needs enough training pressure to show it.

The `--seed` flag was added at the end of this phase, and the first seed repeat revealed a
problem (Section 9).

### Phase 5 — Pause (28 June – 14 August)
No work for about seven weeks.

### Phase 6 — XSum and completion (14–22 August)
XSum (BBC news → one-sentence summary) was added as a second summarization task, deliberately
capped at **exactly 14,731 rows to match Samsum**, so both grids receive an identical number of
optimizer steps and are directly comparable. The full 3×4 grid was completed, plus two seed
repeats.

### Phase 7 — Review-2 (September 2026)
Results consolidated, presentation and report prepared.

---

## 7. Methodology — decisions and why (please don't relitigate these)

**One base model, quantized at load.** All three arms start from the same full-precision
`unsloth/Llama-3.2-3B` and are quantized when loaded. Bit-width is therefore the *only* difference
between arms.

**Precision-matched baselines.** Quantization costs accuracy before any training happens —
4-bit starts at 55.98 MMLU while FP16 starts at 57.74. Each bit-width is compared only against
*itself*, un-fine-tuned. A single shared baseline would count that 1.8-point quantization penalty
as forgetting and the whole ladder would be an artefact of the metric.

**The Forgetting Score.**
```
FS = (MMLU_baseline − MMLU_finetuned) / MMLU_baseline × 100
```
Relative, not absolute, because the three baselines differ. Positive = capability lost.

**MMLU only, not the benchmark average.** MMLU falls in 25 of 26 runs while ARC-Easy *rises* in
26 of 26 — average them and they cancel to nothing. MMLU is also the knowledge benchmark, and
knowledge is what fine-tuning actually erodes.

**Runtime precision verification.** `train.py` inspects the loaded model's layer classes
(`Linear4bit` / `Linear8bitLt`) and asserts it is genuinely at the requested bit-width. The 8-bit
path in Unsloth has a known defect where the config can be silently dropped, handing you a
full-precision model. Without this check an entire INT8 column could have secretly been FP16, and
nothing would have crashed.

**Completion-only loss.** The prompt is masked; the model is graded only on the answer. Without
this it spends its learning capacity memorising questions, which scrambles general knowledge
without teaching the task.

**Matched row counts across tasks.** XSum capped at Samsum's 14,731 so optimizer-step counts
match. Never compare runs of different training sizes — we learned this the hard way when a
25k SQL run appeared to beat a 10k run on the quant ladder and it was purely a size artefact.

**Three epochs, realistic regime.** We deliberately do *not* over-train to amplify the signal.
Cranking epochs would push into an over-fitting regime nobody uses and undermine the point of
producing a practical recipe. When effects are noise-sized, fight noise on the *measurement* side.

**Fixed evaluation depth.** MMLU 5-shot at 100 per subject (~5,700 questions); HellaSwag,
WinoGrande, ARC-Easy 0-shot at 1,000 each. Identical on every run and baseline, forever.

**MedQA was tested and rejected as a probe.** It overlaps MMLU's own medical subjects, so
fine-tuning on it pushes MMLU *up* (FS **−1.32**). It measures the same knowledge it is supposed
to be independent of. Reported, not hidden — documenting a rejected instrument is part of method.

---

## 8. Results

### Baselines (un-fine-tuned, per precision)

| | MMLU | HellaSwag | WinoGrande | ARC-Easy |
|---|---|---|---|---|
| 4-bit | 55.98 | 63.9 | 69.0 | 72.7 |
| 8-bit | 57.70 | 64.8 | 68.2 | 72.0 |
| FP16 | 57.74 | 65.0 | 68.9 | 72.3 |

Note 8-bit ≈ FP16 at the starting line — 8-bit quantization is essentially lossless — yet 8-bit
forgets roughly twice as much as FP16 during training. The extra forgetting is a **training-dynamics
effect**, not a weaker starting model.

### Forgetting Score (%) — the two complete grids

**Samsum** (dialogue summarization)

| | 0% | 10% | **20%** | 30% |
|---|---|---|---|---|
| 4-bit | 4.32 | 2.54 | **1.82** | 2.63 |
| 8-bit | 3.43 | 1.98 | **1.58** | 2.18 |
| FP16 | 1.77 | 1.04 | **0.83** | 1.44 |

**XSum** (news summarization)

| | 0% | 10% | **20%** | 30% |
|---|---|---|---|---|
| 4-bit | 4.13 | 3.34 | **2.38** | 2.47 |
| 8-bit | 3.59 | 1.70 | **1.30** | 2.37 |
| FP16 | 3.05 | 1.16 | **0.35** | 0.64 |

**SQL** (partial — *no conclusions are drawn from this grid, see Section 9*)

| | 0% | 10% | 20% | 30% |
|---|---|---|---|---|
| 4-bit | 2.88 | 2.54 | 2.32 | 2.38 |
| 8-bit | 2.70 | 1.33 | 1.98 | — |
| FP16 | 1.56 | 1.16 | 2.16 | — |

**MedQA:** 4-bit / 0% replay → FS **−1.32** (rejected probe)

### Task performance — replay costs nothing

| Task | Base model | All 12 fine-tuned configurations |
|---|---|---|
| Samsum ROUGE-L | 15.3 – 17.7 | **44.3 – 46.8** |
| XSum ROUGE-L | 11.3 – 12.0 | **31.8 – 33.9** |
| SQL exact-match | 5.6 – 8.2 | 85.8 – 88.0 |

Every configuration lands inside a ~2.5-point band while forgetting varies by a factor of twelve.

### Benchmark decomposition — across 26 runs (Samsum + XSum + MedQA)

| Benchmark | Mean change | Runs that rose |
|---|---|---|
| MMLU (5-shot, knowledge) | **−1.19** | **1 / 26** |
| ARC-Easy (0-shot, reasoning) | **+2.88** | **26 / 26** |
| HellaSwag (0-shot, completion) | +1.38 | 24 / 26 |
| WinoGrande (0-shot, commonsense) | +0.13 | 14 / 26 |

---

## 9. Findings — what is supported, and what is not

**All statistics below use Samsum and XSum only.** SQL is excluded from every claim (Section 9.3).

### 9.1 Supported

| # | Finding | Evidence |
|---|---|---|
| 1 | Lower precision forgets more (4-bit > 8-bit > FP16) | **8 / 8** complete rows ordered correctly |
| 2 | The optimal replay ratio does **not** depend on bit-width — it is ~20% everywhere | **6 / 6** curves minimise at 20% |
| 3 | Exceeding 20% replay makes forgetting worse | **6 / 6** cells, sign test **p = 0.016** |
| 4 | Replay never closes the quantization gap | **8 / 8** comparisons: 4-bit ends worse than FP16 at every ratio |
| 5 | Task performance is unaffected by bit-width or replay ratio | every run, both grids |
| 6 | Forgetting is knowledge-specific, not general degradation | MMLU down 25/26, ARC-Easy up **26/26** |

**The practical recipe:**
> Use ~20% general-data replay regardless of bit-width. Do not exceed it. Expect a 4-bit model to
> retain less general capability than FP16 even after replay.

### 9.2 NOT supported — do not put this in the paper yet

**"How much the 20% recovers depends on bit-width."** One dataset supports it, one contradicts it:

| Recovery at 20% replay | 4-bit | 8-bit | FP16 |
|---|---|---|---|
| Samsum | 58% | 54% | 53% ← flat |
| XSum | 42% | 64% | 89% ← gradient |

They disagree in *direction*, not just magnitude — 4-bit recovers the *most* on Samsum and the
*least* on XSum. Also, XSum's whole gradient hinges on one cell (16-bit / 20% = **0.35**) that has
never been repeated, and our seed noise (±0.68) is larger than the distance from 0.35 to its
neighbours. At +1σ the recovery figures become 66% vs 59% and the effect disappears entirely.

**Seed-repeating XSum 16-bit 20% is the single highest-value run remaining.** It decides whether
this claim exists.

### 9.3 Why SQL is excluded from all claims

SQL's entire result spread sits inside a ~1.5-point band while our measured seed noise is ±0.68.
Its cell-to-cell ordering therefore cannot be distinguished from noise without seed repeats on
every cell — effectively a whole new campaign.

**SQL is still reported in the paper**, as a task where effects were too small to resolve. We do
not draw conclusions from it, and we do not delete it. Deleting a dataset because its numbers were
inconvenient is exactly what would look like cherry-picking to a reviewer.

---

## 10. Known confounds and limitations

**Replay adds data rather than replacing it.** 20% replay means 1.20× the rows, so higher replay
also means more optimizer steps. The 20%→30% degradation could be "more training causes more
drift" rather than a property of the replay composition. Evidence leans against pure step-count —
if it were step count alone, every dataset should degrade proportionally, and they don't (Samsum
rises ~0.67 on average, XSum 4-bit only 0.09). **A fixed-total-steps control would settle it.**

**Run-to-run noise is ±0.68 FS**, estimated from only two seed pairs. Individual gaps smaller
than about 1.9 cannot be trusted on their own — which is why every claim above rests on
directional consistency across many cells rather than on any single comparison.

**Grids share baselines.** All 4-bit cells are compared against one 4-bit baseline measurement, so
the rows are not fully independent tests. Worth stating in the paper.

**Both complete grids are abstractive summarization.** Some of their agreement may come from being
the same kind of task. Claims should be scoped to summarization-style generation.

**One replay source.** All replay is Alpaca-cleaned. The claim is "20% Alpaca replay", not "20%
replay universally". Dolly-15k is downloaded and reserved for this ablation.

**One model, one size.** Llama-3.2-3B only. Nothing here is verified to scale.

---

## 11. What's next

In priority order.

| # | Task | Cost | Why |
|---|---|---|---|
| 1 | **Seed repeat: XSum 16-bit 20%** | ~8.4h | Decides whether the recovery-magnitude claim exists at all. Everything else is built on solid ground; this isn't. |
| 2 | **Seed repeat: XSum 4-bit 20%** | ~8.3h | The other half of that comparison; also improves the noise estimate from 2 pairs to 4 |
| 3 | **Per-subject MMLU capture** | code + ~2.5h | ⚠️ **Not free.** `evaluate_forgetting.py` currently discards the 57 per-subject scores. Needs a small code change to save them, then re-running baselines. Would test whether the drop concentrates in knowledge-heavy subjects, confirming Finding 6's mechanism. |
| 4 | **Fixed-total-steps control** | ~20h | Settles the confound in Section 10 — the main methodological criticism available to a reviewer |
| 5 | **Dolly vs Alpaca replay ablation** | ~40h | Tests whether 20% is a property of the mixture or of Alpaca. **One curve = 4 runs at a single quant level, not 12** — the quant ladder is already established and never needs re-proving |
| 6 | Finish SQL (8-bit and 16-bit at 30%) | ~10h | Grid completeness only; no new insight expected |
| 7 | Manuscript | — | Target IEEE Access, Scopus-indexed |

**Important costing note:** new questions need *one replay curve* (4 runs), not a full 3×4 grid
(12 runs). The bit-width ladder is settled across 8/8 rows and does not need re-establishing on
new data. Budgeting 120 GPU-hours for a new dataset when 40 will answer the question is the most
likely way to waste the remaining quota.

---

## 12. Glossary

| Term | Meaning |
|---|---|
| **QLoRA** | Fine-tuning where the base model is quantized and frozen, and only small low-rank adapter matrices are trained |
| **LoRA rank / alpha** | Size and scaling of those adapter matrices. We use r=16, α=32 on 7 projection matrices |
| **NF4 / INT8 / FP16** | 4-bit normal-float, 8-bit integer, 16-bit float — our three precision arms |
| **Replay ratio** | Fraction of general Alpaca data added on top of the task data. 20% = 1.20× rows |
| **Forgetting Score (FS)** | Percentage of MMLU accuracy lost, against a precision-matched baseline |
| **Precision-matched baseline** | The same model at the same bit-width, un-fine-tuned |
| **Completion-only loss** | Training loss computed on the answer tokens only, with the prompt masked |
| **MMLU** | 57-subject knowledge benchmark, 5-shot. Our primary forgetting probe |
| **ARC-Easy / HellaSwag / WinoGrande** | 0-shot reasoning and commonsense benchmarks |
| **Sign test** | Statistical test asking how unlikely it is that N independent comparisons all fall the same way by chance |
