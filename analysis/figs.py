import csv, re, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 200})
COL = {4: "#B3412B", 8: "#E08A1E", 16: "#1F3A5F"}
LAB = {4: "4-bit (NF4)", 8: "8-bit (INT8)", 16: "16-bit (FP16)"}
BL = {4: {"mmlu": 55.98, "hellaswag": 63.9, "winogrande": 69.0, "arc_easy": 72.7},
      8: {"mmlu": 57.7, "hellaswag": 64.8, "winogrande": 68.2, "arc_easy": 72.0},
      16: {"mmlu": 57.74, "hellaswag": 65.0, "winogrande": 68.9, "arc_easy": 72.3}}
A = list(csv.DictReader(open(r"../results/all_results.csv")))
T = {r["run"]: r for r in csv.DictReader(open(r"../results/task_results.csv"))}
FS = {}; ALL = {}
for r in A:
    ALL[r["run"]] = r
    m = re.match(r"llama3b_(\d+)bit_replay(\d+)pct_(\w+?)$", r["run"])
    if r["mode"] == "finetuned" and m:
        FS[(m[3], int(m[1]), int(m[2]))] = float(r["forgetting_score"])
OUT = "figs/"
plt.rcParams["pdf.fonttype"] = 42
def save(fig, name):
    fig.savefig(OUT + name + ".png", bbox_inches="tight", dpi=600)
    fig.savefig(OUT + name + ".pdf", bbox_inches="tight")

import os; os.makedirs(OUT, exist_ok=True)
R = [0, 10, 20, 30]

# Fig 1: FS vs replay, three tasks
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
titles = {"samsum": "Samsum", "xsum": "XSum", "sql": "SQL (within noise)"}
extra = {"samsum": [(4, 0, 3.63)], "xsum": [(4, 0, 3.79), (4, 20, 2.82), (16, 20, 1.97)], "sql": [(16, 20, 3.34)]}
for ax, t in zip(axs, ("samsum", "xsum", "sql")):
    for q in (4, 8, 16):
        ax.plot(R, [FS[(t, q, r)] for r in R], marker="o", ms=4, lw=1.6, color=COL[q], label=LAB[q])
    for q, r, v in extra[t]:
        ax.scatter([r], [v], marker="D", s=26, facecolor="white", edgecolor=COL[q], linewidth=1.3, zorder=5)
    ax.set_title(titles[t], fontsize=9); ax.set_xticks(R); ax.set_xlabel("Replay ratio (%)")
axs[0].set_ylabel("Forgetting Score (%)")
h, l = axs[0].get_legend_handles_labels()
h.append(plt.Line2D([], [], marker="D", ls="", mfc="white", mec="gray")); l.append("second seed")
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.08, 1, 1)); save(fig, "fig1_curves"); plt.close(fig)

# Fig 2: additive effects forest
S = json.load(open("stats.json"))["samsum_xsum"]["coef"]
names = [("p4", "4-bit vs 16-bit"), ("p8", "8-bit vs 16-bit"), ("r10", "10% replay vs 0%"), ("r20", "20% replay vs 0%"), ("r30", "30% replay vs 0%")]
fig, ax = plt.subplots(figsize=(4.6, 2.6))
for i, (k, lab) in enumerate(names[::-1]):
    b, lo, hi = S[k]
    c = "#B3412B" if k.startswith("p") else "#1F3A5F"
    ax.errorbar(b, i, xerr=[[b - lo], [hi - b]], fmt="o", color=c, capsize=3, ms=5)
ax.set_yticks(range(5)); ax.set_yticklabels([n[1] for n in names[::-1]])
ax.axvline(0, color="black", lw=0.8); ax.set_xlabel("Change in Forgetting Score (points), 95% CI")
ax.grid(axis="y", alpha=0)
fig.tight_layout(); save(fig, "fig2_effects"); plt.close(fig)

# Fig 3: fixed steps and Dolly (XSum 4-bit)
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
x = np.arange(3); w = 0.36
normal = [4.13, 2.38, 2.47]; fixed = [4.13, 1.25, 0.71]
axs[0].bar(x - w / 2, normal, w, color="#B0B7C3", label="Normal (additive) training")
axs[0].bar(x + w / 2, fixed, w, color="#1F3A5F", label="Fixed optimizer steps")
axs[0].set_xticks(x); axs[0].set_xticklabels(["0%", "20%", "30%"]); axs[0].set_xlabel("Replay ratio"); axs[0].set_ylabel("Forgetting Score (%)")
axs[0].legend(frameon=False, fontsize=7.5); axs[0].set_title("Controlling training length", fontsize=9)
axs[1].plot([0, 10, 20, 30], [4.13, 3.34, 2.38, 2.47], marker="o", color="#B3412B", label="Alpaca replay")
axs[1].plot([0, 10, 20], [4.13, 2.97, 2.50], marker="s", color="#2E7D5B", label="Dolly replay")
axs[1].set_xticks(R); axs[1].set_xlabel("Replay ratio (%)"); axs[1].set_title("Replay source (XSum, 4-bit)", fontsize=9)
axs[1].legend(frameon=False, fontsize=7.5)
fig.tight_layout(); save(fig, "fig3_controls"); plt.close(fig)

# Fig 4: benchmark change after fine-tuning
d = {k: [] for k in ("mmlu", "hellaswag", "winogrande", "arc_easy")}
for run, r in ALL.items():
    if r["mode"] != "finetuned": continue
    q = int(r["quant_bits"])
    for k in d: d[k].append(float(r[k]) - BL[q][k])
fig, ax = plt.subplots(figsize=(4.8, 2.7))
labs = ["MMLU\n(knowledge)", "HellaSwag", "WinoGrande", "ARC-Easy\n(reasoning)"]
data = [d[k] for k in ("mmlu", "hellaswag", "winogrande", "arc_easy")]
bp = ax.boxplot(data, widths=0.5, patch_artist=True, medianprops=dict(color="black"))
for p, c in zip(bp["boxes"], ["#B3412B", "#B0B7C3", "#B0B7C3", "#1F3A5F"]): p.set_facecolor(c); p.set_alpha(0.8)
ax.axhline(0, color="black", lw=0.8); ax.set_xticklabels(labs); ax.set_ylabel("Score change vs. baseline (points)")
fig.tight_layout(); save(fig, "fig4_benchmarks"); plt.close(fig)
print({k: (round(float(np.mean(v)), 2), sum(x > 0 for x in v), len(v)) for k, v in d.items()})

# Fig 5: task score vs forgetting
fig, ax = plt.subplots(figsize=(4.6, 2.8))
mk = {"samsum": "o", "xsum": "s"}
for run, r in ALL.items():
    m = re.match(r"llama3b_(\d+)bit_replay(\d+)pct_(samsum|xsum)$", run)
    if r["mode"] != "finetuned" or not m or run not in T: continue
    q = int(m[1]); ax.scatter(float(T[run]["rougeL"]), float(r["forgetting_score"]), color=COL[q], marker=mk[m[3]], s=22, alpha=0.85)
ax.set_xlabel("Task ROUGE-L"); ax.set_ylabel("Forgetting Score (%)")
hs = [plt.Line2D([], [], marker="o", ls="", color=COL[q]) for q in (4, 8, 16)] + [plt.Line2D([], [], marker="o", ls="", color="gray"), plt.Line2D([], [], marker="s", ls="", color="gray")]
ax.legend(hs, ["4-bit", "8-bit", "16-bit", "Samsum", "XSum"], frameon=False, fontsize=7, ncol=2)
fig.tight_layout(); save(fig, "fig5_tradeoff"); plt.close(fig)
