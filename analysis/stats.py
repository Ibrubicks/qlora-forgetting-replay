import csv, re, json, itertools
import numpy as np
from scipy import stats

rows = list(csv.DictReader(open(r"../results/all_results.csv")))
BL = {4: 55.98, 8: 57.7, 16: 57.74}
cells = {}
for r in rows:
    m = re.match(r"llama3b_(\d+)bit_replay(\d+)pct_(\w+?)$", r["run"])
    if r["mode"] != "finetuned" or not m:
        continue
    q, rp, t = int(m[1]), int(m[2]), m[3]
    cells[(t, q, rp)] = float(r["forgetting_score"])
FS = cells

def design(data, interaction):
    keys = sorted(data)
    X, y = [], []
    tasks = sorted({k[0] for k in keys})
    for (t, q, rp) in keys:
        row = [1.0]
        row += [1.0 if t == tt else 0.0 for tt in tasks[1:]]
        row += [1.0 if q == 4 else 0.0, 1.0 if q == 8 else 0.0]          # ref 16-bit
        row += [1.0 if rp == v else 0.0 for v in (10, 20, 30)]            # ref 0%
        if interaction:
            for qq in (4, 8):
                for v in (10, 20, 30):
                    row.append(1.0 if (q == qq and rp == v) else 0.0)
        X.append(row); y.append(data[(t, q, rp)])
    names = ["const"] + [f"task_{t}" for t in tasks[1:]] + ["p4", "p8", "r10", "r20", "r30"]
    if interaction:
        names += [f"p{qq}xr{v}" for qq in (4, 8) for v in (10, 20, 30)]
    return np.array(X), np.array(y), names

def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    rss = float(res @ res); df = len(y) - np.linalg.matrix_rank(X)
    return beta, rss, df

def analyse(label, data):
    X0, y, n0 = design(data, False); X1, _, n1 = design(data, True)
    b0, rss0, df0 = ols(X0, y); b1, rss1, df1 = ols(X1, y)
    out = {"label": label, "n": len(y), "df_add": df0, "df_int": df1,
           "resid_sd_additive": (rss0 / df0) ** 0.5}
    if df1 > 0:
        F = ((rss0 - rss1) / (df0 - df1)) / (rss1 / df1)
        out["F_interaction"] = F; out["p_interaction"] = float(1 - stats.f.cdf(F, df0 - df1, df1))
        out["df_num"] = df0 - df1
    # CIs for additive model
    s2 = rss0 / df0
    cov = s2 * np.linalg.inv(X0.T @ X0)
    se = np.sqrt(np.diag(cov)); tcrit = stats.t.ppf(0.975, df0)
    out["coef"] = {n: (float(b), float(b - tcrit * s), float(b + tcrit * s)) for n, b, s in zip(n0, b0, se)}
    return out

main = {k: v for k, v in FS.items() if k[0] in ("samsum", "xsum")}
res = {}
res["samsum_xsum"] = analyse("Samsum+XSum seed42", main)
res["all3"] = analyse("Samsum+XSum+SQL seed42", {k: v for k, v in FS.items() if k[0] in ("samsum", "xsum", "sql")})
# sensitivity: average the two seeds for XSum 16-bit 20% (0.35, 1.97) and 4-bit 0%/20% (seed 777)
sens = dict(main)
sens[("xsum", 16, 20)] = (0.35 + 1.97) / 2
sens[("xsum", 4, 0)] = (4.13 + 3.79) / 2
sens[("xsum", 4, 20)] = (2.38 + 2.82) / 2
res["seed_averaged"] = analyse("Samsum+XSum seed-averaged", sens)

# descriptive checks
d = {}
d["4bit_gt_16bit_rows_all3"] = sum(FS[(t, 4, r)] > FS[(t, 16, r)] for t in ("samsum", "xsum", "sql") for r in (0, 10, 20, 30))
d["ladder_strict_samsum_xsum"] = sum(FS[(t, 4, r)] > FS[(t, 8, r)] > FS[(t, 16, r)] for t in ("samsum", "xsum") for r in (0, 10, 20, 30))
def best(data, t, q): return min((0, 10, 20, 30), key=lambda r: data[(t, q, r)])
d["best_ratio_seed42"] = {f"{t}_{q}": best(FS, t, q) for t in ("samsum", "xsum", "sql") for q in (4, 8, 16)}
d["best_ratio_seedavg_xsum"] = {f"xsum_{q}": best(sens, "xsum", q) for q in (4, 8, 16)}
d["replay_any_lt_0_curves"] = sum(min(FS[(t, q, r)] for r in (10, 20, 30)) < FS[(t, q, 0)] for t in ("samsum", "xsum", "sql") for q in (4, 8, 16))
d["r20_lt_r0_curves"] = sum(FS[(t, q, 20)] < FS[(t, q, 0)] for t in ("samsum", "xsum", "sql") for q in (4, 8, 16))
d["r30_gt_r20_cells"] = sum(FS[(t, q, 30)] > FS[(t, q, 20)] for t in ("samsum", "xsum") for q in (4, 8, 16))
# mean effects
def m(data, f): v = [data[k] for k in data if f(k)]; return float(np.mean(v))
d["mean_FS_by_precision_samsum_xsum"] = {q: m(main, lambda k, q=q: k[1] == q) for q in (4, 8, 16)}
d["mean_FS_by_replay_samsum_xsum"] = {r: m(main, lambda k, r=r: k[2] == r) for r in (0, 10, 20, 30)}
# seed pair noise
pairs = [(4.13, 3.79), (2.38, 2.82), (4.32, 3.63), (2.16, 3.34), (0.35, 1.97)]
dd = np.array([abs(a - b) for a, b in pairs]); d["pooled_sd"] = float(np.sqrt((dd ** 2).mean() / 2))
# paired replay effect at 4-bit XSum, two seeds
d["xsum4_replay_effect_seed42"] = 4.13 - 2.38; d["xsum4_replay_effect_seed777"] = 3.79 - 2.82
res["descriptive"] = d
json.dump(res, open("stats.json", "w"), indent=1, default=str)
for k in ("samsum_xsum", "all3", "seed_averaged"):
    r = res[k]
    print(k, "n=", r["n"], "resid_sd_add=%.3f" % r["resid_sd_additive"], "F=%.2f (df %s, %s) p=%.3f" % (r["F_interaction"], r["df_num"], r["df_int"], r["p_interaction"]))
    for n in ("p4", "p8", "r10", "r20", "r30"):
        b, lo, hi = r["coef"][n]; print("   %-4s %6.2f  [%5.2f, %5.2f]" % (n, b, lo, hi))
print(json.dumps(d, indent=1, default=str))
