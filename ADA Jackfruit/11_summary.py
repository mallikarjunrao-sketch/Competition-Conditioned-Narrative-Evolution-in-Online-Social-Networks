"""Step 11: one table for the report, with Holm correction for the number of outcomes tested (family = all outcomes of the same tag/set
in the primary test window). Reads results/pvalues.csv (written by step 7) and results/classification.csv (step 8)."""
import argparse, numpy as np, pandas as pd
import config as C
ap = argparse.ArgumentParser(); ap.add_argument("--primary-start", default=C.TEST_START); a = ap.parse_args()

def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(o): run = max(run, (m - rank) * p[i]); adj[i] = min(1.0, run)
    return adj

pv = pd.read_csv(C.RESULTS / "pvalues.csv").drop_duplicates(["tag", "target", "test_start", "set"], keep="last")
prim = pv[pv.test_start == a.primary_start].copy()
prim["p_holm"] = np.nan
for _, idx in prim.groupby(["tag", "set"]).groups.items(): prim.loc[idx, "p_holm"] = holm(prim.loc[idx, "placebo_p"].values)
cols = ["tag", "target", "set", "n_test", "gain_ridge", "gain_hgb", "ci_lo_rel", "ci_hi_rel", "placebo_p", "p_holm"]
print(f"PRIMARY WINDOW (test from {a.primary_start}); gains are relative MSE reduction vs baseline; p_holm corrects across outcomes\n")
print(prim[cols].sort_values(["tag", "set", "target"]).round(4).to_string(index=False))
prim[cols].to_csv(C.RESULTS / "summary_table.csv", index=False)
rob = pv[pv.test_start != a.primary_start]
if len(rob): print("\nROBUSTNESS WINDOWS (uncorrected, descriptive)\n", rob[["tag", "target", "test_start", "set", "gain_ridge", "gain_hgb", "placebo_p"]].round(4).to_string(index=False))
try:
    cl = pd.read_csv(C.RESULTS / "classification.csv"); cl["p_holm"] = holm(cl.placebo_p.values)
    print("\nCLASSIFICATION (AUC / macro-F1)\n", cl.round(4).to_string(index=False)); cl.to_csv(C.RESULTS / "classification_holm.csv", index=False)
except FileNotFoundError: pass
