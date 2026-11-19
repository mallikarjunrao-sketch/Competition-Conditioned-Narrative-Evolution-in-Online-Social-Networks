"""Step 7 (Both): does adding X to the baseline improve forecasts on held-out weeks?  X defaults to the COMPETITION features.
Reports: baseline vs +X (ridge and HGB), component ablation, paired bootstrap over weeks, and a placebo (block-shuffled X).
Every run appends its numbers to results/pvalues.csv (used by step 11 for multiple-testing correction).

  python 07_main_test.py --target drift_db --n-placebo 500                      # competition -> transformation
  python 07_main_test.py --target d_volume --tag transformation->diffusion --n-placebo 500 \
         --extra drift_db,abs_sent_change,abs_stance_change,entity_turnover,abs_frag_change,abs_share_change,abs_merge_change --extra2 ""
  add  --test-start 2022-11-01  for a robustness window;  --placebo-model hgb  for a tree-model placebo"""
import argparse, functools
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import config as C
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--target", default="drift_db"); ap.add_argument("--n-placebo", type=int, default=500)
ap.add_argument("--block", type=int, default=4); ap.add_argument("--n-boot", type=int, default=1000)
ap.add_argument("--test-start", default=C.TEST_START); ap.add_argument("--placebo-model", default="ridge", choices=["ridge", "hgb"])
ap.add_argument("--extra", default=",".join(COMP_COLS), help="columns whose added value is tested (all together, plus ablation)")
ap.add_argument("--extra2", default="pressure_eq,cpa", help="alternative SINGLE-feature versions, each tested on its own")
ap.add_argument("--tag", default="competition")
ap.add_argument("--extra-base", default="", help="EXPLORATORY: extra control columns added to the baseline of every model (e.g. shift_country,shift_channel)")
a = ap.parse_args()
if pd.Timestamp(a.test_start) <= pd.Timestamp(C.TRAIN_END): raise SystemExit("--test-start must be after TRAIN_END")
rng = np.random.default_rng(C.SEED); wf = functools.partial(walk_forward, test_start=a.test_start)
SUF = "" if a.test_start == C.TEST_START else f"_from{a.test_start}"
df = add_features(pd.read_parquet(C.DATA / "panel_pressure.parquet"), a.target)
extra = [c for c in a.extra.split(",") if c]; extra2 = [c for c in a.extra2.split(",") if c]
miss = [c for c in extra + extra2 if c not in df]
if miss: raise SystemExit(f"columns not in panel: {miss}")
xb = [c for c in a.extra_base.split(",") if c]
if [c for c in xb if c not in df]: raise SystemExit("extra-base columns missing: run 06b_composition_controls.py first")
base = baseline_feats(df, a.target) + xb
sets = {"base": base, "base+ALL": base + extra}
for c in extra2: sets[f"base+{c}"] = base + [c]
if len(extra) > 1:
    for c in extra: sets[f"abl: drop {c}"] = base + [x for x in extra if x != c]
MAIN = ["base+ALL"] + [f"base+{c}" for c in extra2]
keep = {s: {} for s in MAIN}

def mse_on(p, idx): return float(((df.loc[idx, "y"] - p.loc[idx]) ** 2).mean())

for mname in ("ridge", "hgb"):
    preds = {"persistence": persistence(df, a.target, a.test_start)} | {k: wf(df, f, MODELS[mname]) for k, f in sets.items()}
    res, idx = score(df, preds); b = res.loc["base", "mse"]; res["gain_vs_base"] = 1 - res.mse / b
    print(f"\n=== [{a.tag}] {a.target} | test from {a.test_start} | model={mname} | {len(idx)} held-out narrative-weeks ===\n{res.round(5)}")
    for s in MAIN:
        dd = pd.DataFrame({"w": df.loc[idx, "week"], "d": (df.loc[idx, "y"] - preds["base"].loc[idx]) ** 2 - (df.loc[idx, "y"] - preds[s].loc[idx]) ** 2}).groupby("w").d.mean().values
        bs = np.array([rng.choice(dd, len(dd)).mean() for _ in range(a.n_boot)])
        keep[s][f"gain_{mname}"] = res.loc[s, "gain_vs_base"]
        if mname == "ridge": keep[s]["ci_lo_rel"], keep[s]["ci_hi_rel"] = np.percentile(bs, 2.5) / b, np.percentile(bs, 97.5) / b
        print(f"   {s}: mean MSE reduction {dd.mean():.2e}, 95% CI [{np.percentile(bs,2.5):.2e}, {np.percentile(bs,97.5):.2e}]  (CI above 0 => helps)")
    res.assign(model_type=mname).to_csv(C.RESULTS / f"main_{a.target}{SUF}_{mname}.csv")

# ---------- placebo ----------
RE, M = 4, MODELS[a.placebo_model]
pb = wf(df, base, M, refit_every=RE); pcs = {s: wf(df, sets[s], M, refit_every=RE) for s in MAIN}
idx = pb.dropna().index
for p in pcs.values(): idx = idx.intersection(p.dropna().index)
b0 = mse_on(pb, idx); real = {s: 1 - mse_on(pcs[s], idx) / b0 for s in MAIN}
PL = list(dict.fromkeys(extra + extra2)); gains = {s: [] for s in MAIN}
for _ in range(a.n_placebo):
    dfb = shuffle_cols(df, PL, a.block, rng)
    for s in MAIN: gains[s].append(1 - mse_on(wf(dfb, sets[s], M, refit_every=RE), idx) / b0)
recs = []
print()
for s in MAIN:
    g_ = np.array(gains[s]); p = (1 + (g_ >= real[s]).sum()) / (1 + len(g_))
    print(f"PLACEBO [{s}] ({a.placebo_model}, {a.n_placebo} block-shuffles, block={a.block}wk): real gain={real[s]:+.4f} | placebo mean={g_.mean():+.4f}, "
          f"95th pct={np.percentile(g_, 95):+.4f} | one-sided p={p:.3f}")
    recs.append(dict(tag=a.tag, target=a.target, test_start=a.test_start, set=s, n_test=len(idx), placebo_model=a.placebo_model,
                     placebo_gain=real[s], placebo_mean=g_.mean(), placebo_p95=np.percentile(g_, 95), placebo_p=p, **keep[s]))
    if s == "base+ALL":
        plt.figure(figsize=(6, 3.5)); plt.hist(g_, bins=30, alpha=.8); plt.axvline(real[s], color="r", label=f"real ({real[s]:+.3f}), p={p:.3f}")
        plt.xlabel("relative MSE gain over baseline"); plt.title(f"Placebo: {a.tag} -> {a.target}"); plt.legend(); plt.tight_layout()
        plt.savefig(C.FIGS / f"placebo_{a.target}{SUF}.png", dpi=130); plt.close()
pv = C.RESULTS / "pvalues.csv"; pd.DataFrame(recs).to_csv(pv, mode="a", header=not pv.exists(), index=False)
print("appended to results/pvalues.csv.  A null result with ablation + placebo evidence is a valid finding.")
