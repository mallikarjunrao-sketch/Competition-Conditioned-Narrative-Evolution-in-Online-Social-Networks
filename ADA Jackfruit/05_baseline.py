"""Step 5 (Person B): baseline (history + content + simple network + controls) vs persistence, on held-out later weeks."""
import argparse, pandas as pd
import config as C
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--panel", default=str(C.DATA / "panel.parquet"))
ap.add_argument("--targets", nargs="+", default=["drift"])
a = ap.parse_args()

allres = []
for target in a.targets:
    df = add_features(pd.read_parquet(a.panel), target)
    feats = baseline_feats(df, target)
    preds = {"persistence": persistence(df, target)}
    preds["train_mean"] = walk_forward(df, ["h_lag1"], MODELS["mean"])
    for m in ("ridge", "hgb"): preds[m] = walk_forward(df, feats, MODELS[m])
    res, idx = score(df, preds)
    print(f"\n=== target: {target}  (predict t+{C.DELTA_BINS}; {len(idx)} held-out narrative-weeks) ===\n{res.round(5)}")
    allres.append(res.assign(target=target))
pd.concat(allres).to_csv(C.RESULTS / "baseline.csv")
print("\nCHECKPOINT: if ridge/hgb can't beat persistence at all, fix features/targets NOW (week 2), before adding competition.")
