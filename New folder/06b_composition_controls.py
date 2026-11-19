"""EXPLORATORY robustness check (run AFTER the pre-specified tests are finished and reported).
Adds to data/panel_pressure.parquet two 'composition shift' columns: how much the narrative's own country mix / outlet mix changed
from week t-1 to t (1 - cosine). If the apparent gain from competition comes only from shifts in WHO is posting, adding these to the baseline
(07_main_test.py --extra-base shift_country,shift_channel) should make the gain disappear. Idempotent: safe to rerun."""
import numpy as np, pandas as pd
import config as C

P = pd.read_parquet(C.DATA / "panel_pressure.parquet"); P["week"] = pd.to_datetime(P["week"])
T = pd.read_parquet(C.DATA / "tweets_en.parquet", columns=["narrative", "week", "channel", "country"]); T["week"] = pd.to_datetime(T["week"])

def own_shift(col):
    if T[col].notna().sum() == 0: return pd.DataFrame(columns=["narrative", "week", f"shift_{col}"])
    cnt = T.groupby(["narrative", "week", col]).size().unstack(col, fill_value=0); out = []
    for nar, d in cnt.groupby(level=0):
        d = d.droplevel(0).sort_index(); M, w = d.to_numpy(float), d.index
        for i in range(1, len(w)):
            if w[i] - w[i - 1] != pd.Timedelta(days=7): continue
            den = np.linalg.norm(M[i]) * np.linalg.norm(M[i - 1])
            if den > 0: out.append((nar, w[i], 1 - M[i] @ M[i - 1] / den))
    return pd.DataFrame(out, columns=["narrative", "week", f"shift_{col}"])

for col in ("country", "channel"):
    P = P.drop(columns=[f"shift_{col}"], errors="ignore").merge(own_shift(col), on=["narrative", "week"], how="left")
P.to_parquet(C.DATA / "panel_pressure.parquet", index=False)
print(P[["shift_country", "shift_channel"]].describe().round(3))
print("added shift_country, shift_channel to data/panel_pressure.parquet")
