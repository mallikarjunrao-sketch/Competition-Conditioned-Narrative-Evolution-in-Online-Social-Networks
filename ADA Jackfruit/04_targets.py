"""Step 4 (Person A): narrative x week panel with per-cell features and the transformation measures.

Outcomes (all measured AT week t relative to t-1; the model predicts them at t+DELTA):
  drift              1 - cosine(centroid_t, centroid_{t-1})       semantic drift
  abs_sent_change    |mean sentiment_t - mean sentiment_{t-1}|
  abs_stance_change  |mean stance_t - mean stance_{t-1}|
  entity_turnover    1 - Jaccard(top-20 entities_t, top-20 entities_{t-1})
Cells with fewer than MIN_TWEETS_CELL tweets get NaN features (centroids of tiny cells are noise).
"""
import argparse, numpy as np, pandas as pd
from collections import Counter
from scipy.stats import entropy, spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import config as C

ap = argparse.ArgumentParser()
ap.add_argument("--tweets", default=str(C.DATA / "tweets_en.parquet"))
ap.add_argument("--emb", default=str(C.DATA / "emb.npy"))
a = ap.parse_args()

df = pd.read_parquet(a.tweets); E = np.load(a.emb); assert len(df) == len(E)
df["week"] = pd.to_datetime(df["week"])
df["sent_score"] = df.sentiment.map({"positive": 1, "negative": -1, "neutral": 0}).fillna(0) * df.sentiment_prob.fillna(0)
df["stance_score"] = df.stance_pro_ua_max - df.stance_pro_ru_max          # continuous, in [-1, 1]
df["is_ua"] = (df.stance == "pro_ukraine").astype(float); df["is_ru"] = (df.stance == "pro_russia").astype(float)

weeks = pd.date_range(df.week.min(), df.week.max(), freq="7D"); W = len(weeks)
nars = sorted(df.narrative.unique()); K = len(nars)
grp = df.groupby(["narrative", "week"]).indices
cent = np.full((K, W, E.shape[1]), np.nan, np.float32)
raw = np.full_like(cent, np.nan)            # un-normalised weekly mean vectors
nz = np.full((K, W), np.nan)                # sampling variance of each weekly mean (noise floor)
rows, tops = [], {}
for ni, nar in enumerate(nars):
    for wi, w in enumerate(weeks):
        idx = grp.get((nar, w)); rec = dict(narrative=nar, week=w, n=0 if idx is None else len(idx), reach=0.0)
        if idx is not None:
            sub = df.iloc[idx]; rec["reach"] = float(sub.followers.fillna(0).sum())
            if len(idx) >= C.MIN_TWEETS_CELL:
                c = E[idx].mean(0); m = len(idx); raw[ni, wi] = c
                nz[ni, wi] = (1 - float(c @ c)) * m / (m - 1) / m      # unbiased var of the mean for unit-norm embeddings
                cent[ni, wi] = c / (np.linalg.norm(c) + 1e-12)
                ents = Counter(str(e[0]).lower() for ee in sub.entities for e in ee)
                tops[(nar, w)] = {k for k, _ in ents.most_common(20)}
                rec.update(sent=sub.sent_score.mean(), stance=sub.stance_score.mean(), pro_ua_share=sub.is_ua.mean(),
                           pro_ru_share=sub.is_ru.mean(), ent_div=len(ents) / max(sum(ents.values()), 1),
                           n_channels=sub.channel.nunique(), country_entropy=float(entropy(sub.country.value_counts())),
                           verified_share=sub.verified.astype(float).mean(),
                           outlet_entropy=float(entropy(sub.channel.value_counts())), disp=(1 - float(c @ c)) * m / (m - 1),
                           **({"engage": float(np.log1p(sub.engagement.fillna(0)).mean())} if "engagement" in sub else {}))
        rows.append(rec)
P = pd.DataFrame(rows).sort_values(["narrative", "week"]).reset_index(drop=True)
for c in ["sent", "stance", "pro_ua_share", "pro_ru_share", "ent_div", "n_channels", "country_entropy", "verified_share", "outlet_entropy", "disp"]:
    if c not in P: P[c] = np.nan

dr = np.full((K, W), np.nan); dr[:, 1:] = 1 - np.einsum("nwd,nwd->nw", cent[:, 1:], cent[:, :-1])
P["drift"] = dr.ravel()
# bias-corrected drift: squared shift of the weekly mean minus the sampling-noise expected from n_t and n_{t-1}
dd = np.full((K, W), np.nan); dd[:, 1:] = ((raw[:, 1:] - raw[:, :-1]) ** 2).sum(-1) - nz[:, 1:] - nz[:, :-1]
P["drift_db"] = dd.ravel()
g = P.groupby("narrative")
P["abs_sent_change"] = (P["sent"] - g["sent"].shift(1)).abs()
P["abs_stance_change"] = (P["stance"] - g["stance"].shift(1)).abs()
tv = []
for nar, d in P.groupby("narrative"):
    prev = None
    for w in d.week:
        cur = tops.get((nar, w)); tv.append(np.nan if (cur is None or prev is None) else 1 - len(cur & prev) / max(len(cur | prev), 1)); prev = cur
P["entity_turnover"] = tv     # P is sorted narrative,week so order matches

P["total_volume"] = P.groupby("week")["n"].transform("sum")
wk = P.groupby("week")["total_volume"].first()
trail = wk.shift(1).rolling(8, min_periods=4).median()                 # causal: uses only past weeks
P["event_week"] = P.week.map((wk > 1.5 * trail).astype(float))

# ---- structural transformation (fragmentation / displacement / merging) ----
P["share"] = P["n"] / P["total_volume"]
mx = np.full((K, W), np.nan)                 # similarity of each narrative to its nearest rival (convergence / merging)
for wi in range(W):
    for ni in range(K):
        if np.isnan(cent[ni, wi, 0]): continue
        s_ = [float(cent[ni, wi] @ cent[m_, wi]) for m_ in range(K) if m_ != ni and not np.isnan(cent[m_, wi, 0])]
        if s_: mx[ni, wi] = max(s_)
P["max_sim"] = mx.ravel()
g = P.groupby("narrative")
P["abs_frag_change"] = (P["disp"] - g["disp"].shift(1)).abs()           # internal fragmentation / consolidation
P["abs_share_change"] = (P["share"] - g["share"].shift(1)).abs()        # displacement: gain/loss of attention share
P["abs_merge_change"] = (P["max_sim"] - g["max_sim"].shift(1)).abs()    # convergence toward / divergence from nearest rival
# ---- diffusion outcomes (what the public release supports: volume, outlet breadth, geographic breadth, engagement if present) ----
P["d_volume"] = np.log1p(P["n"]) - np.log1p(g["n"].shift(1))
P["d_breadth"] = P["outlet_entropy"] - g["outlet_entropy"].shift(1)
P["d_geo"] = P["country_entropy"] - g["country_entropy"].shift(1)
if "engage" in P: P["d_engage"] = P["engage"] - g["engage"].shift(1)

P.to_parquet(C.DATA / "panel.parquet", index=False)
np.savez(C.DATA / "centroids.npz", cent=cent, weeks=weeks.values.astype("datetime64[ns]"), narratives=np.array(nars))
print(f"panel: {len(P)} cells ({K} narratives x {W} weeks); valid-drift cells: {P.drift.notna().sum()}")
print(P.groupby("narrative")[["n", "drift_db", "abs_sent_change", "abs_stance_change", "entity_turnover", "abs_frag_change", "abs_share_change", "abs_merge_change"]].median().round(3))

v = P.dropna(subset=["drift"])
vd = P.dropna(subset=["drift_db"])
print(f"Sanity: Spearman(drift_db, log n) = {spearmanr(vd.drift_db, np.log1p(vd.n))[0]:.2f}  (should be much closer to 0 than the raw drift above)")
print(f"\nSanity: Spearman(drift, log n) = {spearmanr(v.drift, np.log1p(v.n))[0]:.2f}  "
      f"(strongly negative => small cells look 'driftier' because of noise; keep log_n as a control)")
cut = v.drift.quantile(.75)
print(f"CHECKPOINT (end of week 2): drift median={v.drift.median():.4f}, p75={cut:.4f}. If almost nothing moves, "
      "switch to a binary 'large change' target (top quartile) or a coarser bin.")
OUT = ["drift_db", "abs_sent_change", "abs_stance_change", "entity_turnover", "abs_frag_change", "abs_share_change", "abs_merge_change", "d_volume"]
fig, ax = plt.subplots(2, 4, figsize=(16, 6))
for x, c in zip(ax.ravel(), OUT):
    P[c].dropna().hist(ax=x, bins=30); x.set_title(c)
plt.tight_layout(); plt.savefig(C.FIGS / "targets_dist.png", dpi=120)
