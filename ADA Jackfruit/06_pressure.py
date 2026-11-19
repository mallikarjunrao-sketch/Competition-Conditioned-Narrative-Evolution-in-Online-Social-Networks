"""Step 6 (Person B): competition pressure on narrative A at week t, built ONLY from week<=t information.

SIX components (each a volume-weighted average over the OTHER narratives B):
  comp_stance  : 0.5*|stance_A - stance_B|                 stance opposition
  comp_sem     : max(0, cos(centroid_A, centroid_B))       semantic relation
  comp_growth  : growth_B - growth_A                       relative growth (growth = log1p(n_t) - log1p(n_{t-1}))
  comp_reach   : log((sum_B reach_B + 1)/(reach_A + 1))    follower reach of rivals relative to A
  comp_overlap : cos(country mix_A, country mix_B)         audience overlap  [PROXY: countries of the outlets' audiences]
  comp_network : cos(outlet mix_A, outlet mix_B)           network proximity [PROXY: narratives pushed by the same outlets
                                                           are close in the outlet-narrative bipartite graph]
PLUS the proposal's multiplicative formula  CPA_A(t) = sum_{B!=A} w_AB * G_B(t) * O_AB(t) * R_B(t)  with
  w_AB = mean(semantic, audience, network relation)   G_B = max(0, growth_B)   O_AB = stance opposition   R_B = B's share of total reach
  (square-rooted to tame skew). Everything is z-scored with EXPANDING statistics (no future info).
pressure_eq = equal-weight mean of the six z-scored components.
NOTE: true user-level audience overlap / follower-graph proximity needs the rehydrated tweets of Dataset 1 (not obtainable); the two proxies are
outlet-level and must be described as such."""
import warnings
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import config as C
from common import add_features, COMP_COLS

warnings.filterwarnings("ignore")
P = pd.read_parquet(C.DATA / "panel.parquet"); Z = np.load(C.DATA / "centroids.npz")
cent, weeks, nars = Z["cent"], pd.to_datetime(Z["weeks"]), list(Z["narratives"]); K, W = cent.shape[:2]
def mat(col): return P.pivot(index="narrative", columns="week", values=col).reindex(index=nars, columns=weeks).to_numpy(float)
n, reach, stance = mat("n"), mat("reach"), mat("stance")
g = np.full((K, W), np.nan); g[:, 1:] = np.log1p(n[:, 1:]) - np.log1p(n[:, :-1])

T = pd.read_parquet(C.DATA / "tweets_en.parquet", columns=["narrative", "week", "channel", "country"]); T["week"] = pd.to_datetime(T["week"])
def pair_cos(col):
    S = np.full((W, K, K), np.nan)
    if T[col].notna().sum() == 0: return S
    cnt = T.groupby(["narrative", "week", col]).size().unstack(col, fill_value=0)
    M, idx = cnt.to_numpy(float), {k: i for i, k in enumerate(cnt.index)}
    for wi, w in enumerate(weeks):
        for ni, a in enumerate(nars):
            for mi, b in enumerate(nars):
                if ni == mi or (a, w) not in idx or (b, w) not in idx: continue
                x, y = M[idx[(a, w)]], M[idx[(b, w)]]; d = np.linalg.norm(x) * np.linalg.norm(y)
                if d > 0: S[wi, ni, mi] = x @ y / d
    return S
S_aud, S_net = pair_cos("country"), pair_cos("channel")

def wavg(v, w):
    ok = ~np.isnan(v) & (w > 0)
    return np.nan if not ok.any() else float(np.average(v[ok], weights=w[ok]))

out = []
for wi, w in enumerate(weeks):
    for ni, nar in enumerate(nars):
        if np.isnan(cent[ni, wi, 0]): continue
        comp = [m for m in range(K) if m != ni]; wt = n[comp, wi]
        sims = np.array([max(0.0, float(cent[ni, wi] @ cent[m, wi])) if not np.isnan(cent[m, wi, 0]) else np.nan for m in comp])
        opp = 0.5 * np.abs(stance[ni, wi] - stance[comp, wi]); aud, net = S_aud[wi, ni, comp], S_net[wi, ni, comp]
        rel = np.nanmean(np.vstack([sims, aud, net]), axis=0)
        terms = rel * np.maximum(0, g[comp, wi]) * opp * (reach[comp, wi] / max(reach[:, wi].sum(), 1.0))
        out.append(dict(narrative=nar, week=w,
                        comp_stance=wavg(opp, wt), comp_sem=wavg(sims, wt), comp_growth=wavg(g[comp, wi] - g[ni, wi], wt),
                        comp_reach=float(np.log((reach[comp, wi].sum() + 1) / (reach[ni, wi] + 1))),
                        comp_overlap=wavg(aud, wt), comp_network=wavg(net, wt),
                        cpa=float(np.sqrt(np.nansum(terms))) if np.isfinite(terms).any() else np.nan))
C_ = pd.DataFrame(out)
for c in COMP_COLS + ["cpa"]:                                           # expanding z-score
    z = pd.Series(np.nan, index=C_.index)
    for w in sorted(C_.week.unique()):
        past = C_.loc[C_.week <= w, c].dropna(); cur = C_.week == w
        z[cur] = 0.0 if len(past) < 5 or past.std() == 0 else (C_.loc[cur, c] - past.mean()) / past.std()
    C_[c] = z
C_["pressure_eq"] = C_[COMP_COLS].mean(axis=1, skipna=True)
P = P.merge(C_, on=["narrative", "week"], how="left"); P.to_parquet(C.DATA / "panel_pressure.parquet", index=False)

df = add_features(P, "drift_db").dropna(subset=["y"]); rows = []
for c in COMP_COLS + ["cpa", "pressure_eq"]:
    d = df.dropna(subset=[c])
    if len(d) < 10: rows.append(dict(component=c, spearman_raw=np.nan, spearman_within_narr=np.nan, n=len(d))); continue
    dm = d.assign(cw=d[c] - d.groupby("narrative")[c].transform("mean"), yw=d.y - d.groupby("narrative").y.transform("mean"))
    rows.append(dict(component=c, spearman_raw=spearmanr(d[c], d.y)[0], spearman_within_narr=spearmanr(dm.cw, dm.yw)[0], n=len(d)))
chk = pd.DataFrame(rows).set_index("component").round(3); print(chk); chk.to_csv(C.RESULTS / "component_checks.csv")
print("\nwrote data/panel_pressure.parquet")
