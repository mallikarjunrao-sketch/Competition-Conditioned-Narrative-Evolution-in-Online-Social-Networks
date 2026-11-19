"""Step 8: the classification versions of the test (the one-pager's F1/AUC criterion).
 (a) LARGE-CHANGE: for each outcome, label next-week change 'large' if above the 75th percentile of the TRAINING period (threshold fixed before test_start);
     metrics AUC and F1 (F1 uses the top-q predictions, q = training prevalence), baseline vs baseline+competition, with a block-shuffle placebo on AUC.
 (b) TRANSFORMATION TYPE: which of the 5 dimensions (semantic/emotional/stance/entity/structural) changes most next week (argmax of training-standardised
     z-scores) -> macro-F1, baseline vs +competition, with placebo.
  python 08_classification.py --n-placebo 200"""
import argparse
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import config as C
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--outcomes", default=",".join(OUTCOMES)); ap.add_argument("--model", default="logit", choices=["logit", "hgb"])
ap.add_argument("--n-placebo", type=int, default=200); ap.add_argument("--block", type=int, default=4)
ap.add_argument("--test-start", default=C.TEST_START); ap.add_argument("--extra", default=",".join(COMP_COLS))
a = ap.parse_args()
rng = np.random.default_rng(C.SEED); extra = a.extra.split(","); MK = CLF[a.model]; RE = 4
panel = pd.read_parquet(C.DATA / "panel_pressure.parquet")
D = pd.Timedelta(weeks=C.DELTA_BINS)

def wf_clf(df, feats, y, refit_every=RE, min_train=40):
    classes = sorted(y.dropna().unique()); P = pd.DataFrame(0.0, index=df.index, columns=classes); done = pd.Series(False, index=df.index); model = None
    for i, t in enumerate([w for w in sorted(df.week.unique()) if w >= pd.Timestamp(a.test_start)]):
        te = df.index[(df.week == t) & y.notna()]
        if len(te) == 0: continue
        if model is None or i % refit_every == 0:
            tr = df.index[(df.week <= t - D) & y.notna()]
            if len(tr) < min_train or y[tr].nunique() < 2: continue
            model = MK().fit(df.loc[tr, feats], y[tr])
        if model is None: continue
        P.loc[te, list(model.classes_)] = model.predict_proba(df.loc[te, feats]); done[te] = True
    return P[done]

def bin_scores(y, p, prev):
    yhat = (p >= np.quantile(p, 1 - prev)).astype(int)
    return roc_auc_score(y, p), f1_score(y, yhat)

rows = []
for out in [o for o in a.outcomes.split(",") if o]:
    df = add_features(panel, out); base = baseline_feats(df, out)
    tr_mask = (df.week + D) < pd.Timestamp(a.test_start)
    cut = df.loc[tr_mask, "y"].quantile(.75); y = (df["y"] >= cut).astype(float).where(df["y"].notna()); prev = y[tr_mask].mean()
    Pb, Pc = wf_clf(df, base, y), wf_clf(df, base + extra, y)
    idx = Pb.index.intersection(Pc.index)
    if len(idx) < 20 or y[idx].nunique() < 2: print(f"{out}: too few test rows"); continue
    auc_b, f1_b = bin_scores(y[idx], Pb.loc[idx, 1.0], prev); auc_c, f1_c = bin_scores(y[idx], Pc.loc[idx, 1.0], prev)
    pl = []
    for _ in range(a.n_placebo):
        Ps = wf_clf(shuffle_cols(df, extra, a.block, rng), base + extra, y).reindex(idx)
        pl.append(roc_auc_score(y[idx], Ps[1.0].fillna(.5)) - auc_b)
    pl = np.array(pl); p = (1 + (pl >= auc_c - auc_b).sum()) / (1 + len(pl))
    rows.append(dict(task="large_change", outcome=out, n=len(idx), prevalence=y[idx].mean(), auc_base=auc_b, auc_comp=auc_c, auc_gain=auc_c - auc_b,
                     f1_base=f1_b, f1_comp=f1_c, placebo_p=p))
    print(f"{out:18s} n={len(idx)} AUC {auc_b:.3f} -> {auc_c:.3f} (gain {auc_c-auc_b:+.3f}) | F1 {f1_b:.3f} -> {f1_c:.3f} | placebo p={p:.3f}")

# ---- (b) dominant transformation type ----
dfm = add_features(panel, "drift_db"); tr_mask = (dfm.week + D) < pd.Timestamp(a.test_start); Zs = {}
for dim, cols in DIMENSIONS.items():
    zz = []
    for c in cols:
        nxt = dfm.groupby("narrative")[c].shift(-C.DELTA_BINS); zz.append((nxt - nxt[tr_mask].mean()) / nxt[tr_mask].std())
    Zs[dim] = pd.concat(zz, axis=1).mean(axis=1)
Zs = pd.DataFrame(Zs); okz = Zs.notna().all(axis=1); ymc = Zs[okz].idxmax(axis=1).reindex(Zs.index)
bm = baseline_feats(dfm, "drift_db") + [c for c in OUTCOMES if c != "drift_db"]
Pb, Pc = wf_clf(dfm, bm, ymc), wf_clf(dfm, bm + extra, ymc); idx = Pb.index.intersection(Pc.index)
if len(idx) >= 20:
    f = lambda Pm: f1_score(ymc[idx], Pm.loc[idx].idxmax(axis=1), average="macro")
    fb, fc = f(Pb), f(Pc); pl = []
    for _ in range(a.n_placebo):
        Ps = wf_clf(shuffle_cols(dfm, extra, a.block, rng), bm + extra, ymc).reindex(idx).fillna(0)
        pl.append(f1_score(ymc[idx], Ps.idxmax(axis=1), average="macro") - fb)
    pl = np.array(pl); p = (1 + (pl >= fc - fb).sum()) / (1 + len(pl))
    print(f"\nTRANSFORMATION TYPE (5 classes, n={len(idx)}): macro-F1 {fb:.3f} -> {fc:.3f} (gain {fc-fb:+.3f}) | accuracy {accuracy_score(ymc[idx], Pb.loc[idx].idxmax(axis=1)):.3f} -> "
          f"{accuracy_score(ymc[idx], Pc.loc[idx].idxmax(axis=1)):.3f} | placebo p={p:.3f}\nclass counts in test: {ymc[idx].value_counts().to_dict()}")
    rows.append(dict(task="type", outcome="dominant_dimension", n=len(idx), f1_base=fb, f1_comp=fc, auc_gain=np.nan, placebo_p=p))
pd.DataFrame(rows).to_csv(C.RESULTS / "classification.csv", index=False); print("\nsaved results/classification.csv")
