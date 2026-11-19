"""Shared feature building + walk-forward evaluation (used by steps 05-07)."""
import numpy as np, pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
import config as C

COMP_COLS = ["comp_stance", "comp_sem", "comp_growth", "comp_reach", "comp_overlap", "comp_network"]
OUTCOMES = ["drift_db", "abs_sent_change", "abs_stance_change", "entity_turnover",
            "abs_frag_change", "abs_share_change", "abs_merge_change"]        # the pre-specified family (5 dimensions, 7 measures)
DIMENSIONS = {"semantic": ["drift_db"], "emotional": ["abs_sent_change"], "stance": ["abs_stance_change"],
              "entity": ["entity_turnover"], "structural": ["abs_frag_change", "abs_share_change", "abs_merge_change"]}
DIFFUSION = ["d_volume", "d_breadth", "d_geo"]

MODELS = {
    "mean": lambda: make_pipeline(SimpleImputer(strategy="median"), DummyRegressor(strategy="mean")),
    "ridge": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0)),
    "hgb": lambda: HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=3,
                                                 l2_regularization=1.0, random_state=C.SEED),
}


def add_features(panel, target):
    df = panel.sort_values(["narrative", "week"]).reset_index(drop=True).copy()
    g = df.groupby("narrative")[target]
    df["h_lag1"], df["h_lag2"] = g.shift(1), g.shift(2)
    df["h_roll4"] = g.transform(lambda s: s.rolling(4, min_periods=2).mean())
    df["log_n"] = np.log1p(df["n"]); df["log_reach"] = np.log1p(df["reach"])
    df["log_channels"] = np.log1p(df["n_channels"]); df["log_total_volume"] = np.log1p(df["total_volume"])
    for c in ("stance", "sent"):                      # mean-reversion controls: current level minus its own past 8-week mean
        df[f"dev_{c}"] = df[c] - df.groupby("narrative")[c].transform(lambda s: s.shift(1).rolling(8, min_periods=3).mean())
    df["y"] = g.shift(-C.DELTA_BINS)               # outcome DELTA weeks ahead (grid is complete per narrative)
    return pd.concat([df, pd.get_dummies(df["narrative"], prefix="nar", dtype=float)], axis=1)


def baseline_feats(df, target):
    hist = [target, "h_lag1", "h_lag2", "h_roll4"]
    content = ["sent", "stance", "pro_ua_share", "pro_ru_share", "ent_div", "log_n", "dev_stance", "dev_sent"]
    network = ["log_channels", "country_entropy", "verified_share", "log_reach"]
    controls = ["log_total_volume", "event_week"]          # volume + event-week controls live in the baseline
    return hist + content + network + controls + [c for c in df.columns if c.startswith("nar_")]


def walk_forward(df, feats, make_model, test_start=C.TEST_START, delta=C.DELTA_BINS, refit_every=1, min_train=40):
    """Rolling-origin. At origin week t: train only on rows whose TARGET week (w+delta) <= t, predict row t -> t+delta."""
    test_weeks = [w for w in sorted(df.week.unique()) if w >= pd.Timestamp(test_start)]
    preds, model = pd.Series(np.nan, index=df.index), None
    for i, t in enumerate(test_weeks):
        test = df[(df.week == t) & df["y"].notna()]
        if test.empty: continue
        if model is None or i % refit_every == 0:
            train = df[(df.week <= t - pd.Timedelta(weeks=delta)) & df["y"].notna()]
            if len(train) < min_train: continue
            model = make_model().fit(train[feats], train["y"])
        if model is not None: preds.loc[test.index] = model.predict(test[feats])
    return preds


def persistence(df, target, test_start=C.TEST_START):
    return df[target].where(df.week >= pd.Timestamp(test_start))     # "next week = this week"


def score(df, preds):
    idx = None
    for p in preds.values():
        i = p.dropna().index; idx = i if idx is None else idx.intersection(i)
    y = df.loc[idx, "y"]; rows = []
    for name, p in preds.items():
        e = y - p.loc[idx]; rows.append(dict(model=name, n=len(idx), mae=e.abs().mean(), mse=(e ** 2).mean()))
    out = pd.DataFrame(rows).set_index("model")
    if "persistence" in out.index: out["skill_vs_persist"] = 1 - out.mse / out.loc["persistence", "mse"]
    return out, idx


# ---------- helpers shared by steps 7 and 8 ----------
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
CLF = {
    "logit": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.3, max_iter=2000)),
    "hgb": lambda: HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=3, l2_regularization=1.0, random_state=C.SEED),
}


def block_perm(L, block, rng):
    blocks = [np.arange(i, min(i + block, L)) for i in range(0, L, block)]
    rng.shuffle(blocks)
    return np.concatenate(blocks)


def shuffle_cols(df, cols, block, rng):
    """Placebo: permute the given columns in time blocks within each narrative (same permutation for all cols)."""
    out, vals = df.copy(), df[cols].to_numpy().copy()
    nar = df["narrative"].to_numpy()
    for nn in np.unique(nar):
        pos = np.where(nar == nn)[0]
        vals[pos] = vals[pos][block_perm(len(pos), block, rng)]
    out[cols] = vals
    return out
