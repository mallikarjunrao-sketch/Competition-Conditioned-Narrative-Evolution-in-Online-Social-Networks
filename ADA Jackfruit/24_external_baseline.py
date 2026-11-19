
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path("/content/drive/MyDrive/ADA Jackfruit")
DATA = ROOT / "data_rep"
RESULTS = ROOT / "results_rep"

RESULTS.mkdir(parents=True, exist_ok=True)

PANEL = DATA / "panel_pressure_external.parquet"

TEST_START = pd.Timestamp("2023-01-01")
DELTA = 1
MIN_TRAIN = 40
SEED = 42

TARGETS = [
    "drift_db",
    "entity_turnover",
    "abs_frag_change",
    "abs_share_change",
    "abs_merge_change",
    "d_volume",
    "d_breadth",
    "d_engage",
]

MODELS = {
    "mean": lambda: make_pipeline(
        SimpleImputer(strategy="median"),
        DummyRegressor(strategy="mean")
    ),

    "ridge": lambda: make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=10.0)
    ),

    "hgb": lambda: HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.05,
        max_depth=3,
        l2_regularization=1.0,
        random_state=SEED
    ),
}


def add_external_features(panel, target):
    df = (
        panel.sort_values(["narrative", "week"])
        .reset_index(drop=True)
        .copy()
    )

    g = df.groupby("narrative")[target]

    df["h_lag1"] = g.shift(1)
    df["h_lag2"] = g.shift(2)

    df["h_roll4"] = g.transform(
        lambda s: s.rolling(4, min_periods=2).mean()
    )

    df["log_n"] = np.log1p(df["n"])
    df["log_channels"] = np.log1p(df["n_channels"])
    df["log_total_volume"] = np.log1p(df["total_volume"])

    df["y"] = g.shift(-DELTA)

    dummies = pd.get_dummies(
        df["narrative"],
        prefix="nar",
        dtype=float
    )

    return pd.concat([df, dummies], axis=1)


def external_baseline_features(df):
    return [
        "narrative_target",
        "h_lag1",
        "h_lag2",
        "h_roll4",
        "ent_div",
        "log_n",
        "log_channels",
        "log_total_volume",
        "event_week",
    ] + [
        c for c in df.columns
        if c.startswith("nar_")
    ]


def walk_forward(df, features, make_model):
    test_weeks = [
        w for w in sorted(df.week.unique())
        if w >= TEST_START
    ]

    predictions = pd.Series(np.nan, index=df.index)
    model = None

    for i, t in enumerate(test_weeks):

        test = df[
            (df.week == t) &
            df["y"].notna()
        ]

        if test.empty:
            continue

        train = df[
            (df.week <= t - pd.Timedelta(weeks=DELTA)) &
            df["y"].notna()
        ]

        if len(train) < MIN_TRAIN:
            continue

        model = make_model()

        model.fit(
            train[features],
            train["y"]
        )

        predictions.loc[test.index] = model.predict(
            test[features]
        )

    return predictions


def persistence(df, target):
    return df[target].where(df.week >= TEST_START)


def score(df, predictions):
    idx = None

    for pred in predictions.values():
        valid = pred.dropna().index
        idx = valid if idx is None else idx.intersection(valid)

    y = df.loc[idx, "y"]

    rows = []

    for name, pred in predictions.items():
        error = y - pred.loc[idx]

        rows.append({
            "model": name,
            "n": len(idx),
            "mae": error.abs().mean(),
            "mse": (error ** 2).mean()
        })

    result = pd.DataFrame(rows).set_index("model")

    if "persistence" in result.index:
        result["skill_vs_persist"] = (
            1 - result.mse / result.loc["persistence", "mse"]
        )

    return result, idx


def main():

    print("Loading Dataset 3 pressure panel...")
    panel = pd.read_parquet(PANEL)

    print("Panel rows:", len(panel))

    required = [
        "narrative", "week", "n", "n_channels",
        "total_volume", "ent_div", "event_week"
    ]

    missing = [c for c in required if c not in panel.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    all_results = []

    for target in TARGETS:

        print("\n" + "=" * 75)
        print("TARGET:", target)
        print("=" * 75)

        df = add_external_features(panel, target)

        # Current target is retained as an explicit historical predictor,
        # matching the original baseline's use of the current target.
        df["narrative_target"] = df[target]

        features = external_baseline_features(df)

        predictions = {
            "persistence": persistence(df, target)
        }

        predictions["train_mean"] = walk_forward(
            df, ["h_lag1"], MODELS["mean"]
        )

        for model_name in ["ridge", "hgb"]:
            predictions[model_name] = walk_forward(
                df, features, MODELS[model_name]
            )

        result, idx = score(df, predictions)

        print(
            f"Held-out narrative-weeks: {len(idx)}"
        )
        print(result.round(5))

        result = result.assign(target=target)
        all_results.append(result)

    combined = pd.concat(all_results)

    output = RESULTS / "24_external_baseline.csv"
    combined.to_csv(output)

    print("\n=== EXTERNAL BASELINE COMPLETE ===")
    print("Saved:", output)


if __name__ == "__main__":
    main()
