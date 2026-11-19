
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path("/content/drive/MyDrive/ADA Jackfruit")
DATA = ROOT / "data_rep"
RESULTS = ROOT / "results_rep"

RESULTS.mkdir(parents=True, exist_ok=True)

TARGETS = [
    "drift_db",
    "entity_turnover",
    "abs_frag_change",
    "abs_share_change",
    "abs_merge_change",
    "d_volume",
    "d_breadth",
    "d_engage"
]

PRESSURE = [
    "comp_sem_z",
    "comp_growth_z",
    "comp_network_z"
]

TEST_START = pd.Timestamp("2023-01-01")
DELTA = 1
MIN_TRAIN = 40
REFIT_EVERY = 4

N_PLACEBO = 500
N_BOOT = 1000
BLOCK = 4
SEED = 42

MODELS = {
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
    )
}


def prepare_data(panel, target):

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

    df = pd.concat([df, dummies], axis=1)

    baseline = [
        target,
        "h_lag1",
        "h_lag2",
        "h_roll4",
        "ent_div",
        "log_n",
        "log_channels",
        "log_total_volume",
        "event_week"
    ] + [
        c for c in df.columns if c.startswith("nar_")
    ]

    return df, baseline


def walk_forward(
    df,
    features,
    model_factory,
    refit_every=REFIT_EVERY
):

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

        if model is None or i % refit_every == 0:

            train = df[
                (df.week <= t - pd.Timedelta(weeks=DELTA)) &
                df["y"].notna()
            ]

            if len(train) < MIN_TRAIN:
                continue

            model = model_factory()

            model.fit(
                train[features],
                train["y"]
            )

        if model is not None:
            predictions.loc[test.index] = model.predict(
                test[features]
            )

    return predictions


def block_perm(length, block, rng):

    blocks = [
        np.arange(i, min(i + block, length))
        for i in range(0, length, block)
    ]

    rng.shuffle(blocks)

    return np.concatenate(blocks)


def shuffle_cols(df, cols, block, rng):

    out = df.copy()
    values = df[cols].to_numpy().copy()
    narratives = df["narrative"].to_numpy()

    for narrative in np.unique(narratives):

        positions = np.where(narratives == narrative)[0]

        values[positions] = values[positions][
            block_perm(len(positions), block, rng)
        ]

    out[cols] = values

    return out


def mse(df, prediction, idx):

    return float(
        (
            df.loc[idx, "y"] -
            prediction.loc[idx]
        ).pow(2).mean()
    )


def main():

    rng = np.random.default_rng(SEED)

    panel = pd.read_parquet(
        DATA / "panel_pressure_external.parquet"
    )

    all_records = []

    output = RESULTS / "27_external_corrected_placebo.csv"

    for target in TARGETS:

        print("\n" + "=" * 85)
        print("TARGET:", target)
        print("=" * 85)

        df, baseline = prepare_data(panel, target)

        for model_name, model_factory in MODELS.items():

            print("\nMODEL:", model_name)

            # Both real models use four-week refitting.
            base_pred = walk_forward(
                df,
                baseline,
                model_factory,
                refit_every=REFIT_EVERY
            )

            pressure_pred = walk_forward(
                df,
                baseline + PRESSURE,
                model_factory,
                refit_every=REFIT_EVERY
            )

            idx = base_pred.dropna().index.intersection(
                pressure_pred.dropna().index
            )

            if len(idx) == 0:
                raise RuntimeError(
                    f"No common held-out rows for {target}, {model_name}"
                )

            base_mse = mse(df, base_pred, idx)
            pressure_mse = mse(df, pressure_pred, idx)

            real_gain = 1 - pressure_mse / base_mse

            # Paired bootstrap over held-out weeks.
            weekly = pd.DataFrame({
                "week": df.loc[idx, "week"],
                "difference": (
                    (df.loc[idx, "y"] - base_pred.loc[idx]) ** 2
                    -
                    (df.loc[idx, "y"] - pressure_pred.loc[idx]) ** 2
                )
            }).groupby("week")["difference"].mean().to_numpy()

            bootstrap = np.array([
                rng.choice(
                    weekly,
                    size=len(weekly),
                    replace=True
                ).mean()
                for _ in range(N_BOOT)
            ])

            ci_lo, ci_hi = np.percentile(
                bootstrap, [2.5, 97.5]
            )

            print("Held-out rows:", len(idx))
            print("Four-week baseline MSE:", base_mse)
            print("Four-week pressure MSE:", pressure_mse)
            print("Real gain:", round(real_gain, 6))
            print(
                "Paired bootstrap 95% CI:",
                (round(ci_lo, 8), round(ci_hi, 8))
            )

            # Placebo: shuffle all three pressure components.
            placebo_gains = []

            for iteration in range(N_PLACEBO):

                shuffled = shuffle_cols(
                    df,
                    PRESSURE,
                    BLOCK,
                    rng
                )

                placebo_pred = walk_forward(
                    shuffled,
                    baseline + PRESSURE,
                    model_factory,
                    refit_every=REFIT_EVERY
                )

                if placebo_pred.loc[idx].isna().any():
                    raise RuntimeError(
                        "Placebo prediction coverage differs from real coverage"
                    )

                placebo_mse = mse(
                    shuffled,
                    placebo_pred,
                    idx
                )

                placebo_gain = 1 - placebo_mse / base_mse

                placebo_gains.append(placebo_gain)

                if (iteration + 1) % 100 == 0:
                    print(
                        f"Placebo progress: "
                        f"{iteration + 1}/{N_PLACEBO}"
                    )

            placebo_gains = np.asarray(placebo_gains)

            placebo_p = (
                1 + np.sum(placebo_gains >= real_gain)
            ) / (1 + N_PLACEBO)

            placebo_mean = placebo_gains.mean()
            placebo_p95 = np.percentile(placebo_gains, 95)

            print("\nCORRECTED PLACEBO RESULT")
            print("Real gain:", round(real_gain, 6))
            print("Placebo mean:", round(placebo_mean, 6))
            print("Placebo 95th percentile:", round(placebo_p95, 6))
            print("One-sided placebo p:", round(placebo_p, 6))

            all_records.append({
                "target": target,
                "model": model_name,
                "n_test": len(idx),
                "refit_every": REFIT_EVERY,
                "baseline_mse": base_mse,
                "pressure_mse": pressure_mse,
                "real_gain": real_gain,
                "bootstrap_ci_lo": ci_lo,
                "bootstrap_ci_hi": ci_hi,
                "placebo_mean": placebo_mean,
                "placebo_p95": placebo_p95,
                "placebo_p": placebo_p
            })

        # Save a checkpoint after every completed outcome.
        pd.DataFrame(all_records).to_csv(
            output,
            index=False
        )

        print("\nCheckpoint saved:", output)

    print("\n=== CORRECTED EXTERNAL PLACEBO TESTS COMPLETE ===")
    print("Final results:", output)


if __name__ == "__main__":
    main()
