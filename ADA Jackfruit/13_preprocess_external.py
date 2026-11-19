"""Dataset 3 preprocessing using the original project's frozen cleaning rules."""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


_URL = re.compile(r"https?://\S+|www\.\S+|\bt\.co/\S+|\burl_\d+\b", re.I)


def clean_text(t: str) -> str:
    """Exact cleaning function from 02_embed_cluster.py."""
    t = _URL.sub(" ", t)
    t = re.sub(r"\bRT\b", " ", t)
    t = re.sub(r"\bread more\b\s*:?", " ", t, flags=re.I)
    t = re.sub(r"[#@]\s*(?=\w)", "", t)
    return re.sub(r"\s+", " ", t).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Reading Dataset 3 sample...", flush=True)
    df = pd.read_parquet(args.src)

    initial = len(df)
    assert df["tweet_id"].is_unique, "Duplicate tweet IDs detected."
    assert df["text_en"].notna().all(), "Missing text detected."

    # Preserve original event-window and English-language rules.
    df["week"] = pd.to_datetime(df["week"])
    df = df[
        (df["lang"] == "en")
        & (df["week"] >= pd.Timestamp("2022-02-01") - pd.Timedelta(days=6))
        & (df["week"] <= pd.Timestamp("2023-05-28"))
    ].copy()

    print(f"Rows after original filters: {len(df):,}", flush=True)

    # Exact original text cleaning.
    df["text_clean"] = df["text_en"].map(clean_text)

    before_length_filter = len(df)
    df = df[
        df["text_clean"].str.split().str.len() >= 4
    ].reset_index(drop=True)

    print(
        f"Removed short texts: {before_length_filter - len(df):,}",
        flush=True
    )
    print(f"Cleaned tweets retained: {len(df):,}", flush=True)

    # Match np.unique ordering used by the original embedding script.
    unique_texts, inverse = np.unique(
        df["text_clean"].to_numpy(),
        return_inverse=True
    )

    df["text_idx"] = inverse.astype(np.int32)

    cleaned_path = out_dir / "tweets_external_clean.parquet"
    unique_path = out_dir / "external_unique_texts.parquet"

    # Store each unique text once for the embedding stage.
    unique_df = pd.DataFrame({
        "text_idx": np.arange(len(unique_texts), dtype=np.int32),
        "text_clean": unique_texts
    })

    df.to_parquet(cleaned_path, index=False, compression="zstd")
    unique_df.to_parquet(unique_path, index=False, compression="zstd")

    print("\n=== PREPROCESSING COMPLETE ===")
    print(f"Original sample: {initial:,}")
    print(f"Cleaned tweets: {len(df):,}")
    print(f"Unique cleaned texts: {len(unique_texts):,}")
    print(f"Repeated cleaned texts: {len(df) - len(unique_texts):,}")
    print(f"Cleaned dataset: {cleaned_path}")
    print(f"Unique text corpus: {unique_path}")


if __name__ == "__main__":
    main()
