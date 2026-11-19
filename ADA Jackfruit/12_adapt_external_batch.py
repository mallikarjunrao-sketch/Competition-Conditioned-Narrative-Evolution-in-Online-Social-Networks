
import argparse
import os
import re
import gc
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds


def parse_entities(value):
    """Convert hashtags into the original entities representation."""
    if value is None or pd.isna(value):
        return [["none", "NONE"]]

    tags = re.findall(r"[A-Za-z0-9_]+", str(value))
    tags = [tag.lower() for tag in tags[:10]]

    return [[tag, "TAG"] for tag in tags] or [["none", "NONE"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data_rep/dataset3_csv_manifest.txt")
    ap.add_argument("--original", default="data/tweets.parquet")
    ap.add_argument("--out", default="data_rep/tweets_external_sample.parquet")
    ap.add_argument("--fraction", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunk-size", type=int, default=100_000)
    ap.add_argument("--memory", default="4GB")
    args = ap.parse_args()

    if not 0 < args.fraction <= 1:
        raise ValueError("fraction must be in (0, 1].")

    paths = [
        line.strip()
        for line in Path(args.manifest).read_text().splitlines()
        if line.strip() and Path(line.strip()).is_file()
    ]

    if not paths:
        raise RuntimeError("No CSV files found in manifest.")

    print(f"Reading {len(paths)} CSV files.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    raw_path = str(Path(args.out).with_name("_external_sample_raw.parquet"))

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{args.memory}'")
    con.execute("SET threads=2")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET temp_directory='/content/duckdb_tmp'")

    # Load original tweet IDs only.
    original = pd.read_parquet(args.original, columns=["tweet_id"])
    original_ids = pd.to_numeric(
        original["tweet_id"], errors="coerce"
    ).dropna().astype("int64").drop_duplicates()

    con.register("original_df", pd.DataFrame({"tweet_id": original_ids}))
    con.execute("""
        CREATE TEMP TABLE original_ids AS
        SELECT tweet_id FROM original_df
    """)

    del original, original_ids
    gc.collect()

    # First pass: find unique eligible external IDs and dates.
    print("\nPass 1/2: filtering, deduplicating and excluding overlap...")

    con.execute("""
        CREATE TEMP TABLE eligible_ids AS
        SELECT
            TRY_CAST(tweetid AS BIGINT) AS tweet_id,
            TRY_CAST(tweetcreatedts AS TIMESTAMPTZ) AS tweet_date
        FROM read_csv(
            ?,
            union_by_name=true,
            all_varchar=true,
            ignore_errors=true
        )
        WHERE LOWER(language) = 'en'
          AND TRY_CAST(tweetid AS BIGINT) IS NOT NULL
          AND TRY_CAST(tweetcreatedts AS TIMESTAMPTZ)
              >= TIMESTAMPTZ '2022-02-01 00:00:00+00'
          AND TRY_CAST(tweetcreatedts AS TIMESTAMPTZ)
              < TIMESTAMPTZ '2023-05-29 00:00:00+00'
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY TRY_CAST(tweetid AS BIGINT)
            ORDER BY TRY_CAST(tweetcreatedts AS TIMESTAMPTZ)
        ) = 1
    """, [paths])

    con.execute("""
        CREATE TEMP TABLE clean_ids AS
        SELECT e.*
        FROM eligible_ids e
        ANTI JOIN original_ids o USING (tweet_id)
    """)

    population = con.execute(
        "SELECT COUNT(*) FROM clean_ids"
    ).fetchone()[0]

    sample_size = round(population * args.fraction)

    print(f"Eligible unique non-overlapping tweets: {population:,}")
    print(f"Sampling fraction: {args.fraction:.1%}")
    print(f"Exact sample size: {sample_size:,}")
    print(f"Seed: {args.seed}")

    # Deterministic sampling. Seed is included in the hash input.
    print("\nSelecting deterministic sample...")

    con.execute(f"""
        CREATE TEMP TABLE selected_ids AS
        SELECT tweet_id
        FROM clean_ids
        ORDER BY MD5(CAST(tweet_id AS VARCHAR) || ':{args.seed}')
        LIMIT {sample_size}
    """)

    # Second pass: retrieve the selected tweets with their required columns.
    print("\nPass 2/2: retrieving selected tweet records...")

    # Use SQL literals here: DuckDB COPY does not accept
    # parameter placeholders in this query.
    paths_sql = "[" + ", ".join(
        "'" + p.replace("'", "''") + "'" for p in paths
    ) + "]"

    raw_sql = raw_path.replace("'", "''")

    if os.path.exists(raw_path):
        os.remove(raw_path)

    copy_sql = f"""
        COPY (
            SELECT r.*
            FROM (
                SELECT
                    TRY_CAST(tweetid AS BIGINT) AS tweet_id,
                    TRY_CAST(tweetcreatedts AS TIMESTAMPTZ) AS tweet_date,
                    userid,
                    text,
                    language,
                    retweetcount,
                    favorite_count,
                    hashtags
                FROM read_csv(
                    {paths_sql},
                    union_by_name=true,
                    all_varchar=true,
                    ignore_errors=true
                )
                WHERE LOWER(language) = 'en'
                  AND TRY_CAST(tweetid AS BIGINT) IS NOT NULL
                  AND TRY_CAST(tweetcreatedts AS TIMESTAMPTZ)
                      >= TIMESTAMPTZ '2022-02-01 00:00:00+00'
                  AND TRY_CAST(tweetcreatedts AS TIMESTAMPTZ)
                      < TIMESTAMPTZ '2023-05-29 00:00:00+00'
            ) r
            INNER JOIN selected_ids s USING (tweet_id)
        )
        TO '{raw_sql}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """

    con.execute(copy_sql)

    # Convert selected raw records to the original adapter schema in batches.
    print("\nConverting selected records to research schema...")

    parquet_file = pq.ParquetFile(raw_path)
    writer = None
    written = 0
    seen_ids = set()

    for batch in parquet_file.iter_batches(batch_size=args.chunk_size):
        d = batch.to_pandas()

        dates = pd.to_datetime(d["tweet_date"], utc=True, errors="coerce")
        valid = dates.notna() & d["text"].notna()
        d = d.loc[valid].copy()
        dates = dates.loc[valid]

        if d.empty:
            continue

        # Deduplicate across batches without a global SQL sort.
        keep = ~d["tweet_id"].isin(seen_ids)
        d = d.loc[keep].copy()
        dates = dates.loc[keep]

        if d.empty:
            continue

        seen_ids.update(d["tweet_id"].tolist())

        retweets = pd.to_numeric(
            d["retweetcount"], errors="coerce"
        ).fillna(0)

        likes = pd.to_numeric(
            d["favorite_count"], errors="coerce"
        ).fillna(0)

        output = pd.DataFrame({
            "tweet_id": d["tweet_id"].astype("int64"),
            "date": dates,
            "lang": "en",
            "channel": d["userid"].fillna("unknown").astype(str),
            "country": np.nan,
            "verified": False,
            "followers": np.nan,
            "text_en": d["text"].astype(str),
            "stance": "unsure",
            "stance_pro_ru_max": np.nan,
            "stance_pro_ua_max": np.nan,
            "sentiment": np.nan,
            "sentiment_prob": np.nan,
            "entities": d["hashtags"].map(parse_entities),
            "engagement": (retweets + likes).astype(float),
        })

        output["week"] = (
            output["date"]
            .dt.tz_localize(None)
            .dt.to_period("W")
            .dt.start_time
        )

        table = pa.Table.from_pandas(output, preserve_index=False)

        if writer is None:
            writer = pq.ParquetWriter(
                args.out, table.schema, compression="zstd"
            )

        writer.write_table(table)
        written += len(output)

        print(f"Converted {written:,} records", end="\r")

    if writer is not None:
        writer.close()

    if os.path.exists(raw_path):
        os.remove(raw_path)

    print("\n\n--- COMPLETED ---")
    print(f"Eligible population: {population:,}")
    print(f"Selected sample target: {sample_size:,}")
    print(f"Valid records written: {written:,}")
    print(f"Output: {args.out}")

    con.close()


if __name__ == "__main__":
    main()
