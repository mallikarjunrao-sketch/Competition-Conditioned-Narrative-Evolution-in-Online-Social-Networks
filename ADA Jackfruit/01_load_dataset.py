"""
Day-1 starter for "Competition-Conditioned Narrative Evolution" (The Outliers).

Flattens the Hakimov & Cheema Russo-Ukrainian news-tweet dataset (Zenodo JSONL)
into one row per tweet, with:
  - text_en  : English (machine-translated) text rebuilt from the Stanza tokens
  - stance   : pro_russia / pro_ukraine / unsure  (paper's 0.9 rule)
  - sentiment: label + prob
  - entities : list of (text, type)
NOTE: the public release has NO tweet_id -> date mapping. Dates must be recovered
from the tweet_id (Twitter snowflake IDs encode the timestamp, see tweet_id_to_datetime).

Usage:
    python 01_load_dataset.py --src path/to/dataset.jsonl --out data/tweets.parquet
    python 01_load_dataset.py --src ru-ua-news-discourse-twitter/resources/dataset_sample.jsonl --out data/sample.parquet
"""
import argparse, json, random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config import SEED, TIME_BIN, STANCE_THRESH, LANG

random.seed(SEED)
np.random.seed(SEED)

PRO_RU = ["This statement is in favour of Russia",
          "This statement is in favour of war",
          "This statement is in favour of military conflict"]
PRO_UA = ["This statement is against Russia",
          "This statement is in favour of Ukraine",
          "This statement is against war",
          "This statement is against military conflict"]

TWITTER_EPOCH_MS = 1288834974657


def tweet_id_to_datetime(tid: int) -> datetime:
    """Twitter snowflake ID -> UTC timestamp (works for all post-2010 tweets)."""
    ms = (int(tid) >> 22) + TWITTER_EPOCH_MS
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def derive_stance(stance_list, thr=STANCE_THRESH):
    """Paper's rule: entail_prob >= thr on pro-Russia hypotheses -> pro_russia,
    on pro-Ukraine hypotheses -> pro_ukraine, else unsure. Conflicts -> unsure."""
    p = {s["hypothesis"]: float(s["entail_prob"]) for s in stance_list}
    ru = max(p.get(h, 0.0) for h in PRO_RU)
    ua = max(p.get(h, 0.0) for h in PRO_UA)
    hit_ru, hit_ua = ru >= thr, ua >= thr
    if hit_ru and not hit_ua:
        return "pro_russia", ru, ua
    if hit_ua and not hit_ru:
        return "pro_ukraine", ru, ua
    return "unsure", ru, ua


def parse_record(d) -> dict:
    if isinstance(d, str):
        d = json.loads(d)
    toks = [t["text"] for sent in d.get("stanza_output", []) for t in sent]
    stance, ru, ua = derive_stance(d.get("stance", []))
    sent = d.get("sentiment", {}) or {}
    # the release stores only the winning class: {'neutral': '0.5044'}
    sent_label, sent_prob = (next(iter(sent.items())) if sent else (None, None))
    return {
        "tweet_id": int(d["tweet_id"]),
        "date": tweet_id_to_datetime(d["tweet_id"]),
        "lang": d.get("lang"),
        "channel": d.get("channel"),
        "country": d.get("country"),
        "verified": d.get("verified"),
        "followers": d.get("follower_count"),
        "text_en": " ".join(toks),
        "stance": stance,
        "stance_pro_ru_max": ru,
        "stance_pro_ua_max": ua,
        "sentiment": sent_label,
        "sentiment_prob": float(sent_prob) if sent_prob is not None else np.nan,
        "entities": [(e["text"], e["type"]) for e in d.get("stanza_named_entities", [])],
    }


def iter_json_objects(path, chunk_chars=1 << 22):
    """Yield JSON objects from a file that may be (a) one object per line, (b) several objects glued
    together on one line / separated by commas, or (c) one big [ ... ] array. Streams, so a multi-GB
    file does not need to fit in memory."""
    dec = json.JSONDecoder()
    buf, pos, eof, started = "", 0, False, False
    with open(path, encoding="utf-8") as f:
        while True:
            # skip whitespace / commas between objects, and a leading '['
            while True:
                while pos < len(buf) and buf[pos] in " \t\r\n,":
                    pos += 1
                if not started and pos < len(buf):
                    started = True
                    if buf[pos] == "[":
                        pos += 1
                        continue
                break
            if pos < len(buf) and buf[pos] == "]":
                return
            if pos >= len(buf):
                if eof:
                    return
                buf, pos = f.read(chunk_chars), 0
                eof = buf == ""
                continue
            try:
                obj, end = dec.raw_decode(buf, pos)
                yield obj
                pos = end
            except json.JSONDecodeError:
                more = f.read(chunk_chars)          # object continues in the next chunk
                if not more:
                    raise
                buf, pos = buf[pos:] + more, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lang", default=LANG, help="keep only this language code (default from config); use all to keep everything")
    ap.add_argument("--limit", type=int, default=None, help="stop after N records (quick test)")
    args = ap.parse_args()

    rows, seen = [], 0
    for obj in iter_json_objects(args.src):
        seen += 1
        lg = obj.get("lang")
        if not (args.lang == "all" or (args.lang == "nonen" and lg != "en") or lg == args.lang):
            continue
        rows.append(parse_record(obj))
        if seen % 100_000 == 0:
            print(f"  read {seen:,} records, kept {len(rows):,}", flush=True)
        if args.limit and seen >= args.limit:
            break
    print(f"read {seen:,} records, kept {len(rows):,} (lang={args.lang})")
    df = pd.DataFrame(rows)
    df["week"] = df["date"].dt.tz_localize(None).dt.to_period(TIME_BIN).dt.start_time

    # Dedupe: wire-service tweets / exact repeats across channels
    n0 = len(df)
    df = df.drop_duplicates(subset=["text_en", "channel"]).reset_index(drop=True)
    print(f"rows: {n0} -> {len(df)} after (text, channel) dedupe")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(df[["tweet_id", "date", "lang", "country", "stance", "sentiment"]].head())
    print("\nFields:", list(df.columns))
    print("\nTweets/week:\n", df.groupby("week").size().describe())
    print("\nStance:\n", df["stance"].value_counts(normalize=True).round(3))


if __name__ == "__main__":
    main()
