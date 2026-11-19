"""Optional: convert an external tweet CSV (e.g. Dataset 3, Kaggle 'Russia-Ukraine Conflict Twitter Dataset') into the schema of step 01's output,
so steps 02(--centroids)->04->06->07 can replicate on it.  Only text/time/user/engagement exist there (no stance, sentiment, followers, country), so stance/emotion
outcomes and some components are unavailable: the replication is a REDUCED model.  Needs >= ~30 weeks of data with >= ~100 tweets per narrative-week to be useful.
  python 09_adapt_external.py --src kaggle.csv --out data_rep/tweets.parquet
Column names follow that Kaggle file; override with flags if yours differ. Check the date span first!"""
import argparse, numpy as np, pandas as pd
import config as C
ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
for k, v in dict(text="Text", date="Datetime", id="Tweet Id", user="Username", lang="Language", retweets="RetweetCount", likes="LikeCount",
                 replies="ReplyCount", quotes="QuoteCount", hashtags="hashtag").items(): ap.add_argument(f"--{k}", default=v)
a = ap.parse_args()
d = pd.read_csv(a.src, low_memory=False)
dt = pd.to_datetime(d[a.date], errors="coerce", utc=True); ok = dt.notna() & d[a.text].notna()
d, dt = d[ok].reset_index(drop=True), dt[ok].reset_index(drop=True)
num = lambda c: pd.to_numeric(d[c], errors="coerce").fillna(0) if c in d else 0
ent = (d[a.hashtags].fillna("").astype(str).str.replace(r"[\[\]'\",]", " ", regex=True).str.split().map(lambda t: [[x.lower(), "TAG"] for x in t[:10]] or [["none", "NONE"]])
       if a.hashtags in d else pd.Series([[["none", "NONE"]]] * len(d)))
out = pd.DataFrame(dict(tweet_id=pd.to_numeric(d[a.id], errors="coerce").fillna(0).astype("int64"), date=dt, lang=d[a.lang] if a.lang in d else "en",
                        channel=d[a.user].astype(str) if a.user in d else "u", country=np.nan, verified=False, followers=np.nan, text_en=d[a.text].astype(str),
                        stance="unsure", stance_pro_ru_max=np.nan, stance_pro_ua_max=np.nan, sentiment=np.nan, sentiment_prob=np.nan, entities=ent,
                        engagement=num(a.retweets) + num(a.likes) + num(a.replies) + num(a.quotes)))
out["week"] = out["date"].dt.tz_localize(None).dt.to_period("W").dt.start_time
import pathlib; pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True); out.to_parquet(a.out, index=False)
w = out.groupby("week").size()
print(f"{len(out)} tweets, {out.week.min().date()} .. {out.week.max().date()}, {len(w)} weeks; tweets/week median={w.median():.0f}")
print("USABLE for a weekly walk-forward replication only if this spans >= ~30 weeks with enough tweets per week.")
