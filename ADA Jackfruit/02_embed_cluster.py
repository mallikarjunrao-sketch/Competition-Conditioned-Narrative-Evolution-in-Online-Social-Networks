"""Step 2 (Person A): English filter -> embeddings -> narratives defined on the TRAINING period only
-> assign every tweet to a narrative -> draw the hand-labeling sample.

  python 02_embed_cluster.py --tweets data/tweets.parquet --backend st
  python 02_embed_cluster.py --tweets data/synthetic.parquet --backend tfidf     # plumbing test
"""
import argparse, re, numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.feature_extraction.text import TfidfVectorizer
import config as C


_URL = re.compile(r"https?://\S+|www\.\S+|\bt\.co/\S+|\burl_\d+\b", re.I)


def clean_text(t: str) -> str:
    """Strip URLs / 'url_1' placeholders / RT markers / hashtag+mention symbols / 'Read more' boilerplate.
    These tokens carry no narrative content but dominate the embeddings otherwise."""
    t = _URL.sub(" ", t)
    t = re.sub(r"\bRT\b", " ", t)
    t = re.sub(r"\bread more\b\s*:?", " ", t, flags=re.I)
    t = re.sub(r"[#@]\s*(?=\w)", "", t)
    return re.sub(r"\s+", " ", t).strip()


def embed(texts, backend, model_name):
    if backend == "st":
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(model_name)
        return m.encode(texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True).astype(np.float32)
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize
    X = TfidfVectorizer(min_df=2, max_features=20000, stop_words="english").fit_transform(texts)
    return normalize(TruncatedSVD(64, random_state=C.SEED).fit_transform(X)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tweets", default=str(C.DATA / "tweets.parquet"))
    ap.add_argument("--backend", default="st", choices=["st", "tfidf"])
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--k", type=int, default=None, help="force number of narratives")
    ap.add_argument("--n_label", type=int, default=100)
    ap.add_argument("--reuse", action="store_true", help="reuse saved data/tweets_en.parquet + data/emb.npy (skip embedding; re-cluster in ~1 min, e.g. with a different --k)")
    ap.add_argument("--lang", default=C.LANG, help="language to keep: en (default), nonen, or all")
    ap.add_argument("--centroids", default=None, help="REPLICATION: assign tweets to these fixed narrative centroids (data/narrative_centroids.npy of the main run) instead of clustering")
    ap.add_argument("--frac", type=float, default=1.0, help="uniform random fraction of tweets to use (e.g. 0.1 for a fast trial run)")
    a = ap.parse_args()

    if a.reuse:
        df = pd.read_parquet(C.DATA / "tweets_en.parquet"); E = np.load(C.DATA / "emb.npy"); assert len(df) == len(E)
        df["week"] = pd.to_datetime(df["week"]); print(f"reusing {len(df)} tweets + saved embeddings (no re-embedding)")
    else:
        df = pd.read_parquet(a.tweets)
        df["week"] = pd.to_datetime(df["week"])
        df = df[((df.lang == a.lang) | (a.lang == "all") | ((a.lang == "nonen") & (df.lang != "en"))) & (df.week >= pd.Timestamp(C.EVENT_START) - pd.Timedelta(days=6)) &
                (df.week <= pd.Timestamp(C.EVENT_END))].reset_index(drop=True)
        if a.frac < 1.0:
            df = df.sample(frac=a.frac, random_state=C.SEED).sort_values("date").reset_index(drop=True)
            print(f"using a {a.frac:.0%} uniform sample")
        print(f"{len(df)} {C.LANG} tweets in window")

        df["text_clean"] = df["text_en"].map(clean_text)
        n_before = len(df)
        df = df[df["text_clean"].str.split().str.len() >= 4].reset_index(drop=True)
        print(f"dropped {n_before - len(df)} tweets with <4 words after cleaning (mostly bare links); {len(df)} left")
        uniq, inv = np.unique(df["text_clean"].to_numpy(), return_inverse=True)      # embed each distinct text once
        print(f"embedding {len(uniq)} distinct texts (of {len(df)} tweets)")
        E = embed(uniq.tolist(), a.backend, a.model)[inv]
    if a.centroids:
        cent = np.load(a.centroids); assert cent.shape[1] == E.shape[1], "embedding dim differs from the centroids' (same --backend/--model as the main run?)"
        df["narrative"] = (E @ cent.T).argmax(1)
        df.to_parquet(C.DATA / "tweets_en.parquet", index=False); np.save(C.DATA / "emb.npy", E)
        print(f"assigned {len(df)} tweets to the FIXED narratives from {a.centroids}:\n{df.narrative.value_counts(normalize=True).sort_index().round(3)}")
        return
    train_idx = np.where(df.week <= pd.Timestamp(C.TRAIN_END))[0]
    rng = np.random.default_rng(C.SEED)
    fit_idx = rng.choice(train_idx, min(len(train_idx), 100_000), replace=False)
    print(f"{len(train_idx)} training-period tweets (fit on {len(fit_idx)})")

    # choose K by silhouette on a sample (or take --k)
    sil_idx = rng.choice(fit_idx, min(len(fit_idx), 5000), replace=False)
    best = {}
    for k in ([a.k] if a.k else C.K_RANGE):
        km = KMeans(k, n_init=10, random_state=C.SEED).fit(E[fit_idx])
        best[k] = (silhouette_score(E[sil_idx], km.predict(E[sil_idx])) if k > 1 else 0, km)
        print(f"  K={k}  silhouette={best[k][0]:.3f}")
    K = a.k or max(best, key=lambda k: best[k][0])
    km = best[K][1]
    print(f"-> using K={K}  (you may override with --k after reading the cluster summaries)")

    # stability: refit with another seed, compare labels on the training sample
    km2 = KMeans(K, n_init=10, random_state=C.SEED + 1).fit(E[fit_idx])
    print(f"stability ARI between two seeds: {adjusted_rand_score(km.predict(E[sil_idx]), km2.predict(E[sil_idx])):.3f}")

    cent = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
    df["narrative"] = (E @ cent.T).argmax(1)            # same cosine rule for ALL tweets, train and test

    # human-readable summaries (training period only)
    tr = df.iloc[train_idx]
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(stop_words="english", min_df=20, max_features=30000)
    X = cv.fit_transform(tr["text_clean"]); terms = np.array(cv.get_feature_names_out())
    nars = sorted(tr.narrative.unique())
    counts = np.vstack([np.asarray(X[(tr.narrative == nar).to_numpy()].sum(0)).ravel() for nar in nars]).astype(float)
    p_c = counts / counts.sum(1, keepdims=True); p_all = counts.sum(0) / counts.sum()
    score = p_c * np.log((p_c + 1e-12) / p_all)            # frequent in this cluster AND over-represented vs. all clusters
    print("\nNarrative summaries (train period):")
    summ = []
    for i, nar in enumerate(nars):
        top = terms[score[i].argsort()[::-1][:10]]
        sub = tr[tr.narrative == nar]; ex = sub.iloc[(E[sub.index] @ cent[nar]).argsort()[::-1][:3]]["text_clean"].tolist()
        print(f"[{nar}] n={len(sub):>6}  distinctive terms: {', '.join(top)}\n     e.g. {ex[0][:130]}")
        summ.append(dict(narrative=nar, n_train=len(sub), top_terms=", ".join(top), examples=" || ".join(ex)))
    pd.DataFrame(summ).to_csv(C.RESULTS / "narratives_summary.csv", index=False)

    df.to_parquet(C.DATA / "tweets_en.parquet", index=False)
    if not a.reuse: np.save(C.DATA / "emb.npy", E)
    np.save(C.DATA / "narrative_centroids.npy", cent)

    # hand-labeling sample: SAME tweets for both annotators, cluster id hidden
    samp = tr.sample(min(a.n_label, len(tr)), random_state=C.SEED)
    out = samp[["tweet_id", "text_clean"]].rename(columns={"text_clean": "text"}); out["label_A"] = ""; out["label_B"] = ""
    path, keydf = C.DATA / "label_sample.csv", samp[["tweet_id", "narrative"]]
    if path.exists():
        prev = pd.read_csv(path)
        if prev[["label_A", "label_B"]].notna().any().any():
            found = df[df.tweet_id.isin(set(prev.tweet_id))]
            if found.tweet_id.nunique() == prev.tweet_id.nunique():
                out = None; keydf = found[["tweet_id", "narrative"]]
                print("label_sample.csv already holds your labels and all those tweets are in this data -> NOT overwritten; label_key.csv rebuilt for them")
            else:
                path = C.DATA / "label_sample_NEW.csv"; print("WARNING: some labeled tweets are missing from this data; wrote label_sample_NEW.csv instead")
    if out is not None: out.to_csv(path, index=False)
    keydf.to_csv(C.DATA / "label_key.csv", index=False)
    print("\nlabel_key.csv updated (cluster ids for the labeled tweets; don't peek before labeling)")


if __name__ == "__main__":
    main()
