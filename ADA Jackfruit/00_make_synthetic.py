"""PLUMBING TEST ONLY. Makes a fake tweets.parquet (same schema as 01's output) so you can run
steps 02-07 before the real data arrives. Competition effects planted here are NOT evidence of anything."""
import numpy as np, pandas as pd
import config as C

rng = np.random.default_rng(C.SEED)
TOPICS = {
 0: ("battlefield", "troops shelling frontline offensive artillery tanks missile advance retreat soldiers drone strike city defense forces attack".split()),
 1: ("energy_sanctions", "sanctions gas oil price embargo pipeline energy economy ruble export supply inflation cap barrel prices market".split()),
 2: ("humanitarian", "refugees aid civilians evacuation children shelter humanitarian border displaced hospital food relief families volunteers corridor".split()),
 3: ("diplomacy", "talks summit negotiations ceasefire minister envoy treaty meeting leaders statement peace proposal delegation agreement diplomats".split()),
 4: ("nuclear", "nuclear plant radiation zaporizhzhia safety inspectors reactor iaea power warning threat weapons atomic facility cooling".split()),
}
COMMON = "war russia ukraine news says report latest today president government".split()
ENTS = {0: [("Bakhmut","GPE"),("Kherson","GPE")], 1: [("EU","ORG"),("OPEC","ORG")], 2: [("UNHCR","ORG"),("Poland","GPE")],
        3: [("Zelensky","PERSON"),("Erdogan","PERSON")], 4: [("IAEA","ORG"),("Zaporizhzhia","GPE")]}
STANCE_MU = {0: -.1, 1: .0, 2: .15, 3: .05, 4: .0}
weeks = pd.date_range("2022-02-07", "2023-05-29", freq="7D")
W, K = len(weeks), len(TOPICS)

logv = np.zeros((K, W)); base = np.log(70)
for k in range(K):
    for w in range(W):
        logv[k, w] = base + (0.7 * (logv[k, w-1] - base) if w else 0) + rng.normal(0, .18)
vol = np.exp(logv); vol[:, rng.choice(W, 5, replace=False)] *= 2.0     # "event weeks"
growth = np.diff(np.log(vol), axis=1, prepend=np.log(vol[:, :1]))
mix = np.full((K, W), .1)                                              # planted effect: rival growth -> drift
for k in range(K):
    comp_g = np.delete(growth, k, axis=0).mean(0)
    for w in range(1, W):
        mix[k, w] = np.clip(.1 + 1.5 * max(0, comp_g[w-1]) + rng.normal(0, .03), 0, .6)

rows = []
for w, wk in enumerate(weeks):
    for k in range(K):
        n = int(rng.poisson(vol[k, w]))
        for _ in range(n):
            words = []
            for _ in range(10):
                r = rng.random()
                if r < .15: words.append(rng.choice(COMMON))
                elif rng.random() < mix[k, w]: words.append(rng.choice(TOPICS[rng.choice([j for j in range(K) if j != k])][1]))
                else: words.append(rng.choice(TOPICS[k][1]))
            ua = float(np.clip(rng.beta(1, 8) + STANCE_MU[k], 0, 1)); ru = float(np.clip(rng.beta(1, 14), 0, 1))
            lab = rng.choice(["neutral", "negative", "positive"], p=[.5, .45, .05] if k != 2 else [.4, .55, .05])
            e = ENTS[k][rng.integers(0, 2)]
            rows.append(dict(
                tweet_id=int(rng.integers(1e18, 2e18)), date=(wk + pd.Timedelta(days=int(rng.integers(0, 7)), hours=int(rng.integers(0, 24)))).tz_localize("UTC"),
                lang="en", channel=f"ch{rng.integers(0, 40)}", country=f"c{rng.integers(0, 10)}", verified=bool(rng.random() < .5),
                followers=float(rng.lognormal(11, 1.2)), text_en=" ".join(words), stance="pro_ukraine" if ua >= .9 else "unsure",
                stance_pro_ru_max=ru, stance_pro_ua_max=ua, sentiment=lab, sentiment_prob=float(rng.uniform(.5, .95)),
                entities=[[e[0], e[1]]], week=wk))
df = pd.DataFrame(rows)
df.to_parquet(C.DATA / "synthetic.parquet", index=False)
print(f"synthetic: {len(df)} tweets, {W} weeks, {K} true topics -> data/synthetic.parquet")
