"""Step 3 (Both): agreement between annotators + cluster validity.
  python 03_label_agreement.py                      # 1st run also writes data/label_adjudicate.csv (the disagreements)
  python 03_label_agreement.py --merge economy=politics   # score with collapsed hand labels (optional)
Adjudication: discuss the disagreements together, fill label_final in data/label_adjudicate.csv, rerun."""
import argparse, pandas as pd
from sklearn.metrics import cohen_kappa_score, adjusted_rand_score, normalized_mutual_info_score
import config as C

ap = argparse.ArgumentParser(); ap.add_argument("--merge", nargs="*", default=[], help="e.g. economy=politics")
a = ap.parse_args(); mp = dict(m.split("=") for m in a.merge)
norm = lambda s: s.astype(str).str.strip().str.lower().replace({"nan": ""})

lab = pd.read_csv(C.DATA / "label_sample.csv").merge(pd.read_csv(C.DATA / "label_key.csv"), on="tweet_id")
for c in ("label_A", "label_B"): lab[c] = norm(lab[c])
lab = lab[(lab.label_A != "") & (lab.label_B != "")].copy()
tcol = "text" if "text" in lab else "text_en"
adj_path = C.DATA / "label_adjudicate.csv"
if not adj_path.exists():
    d = lab[lab.label_A != lab.label_B]
    d[["tweet_id", tcol, "label_A", "label_B"]].assign(label_final="").to_csv(adj_path, index=False)
    print(f"Wrote {len(d)} disagreements to data/label_adjudicate.csv -> discuss together, fill label_final, rerun.\n")

lab["final"] = lab.label_A.where(lab.label_A == lab.label_B, "")
adj = pd.read_csv(adj_path); adj["label_final"] = norm(adj["label_final"]); adj = adj[adj.label_final != ""].set_index("tweet_id").label_final
lab["final"] = lab["final"].where(lab["final"] != "", lab.tweet_id.map(adj).fillna(""))
for c in ("label_A", "label_B", "final"): lab[c] = lab[c].map(lambda x: mp.get(x, x))
if mp: print(f"(hand labels collapsed with {mp})")

print(f"{len(lab)} doubly-labeled tweets")
print(f"HUMAN agreement (independent 1st pass): kappa={cohen_kappa_score(lab.label_A, lab.label_B):.3f}  "
      f"ARI(A,B)={adjusted_rand_score(lab.label_A, lab.label_B):.3f}   <- ARI(A,B) is the realistic ceiling for clusters-vs-humans")
print(f"exact agreement: {(lab.label_A == lab.label_B).mean():.0%}")

def purity(l, c): return pd.crosstab(c, l).max(axis=1).sum() / len(l)
print("\nCluster validity vs hand labels:")
for name, d, col in [("A", lab, "label_A"), ("B", lab, "label_B"), ("agreed-only", lab[lab.label_A == lab.label_B], "label_A"),
                     ("consensus", lab[lab.final != ""], "final")]:
    if len(d) > 5:
        print(f"  {name:12s} n={len(d):3d}  ARI={adjusted_rand_score(d[col], d.narrative):.3f}  "
              f"NMI={normalized_mutual_info_score(d[col], d.narrative):.3f}  purity={purity(d[col], d.narrative):.3f}")
cons = lab[lab.final != ""]
print("\nCluster x consensus-label table:\n", pd.crosstab(cons.narrative, cons.final))
print("\nCHECKPOINT: if consensus ARI < ~0.3 or purity < ~0.6, try another K with: python 02_embed_cluster.py --reuse --k 5")
