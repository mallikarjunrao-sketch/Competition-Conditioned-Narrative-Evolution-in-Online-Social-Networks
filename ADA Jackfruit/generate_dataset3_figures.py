# Dataset 3: generate six report-ready figures in Colab.
# Runtime: ~1–3 min. RAM: typically 1–2 GB. Disk: six PNGs in figs_rep/.
# Network: none. No model training. Only creates/overwrites figs_rep/figure_*.png.

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path("/content/drive/MyDrive/ADA Jackfruit")
DATA, RESULTS, FIGS = ROOT/"data_rep", ROOT/"results_rep", ROOT/"figs_rep"
FIGS.mkdir(parents=True, exist_ok=True)

def col(df, names, what):
    lookup = {str(c).lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lookup: return lookup[n.lower()]
    raise ValueError(f"Missing {what}; columns are {list(df.columns)}")

coverage = pd.read_csv(DATA/"15_dataset3_weekly_coverage.csv")
base = pd.read_csv(RESULTS/"24_external_baseline.csv")
comp = pd.read_csv(RESULTS/"27_external_corrected_placebo.csv")
holm = pd.read_csv(RESULTS/"28_external_holm_correction.csv")
panel = pd.read_parquet(DATA/"panel_external.parquet")
week, n = col(panel, ["week"], "week"), col(panel, ["n","tweet_count","count"], "weekly count")
weekly = panel.groupby(week, as_index=False)[n].sum()
weekly[week] = pd.to_datetime(weekly[week]); weekly = weekly.sort_values(week)
assert len(weekly) == 66, f"Expected 66 weeks, got {len(weekly)}"

labels = pd.read_parquet(DATA/"tweets_external_labeled.parquet", columns=["narrative"])
narr = labels.narrative.value_counts().sort_index()
del labels
assert len(narr) == 4

bm, bt, bs = col(base,["model"],"baseline model"), col(base,["target"],"baseline target"), col(base,["skill_vs_persist"],"baseline skill")
cm, ct, cg = col(comp,["model"],"competition model"), col(comp,["target"],"competition target"), col(comp,["real_gain"],"real gain")
ht, hm = col(holm,["target"],"Holm target"), col(holm,["model"],"Holm model")
hp, ha = col(holm,["placebo_p"],"raw p"), col(holm,["holm_p"],"adjusted p")
assert len(base)==32 and len(comp)==16 and len(holm)==16

names = {
 "drift_db":"Semantic drift", "entity_turnover":"Entity turnover*",
 "abs_frag_change":"Fragmentation", "abs_share_change":"Narrative share",
 "abs_merge_change":"Merge change", "d_volume":"Volume growth",
 "d_breadth":"Breadth change", "d_engage":"Engagement change"
}
order=list(names)
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,
                     "axes.spines.right":False,"savefig.dpi":300})
def save(name):
    p=FIGS/name; plt.savefig(p,bbox_inches="tight",facecolor="white"); plt.close()
    print("Saved",p)

# 1. Acquisition / preprocessing flowchart
fig, ax=plt.subplots(figsize=(12,7)); ax.set_xlim(0,12); ax.set_ylim(0,8); ax.axis("off")
boxes=[
(.4,6.2,3.1,1,"External Kaggle collection\n461 CSV files"),
(4.4,6.2,3.1,1,"Raw scan\n70,876,102 rows"),
(8.5,6.2,3.1,1,"English + event-window\neligibility filters"),
(8.5,4.35,3.1,1,"Deduplication + overlap removal\n43,697,878 unique IDs"),
(4.4,4.35,3.1,1,"Deterministic 3% sample\n1,310,936 tweets"),
(.4,4.35,3.1,1,"Text cleaning + minimum length\n1,305,272 tweets"),
(.4,2.35,3.1,1,"Unique cleaned texts\n732,096"),
(4.4,2.35,3.1,1,"all-MiniLM-L6-v2 embeddings\n384 dimensions"),
(8.5,2.35,3.1,1,"Frozen original centroids\n4 narrative labels"),
(4.4,.35,3.1,1,"Weekly panel\n66 weeks × 4 narratives")]
for x,y,w,h,t in boxes:
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.08",fc="#EEF4F8",ec="#315A7D"))
    ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=9)
arrows=[((3.5,6.7),(4.4,6.7)),((7.5,6.7),(8.5,6.7)),((10,6.2),(10,5.35)),
((8.5,4.85),(7.5,4.85)),((4.4,4.85),(3.5,4.85)),((1.95,4.35),(1.95,3.35)),
((3.5,2.85),(4.4,2.85)),((7.5,2.85),(8.5,2.85)),((10,2.35),(10,1.8)),
((10,1.8),(6,1.8)),((6,1.8),(6,1.35))]
for a,b in arrows: ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=12,color="#526777"))
ax.text(6,7.65,"Dataset 3 acquisition and processing pipeline",ha="center",fontsize=15,weight="bold")
ax.text(6,.05,"Original-collection overlap IDs excluded; independent collection provenance remains unverified.",
        ha="center",fontsize=8)
save("figure_1_acquisition_flow.png")

# 2. Weekly volume
fig,ax=plt.subplots(figsize=(12,4.8))
ax.plot(weekly[week],weekly[n],marker="o",markersize=3,linewidth=1.7)
ax.fill_between(weekly[week],weekly[n],alpha=.15)
ax.set(title="Dataset 3 weekly tweet volume",xlabel="Week",ylabel="Tweets")
ax.grid(axis="y",alpha=.25); fig.autofmt_xdate()
ax.text(.99,.96,f"66 weeks | {int(weekly[n].sum()):,} tweets",transform=ax.transAxes,ha="right",va="top")
save("figure_2_weekly_tweet_volume.png")

# 3. Narrative distribution
fig,ax=plt.subplots(figsize=(8,5)); bars=ax.bar([f"Narrative {i}" for i in narr.index],narr.values)
total=int(narr.sum()); ax.set(title="Tweet distribution across four frozen narratives",ylabel="Tweets")
ax.grid(axis="y",alpha=.25)
for b,v in zip(bars,narr.values):
    ax.text(b.get_x()+b.get_width()/2,b.get_height(),f"{v:,}\n({100*v/total:.1f}%)",ha="center",va="bottom")
ax.set_ylim(0,max(narr.values)*1.18); save("figure_3_narrative_distribution.png")

# 4. Baseline model comparison
p=base.pivot_table(index=bt,columns=bm,values=bs,aggfunc="first").reindex(order)*100
fig,ax=plt.subplots(figsize=(12,6)); x=np.arange(len(order)); width=.19
for j,m in enumerate(["persistence","train_mean","ridge","hgb"]):
    ax.bar(x+(j-1.5)*width,p[m],width,label=m.replace("_"," ").title())
ax.axhline(0,color="black",lw=.8); ax.set_xticks(x)
ax.set_xticklabels([names[k] for k in order],rotation=25,ha="right")
ax.set(title="External dataset: baseline predictive performance",ylabel="Skill vs persistence (%)")
ax.legend(ncol=4,frameon=False,loc="upper center",bbox_to_anchor=(.5,1.12)); ax.grid(axis="y",alpha=.2)
ax.text(.01,-.25,"* Entity turnover uses hashtag-derived entity-like tokens.",transform=ax.transAxes,fontsize=8)
save("figure_4_baseline_comparison.png")

# 5. Competition-pressure gains
p=comp.pivot_table(index=ct,columns=cm,values=cg,aggfunc="first").reindex(order)*100
fig,ax=plt.subplots(figsize=(11,5.8)); x=np.arange(len(order)); width=.34
for j,m in enumerate(["ridge","hgb"]): ax.bar(x+(j-.5)*width,p[m],width,label=m.upper())
ax.axhline(0,color="black",lw=1); ax.set_xticks(x)
ax.set_xticklabels([names[k] for k in order],rotation=25,ha="right")
ax.set(title="Competition-pressure feature gains on Dataset 3",ylabel="MSE gain over baseline (%)")
ax.legend(frameon=False); ax.grid(axis="y",alpha=.2)
ax.text(.01,-.25,"Positive = lower MSE; negative = worse performance. Pressure is a reduced three-component proxy.",
        transform=ax.transAxes,fontsize=8)
save("figure_5_competition_gains.png")

# 6. Raw and Holm-adjusted p-values
q=holm.copy()
q["_label"]=q[ht].map(names).fillna(q[ht].astype(str))+" / "+q[hm].astype(str).str.upper()
q=q.sort_values(hp); y=np.arange(len(q))
fig,ax=plt.subplots(figsize=(10,7))
ax.scatter(q[hp],y,label="Raw placebo p",s=35)
ax.scatter(q[ha],y,label="Holm-adjusted p",marker="s",s=35)
ax.axvline(.05,ls="--",lw=1,color="#A33A3A",label="0.05 threshold")
ax.set_yticks(y); ax.set_yticklabels(q["_label"],fontsize=8); ax.set_xlim(-.02,1.04)
ax.set(title="Placebo p-values before and after Holm correction",xlabel="p-value")
ax.grid(axis="x",alpha=.2); ax.legend(frameon=False,loc="lower right")
save("figure_6_placebo_holm_pvalues.png")

print("\nCOMPLETE. Output:",FIGS)
for f in sorted(FIGS.glob("figure_*.png")): print(f.name, f"{f.stat().st_size/1024:.1f} KB")
print("Checks: weeks =",len(weekly),"; labelled tweets =",total,
      "; baseline rows =",len(base),"; competition rows =",len(comp),
      "; Holm significant =",int((holm[ha]<.05).sum()))
