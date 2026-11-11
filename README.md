# Competition-Conditioned Narrative Evolution in Online Social Networks

**Team:** The Outliers  
**Members:** Anuj Srijan , Mallikarjuna Rao R.V , Deepesh M  
**Institution:** PES University, Bengaluru, India  
**Course:** Advanced Data Analytics (ADA)


---

## 1. The project in 30 seconds

**Research question:** when a rival narrative gains strength in a given week, does that pressure predict how the focal narrative changes in the following week, beyond what its own history, content, and network features already predict?

**Primary data:** news-media tweets about the Russo-Ukrainian conflict from the Zenodo dataset *Unveiling Global Narratives*.

**External data:** the BwandoWando Ukraine–Russian Crisis Twitter collection from Kaggle.

**Method in four steps:**

1. Clean the tweet text and generate sentence embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
2. Assign tweets to four narrative categories using the primary study's frozen narrative centroids.
3. Build weekly narrative-level outcomes and competition features.
4. Compare forecasts with and without competition features using chronological walk-forward evaluation, placebo tests, bootstrap intervals, and multiple-testing correction.

**The claim is predictive, not causal.** The experiments test whether competition features add out-of-sample predictive information. They do not establish that competition causes narrative change.

---

## 2. Dataset sources and why they were used

### Dataset 1 — not used

The original project plan included a dataset of approximately 127 million tweet IDs. It was not used in the final analysis because the tweet IDs could not be rehydrated: access to the Twitter/X API is no longer freely available. As a result, the underlying tweet content needed for this project could not be recovered through the planned process.

Dataset 1 does not contribute to the reported results.

### Dataset 2 — primary study

| Item | Details |
| --- | --- |
| Dataset | *Unveiling Global Narratives: A Multilingual Twitter Dataset of News Media on the Russo-Ukrainian Conflict* |
| Authors | Sherzod Hakimov and Gullal S. Cheema |
| Source | Zenodo |
| Link | https://zenodo.org/records/8043459 |
| DOI | https://doi.org/10.5281/zenodo.8043459 |
| Coverage | February 2022 – May 2023 |
| Size stated by source | 1,524,832 tweets across 60 languages |

Dataset 2 contains posts from news-media channels covering the Russo-Ukrainian conflict. It provides information such as language, channel, country, stance, sentiment, and named entities, subject to the fields available in the released data.

**How we used it:** Dataset 2 was the primary dataset for the main study. English-language posts in the study period were cleaned and represented using sentence embeddings. The original pipeline used the embeddings to define four narratives and then measured weekly semantic, emotional, stance, entity, and structural changes. Competition features were evaluated against historical, content, and network baselines.

### Dataset 3 — reduced external evaluation

| Item | Details |
| --- | --- |
| Dataset | *Ukraine–Russian Crisis Twitter Dataset (1.2M rows collection)* |
| Publisher | BwandoWando |
| Source | Kaggle |
| Link | https://www.kaggle.com/datasets/bwandowando/ukraine-russian-crisis-twitter-dataset-1-2-m-rows |
| Format | Daily CSV tweet records |
| Raw rows scanned | 70,876,102 |
| Timestamp range observed | 24 February 2022 – 14 June 2023 |

**Why we changed the proposed source:** the original one-pager proposed a Baselight-hosted mirror. When we attempted to access it during the external-data stage, it could not be accessed. The reason for the access failure was not independently established. We selected the BwandoWando Kaggle collection because it provided tweet-level text, timestamps, and user identifiers needed for weekly narrative analysis. A separate dataset considered during source selection contained military loss and casualty records rather than individual tweets, so it was not suitable for this tweet-level task.

The Kaggle page does not verify the role of every account. Dataset 3 therefore appears to represent a broader tweet/user population than Dataset 2's news-media population, but this is a population-level interpretation rather than a verified classification of every account.

**How we used it:** Dataset 3 was used for a reduced external evaluation and cross-population generalisation test. It was not treated as an exact replication because the source population and available variables differ from Dataset 2.

---

## 3. Dataset 3 preparation

| Step | Operation | Result |
| --- | --- | --- |
| 1 | Selected tweet ID, user ID, timestamp, language, text, retweet count, favourite count, and hashtags | Required tweet-level fields retained |
| 2 | Kept English-language records in `[2022-02-01, 2023-05-29)` | Study-period subset |
| 3 | Removed duplicate tweet IDs | One record per tweet ID |
| 4 | Removed IDs overlapping the original study data | Non-overlapping external population |
| 5 | Drew a deterministic uniform 3% sample with seed 42 from 43,697,878 eligible unique tweets | 1,310,936 unique tweet IDs |
| 6 | Applied the project's text cleaning and removed texts shorter than four words | 1,305,272 cleaned tweet records |
| 7 | Embedded each distinct cleaned text once and mapped its vector back to tweet records | 732,096 unique cleaned texts |

The final cleaned external panel covers 66 weeks, from the week beginning 21 February 2022 through the week beginning 22 May 2023. Each narrative-week cell used in panel construction had to meet the minimum 30-tweet threshold.

Repeated cleaned texts were retained at tweet level for weekly frequency analysis. The 573,176 repeated text instances may include retweets, copied posts, or viral repeated messages; they should not all be described as confirmed retweets.

---

## 4. Narrative representation

The project uses `sentence-transformers/all-MiniLM-L6-v2` to generate normalized 384-dimensional sentence embeddings.

For Dataset 3, the four narrative centroids from the primary study were kept frozen. No new K-means clustering was trained on the external dataset. Each cleaned text was assigned to the centroid with the highest cosine similarity.

| Label | Narrative |
| ---: | --- |
| 0 | Economy |
| 1 | Military |
| 2 | Humanitarian |
| 3 | Politics |

Dataset 3 tweet-level assignments:

| Narrative | Tweets | Share |
| --- | ---: | ---: |
| Economy | 166,728 | 12.77% |
| Military | 409,183 | 31.35% |
| Humanitarian | 404,231 | 30.97% |
| Politics | 325,130 | 24.91% |

The mean maximum centroid similarity for unique cleaned texts was 0.4835, and the mean margin between the top two centroid similarities was 0.0537. Dataset 3 labels were not manually validated. They should therefore be described as nearest-centroid assignments, not independently validated classifications.

---

## 5. Outcomes and competition features

### Outcomes

The weekly narrative-level panel measures:

- semantic drift between consecutive weekly narrative representations;
- entity turnover;
- change in semantic dispersion (fragmentation proxy);
- change in narrative attention share;
- change in similarity to the nearest rival narrative (merging proxy);
- weekly volume growth;
- change in user/channel breadth; and
- engagement change.

Some Dataset 3 measures are proxies rather than exact equivalents of the primary-study measures. Entity turnover is based on hashtags, not the original named-entity annotations. Breadth is based on user IDs and is not verified publisher/outlet breadth. Engagement is calculated from retweet and favourite counts. Country-based geographic spread, stance change, and sentiment change were unavailable in a suitable form and were not fabricated or replaced with zeros.

### Reduced external competition measure

The original competition framework has six components. Dataset 3 does not provide suitable variables for all six, particularly stance opposition, reach, and country-based audience overlap. The original six-component score and CPA were therefore not calculated for Dataset 3.

The external analysis uses a clearly labelled **reduced three-component competition measure**:

1. **Semantic similarity:** cosine similarity between weekly narrative centroids.
2. **Relative growth:** rival narrative's weekly log-volume growth minus the focal narrative's growth.
3. **User-network similarity:** cosine similarity between user-count vectors constructed from user IDs.

The three components are aggregated across rival narratives using rival weekly tweet counts as weights, standardized using expanding statistics, and averaged with equal component weights. This reduced score is not the original six-component competition score.

---

## 6. Experimental evaluation

The prediction task estimates next-week narrative outcomes from information available up to the prediction point.

The external evaluation uses chronological walk-forward testing. It compares available historical, content, and network predictors against models augmented with the three reduced competition features. The feature-based models include Ridge regression and Histogram-based Gradient Boosting (HGB).

The corrected external evaluation uses:

| Setting | Value |
| --- | --- |
| Test period start | 1 January 2023 |
| Prediction horizon | One week |
| Minimum training history | 40 observations |
| Held-out observations | 80 per outcome |
| Model refitting | Every four weeks |
| Placebo iterations | 500 |
| Placebo block size | Four weeks |
| Bootstrap resamples | 1,000 |
| Multiple-testing correction | Holm correction across 16 model–outcome tests |

An earlier preliminary external run used mismatched refitting schedules for real and placebo models. Those preliminary p-values are not used for inference. The corrected run uses the same four-week refitting schedule for real and placebo evaluations and is the authoritative external evaluation.

---

## 7. Main findings

### Primary study

For semantic drift, adding competition features produced gains of 5.3% for Ridge and 5.0% for HGB. The placebo p-value was 0.012, but the Holm-adjusted p-value was 0.084. The result is suggestive but did not meet the conventional 0.05 threshold after correction. A replication using disjoint tweets did not support the improvement.

### Dataset 3 external evaluation

For semantic drift, the corrected external evaluation found a Ridge gain of −14.03% and an HGB gain of −3.61%. Negative gains mean that adding the reduced competition features worsened prediction. No external result remained significant after Holm correction across the 16 model–outcome tests.

The results do not establish a robust, generalisable predictive benefit from the tested competition features. They also do not prove that competition has no effect on narrative evolution.

---

## 8. Project code and folder structure

| File or folder | Purpose |
| --- | --- |
| `ADA_Final_Codebook.ipynb` | Consolidated executed notebook and main code entry point |
| `config.py` | Project settings |
| `common.py` | Shared features and evaluation functions |
| `00_make_synthetic.py` – `11_summary.py` | Primary-study pipeline scripts |
| `12_adapt_external_batch.py` | Dataset 3 sample adaptation |
| `13_preprocess_external.py` | Dataset 3 text preprocessing |
| `14_embed_external.py` | Dataset 3 embedding generation |
| `24_external_baseline.py` | External baseline evaluation |
| `27_external_corrected_placebo.py` | Corrected external competition and placebo evaluation |
| `generate_dataset3_figures.py` | Dataset 3 figure generation |
| `data/` | Primary-study data |
| `data_ext/` | External/source data area |
| `data_rep/` | Dataset 3 sampled and processed data |
| `figs/`, `figs_primary_full/`, `figs_compctrl/` | Primary-study figures |
| `figs_rep/` | Dataset 3 external-evaluation figures |
| `results/`, `results_primary_full/`, `results_compctrl/` | Primary-study results |
| `results_rep/` | Corrected Dataset 3 external-evaluation results |

Exploratory scripts and backup folders may remain in the working directory. The corrected Step 27 output is authoritative for the external evaluation; preliminary Step 25 and Step 26 results are not used for final inference.

---

## 9. Reproducibility notes

1. Use the executed `ADA_Final_Codebook.ipynb` as the main notebook entry point.
2. Keep the supporting Python scripts, `config.py`, `common.py`, and `requirements.txt` alongside the notebook.
3. Preserve the relative folder structure and filenames expected by the scripts.
4. Dataset 3 sampling is deterministic and uses random seed 42.
5. Keep the original narrative centroids frozen when reproducing Dataset 3 assignments.
6. Reuse saved embeddings and processed files when available rather than unnecessarily rerunning expensive steps.
7. Keep primary-study files and Dataset 3 external-evaluation files in their separate folders.
8. Interpret the external results as a reduced external evaluation, not an exact replication.

---

## 10. References

1. S. Hakimov and G. S. Cheema, “Unveiling Global Narratives: A Multilingual Twitter Dataset of News Media on the Russo-Ukrainian Conflict,” Zenodo, 2023. https://doi.org/10.5281/zenodo.8043459
2. BwandoWando, “Ukraine–Russian Crisis Twitter Dataset (1.2M rows collection),” Kaggle. https://www.kaggle.com/datasets/bwandowando/ukraine-russian-crisis-twitter-dataset-1-2-m-rows
3. N. Reimers and I. Gurevych, “Sentence-BERT: Sentence embeddings using Siamese BERT-networks,” in *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing*, 2019.
