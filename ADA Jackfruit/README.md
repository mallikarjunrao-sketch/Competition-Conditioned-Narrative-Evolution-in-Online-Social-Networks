# Competition-Conditioned Narrative Evolution (The Outliers)
Run everything from this folder. Day-1 decisions live in config.py. Run `python patch_config.py` once (enables replication runs via NARR_RUN).

## Main run
| Step | Command |
|---|---|
| 1 load | `python 01_load_dataset.py --src dataset_zenodo.jsonl --out data/tweets.parquet` |
| 2 embed + cluster (GPU/Colab for the full data) | `python 02_embed_cluster.py --tweets data/tweets.parquet --backend st --k 4` |
| 3 validate (after hand-labelling + adjudication) | `python 03_label_agreement.py` |
| 4 panel + outcomes (5 dimensions, 7 measures, diffusion proxies) | `python 04_targets.py` |
| 5 baseline | `python 05_baseline.py --targets drift_db abs_sent_change abs_stance_change entity_turnover abs_frag_change abs_share_change abs_merge_change` |
| 6 competition pressure (6 components + CPA formula) | `python 06_pressure.py` |
| 7 main test, per outcome | `python 07_main_test.py --target drift_db --n-placebo 500` |
| 8 classification (AUC / F1 / transformation type) | `python 08_classification.py --n-placebo 200` |
| 11 final table with Holm correction | `python 11_summary.py` |

## Other runs
- Robustness window: add `--test-start 2022-11-01` (or 2023-03-01) to step 7.
- Transformation -> diffusion: `python 07_main_test.py --target d_volume --tag transformation->diffusion --extra drift_db,abs_sent_change,abs_stance_change,entity_turnover,abs_frag_change,abs_merge_change --extra2 ""` (also d_breadth, d_geo).
- Replication (disjoint non-English tweets): set NARR_RUN=_rep, then 01 `--lang nonen --out data_rep/tweets.parquet`, 02 `--lang nonen --backend st --frac 0.5 --centroids data/narrative_centroids.npy`, then 04, 06, 07.
- External dataset: 09_adapt_external.py, then as above with NARR_RUN=_ext.
- 00_make_synthetic.py makes fake data for testing the pipeline only.

Proxies: audience overlap = country mix of outlets; network proximity = shared outlets. True user-level versions need rehydrated Dataset 1 (not obtainable).
