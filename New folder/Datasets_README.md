# Datasets Used in the Project

## Dataset 2 – Unveiling Global Narratives

### Dataset Overview

**Title:** Unveiling Global Narratives: A Multilingual Twitter Dataset of News Media on the Russo-Ukrainian Conflict  
**Authors:** Sherzod Hakimov and Gullal S. Cheema  
**Source:** Zenodo  
**Dataset link:** https://zenodo.org/records/8043459  
**DOI:** https://doi.org/10.5281/zenodo.8043459  
**Coverage period:** February 2022 – May 2023  

The dataset contains 1,524,832 tweets in 60 languages collected from news-media channels worldwide covering the Russo-Ukrainian conflict. It includes tweet-level information such as language, news channel, country, stance, sentiment and named entities.

### How We Used Dataset 2

Dataset 2 was used for the primary study, *Competition-Conditioned Narrative Evolution in Online Social Networks*. We used the news-media posts to study how competing narratives evolved over time and whether competition-related signals helped predict subsequent narrative changes.

The data was prepared for analysis by retaining English-language posts in the study period and applying the project's text-cleaning procedure. This removed URLs, retweet markers and boilerplate text such as “read more”, and normalised hashtags, mentions and whitespace. Posts with fewer than four words after cleaning were excluded.

We generated sentence embeddings for the cleaned text using the `sentence-transformers/all-MiniLM-L6-v2` model. The resulting representations were used to assign posts to the project's four narrative groups using the established, frozen narrative centroids. The narrative groups were:

- **Economy**
- **Military**
- **Humanitarian**
- **Politics**

The processed records were organised into weekly narrative-level observations. These observations were used to measure narrative evolution through semantic drift, entity turnover, fragmentation, changes in narrative attention share and merging behaviour. We also examined changes in volume and audience breadth.

For the competition analysis, we constructed time-varying features describing relationships between narratives. These features were evaluated alongside historical, content and network predictors using walk-forward testing on later time periods. Baseline comparisons and placebo tests were used to assess whether competition information improved prediction.

### Dataset Fields Used

The processed project data included fields such as:

| Field | Purpose |
|---|---|
| `tweet_id` | Tweet identifier |
| `date` | Tweet creation date |
| `lang` | Tweet language |
| `channel` | News-media channel/account |
| `country` | Country information, where available |
| `verified` | Account verification indicator |
| `followers` | Follower information, where available |
| `text_en` | English text used for analysis |
| `stance` | Stance annotation |
| `sentiment` | Sentiment annotation |
| `entities` | Extracted entity information |
| `week` | Weekly time index |
| `narrative` | Assigned narrative group |

The availability of individual fields depends on the released and processed data. Missing values were treated as unavailable rather than interpreted as zero.

---

## Dataset 3 – Ukraine–Russian Crisis Twitter Dataset

### Dataset Overview

**Dataset title:** Ukraine–Russian Crisis Twitter Dataset (1.2M rows collection)  
**Publisher:** BwandoWando  
**Source:** Kaggle  
**Dataset link:** https://www.kaggle.com/datasets/bwandowando/ukraine-russian-crisis-twitter-dataset-1-2-m-rows  
**Coverage in the files examined:** 24 February 2022 – 14 June 2023  

Dataset 3 is a collection of tweet-level records related to the Ukraine–Russia conflict. The downloaded collection consisted of daily CSV files. The records included fields such as tweet ID, user ID, creation timestamp, language, tweet text, retweet count, favourite count and hashtags.

### Why We Used Dataset 3 Instead of the Dataset 3 Source Listed Initially

The original one-pager listed a Baselight-hosted mirror as the proposed third dataset:

https://baselight.app/u/kaggle/dataset/tariqsays_russiaukraine_conflict_twitter_dataset

When we attempted to access that mirror during the external-data stage, it could not be accessed. We therefore selected the BwandoWando Kaggle collection as an alternative source.

We selected this alternative because it provided tweet-level records with text, timestamps and user identifiers, which were required for the external analysis of weekly narrative evolution and user-level network proxies. Another dataset considered during source selection contained military loss and casualty records rather than individual tweets, so it was not suitable for this tweet-level analysis.

This was a practical data-access and suitability decision; it was not a change to the research question. The replacement source is documented as Dataset 3. Its independent collection provenance and the account type of every user are not fully established by the Kaggle page, so we do not assume that it represents the same news-media population as Dataset 2.

### How We Prepared Dataset 3

The raw collection contained approximately 70.88 million rows across the CSV files scanned. We applied the following steps to create the external evaluation sample:

1. **Field selection:** We used the tweet ID, user ID, timestamp, language, text, retweet count, favourite count and hashtag fields.
2. **Time and language filtering:** We retained English-language tweets within the study window, from 1 February 2022 up to (but not including) 29 May 2023.
3. **Duplicate and overlap removal:** Duplicate tweet IDs were removed. We also excluded IDs that overlapped with the original study data so that the external evaluation used non-overlapping tweets.
4. **Sampling:** From the eligible non-overlapping population of 43,697,878 unique tweets, we selected a deterministic uniform 3% sample using random seed 42. This produced 1,310,936 tweets.
5. **Text cleaning:** We applied the same text-cleaning procedure used in the original pipeline: URLs, retweet markers and “read more” boilerplate were removed; hashtags and mentions were normalised; whitespace was cleaned; and texts with fewer than four words were excluded.
6. **Text deduplication for embedding:** We embedded each distinct cleaned text once and mapped its embedding back to the corresponding tweet records.

After text cleaning, 1,305,272 tweet records remained, containing 732,096 unique cleaned texts. The final sample contained 1,310,936 unique tweet IDs before the short-text exclusion; the cleaned analysis file contained 1,305,272 records. The weekly panel covered 66 weeks, from the week beginning 21 February 2022 through the week beginning 22 May 2023.

### Narrative Assignment and External Evaluation

We used the same `sentence-transformers/all-MiniLM-L6-v2` embedding model and the original study's four frozen narrative centroids. We did not retrain the narrative clusters on Dataset 3. Each unique cleaned text was assigned to the nearest of the four existing centroids by cosine similarity. The four labels were retained as Economy, Military, Humanitarian and Politics.

We then built a weekly narrative-level panel and measured the available outcomes, including semantic drift, hashtag-derived entity turnover, fragmentation, narrative attention-share change, merging, volume growth and user/channel breadth.

Dataset 3 did not provide all variables required by the original six-component competition score. In particular, stance, sentiment, country-based overlap/geography and reach information were not available in a form suitable for reproducing the original measures. We therefore did not invent or impute these values. The external analysis used a clearly labelled **reduced three-component competition measure**, based on:

- semantic similarity between narrative centroids;
- relative weekly narrative growth; and
- similarity between user-count vectors, using user IDs as the channel/user proxy.

The reduced score is not identical to the original six-component score. Dataset 3 was therefore used as a **reduced external replication and cross-population generalisation test**, not as an exact replication of the primary study.

The corrected external evaluation used walk-forward testing, four-week block-shuffled placebo tests (500 iterations), bootstrap intervals (1,000 resamples) and Holm correction across 16 model–outcome tests. The reduced competition features worsened Ridge prediction of semantic drift by 14.0%; no result remained significant after Holm correction across the 16 external tests. These results do not support a robust predictive benefit from the reduced competition features on Dataset 3.

### Important Dataset 3 Interpretation Notes

- The Kaggle collection appears to cover a broader tweet/user population than the news-media dataset. The Kaggle source does not verify the role of every account, so this is treated as a population difference to consider, not as a definitive account-level classification.
- Hashtag-derived entity turnover is a proxy; it is not equivalent to the named-entity extraction used in the primary dataset.
- Repeated cleaned texts were retained at the tweet level for weekly counts and analysis. The 573,176 repeated text instances may include retweets, copied posts or viral repeated messages; they should not all be labelled as retweets.
- Dataset 3 does not provide all fields needed for the original competition score. The three-component score and its results must be interpreted as reduced external evidence.

## References

1. Hakimov, S., & Cheema, G. S. (2023). *Unveiling Global Narratives: A Multilingual Twitter Dataset of News Media on the Russo-Ukrainian Conflict* (Version v0.1). Zenodo. https://doi.org/10.5281/zenodo.8043459
2. BwandoWando. *Ukraine–Russian Crisis Twitter Dataset (1.2M rows)*. Kaggle. https://www.kaggle.com/datasets/bwandowando/ukraine-russian-crisis-twitter-dataset-1-2-m-rows
