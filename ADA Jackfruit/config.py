"""Single source of truth for the Day-1 decisions. Edit ONCE with your teammate, then freeze."""
from pathlib import Path

ROOT = Path(__file__).parent
import os
_RUN = os.environ.get("NARR_RUN", "")      # "" = main run; "_rep" = replication run
DATA, FIGS, RESULTS = ROOT / f"data{_RUN}", ROOT / f"figs{_RUN}", ROOT / f"results{_RUN}"
for _p in (DATA, FIGS, RESULTS):
    _p.mkdir(exist_ok=True)

SEED = 42
TIME_BIN = "W"                 # weekly bins (Monday start)
DELTA_BINS = 1                 # predict 1 bin ahead
EVENT_START = "2022-02-01"
EVENT_END = "2023-05-28"
TRAIN_END = "2022-09-30"       # narratives are defined ONLY on data up to here
TEST_START = "2023-01-01"      # held-out period for walk-forward evaluation
LANG = "en"                    # first pass: English only (581k tweets)
K_RANGE = range(4, 9)          # 4-8 narratives
MIN_TWEETS_CELL = 30           # min tweets in a narrative-week for stable centroids
STANCE_THRESH = 0.9            # dataset paper's rule
