"""Run ONCE (python patch_config.py). Lets any step write to data_rep/, results_rep/ ... when the env var NARR_RUN=_rep is set,
so a replication run never overwrites your main run. Your other config values are untouched."""
s = open("config.py").read()
old = 'DATA, FIGS, RESULTS = ROOT / "data", ROOT / "figs", ROOT / "results"'
new = ('import os\n_RUN = os.environ.get("NARR_RUN", "")      # "" = main run; "_rep" = replication run\n'
       'DATA, FIGS, RESULTS = ROOT / f"data{_RUN}", ROOT / f"figs{_RUN}", ROOT / f"results{_RUN}"')
if "NARR_RUN" in s: print("config.py already patched")
elif old in s:
    open("config.py", "w").write(s.replace(old, new)); print("config.py patched")
else: print("could not find the DATA/FIGS/RESULTS line - edit config.py by hand (see the docstring)")
