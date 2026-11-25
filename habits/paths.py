"""Where the data and results of the current dataset live.

Every experiment runs on REFIT by default. With the environment variable
CAPSTONE_DATASET=ukdale the same scripts run on UK-DALE instead: its prepared
tables, London's solar output, and its own results folder.

    REFIT    20 homes around Loughborough (Murray et al., 2017)
    UK-DALE  5 homes in London (Kelly & Knottenbelt, 2015)
    CASAS    21 single-resident homes with motion sensors and activity labels
             (Cook et al., 2013), the data of Gad et al. (2025); no power data,
             so only the habit experiments run on it
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = os.environ.get("CAPSTONE_DATASET", "refit").lower()
if DATASET not in ("refit", "ukdale", "casas"):
    raise ValueError(f"CAPSTONE_DATASET must be 'refit', 'ukdale' or 'casas', not {DATASET!r}")

PROCESSED = ROOT / "data" / "processed" / DATASET
RESULTS = ROOT / "data" / "results" if DATASET == "refit" else ROOT / "data" / "results" / DATASET
LOCATION = {"refit": ("Loughborough", 52.77, -1.20), "ukdale": ("London", 51.51, -0.13),
            "casas": ("Pullman", 46.73, -117.18)}[DATASET]  # CASAS: Washington State; no solar experiments
