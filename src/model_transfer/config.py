"""Parameters and random seeds of the analysis published in the article (run v106)."""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

PLOT_TABLE = "20220215_Traits_Spectra_PlotLevel.csv"
TREATMENT_TABLE = "20220215_Traits_Spectra_TreatLevel.csv"

# Dry matter uses plot-level data; both N traits use treatment-level data.
TRAITS = ("DM_dt_ha", "N_perc_DM", "N_kg_ha")
SOURCE_YEARS = (2014, 2017)
DATES = ("20140515", "20141015", "20170510", "20170830", "20171026")
DOY = {"20140515": 135, "20141015": 288, "20170510": 130, "20170830": 242, "20171026": 299}
WAVELENGTHS_NM = np.arange(450, 941, 5)

N_REPETITIONS = 100
VALIDATION_FRACTION = 0.3
OUT_OF_BAG_FRACTION = 0.3333
MIN_FACTORS = 3
MAX_FACTORS = 10

MAX_TARGET_FRACTION = 0.5
N_CLUSTERS = 2
PCA_VARIANCE = 0.995
MIN_PCA_COMPONENTS = 3
UNCERTAINTY_INTERVAL = 0.2
VIP_ALPHA = 0.1

# Index 0 and 1 are the single-year datasets; index 2 is the benchmark with all dates.
SPLIT_SEEDS = (325685, 73256841, 568)


def repetition_seeds():
    """Return one list of bootstrap seeds per trait."""
    rng = np.random.RandomState(23577)
    return [
        [int(round(value)) for value in rng.uniform(low=0, high=1e9, size=N_REPETITIONS)]
        for _ in TRAITS
    ]


def pca_seeds():
    """Return one PCA seed per acquisition date."""
    return np.random.RandomState(965327).uniform(low=0, high=1e9, size=len(DATES)).astype(int)


def kmeans_seeds():
    """Return the k-means seed of each [source year][trait][target date]."""
    # The original run used 32-bit integers, the default on Windows at that time.
    rng = np.random.RandomState(27581425)
    return [
        [rng.randint(0, 1000000000, size=n_targets, dtype=np.int32) for _ in TRAITS]
        for n_targets in (3, 2)
    ]


def kmeans_step_seeds(seed, n_steps):
    """Return the k-means seed of each selection step."""
    return np.random.RandomState(int(seed)).randint(0, 1000000000, size=n_steps, dtype=np.int32)
