"""Export the tables of the original run (2021-2022), used by the tests as expected values.

Only the author can run this script, because it reads the original outputs, which are not
shared. Usage: python tools/export_original_results.py <folder with the original .pkl files>
"""
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model_transfer import config, summary  # noqa: E402
from model_transfer.data import load_tables, target_dates  # noqa: E402

EXPECTED_DIR = ROOT / "tests" / "expected"
original_dir = Path(sys.argv[1])

BASELINE_FILES = {
    "factors": "PLSRFactF",
    "r2_validation": "R2ValF",
    "rmse_validation": "RMSEValF",
    "pred_target": "OutDUnF",
}
TRANSFER_FILES = {
    "factors": "PLSRFactF",
    "r2_independent_validation": "R2IndVF",
    "rmse_independent_validation": "RMSEIndVF",
    "rmse_validation": "RMSEValF",
    "pred_target": "OutDUnF",
    "pred_oob": "EvalUnF",
}


def load(name):
    with open(original_dir / f"{name}.pkl", "rb") as file:
        return pickle.load(file)


def load_baseline(suffix):
    files = {key: load(f"{stem}_{suffix}") for key, stem in BASELINE_FILES.items()}
    return [
        [{key: np.asarray(files[key][dataset][trait]) for key in files} for trait in range(3)]
        for dataset in range(3)
    ]


def load_transfer(mode):
    suffix = f"allO_BootPLS_spike_opt_v106_{mode}"
    files = {key: load(f"{stem}_{suffix}") for key, stem in TRANSFER_FILES.items()}
    selected = load(f"selSampleRefAllF_{suffix}")
    best = load(f"bestSampleSelF_{suffix}")
    results = []
    for dataset, year in enumerate(config.SOURCE_YEARS):
        results.append([])
        for trait, name in enumerate(config.TRAITS):
            dates = target_dates(year)
            steps = [
                [
                    {key: np.asarray(files[key][dataset][trait][date][step]) for key in files}
                    for step in range(len(files["factors"][dataset][trait][date]))
                ]
                for date in range(len(dates))
            ]
            results[dataset].append(
                {
                    "source_year": year,
                    "trait": name,
                    "dates": dates,
                    "selected": [
                        [[int(row) for row in step] for step in per_date]
                        for per_date in selected[dataset][trait]
                    ],
                    "best": [int(value) for value in best[dataset][trait]],
                    "steps": steps,
                }
            )
    return results


def load_transfer_selection(transfer):
    suffix = "varSel_allO_BootPLS_spike_VIP_opt_v01_mean"
    keys = {key: value for key, value in TRANSFER_FILES.items() if not key.startswith("pred_")}
    files = {key: load(f"{stem}_{suffix}") for key, stem in keys.items()}
    decisions = load(f"CIselVIPRefF_{suffix}")
    results = []
    for dataset, per_trait in enumerate(transfer):
        results.append([])
        for trait, case in enumerate(per_trait):
            per_date = []
            for date in range(len(case["dates"])):
                outputs = {key: np.asarray(files[key][dataset][trait][date]) for key in files}
                outputs["bands"] = np.where(np.array(decisions[dataset][trait][date]) == "reject")[0]
                per_date.append(outputs)
            results[dataset].append(per_date)
    return results


def write(table, name):
    table.to_csv(EXPECTED_DIR / f"{name}.csv", index=False, float_format="%.10g")


EXPECTED_DIR.mkdir(exist_ok=True)
tables = load_tables()
uncertainty = load_transfer("stdS")
centroid = load_transfer("centroid")
transfer_selection = load_transfer_selection(uncertainty)
baseline = load_baseline("allO_BootPLS_v01")
baseline_selection = load_baseline("varSel_allO_BootPLS_v01")
for dataset, per_trait in enumerate(baseline_selection):
    retained = load("VIPRetainedF_varSel_allO_BootPLS_v01")[dataset]
    for trait, case in enumerate(per_trait):
        case["bands"] = np.asarray(retained[trait], dtype=int)

write(summary.baseline_table(tables, baseline), "baseline")
write(summary.baseline_table(tables, baseline_selection), "baseline_feature_selection")
write(summary.transfer_steps_table(uncertainty), "transfer_steps_uncertainty")
write(summary.transfer_steps_table(centroid), "transfer_steps_centroid")
write(summary.selected_samples_table(tables, uncertainty), "selected_samples_uncertainty")
write(summary.selected_samples_table(tables, centroid), "selected_samples_centroid")
write(summary.transfer_selection_table(uncertainty, transfer_selection), "transfer_feature_selection")
write(summary.retained_bands_table(baseline_selection, uncertainty, transfer_selection), "retained_bands")
