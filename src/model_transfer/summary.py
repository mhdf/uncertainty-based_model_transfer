"""Turn the outputs of each stage into small tables."""
import numpy as np
import pandas as pd
from scipy import stats

from . import config, plsr
from .data import get_domains, table_of_trait
from .transfer import step_uncertainty

DATASET_NAMES = ("2014", "2017", "all")


def _mode(values):
    return int(stats.mode(np.asarray(values), keepdims=False).mode)


def baseline_table(tables, baseline):
    """Summarize each dataset and trait on its validation set and on the other year."""
    rows = []
    for dataset, per_trait in enumerate(baseline):
        year = config.SOURCE_YEARS[dataset] if dataset < 2 else None
        for trait_index, case in enumerate(per_trait):
            row = {
                "dataset": DATASET_NAMES[dataset],
                "trait": config.TRAITS[trait_index],
                "factors_mode": _mode(case["factors"]),
                "r2_validation": np.mean(case["r2_validation"]),
                "rmse_validation": np.mean(case["rmse_validation"]),
                "rmse_validation_std": np.std(case["rmse_validation"]),
                "r2_other_year": np.nan,
                "rmse_other_year": np.nan,
                "rmse_other_year_std": np.nan,
            }
            if year is not None:
                _, target = get_domains(tables, trait_index, year)
                scores = np.array(
                    [plsr.r2_rmse(target.y, pred) for pred in case["pred_target"].T]
                )
                row["r2_other_year"] = scores[:, 0].mean()
                row["rmse_other_year"] = scores[:, 1].mean()
                row["rmse_other_year_std"] = scores[:, 1].std()
            rows.append(row)
    return pd.DataFrame(rows)


def _model_row(case, date, outputs):
    return {
        "source_year": case["source_year"],
        "trait": case["trait"],
        "target_date": date,
        "factors_mode": _mode(outputs["factors"]),
        "r2_independent": np.mean(outputs["r2_independent_validation"]),
        "rmse_independent": np.mean(outputs["rmse_independent_validation"]),
        "rmse_independent_std": np.std(outputs["rmse_independent_validation"]),
        "rmse_source_validation": np.mean(outputs["rmse_validation"]),
        "rmse_source_validation_std": np.std(outputs["rmse_validation"]),
    }


def _cases(results):
    return [case for per_trait in results for case in per_trait if case is not None]


def transfer_steps_table(results):
    """Summarize every selection step of a transfer run, for all target dates."""
    rows = []
    for case in _cases(results):
        for date_index, date in enumerate(case["dates"]):
            for step, outputs in enumerate(case["steps"][date_index]):
                row = _model_row(case, date, outputs)
                row["samples_added"] = (step + 1) * config.N_CLUSTERS
                row["selected_as_optimal"] = step == case["best"][date_index]
                row["uncertainty"] = step_uncertainty(outputs)
                rows.append(row)
    return pd.DataFrame(rows)


def optimal_ratios(case, date_index):
    """Compare the step selected for a target date with the steps of lowest error and spread.

    rmse_ratio divides the lowest average RMSE among all steps by the average RMSE of the
    selected step. std_ratio does the same with the standard deviation of the RMSE, which is
    what was printed in Figure 3 of the article.
    """
    steps = case["steps"][date_index]
    means = [np.mean(step["rmse_independent_validation"]) for step in steps]
    spreads = [np.std(step["rmse_independent_validation"]) for step in steps]
    best = case["best"][date_index]
    return {"rmse_ratio": np.min(means) / means[best], "std_ratio": np.min(spreads) / spreads[best]}


def optimal_ratios_table(results):
    """Tabulate optimal_ratios for all the target dates."""
    rows = []
    for case in _cases(results):
        for date_index, date in enumerate(case["dates"]):
            rows.append(
                {
                    "source_year": case["source_year"],
                    "trait": case["trait"],
                    "target_date": date,
                    **optimal_ratios(case, date_index),
                }
            )
    return pd.DataFrame(rows)


def selected_samples_table(tables, results):
    """List the target samples chosen at each selection step."""
    rows = []
    for case in _cases(results):
        table = table_of_trait(tables, config.TRAITS.index(case["trait"]))
        target = table[table["Year"] != case["source_year"]].reset_index(drop=True)
        for date_index, date in enumerate(case["dates"]):
            for step, chosen in enumerate(case["selected"][date_index]):
                for row_number in chosen:
                    rows.append(
                        {
                            "source_year": case["source_year"],
                            "trait": case["trait"],
                            "target_date": date,
                            "step": step + 1,
                            "target_row": row_number,
                            "treatment": target.loc[row_number, "Treatment"],
                            "replicate": target.loc[row_number, "Replicate"] if "Replicate" in target else "",
                        }
                    )
    return pd.DataFrame(rows)


def transfer_selection_table(transfer, selection):
    """Summarize the transferred models refitted with the retained bands."""
    rows = []
    for case, per_date in zip(_cases(transfer), [item for group in selection for item in group if item]):
        for date, outputs in zip(case["dates"], per_date):
            row = _model_row(case, date, outputs)
            row["n_bands"] = len(outputs["bands"])
            rows.append(row)
    return pd.DataFrame(rows)


def retained_bands_table(baseline_selection, transfer, transfer_selection):
    """List the wavelengths retained for each model."""
    rows = []
    for dataset, per_trait in enumerate(baseline_selection):
        for trait_index, case in enumerate(per_trait):
            rows.append(
                {
                    "model": DATASET_NAMES[dataset],
                    "trait": config.TRAITS[trait_index],
                    "target_date": "",
                    "wavelengths_nm": " ".join(map(str, config.WAVELENGTHS_NM[case["bands"]])),
                }
            )
    for dataset, per_trait in enumerate(transfer_selection):
        for trait_index, per_date in enumerate(per_trait):
            if per_date is None:
                continue
            dates = transfer[dataset][trait_index]["dates"]
            for date, outputs in zip(dates, per_date):
                rows.append(
                    {
                        "model": f"{DATASET_NAMES[dataset]}+",
                        "trait": config.TRAITS[trait_index],
                        "target_date": date,
                        "wavelengths_nm": " ".join(map(str, config.WAVELENGTHS_NM[outputs["bands"]])),
                    }
                )
    return pd.DataFrame(rows)


def write_csv(table, name, folder=config.RESULTS_DIR):
    folder.mkdir(parents=True, exist_ok=True)
    table.to_csv(folder / f"{name}.csv", index=False, float_format="%.10g")
