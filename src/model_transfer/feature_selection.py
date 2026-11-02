"""Feature selection through VIP confidence intervals (Section 2.6 of the article)."""
import numpy as np
from scipy import stats

from . import config, plsr
from .baseline import run_baseline
from .data import get_domains
from .transfer import empty_outputs, fit_repetition, store_repetition, training_rows


def vip_confidence_interval(vip_values, alpha=config.VIP_ALPHA):
    """Return the mean VIP of each band and the bounds of its Student's interval."""
    n_rep = vip_values.shape[1]
    mean = np.nanmean(vip_values, axis=1)
    deviation = np.sqrt(np.nansum((vip_values.T - mean) ** 2, axis=0) / (n_rep - 1))
    margin = stats.t.ppf(1 - alpha / 2, df=n_rep - 1) * deviation
    return mean, mean - margin, mean + margin


def retained_bands(vip_values):
    """Return the bands whose lower VIP confidence bound is above one."""
    _, lower, _ = vip_confidence_interval(vip_values)
    return np.where(lower > 1)[0]


def run_baseline_selection(tables, baseline, progress=print):
    """Refit the baseline models using only the retained bands."""
    bands = [[retained_bands(case["vip"]) for case in dataset] for dataset in baseline]
    return run_baseline(tables, bands=bands, progress=progress)


def run_transfer_selection(tables, transfer, cases=None, progress=print):
    """Refit each transferred model, at its optimal step, using only the retained bands.

    Returns results[dataset][trait][target date].
    """
    seeds_all = config.repetition_seeds()
    results = []
    for dataset, per_trait in enumerate(transfer):
        results.append([])
        for trait_index, case in enumerate(per_trait):
            if case is None or (cases is not None and (dataset, trait_index) not in cases):
                results[dataset].append(None)
                continue
            source, target = get_domains(tables, trait_index, case["source_year"])
            n_bands = source.X.shape[1]
            per_date = []
            for date_index, date in enumerate(case["dates"]):
                progress(
                    f"feature selection: source {case['source_year']}, {case['trait']}, target {date}"
                )
                best = case["best"][date_index]
                bands = retained_bands(case["steps"][date_index][best]["vip"])
                # The current date enters with its optimal samples only.
                selected = case["selected"][: date_index] + [
                    case["selected"][date_index][: best + 1]
                ]
                added, unused = training_rows(target, case["dates"], selected, case["best"], date_index)
                outputs = empty_outputs(len(source), len(target), n_bands)
                for rep, seed in enumerate(seeds_all[trait_index]):
                    oob = plsr.out_of_bag_rows(source, case["calibration"], seed)
                    fitted = fit_repetition(
                        source, target, case["calibration"], case["validation"], oob, added, unused, bands
                    )
                    store_repetition(outputs, rep, fitted, oob, case["validation"], added, unused, bands)
                outputs["bands"] = bands
                per_date.append(outputs)
            results[dataset].append(per_date)
    return results
