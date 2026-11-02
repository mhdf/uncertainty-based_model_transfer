"""Bootstrap PLSR for the 2014, 2017 and all-dates datasets (Section 2.4 of the article)."""
import numpy as np

from . import config, plsr
from .data import get_domains


def _fit_case(source, target, dataset, trait_index, bands=None):
    """Run the bootstrap for one dataset and trait, optionally on a subset of bands."""
    seeds = config.repetition_seeds()[trait_index]
    n_bands = source.X.shape[1]
    bands = np.arange(n_bands) if bands is None else np.asarray(bands)
    calibration, validation = plsr.split_calibration_validation(source, config.SPLIT_SEEDS[dataset])

    n_rep = config.N_REPETITIONS
    result = {
        "calibration": calibration,
        "validation": validation,
        "bands": bands,
        "factors": np.zeros(n_rep, dtype=int),
        "r2_oob": np.zeros(n_rep),
        "rmse_oob": np.zeros(n_rep),
        "r2_validation": np.zeros(n_rep),
        "rmse_validation": np.zeros(n_rep),
        "pred_fit": np.full((len(source), n_rep), np.nan),
        "pred_oob": np.full((len(source), n_rep), np.nan),
        "pred_validation": np.full((len(source), n_rep), np.nan),
        "pred_target": None if target is None else np.full((len(target), n_rep), np.nan),
        "vip": np.full((n_bands, n_rep), np.nan),
    }

    X, y = source.X[:, bands], source.y
    for rep, seed in enumerate(seeds):
        oob = plsr.out_of_bag_rows(source, calibration, seed)
        fit_rows = np.setdiff1d(calibration, oob)
        candidates = plsr.factor_candidates(len(fit_rows), len(bands))
        errors = [
            plsr.rmse(y[oob], plsr.predict(plsr.fit(X[fit_rows], y[fit_rows], n), X[oob]))
            for n in candidates
        ]
        n_factors = candidates[int(np.argmin(errors))]
        model = plsr.fit(X[fit_rows], y[fit_rows], n_factors)

        result["factors"][rep] = n_factors
        result["pred_fit"][fit_rows, rep] = plsr.predict(model, X[fit_rows])
        result["pred_oob"][oob, rep] = plsr.predict(model, X[oob])
        result["pred_validation"][validation, rep] = plsr.predict(model, X[validation])
        if target is not None:
            result["pred_target"][:, rep] = plsr.predict(model, target.X[:, bands])
        result["vip"][bands, rep] = plsr.vip(model)
        result["r2_oob"][rep], result["rmse_oob"][rep] = plsr.r2_rmse(
            y[oob], result["pred_oob"][oob, rep]
        )
        result["r2_validation"][rep], result["rmse_validation"][rep] = plsr.r2_rmse(
            y[validation], result["pred_validation"][validation, rep]
        )
    return result


def run_baseline(tables, bands=None, progress=print):
    """Return results[dataset][trait]; dataset 0 is 2014, 1 is 2017 and 2 is all dates.

    bands optionally holds the band subset of each [dataset][trait].
    """
    results = []
    for dataset in range(3):
        year = config.SOURCE_YEARS[dataset] if dataset < 2 else None
        per_trait = []
        for trait_index, trait in enumerate(config.TRAITS):
            progress(f"baseline: {year or 'all dates'}, {trait}")
            source, target = get_domains(tables, trait_index, year)
            subset = None if bands is None else bands[dataset][trait_index]
            per_trait.append(_fit_case(source, target, dataset, trait_index, subset))
        results.append(per_trait)
    return results
