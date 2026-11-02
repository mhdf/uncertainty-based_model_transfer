"""Model transfer through active sample selection (Section 2.5 of the article)."""
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

from . import config, plsr
from .data import band_columns, get_domains, target_dates

METRICS = (
    "oob",  # out-of-bag source samples
    "added",  # target samples added to the training data
    "validation",  # source validation samples
    "target",  # target samples not used for training
    "independent",  # target samples not used for training + out-of-bag source samples
    "independent_validation",  # target samples not used for training + source validation samples
)


def pca_scores(tables):
    """Return, per spatial level and date, the PCA scores and explained variance ratio."""
    seeds = config.pca_seeds()
    output = {}
    for level, table in tables.items():
        bands = band_columns(table)
        output[level] = {}
        for index, date in enumerate(config.DATES):
            spectra = table.loc[table["Date"] == date, bands].to_numpy()
            scaler = RobustScaler(with_centering=True, with_scaling=False).fit(spectra)
            spectra = scaler.transform(spectra)
            model = PCA(
                n_components=min(len(bands), len(spectra)),
                svd_solver="full",
                random_state=int(seeds[index]),
            ).fit(spectra)
            output[level][date] = (model.transform(spectra), model.explained_variance_ratio_)
    return output


def n_pca_components(variance_ratio):
    """Return the components reaching the cumulative variance threshold."""
    keep = np.where(np.cumsum(variance_ratio) >= config.PCA_VARIANCE)[0].min() + 1
    return max(int(keep), config.MIN_PCA_COMPONENTS)


def _kmeans_plus_plus(X, rng):
    """Return initial centers using the k-means++ seeding of scikit-learn before version 1.3."""
    n_samples = X.shape[0]
    n_trials = 2 + int(np.log(config.N_CLUSTERS))
    first = int(rng.randint(n_samples, dtype=np.int32))
    centers = [X[first]]
    closest = ((X - X[first]) ** 2).sum(axis=1)
    potential = closest.sum()
    for _ in range(1, config.N_CLUSTERS):
        draws = rng.random_sample(n_trials) * potential
        candidates = np.clip(np.searchsorted(np.cumsum(closest), draws), None, n_samples - 1)
        distances = ((X[candidates][:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
        distances = np.minimum(closest, distances)
        potentials = distances.sum(axis=1)
        best = int(np.argmin(potentials))
        potential, closest = potentials[best], distances[best]
        centers.append(X[candidates[best]])
    return np.array(centers)


def kmeans_labels(scores, seed):
    """Cluster the samples, keeping the best of ten k-means runs.

    scikit-learn 1.3 changed how k-means++ draws the first center, so the same seed can
    end in another clustering. The original seeding is kept to reproduce the published run.
    """
    rng = np.random.RandomState(int(seed))
    center = scores.mean(axis=0)
    best = None
    for _ in range(10):
        model = KMeans(
            n_clusters=config.N_CLUSTERS,
            init=_kmeans_plus_plus(scores - center, rng) + center,
            n_init=1,
            max_iter=300,
            tol=0.0001,
            algorithm="elkan",
        ).fit(scores)
        if best is None or model.inertia_ < best.inertia_:
            best = model
    return best.labels_


def select_in_clusters(scores, uncertainty, rows, seed, selection):
    """Pick one sample from each k-means cluster and return their row numbers."""
    labels = kmeans_labels(scores, seed)
    chosen = []
    for cluster in range(config.N_CLUSTERS):
        members = np.where(labels == cluster)[0]
        if selection == "centroid":
            centroid = scores[members].mean(axis=0, keepdims=True)
            position = int(np.argmin(cdist(scores[members], centroid)))
        else:
            position = int(np.nanargmax(uncertainty[members]))
        chosen.append(int(rows[members[position]]))
    return chosen


def choose_number_of_steps(uncertainty):
    """Return the selection step considered optimal for a target date.

    Keeps the steps within an interval around the average uncertainty, whose half-width
    is a fraction of the total uncertainty reduction. While no step is inside it, the
    highest uncertainty is dropped. The median of the kept steps is returned.
    """
    values = list(uncertainty)
    half_width = config.UNCERTAINTY_INTERVAL * (np.nanmax(values) - np.nanmin(values))
    remaining = values
    while True:
        center = np.nanmean(remaining)
        inside = [v for v in remaining if center - half_width <= v <= center + half_width]
        if inside:
            break
        remaining = [v for v in remaining if v != np.nanmax(remaining)]
    steps = np.where([value in inside for value in values])[0]
    return int(np.floor(np.nanmedian(steps)))


def published_factor_choice(rmse_source, rmse_added):
    """Return the position of the chosen number of factors.

    This keeps a quirk of the original code so that the published results are reproduced:
    the summed errors are ordered by decreasing error on the added samples, but the
    position of their minimum is then read as a position in the list of factor numbers.
    The two orders often coincide, because that error tends to fall as factors are added.
    """
    order = sorted(enumerate(rmse_added), key=lambda item: item[1], reverse=True)
    joint = [rmse_added[index] + rmse_source[index] for index, _ in order]
    return int(np.nanargmin(joint))


def fit_repetition(source, target, calibration, validation, oob, added, unused, bands):
    """Fit one bootstrap repetition of a transferred model and return its outputs."""
    fit_rows = np.setdiff1d(calibration, oob)
    X_source, X_target = source.X[:, bands], target.X[:, bands]
    X_train = np.vstack((X_source[fit_rows], X_target[added]))
    y_train = np.concatenate((source.y[fit_rows], target.y[added]))

    candidates = plsr.factor_candidates(len(y_train), len(bands))
    errors_source, errors_added = [], []
    for n_factors in candidates:
        model = plsr.fit(X_train, y_train, n_factors)
        errors_source.append(plsr.rmse(source.y[oob], plsr.predict(model, X_source[oob])))
        errors_added.append(plsr.rmse(target.y[added], plsr.predict(model, X_target[added])))
    n_factors = candidates[published_factor_choice(errors_source, errors_added)]
    model = plsr.fit(X_train, y_train, n_factors)

    pred = {
        "fit": plsr.predict(model, X_source[fit_rows]),
        "oob": plsr.predict(model, X_source[oob]),
        "validation": plsr.predict(model, X_source[validation]),
        "added": plsr.predict(model, X_target[added]),
        "target": plsr.predict(model, X_target[unused]),
    }
    y_target, y_oob, y_val = target.y[unused], source.y[oob], source.y[validation]
    scores = {
        "oob": plsr.r2_rmse(y_oob, pred["oob"]),
        "added": plsr.r2_rmse(target.y[added], pred["added"]),
        "validation": plsr.r2_rmse(y_val, pred["validation"]),
        "target": plsr.r2_rmse(y_target, pred["target"]),
        "independent": plsr.r2_rmse(
            np.append(y_target, y_oob), np.append(pred["target"], pred["oob"])
        ),
        "independent_validation": plsr.r2_rmse(
            np.append(y_target, y_val), np.append(pred["target"], pred["validation"])
        ),
    }
    return {
        "factors": n_factors,
        "fit_rows": fit_rows,
        "pred": pred,
        "scores": scores,
        "vip": plsr.vip(model),
    }


def empty_outputs(n_source, n_target, n_bands):
    """Create the arrays that hold one model across the bootstrap repetitions."""
    n_rep = config.N_REPETITIONS
    outputs = {f"r2_{name}": np.zeros(n_rep) for name in METRICS}
    outputs.update({f"rmse_{name}": np.zeros(n_rep) for name in METRICS})
    outputs["factors"] = np.zeros(n_rep, dtype=int)
    for name, n_rows in (
        ("pred_fit", n_source),
        ("pred_oob", n_source),
        ("pred_validation", n_source),
        ("pred_added", n_target),
        ("pred_target", n_target),
        ("vip", n_bands),
    ):
        outputs[name] = np.full((n_rows, n_rep), np.nan)
    return outputs


def store_repetition(outputs, rep, fitted, oob, validation, added, unused, bands):
    """Copy the outputs of one bootstrap repetition into the arrays of a model."""
    outputs["factors"][rep] = fitted["factors"]
    outputs["pred_fit"][fitted["fit_rows"], rep] = fitted["pred"]["fit"]
    outputs["pred_oob"][oob, rep] = fitted["pred"]["oob"]
    outputs["pred_validation"][validation, rep] = fitted["pred"]["validation"]
    outputs["pred_added"][added, rep] = fitted["pred"]["added"]
    outputs["pred_target"][unused, rep] = fitted["pred"]["target"]
    outputs["vip"][bands, rep] = fitted["vip"]
    for name in METRICS:
        outputs[f"r2_{name}"][rep], outputs[f"rmse_{name}"][rep] = fitted["scores"][name]


def training_rows(target, dates, selected, best, current):
    """Return the target rows used for training and the rows still unused.

    Previous dates contribute their optimal samples; the current date contributes all the
    samples selected so far. Later dates do not take part yet.
    """
    added, unused = [], []
    for index, date in enumerate(dates[: current + 1]):
        rows = np.where(target.dates == date)[0]
        steps = selected[index] if index == current else selected[index][: best[index] + 1]
        chosen = [row for step in steps for row in step]
        added.extend(chosen)
        unused.extend(row for row in rows if row not in chosen)
    return np.array(added, dtype=int), np.array(unused, dtype=int)


def step_uncertainty(outputs):
    """Return the prediction uncertainty that guides the choice of the number of samples."""
    total = 0.0
    for predictions in (outputs["pred_target"], outputs["pred_oob"]):
        rows = ~np.isnan(predictions[:, 0])
        total += np.nanmean(np.nanstd(predictions[rows], axis=1))
    return total


def _run_case(tables, baseline, scores, dataset, trait_index, selection, best_fixed, progress):
    year = config.SOURCE_YEARS[dataset]
    source, target = get_domains(tables, trait_index, year)
    level = "plot" if trait_index == 0 else "treatment"
    dates = target_dates(year)
    n_bands = source.X.shape[1]
    bands = np.arange(n_bands)
    seeds = config.repetition_seeds()[trait_index]
    kmeans_seeds = config.kmeans_seeds()[dataset][trait_index]
    calibration, validation = plsr.split_calibration_validation(source, config.SPLIT_SEEDS[dataset])
    start_uncertainty = np.nanstd(baseline[dataset][trait_index]["pred_target"], axis=1)

    selected, best, steps_all = [], [], []
    for date_index, date in enumerate(dates):
        progress(f"transfer ({selection}): source {year}, {config.TRAITS[trait_index]}, target {date}")
        rows = np.where(target.dates == date)[0]
        date_scores, variance_ratio = scores[level][date]
        date_scores = date_scores[:, : n_pca_components(variance_ratio)]

        max_samples = int(np.ceil(len(rows) * config.MAX_TARGET_FRACTION))
        max_samples -= max_samples % config.N_CLUSTERS
        n_steps = max_samples // config.N_CLUSTERS
        step_seeds = config.kmeans_step_seeds(kmeans_seeds[date_index], n_steps)

        selected.append([])
        steps = []
        for step in range(n_steps):
            taken = {row for chosen in selected[date_index] for row in chosen}
            available = np.array([row not in taken for row in rows])
            if step == 0:
                uncertainty = start_uncertainty[rows[available]]
            else:
                previous = steps[step - 1]["pred_target"][rows[available]]
                uncertainty = np.nanstd(previous, axis=1)
            selected[date_index].append(
                select_in_clusters(
                    date_scores[available], uncertainty, rows[available], step_seeds[step], selection
                )
            )

            added, unused = training_rows(target, dates, selected, best, date_index)
            outputs = empty_outputs(len(source), len(target), n_bands)
            for rep, seed in enumerate(seeds):
                oob = plsr.out_of_bag_rows(source, calibration, seed)
                fitted = fit_repetition(
                    source, target, calibration, validation, oob, added, unused, bands
                )
                store_repetition(outputs, rep, fitted, oob, validation, added, unused, bands)
            steps.append(outputs)

        if best_fixed is not None:
            best.append(best_fixed[date_index])
        else:
            best.append(choose_number_of_steps([step_uncertainty(item) for item in steps]))
        steps_all.append(steps)

    return {
        "source_year": year,
        "trait": config.TRAITS[trait_index],
        "calibration": calibration,
        "validation": validation,
        "dates": dates,
        "selected": selected,
        "best": best,
        "steps": steps_all,
    }


def run_transfer(tables, baseline, selection="uncertainty", best=None, cases=None, progress=print):
    """Return results[dataset][trait]; dataset 0 uses 2014 as source and 1 uses 2017.

    baseline holds the output of run_baseline, which gives the starting uncertainty.
    selection is "uncertainty" (largest uncertainty in each cluster) or "centroid" (closest
    to each cluster centroid). The centroid comparison takes best, the number of samples
    found with the uncertainty criterion. cases optionally limits the run to some
    (dataset, trait) pairs; the other positions are None.
    """
    if selection == "centroid" and best is None:
        raise ValueError("the centroid comparison needs the best steps of the uncertainty run")
    scores = pca_scores(tables)
    results = [[None] * len(config.TRAITS) for _ in config.SOURCE_YEARS]
    for dataset in range(len(config.SOURCE_YEARS)):
        for trait_index in range(len(config.TRAITS)):
            if cases is not None and (dataset, trait_index) not in cases:
                continue
            best_fixed = None if best is None else best[dataset][trait_index]
            results[dataset][trait_index] = _run_case(
                tables, baseline, scores, dataset, trait_index, selection, best_fixed, progress
            )
    return results
