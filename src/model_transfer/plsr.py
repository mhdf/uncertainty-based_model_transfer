"""Shared PLSR helpers: model fitting, scores, VIP and repeated data splits."""
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupShuffleSplit

from . import config


def fit(X, y, n_factors):
    """Fit a PLSR model with autoscaling."""
    return PLSRegression(n_components=n_factors, scale=True).fit(X, y.reshape(-1, 1))


def predict(model, X):
    return model.predict(X).ravel()


def r2_rmse(observed, predicted):
    """Return R2 (as 1 - SSE/SST) and RMSE."""
    sse = ((observed - predicted) ** 2).sum()
    sst = ((observed - observed.mean()) ** 2).sum()
    return 1 - sse / sst, np.sqrt(sse / len(observed))


def rmse(observed, predicted):
    return np.sqrt(((observed - predicted) ** 2).sum() / len(observed))


def factor_candidates(n_samples, n_bands):
    """Return the numbers of PLS factors to try for a training set."""
    limit = min(n_samples, n_bands, config.MAX_FACTORS)
    return list(range(config.MIN_FACTORS, limit + 1))


def vip(model):
    """Return the Variable Importance in the Projection of each band."""
    scores, weights, loadings = model.x_scores_, model.x_weights_, model.y_loadings_
    n_bands, n_factors = weights.shape
    explained = np.diag(scores.T @ scores @ loadings.T @ loadings).reshape(n_factors, -1)
    normalized = (weights / np.linalg.norm(weights, axis=0)) ** 2
    return np.sqrt(n_bands * (normalized @ explained).ravel() / explained.sum())


def split_calibration_validation(domain, seed):
    """Split the groups into calibration and validation row numbers."""
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=config.VALIDATION_FRACTION, random_state=seed
    )
    calibration, validation = next(splitter.split(domain.X, domain.y, domain.groups))
    return calibration, validation


def out_of_bag_rows(domain, calibration, seed):
    """Return the calibration rows left out of one bootstrap repetition."""
    n_dates = len(np.unique(domain.dates))
    size = int(len(calibration) * config.OUT_OF_BAG_FRACTION / n_dates)
    groups = [domain.groups[row] for row in calibration]
    splitter = GroupShuffleSplit(n_splits=1, test_size=size, random_state=seed)
    _, positions = next(splitter.split(domain.X[calibration], domain.y[calibration], groups))
    return calibration[positions]
