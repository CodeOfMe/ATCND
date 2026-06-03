"""
Adapters: high-level helpers for specific libraries.
Wrap common "find optimal integer parameter" patterns from
numpy, scipy, sklearn, pandas, matplotlib into ATCND search calls.
"""

import numpy as np
from typing import Any, Callable, Optional, Tuple, Union
from .search import search, SearchResult


def search_model(
    model_class,
    X,
    param_name: str = "n_clusters",
    k_min: int = 2,
    k_max: int = 50,
    strategy: str = "binary",
    metric: Union[str, Callable] = "silhouette",
    fit_kwargs: dict = None,
    max_iter: int = 30,
    n_candidates: int = 3,
) -> SearchResult:
    fit_kwargs = fit_kwargs or {}

    def f(k):
        model = model_class(**{param_name: k}, **fit_kwargs)
        model.fit(X)
        if callable(metric):
            return metric(model, X)
        if metric == "silhouette":
            from sklearn.metrics import silhouette_score
            labels = model.predict(X) if hasattr(model, 'predict') else model.labels_
            X_dense = X.toarray() if hasattr(X, 'toarray') else np.asarray(X)
            return silhouette_score(X_dense, labels)
        if metric == "inertia":
            return -model.inertia_
        if metric == "reconstruction":
            if hasattr(model, 'score'):
                return model.score(X)
            if hasattr(model, 'inertia_'):
                return -model.inertia_
            return 0.0
        raise ValueError(f"Unknown metric: {metric}")

    return search(f, k_min=k_min, k_max=k_max, strategy=strategy,
                  max_iter=max_iter, n_candidates=n_candidates)


def search_neighbors(
    X,
    k_min: int = 1,
    k_max: int = 30,
    strategy: str = "binary",
    metric: Union[str, Callable] = "accuracy",
    y=None,
    max_iter: int = 30,
    n_candidates: int = 3,
) -> SearchResult:
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.model_selection import cross_val_score

    is_classifier = y is not None and len(np.unique(y)) <= len(y) // 2

    def f(k):
        if is_classifier:
            model = KNeighborsClassifier(n_neighbors=k)
        else:
            model = KNeighborsRegressor(n_neighbors=k)
        if y is not None:
            scores = cross_val_score(model, X, y, cv=5)
            return scores.mean()
        raise ValueError("y is required for KNN search")

    return search(f, k_min=k_min, k_max=k_max, strategy=strategy,
                  max_iter=max_iter, n_candidates=n_candidates)


def search_bins(
    data,
    k_min: int = 3,
    k_max: int = 100,
    strategy: str = "binary",
    method: str = "freedman",
    max_iter: int = 30,
    n_candidates: int = 3,
) -> SearchResult:
    def f(n_bins):
        counts, _ = np.histogram(data, bins=n_bins)
        if method == "freedman":
            from scipy.stats import entropy
            probs = counts / counts.sum() if counts.sum() > 0 else counts
            probs = probs[probs > 0]
            return entropy(probs)
        if method == "sturges":
            return -np.sum(np.diff(counts) ** 2)
        if method == "custom":
            return float(counts.std())
        raise ValueError(f"Unknown method: {method}")

    return search(f, k_min=k_min, k_max=k_max, strategy=strategy,
                  max_iter=max_iter, n_candidates=n_candidates)


def search_components(
    X,
    k_min: int = 1,
    k_max: int = 30,
    strategy: str = "binary",
    variance_threshold: float = 0.95,
    max_iter: int = 30,
    n_candidates: int = 3,
) -> SearchResult:
    from sklearn.decomposition import PCA

    def f(n_components):
        pca = PCA(n_components=n_components)
        pca.fit(X)
        return sum(pca.explained_variance_ratio_)

    result = search(f, k_min=k_min, k_max=k_max, strategy=strategy,
                     max_iter=max_iter, n_candidates=n_candidates)

    if variance_threshold is not None:
        cumvar = {k: result.all_scores[k] for k in sorted(result.all_scores)}
        for k in sorted(cumvar):
            if cumvar[k] >= variance_threshold:
                result.optimal_k = k
                result.optimal_score = cumvar[k]
                break

    return result


def search_knots(
    x, y,
    k_min: int = 3,
    k_max: int = 20,
    strategy: str = "binary",
    degree: int = 3,
    max_iter: int = 30,
    n_candidates: int = 3,
) -> SearchResult:
    from scipy.interpolate import BSpline, make_interp_spline
    from sklearn.metrics import mean_squared_error

    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    def f(n_knots):
        n_knots = max(degree + 2, n_knots)
        try:
            idx = np.linspace(0, len(x_sorted) - 1, n_knots, dtype=int)
            knots = x_sorted[idx]
            spl = make_interp_spline(x_sorted, y_sorted, k=degree, t=knots)
            y_pred = spl(x_sorted)
            return -mean_squared_error(y_sorted, y_pred)
        except Exception:
            return -1e10

    return search(f, k_min=k_min, k_max=k_max, strategy=strategy,
                  max_iter=max_iter, n_candidates=n_candidates)


def search_window(
    series,
    k_min: int = 2,
    k_max: int = 100,
    strategy: str = "binary",
    method: str = "smoothness",
    max_iter: int = 30,
    n_candidates: int = 3,
) -> SearchResult:
    s = np.asarray(series, dtype=float)

    def f(w):
        w = max(1, w)
        if method == "smoothness":
            kernel = np.ones(w) / w
            smoothed = np.convolve(s, kernel, mode='valid')
            if len(smoothed) < 3:
                return -1e10
            diff2 = np.diff(smoothed, 2)
            return -np.var(diff2)
        if method == "ssd":
            kernel = np.ones(w) / w
            smoothed = np.convolve(s, kernel, mode='valid')
            n = min(len(smoothed), len(s))
            return -np.sum((smoothed[:n] - s[:n]) ** 2) / w
        raise ValueError(f"Unknown method: {method}")

    return search(f, k_min=k_min, k_max=k_max, strategy=strategy,
                  max_iter=max_iter, n_candidates=n_candidates)


def search_param(
    objective: Callable[[int], float],
    k_min: int = 2,
    k_max: int = 50,
    strategy: str = "binary",
    max_iter: int = 30,
    n_candidates: int = 3,
) -> SearchResult:
    return search(objective, k_min=k_min, k_max=k_max, strategy=strategy,
                  max_iter=max_iter, n_candidates=n_candidates)