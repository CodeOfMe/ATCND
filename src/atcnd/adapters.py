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
    method: str = "aic",
    max_iter: int = 30,
    n_candidates: int = 3,
) -> SearchResult:
    from scipy.stats import norm

    def f(n_bins):
        counts, edges = np.histogram(data, bins=n_bins)
        if method == "aic":
            n = len(data)
            widths = np.diff(edges)
            if np.any(widths <= 0) or counts.sum() == 0:
                return -1e10
            densities = counts / (n * widths)
            loglik = np.sum(counts[counts > 0] * np.log(densities[counts > 0]))
            return loglik - n_bins
        if method == "balancing":
            if counts.sum() == 0 or len(counts) < 2:
                return -1e10
            probs = counts / counts.sum()
            uniform = 1.0 / len(counts)
            return -np.sum((probs - uniform) ** 2)
        if method == "freedman":
            from scipy.stats import entropy
            probs = counts / counts.sum() if counts.sum() > 0 else counts
            probs = probs[probs > 0]
            return -entropy(probs)
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
    from scipy.interpolate import make_interp_spline
    from sklearn.metrics import mean_squared_error

    order = np.argsort(x)
    x_sorted = np.asarray(x)[order]
    y_sorted = np.asarray(y)[order]
    n = len(x_sorted)

    def f(n_internal):
        n_internal = max(degree + 1, n_internal)
        try:
            n_total = 2 * (degree + 1) + n_internal
            if n_total > n:
                return -1e10
            idx = np.linspace(degree + 1, n - degree - 2, n_internal, dtype=int)
            idx = np.unique(idx)
            knots = x_sorted[idx]
            spl = make_interp_spline(x_sorted, y_sorted, k=degree, t=knots)
            y_pred = spl(x_sorted)
            mse = mean_squared_error(y_sorted, y_pred)
            penalty = n_internal / n
            return -(mse + 0.01 * penalty)
        except Exception:
            return -1e10

    return search(f, k_min=k_min, k_max=k_max, strategy=strategy,
                  max_iter=max_iter, n_candidates=n_candidates)


def search_window(
    series,
    k_min: int = 2,
    k_max: int = 100,
    strategy: str = "binary",
    method: str = "bic",
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
        if method == "bic":
            kernel = np.ones(w) / w
            smoothed = np.convolve(s, kernel, mode='valid')
            n = min(len(smoothed), len(s))
            ssd = np.sum((smoothed[:n] - s[:n]) ** 2)
            return -(ssd + w * np.log(len(s)))
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


def search_hidden(
    train_loader,
    input_dim: int,
    output_dim: int,
    k_min: int = 8,
    k_max: int = 512,
    strategy: str = "binary",
    max_iter: int = 30,
    n_candidates: int = 3,
    epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cpu",
) -> SearchResult:
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        raise ImportError("PyTorch is required for search_hidden. Install with: pip install atcnd[torch]")

    def f(n_hidden):
        model = nn.Sequential(
            nn.Linear(input_dim, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, output_dim),
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss() if output_dim > 1 else nn.MSELoss()
        model.train()
        for _ in range(epochs):
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss_fn(model(xb), yb).backward()
                opt.step()
        model.eval()
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                total_loss += loss_fn(model(xb), yb).item() * len(xb)
                n += len(xb)
        return -total_loss / max(n, 1)

    return search(f, k_min=k_min, k_max=k_max, strategy=strategy,
                  max_iter=max_iter, n_candidates=n_candidates)


def search_layers(
    train_loader,
    input_dim: int,
    output_dim: int,
    hidden_dim: int = 64,
    k_min: int = 1,
    k_max: int = 10,
    strategy: str = "binary",
    max_iter: int = 30,
    n_candidates: int = 3,
    epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cpu",
) -> SearchResult:
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        raise ImportError("PyTorch is required for search_layers. Install with: pip install atcnd[torch]")

    def f(n_layers):
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        model = nn.Sequential(*layers).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss() if output_dim > 1 else nn.MSELoss()
        model.train()
        for _ in range(epochs):
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss_fn(model(xb), yb).backward()
                opt.step()
        model.eval()
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                total_loss += loss_fn(model(xb), yb).item() * len(xb)
                n += len(xb)
        return -total_loss / max(n, 1)

    return search(f, k_min=k_min, k_max=k_max, strategy=strategy,
                  max_iter=max_iter, n_candidates=n_candidates)


def search_trees(
    X, y,
    k_min: int = 10,
    k_max: int = 500,
    strategy: str = "binary",
    cv: int = 5,
    max_iter: int = 30,
    n_candidates: int = 3,
    task: str = "auto",
) -> SearchResult:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import cross_val_score

    is_classifier = task == "classifier" or (task == "auto" and len(np.unique(y)) <= len(y) // 2)

    def f(n_trees):
        if is_classifier:
            model = RandomForestClassifier(n_estimators=n_trees, random_state=42, n_jobs=-1)
        else:
            model = RandomForestRegressor(n_estimators=n_trees, random_state=42, n_jobs=-1)
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy" if is_classifier else "r2")
        return scores.mean()

    return search(f, k_min=k_min, k_max=k_max, strategy=strategy,
                  max_iter=max_iter, n_candidates=n_candidates)


def search_dbscan_eps(
    X,
    eps_min: int = 1,
    eps_max: int = 100,
    strategy: str = "binary",
    max_iter: int = 30,
    n_candidates: int = 3,
) -> SearchResult:
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import silhouette_score

    def f(eps_scaled):
        eps = eps_scaled / 10.0
        model = DBSCAN(eps=eps, min_samples=5)
        labels = model.fit_predict(X)
        n_clusters = len(set(labels) - {-1})
        if n_clusters < 2:
            return -1.0
        n_unique = len(set(labels))
        if n_unique >= len(labels):
            return -1.0
        core_mask = labels != -1
        if core_mask.sum() < 2:
            return -1.0
        return silhouette_score(X[core_mask], labels[core_mask])

    return search(f, k_min=eps_min, k_max=eps_max, strategy=strategy,
                  max_iter=max_iter, n_candidates=n_candidates)


def search_gmm_components(
    X,
    k_min: int = 2,
    k_max: int = 20,
    strategy: str = "binary",
    max_iter: int = 30,
    n_candidates: int = 3,
) -> SearchResult:
    from sklearn.mixture import GaussianMixture

    def f(n_components):
        model = GaussianMixture(n_components=n_components, random_state=42)
        model.fit(X)
        return model.bic(X)

    result = search(f, k_min=k_min, k_max=k_max, strategy=strategy,
                    max_iter=max_iter, n_candidates=n_candidates)

    best_k = min(result.all_scores, key=lambda k: result.all_scores[k])
    result.optimal_k = best_k
    result.optimal_score = result.all_scores[best_k]
    return result


def search_dataframe_bins(
    df,
    column: str,
    k_min: int = 3,
    k_max: int = 100,
    strategy: str = "binary",
    method: str = "aic",
    max_iter: int = 30,
    n_candidates: int = 3,
) -> SearchResult:
    data = df[column].dropna().values

    def f(n_bins):
        counts, edges = np.histogram(data, bins=n_bins)
        if method == "aic":
            n = len(data)
            widths = np.diff(edges)
            if np.any(widths <= 0) or counts.sum() == 0:
                return -1e10
            densities = counts / (n * widths)
            loglik = np.sum(counts[counts > 0] * np.log(densities[counts > 0]))
            return loglik - n_bins
        if method == "balancing":
            if counts.sum() == 0 or len(counts) < 2:
                return -1e10
            probs = counts / counts.sum()
            uniform = 1.0 / len(counts)
            return -np.sum((probs - uniform) ** 2)
        if method == "entropy":
            from scipy.stats import entropy
            probs = counts / counts.sum() if counts.sum() > 0 else counts
            probs = probs[probs > 0]
            return -entropy(probs)
        if method == "std":
            return -np.std(counts)
        raise ValueError(f"Unknown method: {method}")

    return search(f, k_min=k_min, k_max=k_max, strategy=strategy,
                  max_iter=max_iter, n_candidates=n_candidates)


def search_rolling_window(
    series,
    k_min: int = 2,
    k_max: int = 100,
    strategy: str = "binary",
    method: str = "bic",
    max_iter: int = 30,
    n_candidates: int = 3,
) -> SearchResult:
    import pandas as pd
    s = pd.Series(series) if not isinstance(series, pd.Series) else series

    def f(window):
        window = max(1, window)
        if method == "smoothness":
            smoothed = s.rolling(window=window, center=True).mean().dropna()
            if len(smoothed) < 3:
                return -1e10
            diff2 = np.diff(smoothed.values, 2)
            return -np.var(diff2)
        if method == "bic":
            smoothed = s.rolling(window=window, center=True).mean().dropna()
            n = min(len(smoothed), len(s))
            ssd = np.sum((smoothed.values[:n] - s.values[:n]) ** 2)
            return -(ssd + window * np.log(len(s)))
        if method == "ssd":
            smoothed = s.rolling(window=window, center=True).mean().dropna()
            n = min(len(smoothed), len(s))
            return -np.sum((smoothed.values[:n] - s.values[:n]) ** 2) / window
        raise ValueError(f"Unknown method: {method}")

    return search(f, k_min=k_min, k_max=k_max, strategy=strategy,
                  max_iter=max_iter, n_candidates=n_candidates)


def search_nmf_topics(
    texts,
    k_min: int = 2,
    k_max: int = 30,
    strategy: str = "binary",
    metric: str = "coherence",
    max_iter: int = 30,
    n_candidates: int = 3,
    hot_start: Optional[int] = None,
) -> SearchResult:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import NMF
    from sklearn.metrics import silhouette_score

    vec = TfidfVectorizer(max_df=0.95, min_df=2, stop_words="english", max_features=5000)
    X = vec.fit_transform(texts)
    feature_names = vec.get_feature_names_out()

    def f(k):
        model = NMF(n_components=k, init="nndsvd", random_state=42, max_iter=500)
        model.fit(X)
        if metric == "coherence":
            topics = []
            for idx in range(k):
                comp = model.components_[idx]
                top_indices = comp.argsort()[-20:][::-1]
                topics.append([feature_names[i] for i in top_indices])
            try:
                from gensim.models import CoherenceModel
                from gensim.corpora import Dictionary
                tokenized = [t.lower().split() for t in texts]
                dictionary = Dictionary(tokenized)
                dictionary.filter_extremes(no_below=1, no_above=1.0)
                cm = CoherenceModel(topics=topics, texts=tokenized, dictionary=dictionary, coherence="c_v")
                return cm.get_coherence()
            except Exception:
                diversity = np.mean([len(set(t)) / max(len(t), 1) for t in topics])
                return diversity
        if metric == "silhouette":
            W = model.transform(X)
            labels = np.argmax(W, axis=1)
            X_dense = X.toarray()
            n_unique = len(set(labels))
            if n_unique < 2 or n_unique >= len(labels):
                return -1.0
            return silhouette_score(X_dense, labels)
        if metric == "reconstruction":
            W = model.transform(X)
            X_dense = X.toarray()
            return -np.sum((X_dense - W @ model.components_) ** 2)
        raise ValueError(f"Unknown metric: {metric}")

    return search(f, k_min=k_min, k_max=k_max, strategy=strategy,
                  max_iter=max_iter, n_candidates=n_candidates, hot_start=hot_start)