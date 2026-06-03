"""
Adaptive strategy selection: analyze data characteristics and recommend
the best strategy + metric combination for K-Means clustering.

Heuristics derived from empirical benchmarks:
- High-dimensional dense data: silhouette + binary/predictive
- Low-dimensional well-separated: silhouette + fibonacci
- Overlapping clusters: silhouette_drop + binary
- Need parsimony: bic + golden_section
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from .search import search, estimate_k_n_clusters, SearchResult


@dataclass
class DataProfile:
    n_samples: int
    n_features: int
    sparsity: float
    intrinsic_dim: int
    separation_ratio: float
    skewness: float


@dataclass
class AdaptiveRecommendation:
    strategy: str
    metric: str
    confidence: float
    profile: DataProfile
    all_recommendations: List[Dict[str, Any]] = field(default_factory=list)


def profile_data(X) -> DataProfile:
    """Extract statistical profile from data matrix."""
    X_arr = np.asarray(X.toarray() if hasattr(X, 'toarray') else X)
    n_samples, n_features = X_arr.shape

    nnz = np.count_nonzero(X_arr)
    sparsity = 1.0 - nnz / max(n_samples * n_features, 1)

    from sklearn.decomposition import PCA
    max_components = min(n_features, n_samples, 50)
    pca = PCA(n_components=max_components, random_state=42)
    pca.fit(X_arr)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    intrinsic_dim = int(np.searchsorted(cumvar, 0.95) + 1)
    intrinsic_dim = max(1, min(intrinsic_dim, n_features))

    eigenvalues = pca.explained_variance_
    if len(eigenvalues) >= 2:
        ratio_first = eigenvalues[0] / max(eigenvalues[1], 1e-10)
        if ratio_first > 10:
            separation_ratio = 0.3
        elif ratio_first > 3:
            separation_ratio = 0.6
        else:
            separation_ratio = 0.9
    else:
        separation_ratio = 0.5

    if n_samples > 100:
        sample_idx = np.random.choice(n_samples, min(100, n_samples), replace=False)
        X_sample = X_arr[sample_idx]
    else:
        X_sample = X_arr

    norms = np.linalg.norm(X_sample, axis=1)
    if norms.std() > 0:
        skewness = float(np.mean(np.abs(norms - norms.mean()) ** 3) / (norms.std() ** 3 + 1e-10))
    else:
        skewness = 0.0

    return DataProfile(
        n_samples=n_samples,
        n_features=n_features,
        sparsity=sparsity,
        intrinsic_dim=intrinsic_dim,
        separation_ratio=separation_ratio,
        skewness=skewness,
    )


def _score_combination(strategy: str, metric: str, profile: DataProfile) -> float:
    """Score a strategy+metric combination for the given data profile. Returns 0-1."""
    score = 0.5

    # Dimension-based adjustments
    if profile.n_features > 30:
        # High-dimensional: silhouette works well (our benchmark confirms)
        if metric == "silhouette":
            score += 0.25
        elif metric == "silhouette_drop":
            score += 0.15
        elif metric == "bic":
            score -= 0.1  # BIC tends to overfit in high-dim
    elif profile.n_features <= 15:
        # Low-dimensional: alternative metrics help
        if metric == "silhouette_drop":
            score += 0.25  # Exact match on Wine (dim=13)
        elif metric == "bic":
            score += 0.1
        elif metric == "silhouette":
            score -= 0.05  # silhouette prefers K=2 on low-dim

    # Separation-based adjustments
    if profile.separation_ratio < 0.4:
        # Well-separated clusters
        if metric == "silhouette":
            score += 0.2
        if strategy in ("binary", "predictive"):
            score += 0.1  # Unimodal objective → fast convergers
    elif profile.separation_ratio > 0.7:
        # Overlapping clusters
        if metric == "silhouette_drop":
            score += 0.15
        if strategy in ("golden_section", "fibonacci"):
            score += 0.05  # More robust to non-unimodal

    # Sample size adjustments
    if profile.n_samples > 5000:
        # Large dataset: fewer evaluations matters more
        if strategy == "predictive":
            score += 0.15
        elif strategy == "binary":
            score += 0.1
        elif strategy == "grid":
            score -= 0.2
    elif profile.n_samples < 200:
        # Small dataset: can afford more evaluations
        if strategy == "fibonacci":
            score += 0.1  # Optimal worst-case for discrete
        elif strategy == "grid":
            score -= 0.05

    # Sparsity adjustments
    if profile.sparsity > 0.7:
        if metric == "bic":
            score += 0.1
        if strategy == "golden_section":
            score += 0.05

    # Strategy-metric synergy
    if strategy == "predictive" and metric == "silhouette":
        score += 0.05  # PCA hot-start + silhouette is validated
    if strategy == "fibonacci" and metric in ("silhouette", "silhouette_drop"):
        score += 0.05  # Robust to non-unimodal

    return max(0.0, min(1.0, score))


STRATEGIES = ["binary", "golden_section", "ternary", "fibonacci",
              "interpolation", "exponential", "predictive"]
METRICS = ["silhouette", "silhouette_knee", "bic", "combined", "silhouette_drop"]


def adaptive_select(X, k_min: int = 2, k_max: int = 50) -> AdaptiveRecommendation:
    """Analyze data and recommend the best strategy + metric combination.

    Returns an AdaptiveRecommendation with the top choice and a ranked list
    of all 35 strategy-metric combinations.
    """
    profile = profile_data(X)

    combinations = []
    for strategy in STRATEGIES:
        for metric in METRICS:
            score = _score_combination(strategy, metric, profile)
            combinations.append({
                "strategy": strategy,
                "metric": metric,
                "score": score,
            })

    combinations.sort(key=lambda c: c["score"], reverse=True)

    best = combinations[0]
    top_score = best["score"]
    second_score = combinations[1]["score"]
    confidence = top_score - second_score + 0.3  # baseline confidence
    confidence = max(0.1, min(1.0, confidence))

    return AdaptiveRecommendation(
        strategy=best["strategy"],
        metric=best["metric"],
        confidence=confidence,
        profile=profile,
        all_recommendations=combinations,
    )


def adaptive_search(
    X,
    k_min: int = 2,
    k_max: int = 50,
    model_class=None,
    param_name: str = "n_clusters",
    fit_kwargs: dict = None,
) -> SearchResult:
    """Automatically select best strategy+metric and run search.

    Uses data profile analysis to choose strategy and metric,
    then delegates to search_model with the recommended combination.
    """
    rec = adaptive_select(X, k_min=k_min, k_max=k_max)

    if model_class is None:
        from sklearn.cluster import KMeans
        model_class = KMeans

    from .adapters import search_model
    result = search_model(
        model_class=model_class,
        X=X,
        param_name=param_name,
        k_min=k_min,
        k_max=k_max,
        strategy=rec.strategy,
        metric=rec.metric,
        fit_kwargs=fit_kwargs,
    )

    result.strategy = f"adaptive({rec.strategy}+{rec.metric})"
    return result