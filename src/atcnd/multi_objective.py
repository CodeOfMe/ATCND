"""
Multi-objective optimization: simultaneously optimize multiple quality
metrics (e.g., silhouette + BIC) and return Pareto-optimal K candidates.

Approach:
- Evaluate each metric independently via grid or structured search
- Normalize scores to [0, 1] range
- Compute weighted score or Pareto frontier
- Return ranked candidate set
"""

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from .search import search, grid_search, SearchResult
from .adapters import search_model


@dataclass
class MultiObjectiveResult:
    optimal_k: int
    optimal_score: float
    all_scores: Dict[int, float]
    per_metric_scores: Dict[str, Dict[int, float]]
    per_metric_results: Dict[str, SearchResult]
    pareto_ks: List[int] = field(default_factory=list)
    pareto_scores: Dict[str, List[float]] = field(default_factory=dict)
    normalized_scores: Dict[str, Dict[int, float]] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)
    search_history: List[Dict[str, Any]] = field(default_factory=list)
    strategy: str = ""


def _normalize_scores(scores: Dict[int, float]) -> Dict[int, float]:
    """Min-max normalize scores to [0, 1]."""
    if not scores:
        return {}
    vals = list(scores.values())
    vmin, vmax = min(vals), max(vals)
    if abs(vmax - vmin) < 1e-12:
        return {k: 1.0 for k in scores}
    # For metrics where lower is better (BIC), flip
    return {k: (v - vmin) / (vmax - vmin) for k, v in scores.items()}


def _normalize_scores_flipped(scores: Dict[int, float]) -> Dict[int, float]:
    """Normalize and flip: lower original → higher normalized."""
    if not scores:
        return {}
    vals = list(scores.values())
    vmin, vmax = min(vals), max(vals)
    if abs(vmax - vmin) < 1e-12:
        return {k: 1.0 for k in scores}
    return {k: (vmax - v) / (vmax - vmin) for k, v in scores.items()}


def _is_pareto_dominated(obj_a: np.ndarray, obj_b: np.ndarray) -> bool:
    """Return True if obj_a is dominated by obj_b (all objectives ≤, at least one <)."""
    return np.all(obj_b >= obj_a) and np.any(obj_b > obj_a)


def _pareto_frontier(ks: List[int], objective_matrix: np.ndarray) -> List[int]:
    """Find Pareto-optimal K values (non-dominated set)."""
    n = len(ks)
    pareto = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i != j and _is_pareto_dominated(objective_matrix[i], objective_matrix[j]):
                dominated = True
                break
        if not dominated:
            pareto.append(ks[i])
    return sorted(pareto)


METRIC_PROPS = {
    "silhouette": {"higher_better": True, "flip": False},
    "silhouette_knee": {"higher_better": True, "flip": False},
    "bic": {"higher_better": False, "flip": True},
    "combined": {"higher_better": True, "flip": False},
    "silhouette_drop": {"higher_better": True, "flip": False},
}


def multi_objective_search(
    model_class,
    X,
    metrics: List[str] = None,
    weights: Optional[Dict[str, float]] = None,
    param_name: str = "n_clusters",
    k_min: int = 2,
    k_max: int = 50,
    strategy: str = "grid",
    fit_kwargs: dict = None,
    pareto: bool = True,
) -> MultiObjectiveResult:
    """Run multi-objective K selection: evaluate multiple metrics simultaneously.

    Args:
        model_class: Sklearn-style model class (e.g., KMeans).
        X: Data matrix.
        metrics: List of metric names to optimize. Default: ["silhouette", "bic"].
        weights: Optional weight dict for weighted sum. Default: equal weights.
        param_name: Model parameter name for K.
        k_min, k_max: Search range.
        strategy: Search strategy for each metric evaluation.
        fit_kwargs: Additional keyword arguments for model.fit().
        pareto: Whether to compute Pareto frontier.

    Returns:
        MultiObjectiveResult with combined scores, per-metric breakdown,
        and optional Pareto-optimal K set.
    """
    if metrics is None:
        metrics = ["silhouette", "bic"]
    if fit_kwargs is None:
        fit_kwargs = {}

    if weights is None:
        weights = {m: 1.0 / len(metrics) for m in metrics}

    per_metric_results = {}
    per_metric_scores = {}
    normalized = {}

    for metric in metrics:
        result = search_model(
            model_class=model_class,
            X=X,
            param_name=param_name,
            k_min=k_min,
            k_max=k_max,
            strategy=strategy,
            metric=metric,
            fit_kwargs=fit_kwargs,
        )
        per_metric_results[metric] = result
        per_metric_scores[metric] = result.all_scores

        props = METRIC_PROPS.get(metric, {"higher_better": True, "flip": False})
        if props["flip"]:
            normalized[metric] = _normalize_scores_flipped(result.all_scores)
        else:
            normalized[metric] = _normalize_scores(result.all_scores)

    common_ks = set(per_metric_scores[metrics[0]].keys())
    for metric in metrics[1:]:
        common_ks &= set(per_metric_scores[metric].keys())
    common_ks = sorted(common_ks)

    combined_scores = {}
    for k in common_ks:
        total = 0.0
        for metric in metrics:
            w = weights.get(metric, 1.0 / len(metrics))
            total += w * normalized[metric].get(k, 0.0)
        combined_scores[k] = total

    best_k = max(combined_scores, key=combined_scores.get)
    best_score = combined_scores[best_k]

    pareto_ks = []
    pareto_scores = {}
    if pareto and len(metrics) >= 2 and common_ks:
        obj_matrix = np.zeros((len(common_ks), len(metrics)))
        for i, k in enumerate(common_ks):
            for j, metric in enumerate(metrics):
                obj_matrix[i, j] = normalized[metric].get(k, 0.0)
        pareto_ks = _pareto_frontier(common_ks, obj_matrix)
        pareto_scores = {}
        for metric in metrics:
            pareto_scores[metric] = [per_metric_scores[metric].get(k, 0.0) for k in pareto_ks]

    search_history = []
    for metric in metrics:
        for entry in per_metric_results[metric].search_history:
            search_history.append({**entry, "metric": metric})

    return MultiObjectiveResult(
        optimal_k=best_k,
        optimal_score=best_score,
        all_scores=combined_scores,
        per_metric_scores=per_metric_scores,
        per_metric_results=per_metric_results,
        pareto_ks=pareto_ks,
        pareto_scores=pareto_scores,
        normalized_scores=normalized,
        weights=weights,
        search_history=search_history,
        strategy=f"multi_objective({'+'.join(metrics)})",
    )