"""
Pure search algorithms for discrete integer optimization.
Decoupled from any model or metric — operates on any callable f: int -> float.
"""

import math
from typing import Callable, Dict, List, Any, Optional
from dataclasses import dataclass, field
import numpy as np


@dataclass
class SearchResult:
    optimal_k: int
    optimal_score: float
    all_scores: Dict[int, float]
    search_history: List[Dict[str, Any]] = field(default_factory=list)
    candidate_ks: List[int] = field(default_factory=list)
    candidate_scores: List[float] = field(default_factory=list)
    strategy: str = ""


def _update_candidates(k, score, candidate_ks, candidate_scores, n_candidates):
    if len(candidate_ks) < n_candidates:
        candidate_ks.append(k)
        candidate_scores.append(score)
    else:
        min_idx = int(np.argmin(candidate_scores))
        if score > candidate_scores[min_idx]:
            candidate_ks[min_idx] = k
            candidate_scores[min_idx] = score
    paired = sorted(zip(candidate_scores, candidate_ks), reverse=True)
    candidate_scores[:] = [s for s, _ in paired]
    candidate_ks[:] = [k for _, k in paired]


def binary_search(
    f: Callable[[int], float],
    k_min: int = 2,
    k_max: int = 50,
    max_iter: int = 30,
    n_candidates: int = 3,
) -> SearchResult:
    all_scores: Dict[int, float] = {}
    search_history: List[Dict[str, Any]] = []
    candidate_ks: List[int] = []
    candidate_scores: List[float] = []
    best_k = k_min
    best_score = -np.inf

    def eval_k(k):
        if k not in all_scores:
            all_scores[k] = f(k)
        return all_scores[k]

    score_l = eval_k(k_min)
    search_history.append({"iteration": 0, "k": k_min, "score": score_l, "phase": "boundary"})
    _update_candidates(k_min, score_l, candidate_ks, candidate_scores, n_candidates)

    score_r = eval_k(k_max)
    search_history.append({"iteration": 0, "k": k_max, "score": score_r, "phase": "boundary"})
    _update_candidates(k_max, score_r, candidate_ks, candidate_scores, n_candidates)

    if score_l >= score_r:
        best_k, best_score = k_min, score_l
    else:
        best_k, best_score = k_max, score_r

    left, right = k_min, k_max
    for it in range(1, max_iter + 1):
        if right - left <= 1:
            break
        mid = (left + right) // 2
        score_m = eval_k(mid)
        search_history.append({"iteration": it, "k": mid, "score": score_m, "phase": "binary_search"})
        _update_candidates(mid, score_m, candidate_ks, candidate_scores, n_candidates)
        if score_m > best_score:
            best_score, best_k = score_m, mid

        score_ml = eval_k(max(left, mid - 1))
        if score_m >= score_ml:
            left = mid
        else:
            right = mid

    delta = max(1, (k_max - k_min) // 8)
    k_lo = max(k_min, best_k - delta)
    k_hi = min(k_max, best_k + delta)
    for k in range(k_lo, k_hi + 1):
        if k in all_scores:
            continue
        score = eval_k(k)
        search_history.append({"iteration": it, "k": k, "score": score, "phase": "refinement"})
        _update_candidates(k, score, candidate_ks, candidate_scores, n_candidates)
        if score > best_score:
            best_score, best_k = score, k

    return SearchResult(
        optimal_k=best_k, optimal_score=best_score,
        all_scores=all_scores, search_history=search_history,
        candidate_ks=list(candidate_ks), candidate_scores=list(candidate_scores),
        strategy="binary",
    )


def golden_section_search(
    f: Callable[[int], float],
    k_min: int = 2,
    k_max: int = 50,
    max_iter: int = 30,
    n_candidates: int = 3,
) -> SearchResult:
    all_scores: Dict[int, float] = {}
    search_history: List[Dict[str, Any]] = []
    candidate_ks: List[int] = []
    candidate_scores: List[float] = []
    best_k = k_min
    best_score = -np.inf
    phi = (1 + math.sqrt(5)) / 2

    left, right = k_min, k_max
    it = 0

    while right - left > 1 and it < max_iter:
        it += 1
        k1 = int(right - (right - left) / phi)
        k2 = int(left + (right - left) / phi)

        for k in [k1, k2]:
            if k not in all_scores:
                all_scores[k] = f(k)
            _update_candidates(k, all_scores[k], candidate_ks, candidate_scores, n_candidates)
            if all_scores[k] > best_score:
                best_score, best_k = all_scores[k], k
            search_history.append({"iteration": it, "k": k, "score": all_scores[k], "phase": "golden_section"})

        if all_scores.get(k1, -np.inf) < all_scores.get(k2, -np.inf):
            left = k1
        else:
            right = k2

    return SearchResult(
        optimal_k=best_k, optimal_score=best_score,
        all_scores=all_scores, search_history=search_history,
        candidate_ks=list(candidate_ks), candidate_scores=list(candidate_scores),
        strategy="golden_section",
    )


def ternary_search(
    f: Callable[[int], float],
    k_min: int = 2,
    k_max: int = 50,
    max_iter: int = 30,
    n_candidates: int = 3,
) -> SearchResult:
    all_scores: Dict[int, float] = {}
    search_history: List[Dict[str, Any]] = []
    candidate_ks: List[int] = []
    candidate_scores: List[float] = []
    best_k = k_min
    best_score = -np.inf

    left, right = k_min, k_max
    it = 0

    while right - left > 2 and it < max_iter:
        it += 1
        m1 = left + (right - left) // 3
        m2 = right - (right - left) // 3

        for k in [m1, m2]:
            if k not in all_scores:
                all_scores[k] = f(k)
            _update_candidates(k, all_scores[k], candidate_ks, candidate_scores, n_candidates)
            if all_scores[k] > best_score:
                best_score, best_k = all_scores[k], k
            search_history.append({"iteration": it, "k": k, "score": all_scores[k], "phase": "ternary_search"})

        if all_scores.get(m1, -np.inf) < all_scores.get(m2, -np.inf):
            left = m1
        else:
            right = m2

    for k in range(left, right + 1):
        if k not in all_scores:
            all_scores[k] = f(k)
            search_history.append({"iteration": it, "k": k, "score": all_scores[k], "phase": "final_sweep"})
            _update_candidates(k, all_scores[k], candidate_ks, candidate_scores, n_candidates)
            if all_scores[k] > best_score:
                best_score, best_k = all_scores[k], k

    return SearchResult(
        optimal_k=best_k, optimal_score=best_score,
        all_scores=all_scores, search_history=search_history,
        candidate_ks=list(candidate_ks), candidate_scores=list(candidate_scores),
        strategy="ternary",
    )


def grid_search(
    f: Callable[[int], float],
    k_min: int = 2,
    k_max: int = 50,
    n_candidates: int = 3,
) -> SearchResult:
    all_scores: Dict[int, float] = {}
    search_history: List[Dict[str, Any]] = []
    candidate_ks: List[int] = []
    candidate_scores: List[float] = []
    best_k = k_min
    best_score = -np.inf

    for k in range(k_min, k_max + 1):
        all_scores[k] = f(k)
        search_history.append({"iteration": k - k_min, "k": k, "score": all_scores[k], "phase": "grid"})
        _update_candidates(k, all_scores[k], candidate_ks, candidate_scores, n_candidates)
        if all_scores[k] > best_score:
            best_score, best_k = all_scores[k], k

    return SearchResult(
        optimal_k=best_k, optimal_score=best_score,
        all_scores=all_scores, search_history=search_history,
        candidate_ks=list(candidate_ks), candidate_scores=list(candidate_scores),
        strategy="grid",
    )


STRATEGIES = {
    "binary": binary_search,
    "golden_section": golden_section_search,
    "ternary": ternary_search,
    "grid": grid_search,
}


def search(
    f: Callable[[int], float],
    k_min: int = 2,
    k_max: int = 50,
    strategy: str = "binary",
    max_iter: int = 30,
    n_candidates: int = 3,
) -> SearchResult:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}. Use one of {list(STRATEGIES.keys())}")
    fn = STRATEGIES[strategy]
    if strategy == "grid":
        return fn(f, k_min=k_min, k_max=k_max, n_candidates=n_candidates)
    return fn(f, k_min=k_min, k_max=k_max, max_iter=max_iter, n_candidates=n_candidates)