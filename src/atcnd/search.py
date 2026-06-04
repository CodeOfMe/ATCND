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


def _fib_sequence(n):
    fibs = [1, 1]
    while fibs[-1] < n:
        fibs.append(fibs[-1] + fibs[-2])
    return fibs


def fibonacci_search(
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

    def record(k, phase, it):
        score = eval_k(k)
        search_history.append({"iteration": it, "k": k, "score": score, "phase": phase})
        _update_candidates(k, score, candidate_ks, candidate_scores, n_candidates)
        return score

    N = k_max - k_min
    fibs = _fib_sequence(N + 1)
    m = 0
    while m < len(fibs) - 1 and fibs[m] < N + 1:
        m += 1
    if m >= len(fibs):
        m = len(fibs) - 1

    left, right = k_min, k_max
    it = 0

    if fibs[m] > 1:
        k1 = left + fibs[m - 2] - 1
        k2 = left + fibs[m - 1] - 1
        k1 = max(k_min, min(k1, k_max))
        k2 = max(k_min, min(k2, k_max))
        if k1 == k2:
            k2 = min(k2 + 1, k_max)
        s1 = record(k1, "fibonacci", it)
        s2 = record(k2, "fibonacci", it)
        if s1 > best_score:
            best_score, best_k = s1, k1
        if s2 > best_score:
            best_score, best_k = s2, k2

        for i in range(m - 2, 0, -1):
            it += 1
            if it > max_iter:
                break
            if s1 < s2:
                left = k1
                k1 = k2
                s1 = s2
                k2 = left + fibs[i] - 1
                k2 = max(left, min(k2, right))
                if k2 == k1:
                    k2 = min(k2 + 1, right)
                s2 = record(k2, "fibonacci", it)
                if s2 > best_score:
                    best_score, best_k = s2, k2
            else:
                right = k2
                k2 = k1
                s2 = s1
                k1 = right - fibs[i] + 1
                k1 = max(left, min(k1, right))
                if k1 == k2:
                    k1 = max(k1 - 1, left)
                s1 = record(k1, "fibonacci", it)
                if s1 > best_score:
                    best_score, best_k = s1, k1

    delta = max(1, (k_max - k_min) // 8)
    k_lo = max(k_min, best_k - delta)
    k_hi = min(k_max, best_k + delta)
    for k in range(k_lo, k_hi + 1):
        if k not in all_scores:
            s = record(k, "refinement", it)
            if s > best_score:
                best_score, best_k = s, k

    return SearchResult(
        optimal_k=best_k, optimal_score=best_score,
        all_scores=all_scores, search_history=search_history,
        candidate_ks=list(candidate_ks), candidate_scores=list(candidate_scores),
        strategy="fibonacci",
    )


def interpolation_search(
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

    def record(k, phase, it):
        score = eval_k(k)
        search_history.append({"iteration": it, "k": k, "score": score, "phase": phase})
        _update_candidates(k, score, candidate_ks, candidate_scores, n_candidates)
        return score

    s_min = record(k_min, "boundary", 0)
    s_max = record(k_max, "boundary", 0)
    if s_min > best_score:
        best_score, best_k = s_min, k_min
    if s_max > best_score:
        best_score, best_k = s_max, k_max

    left, right = k_min, k_max
    for it in range(1, max_iter + 1):
        if right - left <= 2:
            for k in range(left, right + 1):
                if k not in all_scores:
                    s = record(k, "refinement", it)
                    if s > best_score:
                        best_score, best_k = s, k
            break

        sl = all_scores.get(left, -np.inf)
        sr = all_scores.get(right, -np.inf)

        if abs(sr - sl) < 1e-12:
            mid = (left + right) // 2
        else:
            predicted = left + (right - left) * (best_score - sl) / (sr - sl) if sr != sl else (left + right) / 2
            mid = max(left + 1, min(right - 1, int(round(predicted))))

        mid = max(left, min(right, mid))
        if mid in all_scores:
            mid = (left + right) // 2
            if mid in all_scores:
                mid = left + 1
            if mid in all_scores and mid < right:
                mid = right - 1

        s_mid = record(mid, "interpolation", it)
        if s_mid > best_score:
            best_score, best_k = s_mid, mid

        s_left_neighbor = eval_k(max(left, mid - 1)) if mid - 1 >= left else -np.inf
        if mid - 1 >= left and (mid - 1) not in all_scores:
            s_left_neighbor = record(mid - 1, "interpolation", it)
        elif mid - 1 >= left:
            s_left_neighbor = all_scores[mid - 1]
            if s_left_neighbor > best_score:
                best_score, best_k = s_left_neighbor, mid - 1

        if s_mid >= s_left_neighbor:
            left = mid
        else:
            right = mid

    delta = max(1, (k_max - k_min) // 8)
    k_lo = max(k_min, best_k - delta)
    k_hi = min(k_max, best_k + delta)
    for k in range(k_lo, k_hi + 1):
        if k not in all_scores:
            s = record(k, "refinement", it)
            if s > best_score:
                best_score, best_k = s, k

    return SearchResult(
        optimal_k=best_k, optimal_score=best_score,
        all_scores=all_scores, search_history=search_history,
        candidate_ks=list(candidate_ks), candidate_scores=list(candidate_scores),
        strategy="interpolation",
    )


def exponential_search(
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

    def record(k, phase, it):
        score = eval_k(k)
        search_history.append({"iteration": it, "k": k, "score": score, "phase": phase})
        _update_candidates(k, score, candidate_ks, candidate_scores, n_candidates)
        return score

    it = 0
    s0 = record(k_min, "doubling", it)
    if s0 > best_score:
        best_score, best_k = s0, k_min

    bound = k_min + 1
    while bound < k_max:
        it += 1
        s_bound = record(min(bound, k_max), "doubling", it)
        if s_bound > best_score:
            best_score, best_k = s_bound, min(bound, k_max)
        if bound >= k_max or s_bound < best_score * 0.9:
            break
        bound = min(bound * 2, k_max)

    left = max(k_min, bound // 2)
    right = min(k_max, bound)

    for it2 in range(it + 1, max_iter + 1):
        if right - left <= 1:
            break
        mid = (left + right) // 2
        s_mid = record(mid, "binary_search", it2)
        if s_mid > best_score:
            best_score, best_k = s_mid, mid
        s_ml = eval_k(max(left, mid - 1))
        if mid - 1 >= left and (mid - 1) not in all_scores:
            s_ml = record(mid - 1, "binary_search", it2)
        elif mid - 1 >= left:
            s_ml = all_scores[mid - 1]
        if s_mid >= s_ml:
            left = mid
        else:
            right = mid
        it = it2

    delta = max(1, (k_max - k_min) // 8)
    k_lo = max(k_min, best_k - delta)
    k_hi = min(k_max, best_k + delta)
    for k in range(k_lo, k_hi + 1):
        if k not in all_scores:
            s = record(k, "refinement", it)
            if s > best_score:
                best_score, best_k = s, k

    return SearchResult(
        optimal_k=best_k, optimal_score=best_score,
        all_scores=all_scores, search_history=search_history,
        candidate_ks=list(candidate_ks), candidate_scores=list(candidate_scores),
        strategy="exponential",
    )


STRATEGIES = {
    "binary": binary_search,
    "golden_section": golden_section_search,
    "ternary": ternary_search,
    "fibonacci": fibonacci_search,
    "interpolation": interpolation_search,
    "exponential": exponential_search,
    "predictive": None,
    "grid": grid_search,
}


def _parabolic_peak(k1, s1, k2, s2, k3, s3):
    denom = (k1 - k2) * (k1 - k3) * (k2 - k3)
    if abs(denom) < 1e-12:
        return k2
    a = (k3 * (s2 - s1) + k2 * (s1 - s3) + k1 * (s3 - s2)) / denom
    if abs(a) < 1e-12:
        return k2
    b = (s1 - s2) * (k1 - k2) - a * (k1**2 - k2**2)
    b /= (k1 - k2) if abs(k1 - k2) > 1e-12 else 1e-12
    return -b / (2 * a)


def predictive_search(
    f: Callable[[int], float],
    k_min: int = 2,
    k_max: int = 50,
    hot_start: Optional[int] = None,
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

    def record(k, phase, it):
        score = eval_k(k)
        search_history.append({"iteration": it, "k": k, "score": score, "phase": phase})
        _update_candidates(k, score, candidate_ks, candidate_scores, n_candidates)
        return score

    it = 0
    N = k_max - k_min + 1

    if hot_start is not None and k_min + 1 <= hot_start <= k_max - 1:
        lo_probe = max(k_min, hot_start - 1)
        hi_probe = min(k_max, hot_start + 1)
        probe_points = [lo_probe, hot_start, hi_probe]
    elif hot_start is not None and k_min <= hot_start <= k_max:
        probe_points = [hot_start]
        other = hot_start + 1 if hot_start < k_max else hot_start - 1
        probe_points.append(other)
        if len(probe_points) < 3:
            probe_points.append((k_min + k_max) // 2)
    else:
        probe_points = [k_min, k_max]
        mid = (k_min + k_max) // 2
        probe_points.append(mid)
        q1 = k_min + (k_max - k_min) // 4
        q3 = k_min + 3 * (k_max - k_min) // 4
        for p in [q1, q3]:
            if p not in probe_points:
                probe_points.append(p)

    seen = set()
    for p in probe_points:
        if p in seen:
            continue
        seen.add(p)
        s = record(p, "probing", it)
        if s > best_score:
            best_score, best_k = s, p

    for it in range(1, max_iter + 1):
        sorted_ks = sorted(all_scores.keys())
        if len(sorted_ks) < 3:
            mid = (k_min + k_max) // 2
            if mid not in all_scores:
                s = record(mid, "binary_search", it)
                if s > best_score:
                    best_score, best_k = s, mid
            continue

        top3_ks = sorted(sorted_ks, key=lambda k: all_scores[k], reverse=True)[:3]
        top3_ks.sort()
        k1, k2, k3 = top3_ks
        s1, s2, s3 = all_scores[k1], all_scores[k2], all_scores[k3]

        predicted = _parabolic_peak(k1, s1, k2, s2, k3, s3)
        pred_k = int(round(predicted))
        pred_k = max(k_min, min(k_max, pred_k))

        if pred_k in all_scores:
            for offset in [1, -1, 2, -2, 3, -3]:
                alt = pred_k + offset
                if k_min <= alt <= k_max and alt not in all_scores:
                    pred_k = alt
                    break

        if pred_k in all_scores:
            break

        s_pred = record(pred_k, "predictive", it)
        if s_pred > best_score:
            best_score, best_k = s_pred, pred_k

        if len(all_scores) >= max(5, N * 0.3):
            break

        top_ks = sorted(all_scores.keys(), key=lambda k: all_scores[k], reverse=True)
        top_k = top_ks[0]
        gap = 0
        for neighbor in [top_k - 1, top_k + 1]:
            if k_min <= neighbor <= k_max and neighbor not in all_scores:
                gap += 1
        if gap == 0:
            all_neighbors = True
            for neighbor in range(max(k_min, top_k - 2), min(k_max, top_k + 2) + 1):
                if neighbor not in all_scores:
                    all_neighbors = False
                    break
            if all_neighbors:
                break

    delta = max(1, (k_max - k_min) // 8)
    k_lo = max(k_min, best_k - delta)
    k_hi = min(k_max, best_k + delta)
    for k in range(k_lo, k_hi + 1):
        if k not in all_scores:
            s = record(k, "refinement", it)
            if s > best_score:
                best_score, best_k = s, k

    return SearchResult(
        optimal_k=best_k, optimal_score=best_score,
        all_scores=all_scores, search_history=search_history,
        candidate_ks=list(candidate_ks), candidate_scores=list(candidate_scores),
        strategy="predictive",
    )


@dataclass
class SuggestedKRange:
    k_min: int
    k_max: int
    method: str
    rationale: Dict[str, Any]


def suggest_k_range(X, model_type="kmeans"):
    X_arr = np.asarray(X) if not hasattr(X, 'toarray') else np.asarray(X.toarray())
    n, d = X_arr.shape

    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(d, n, 50), random_state=42)
    pca.fit(X_arr)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_components_95 = int(np.searchsorted(cumvar, 0.95) + 1)

    sqrt_n = int(np.sqrt(n))
    log2_n = int(np.log2(n))

    eigenvalues = pca.explained_variance_
    ratio = eigenvalues[1:] / eigenvalues[:-1]
    pca_elbow = int(np.argmax(ratio < 0.5) + 1) if np.any(ratio < 0.5) else n_components_95

    if model_type in ("kmeans", "dbscan", "gmm"):
        upper_bounds = []
        upper_bounds.append(("pca_intrinsic_dim", n_components_95 * 2))
        upper_bounds.append(("sqrt_n", sqrt_n))
        upper_bounds.append(("pca_elbow_doubled", min(pca_elbow * 2, n // 5)))
        rationale = {name: val for name, val in upper_bounds}
        vals = [v for _, v in upper_bounds if v > 0]
        rec_k_max = int(len(vals) / sum(1.0 / v for v in vals) * 1.2) if vals else 10
        rec_k_max = max(rec_k_max, 5)
    else:
        sparsity = (X_arr > 0).sum() / max(n * d, 1)
        avg_doc_len = X_arr.sum(axis=1).mean()
        vocab_richness = (X_arr > 0).sum() / max(n * d, 1)
        topic_estimate = max(int(avg_doc_len * vocab_richness * 5), 3)
        upper_bounds = []
        upper_bounds.append(("pca_intrinsic_dim", n_components_95 * 2))
        upper_bounds.append(("sqrt_n", sqrt_n))
        upper_bounds.append(("topic_estimate_doubled", topic_estimate * 2))
        rationale = {name: val for name, val in upper_bounds}
        vals = [v for _, v in upper_bounds if v > 0]
        rec_k_max = int(len(vals) / sum(1.0 / v for v in vals) * 1.2) if vals else 10
        rec_k_max = max(rec_k_max, 5)

    return SuggestedKRange(
        k_min=2,
        k_max=rec_k_max,
        method=f"min-of-heuristics({model_type})",
        rationale=rationale,
    )


def estimate_k_n_clusters(X, k_min=2, k_max=200):
    X_arr = np.asarray(X) if not hasattr(X, 'toarray') else np.asarray(X.toarray())
    n_samples, n_features = X_arr.shape
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(n_features, n_samples, 50), random_state=42)
    pca.fit(X_arr)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_components_95 = np.searchsorted(cumvar, 0.95) + 1
    eigenvalues = pca.explained_variance_
    ratio = eigenvalues[1:] / eigenvalues[:-1]
    elbow = np.argmax(ratio < 0.5) + 1 if np.any(ratio < 0.5) else n_components_95
    k_est = int(np.clip(elbow + 1, k_min, k_max))
    return k_est


def estimate_k_n_topics(X, k_min=2, k_max=200):
    X_arr = np.asarray(X) if not hasattr(X, 'toarray') else np.asarray(X.toarray())
    n_docs, n_terms = X_arr.shape
    avg_doc_len = X_arr.sum(axis=1).mean()
    vocab_richness = (X_arr > 0).sum() / max(n_docs * n_terms, 1)
    k_est = int(np.clip(avg_doc_len * vocab_richness * 5, k_min, k_max))
    if k_est < 3:
        _, n_nonzero = np.where(X_arr.T > 0)
        topic_cooccurrence = len(n_nonzero) / max(n_docs, 1) * 2
        k_est = int(np.clip(topic_cooccurrence, k_min, k_max))
    return k_est


STRATEGIES = {
    "binary": binary_search,
    "golden_section": golden_section_search,
    "ternary": ternary_search,
    "fibonacci": fibonacci_search,
    "interpolation": interpolation_search,
    "exponential": exponential_search,
    "predictive": predictive_search,
    "grid": grid_search,
}


def search(
    f: Callable[[int], float],
    k_min: int = 2,
    k_max: int = 50,
    strategy: str = "binary",
    max_iter: int = 30,
    n_candidates: int = 3,
    hot_start: Optional[int] = None,
) -> SearchResult:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}. Use one of {list(STRATEGIES.keys())}")
    fn = STRATEGIES[strategy]
    if strategy == "grid":
        return fn(f, k_min=k_min, k_max=k_max, n_candidates=n_candidates)
    if strategy == "predictive":
        return fn(f, k_min=k_min, k_max=k_max, hot_start=hot_start,
                  max_iter=max_iter, n_candidates=n_candidates)
    return fn(f, k_min=k_min, k_max=k_max, max_iter=max_iter, n_candidates=n_candidates)