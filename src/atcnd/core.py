"""
ATCND: Adaptive Topic and Cluster Number Determination

Uses binary search, golden section search, and ternary search within a
configurable range {K_min, ..., K_max} to efficiently find the optimal number
of topics (LDA/NMF) or clusters (K-Means).

Key insight: Instead of requiring the user to specify K manually, ATCND
takes a search range and applies efficient search strategies with quality
metrics to find the optimal K. Binary search achieves O(log(K_max - K_min))
evaluations compared to O(K_max - K_min) for grid search.

Supported models: LDA, NMF, K-Means
Supported metrics: coherence (c_v, u_mass), silhouette, perplexity,
                   reconstruction_error, combined
Search strategies: binary, golden_section, ternary, grid

Returns a ranked set of top-K candidates to handle plateaus, ties,
and multiple optima common in discrete objective landscapes.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Union
from dataclasses import dataclass, field


@dataclass
class ATCNDConfig:
    k_min: int = 2
    k_max: int = 50
    metric: str = "silhouette"
    coherence_type: str = "c_v"
    max_iter: int = 20
    tolerance: float = 0.001
    model_type: str = "kmeans"
    search_strategy: str = "binary"
    random_state: int = 42
    lda_max_iter: int = 200
    lda_doc_topic_prior: Optional[float] = None
    lda_topic_word_prior: Optional[float] = None
    nmf_init: str = "nndsvd"
    nmf_solver: str = "mu"
    nmf_max_iter: int = 500
    nmf_beta_loss: str = "frobenius"
    kmeans_n_init: int = 10
    kmeans_max_iter: int = 300
    kmeans_algorithm: str = "lloyd"
    n_candidates: int = 3
    verbose: bool = False


@dataclass
class ATCNDResult:
    optimal_k: int
    optimal_score: float
    all_scores: Dict[int, float]
    model: Any
    model_type: str
    search_history: List[Dict[str, Any]] = field(default_factory=list)
    vectorizer: Any = None
    feature_names: Any = None
    config: ATCNDConfig = None
    candidate_ks: List[int] = field(default_factory=list)
    candidate_scores: List[float] = field(default_factory=list)


def _preprocess_texts(texts: List[str], config: ATCNDConfig = None):
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

    if config is not None and config.model_type in ("kmeans", "nmf"):
        vec = TfidfVectorizer(
            max_df=0.95, min_df=2, stop_words="english",
            max_features=5000,
        )
    else:
        vec = CountVectorizer(
            max_df=0.95, min_df=2, stop_words="english",
            max_features=5000,
        )
    dtm = vec.fit_transform(texts)
    feature_names = vec.get_feature_names_out()
    return dtm, feature_names, vec


def _compute_coherence(
    model, feature_names, texts: List[str],
    coherence_type: str = "c_v", model_type: str = "lda",
):
    try:
        from gensim.models import CoherenceModel
        from gensim.corpora import Dictionary

        topics = _extract_topics(model, feature_names, model_type, top_n=20)
        tokenized = [t.lower().split() for t in texts]
        dictionary = Dictionary(tokenized)
        dictionary.filter_extremes(no_below=1, no_above=1.0)
        cm = CoherenceModel(
            topics=topics, texts=tokenized,
            dictionary=dictionary, coherence=coherence_type,
        )
        return cm.get_coherence()
    except Exception:
        return _compute_coherence_fast(model, feature_names, model_type)


def _compute_coherence_fast(model, feature_names, model_type="lda", top_n=10):
    topics = _extract_topics(model, feature_names, model_type, top_n)
    if not topics:
        return -1.0
    topic_dists = []
    for topic in topics:
        word_set = set(topic)
        topic_dists.append(len(word_set) / max(len(topic), 1))
    diversity = np.mean(topic_dists) if topic_dists else 0.0
    n_topics = len(topics)
    all_words = set()
    for topic in topics:
        all_words.update(topic)
    uniqueness = len(all_words) / (n_topics * top_n) if n_topics > 0 else 0
    return 0.5 * diversity + 0.5 * uniqueness


def _extract_topics(model, feature_names, model_type="lda", top_n=10):
    topics = []
    n_components = (
        model.n_components if hasattr(model, 'n_components')
        else model.n_clusters if hasattr(model, 'n_clusters')
        else 0
    )
    components = (
        model.components_ if hasattr(model, 'components_')
        else model.cluster_centers_ if hasattr(model, 'cluster_centers_')
        else None
    )
    if components is None:
        return topics
    for idx in range(n_components):
        comp = components[idx]
        if hasattr(comp, 'A1'):
            comp = comp.A1
        comp = np.asarray(comp).ravel()
        top_indices = comp.argsort()[-top_n:][::-1]
        topics.append([feature_names[i] for i in top_indices if i < len(feature_names)])
    return topics


def _compute_perplexity(model, dtm, model_type="lda"):
    if model_type == "lda":
        try:
            return model.perplexity(dtm)
        except Exception:
            try:
                log_likelihood = model.score(dtm)
                n_words = dtm.sum()
                return np.exp(-log_likelihood / n_words)
            except Exception:
                return 1e10
    else:
        try:
            W = model.transform(dtm)
            H = model.components_
            reconstruction = W @ H
            reconstruction = np.maximum(reconstruction, 1e-10)
            dtm_dense = dtm.toarray() if hasattr(dtm, 'toarray') else dtm
            frobenius = np.sum((dtm_dense - reconstruction) ** 2)
            return frobenius / (dtm.shape[0] * dtm.shape[1])
        except Exception:
            return 1e10


def _compute_silhouette(model, X, model_type="kmeans"):
    from sklearn.metrics import silhouette_score
    if hasattr(X, 'toarray'):
        X_dense = X.toarray()
    else:
        X_dense = np.asarray(X)

    if model_type == "kmeans":
        labels = model.predict(X) if hasattr(model, 'predict') else model.labels_
    elif model_type == "lda":
        doc_topic = model.transform(X)
        labels = np.argmax(doc_topic, axis=1)
    elif model_type == "nmf":
        W = model.transform(X)
        labels = np.argmax(W, axis=1)
    else:
        labels = model.predict(X) if hasattr(model, 'predict') else model.labels_

    n_unique = len(np.unique(labels))
    if n_unique < 2 or n_unique >= len(labels):
        return -1.0

    sample_size = min(5000, len(labels))
    if len(labels) > sample_size:
        idx = np.random.choice(len(labels), sample_size, replace=False)
        return silhouette_score(X_dense[idx], labels[idx], metric='euclidean')
    return silhouette_score(X_dense, labels, metric='euclidean')


def _evaluate_k(
    k: int, X, feature_names, texts: Optional[List[str]],
    config: ATCNDConfig,
) -> Tuple[float, Any]:
    from sklearn.decomposition import LatentDirichletAllocation, NMF
    from sklearn.cluster import KMeans

    if config.model_type == "lda":
        model = LatentDirichletAllocation(
            n_components=k,
            max_iter=config.lda_max_iter,
            learning_method="online",
            random_state=config.random_state,
            doc_topic_prior=config.lda_doc_topic_prior,
            topic_word_prior=config.lda_topic_word_prior,
        )
        model.fit(X)
    elif config.model_type == "nmf":
        model = NMF(
            n_components=k,
            init=config.nmf_init,
            solver=config.nmf_solver,
            max_iter=config.nmf_max_iter,
            beta_loss=config.nmf_beta_loss,
            random_state=config.random_state,
        )
        model.fit(X)
    elif config.model_type == "kmeans":
        X_dense = X.toarray() if hasattr(X, 'toarray') else np.asarray(X)
        model = KMeans(
            n_clusters=k,
            n_init=config.kmeans_n_init,
            max_iter=config.kmeans_max_iter,
            random_state=config.random_state,
            algorithm=config.kmeans_algorithm,
        )
        model.fit(X_dense)
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")

    if config.metric == "coherence":
        if texts is None:
            score = _compute_coherence_fast(model, feature_names, config.model_type)
        else:
            score = _compute_coherence(
                model, feature_names, texts, config.coherence_type, config.model_type,
            )
    elif config.metric == "perplexity":
        perp = _compute_perplexity(model, X, config.model_type)
        score = -perp
    elif config.metric == "silhouette":
        score = _compute_silhouette(model, X, config.model_type)
    elif config.metric == "reconstruction":
        if config.model_type == "nmf":
            W = model.transform(X)
            H = model.components_
            recon = W @ H
            X_dense = X.toarray() if hasattr(X, 'toarray') else X
            score = -np.sum((X_dense - recon) ** 2)
        elif config.model_type == "lda":
            score = model.score(X)
        else:
            score = -model.inertia_
    elif config.metric == "combined":
        sil = _compute_silhouette(model, X, config.model_type)
        if texts is not None:
            coh = _compute_coherence(
                model, feature_names, texts, config.coherence_type, config.model_type,
            )
        else:
            coh = _compute_coherence_fast(model, feature_names, config.model_type)
        score = 0.5 * sil + 0.5 * coh
    else:
        raise ValueError(f"Unknown metric: {config.metric}")

    if config.verbose:
        print(f"  K={k}, metric={config.metric}, score={score:.6f}")

    return score, model


def _validate_config(config: ATCNDConfig):
    if config.k_min < 2:
        raise ValueError("k_min must be >= 2")
    if config.k_max <= config.k_min:
        raise ValueError("k_max must be > k_min")
    if config.model_type not in ("lda", "nmf", "kmeans"):
        raise ValueError(f"model_type must be lda, nmf, or kmeans, got {config.model_type}")
    if config.search_strategy not in ("binary", "golden_section", "ternary", "grid"):
        raise ValueError(f"search_strategy must be binary, golden_section, ternary, or grid")
    valid_metrics = ("coherence", "perplexity", "silhouette", "reconstruction", "combined")
    if config.metric not in valid_metrics:
        raise ValueError(f"metric must be one of {valid_metrics}, got {config.metric}")


def _update_candidates(
    k: int, score: float, model: Any,
    candidate_ks: List[int], candidate_scores: List[float],
    n_candidates: int,
):
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


def _make_result(best_k, best_score, all_scores, best_model, config,
                 vectorizer, feature_names, search_history,
                 candidate_ks, candidate_scores):
    return ATCNDResult(
        optimal_k=best_k,
        optimal_score=best_score,
        all_scores=all_scores,
        model=best_model,
        model_type=config.model_type,
        search_history=search_history,
        vectorizer=vectorizer,
        feature_names=feature_names,
        config=config,
        candidate_ks=list(candidate_ks),
        candidate_scores=list(candidate_scores),
    )


def atcnd_search(
    texts: Optional[List[str]] = None,
    X=None,
    config: Optional[ATCNDConfig] = None,
) -> ATCNDResult:
    if config is None:
        config = ATCNDConfig()

    _validate_config(config)

    if X is None and texts is not None:
        X, feature_names, vectorizer = _preprocess_texts(texts, config)
    elif X is not None:
        vectorizer = None
        if hasattr(X, 'shape'):
            feature_names = np.arange(X.shape[1]).astype(str)
        else:
            feature_names = np.arange(X.shape[1]).astype(str)
    else:
        raise ValueError("Either texts or X must be provided")

    strategy = config.search_strategy
    if strategy == "binary":
        return _binary_search(X, feature_names, texts, config, vectorizer)
    elif strategy == "golden_section":
        return _golden_section_search(X, feature_names, texts, config, vectorizer)
    elif strategy == "ternary":
        return _ternary_search(X, feature_names, texts, config, vectorizer)
    elif strategy == "grid":
        return _grid_search(X, feature_names, texts, config, vectorizer)
    else:
        raise ValueError(f"Unknown search_strategy: {strategy}")


def _binary_search(X, feature_names, texts, config, vectorizer):
    all_scores: Dict[int, float] = {}
    search_history: List[Dict[str, Any]] = []
    best_k = config.k_min
    best_score = -np.inf
    best_model = None
    candidate_ks: List[int] = []
    candidate_scores: List[float] = []

    def eval_k(k):
        if k in all_scores:
            return all_scores[k], None
        score, model = _evaluate_k(k, X, feature_names, texts, config)
        all_scores[k] = score
        return score, model

    score_l, model_l = eval_k(config.k_min)
    search_history.append({"iteration": 0, "k": config.k_min, "score": score_l, "phase": "boundary"})
    _update_candidates(config.k_min, score_l, model_l, candidate_ks, candidate_scores, config.n_candidates)

    score_r, model_r = eval_k(config.k_max)
    search_history.append({"iteration": 0, "k": config.k_max, "score": score_r, "phase": "boundary"})
    _update_candidates(config.k_max, score_r, model_r, candidate_ks, candidate_scores, config.n_candidates)

    if score_l >= score_r:
        best_k, best_score, best_model = config.k_min, score_l, model_l
    else:
        best_k, best_score, best_model = config.k_max, score_r, model_r

    left, right = config.k_min, config.k_max
    for it in range(1, config.max_iter + 1):
        if right - left <= 1:
            break
        mid = (left + right) // 2
        score_m, model_m = eval_k(mid)
        search_history.append({"iteration": it, "k": mid, "score": score_m, "phase": "binary_search"})
        _update_candidates(mid, score_m, model_m, candidate_ks, candidate_scores, config.n_candidates)
        if model_m is not None and score_m > best_score:
            best_score, best_k, best_model = score_m, mid, model_m
        elif score_m > best_score:
            best_score, best_k = score_m, mid

        score_mid_left, _ = eval_k(max(left, mid - 1))
        if score_m >= score_mid_left:
            left = mid
        else:
            right = mid

    delta = max(1, (config.k_max - config.k_min) // 8)
    k_lo = max(config.k_min, best_k - delta)
    k_hi = min(config.k_max, best_k + delta)
    for k in range(k_lo, k_hi + 1):
        if k in all_scores:
            continue
        score, model = _evaluate_k(k, X, feature_names, texts, config)
        all_scores[k] = score
        search_history.append({"iteration": it, "k": k, "score": score, "phase": "refinement"})
        _update_candidates(k, score, model, candidate_ks, candidate_scores, config.n_candidates)
        if score > best_score:
            best_score, best_k = score, k
            if model is not None:
                best_model = model

    if best_model is None:
        _, best_model = _evaluate_k(best_k, X, feature_names, texts, config)

    return _make_result(best_k, best_score, all_scores, best_model, config,
                        vectorizer, feature_names, search_history,
                        candidate_ks, candidate_scores)


def _golden_section_search(X, feature_names, texts, config, vectorizer):
    import math
    all_scores: Dict[int, float] = {}
    search_history: List[Dict[str, Any]] = []
    best_k = config.k_min
    best_score = -np.inf
    best_model = None
    candidate_ks: List[int] = []
    candidate_scores: List[float] = []

    phi = (1 + math.sqrt(5)) / 2
    left, right = config.k_min, config.k_max
    it = 0

    while right - left > 1 and it < config.max_iter:
        it += 1
        k1 = int(right - (right - left) / phi)
        k2 = int(left + (right - left) / phi)

        for k in [k1, k2]:
            if k not in all_scores:
                score, model = _evaluate_k(k, X, feature_names, texts, config)
                all_scores[k] = score
                _update_candidates(k, score, model, candidate_ks, candidate_scores, config.n_candidates)
                if score > best_score:
                    best_score, best_k = score, k
                    if model is not None:
                        best_model = model
            search_history.append({"iteration": it, "k": k, "score": all_scores[k], "phase": "golden_section"})

        if all_scores.get(k1, -np.inf) < all_scores.get(k2, -np.inf):
            left = k1
        else:
            right = k2

    if best_model is None:
        _, best_model = _evaluate_k(best_k, X, feature_names, texts, config)
    return _make_result(best_k, best_score, all_scores, best_model, config,
                        vectorizer, feature_names, search_history,
                        candidate_ks, candidate_scores)


def _ternary_search(X, feature_names, texts, config, vectorizer):
    all_scores: Dict[int, float] = {}
    search_history: List[Dict[str, Any]] = []
    best_k = config.k_min
    best_score = -np.inf
    best_model = None
    candidate_ks: List[int] = []
    candidate_scores: List[float] = []

    left, right = config.k_min, config.k_max
    it = 0

    while right - left > 2 and it < config.max_iter:
        it += 1
        m1 = left + (right - left) // 3
        m2 = right - (right - left) // 3

        for k in [m1, m2]:
            if k not in all_scores:
                score, model = _evaluate_k(k, X, feature_names, texts, config)
                all_scores[k] = score
                _update_candidates(k, score, model, candidate_ks, candidate_scores, config.n_candidates)
                if score > best_score:
                    best_score, best_k = score, k
                    if model is not None:
                        best_model = model
            search_history.append({"iteration": it, "k": k, "score": all_scores[k], "phase": "ternary_search"})

        if all_scores.get(m1, -np.inf) < all_scores.get(m2, -np.inf):
            left = m1
        else:
            right = m2

    for k in range(left, right + 1):
        if k not in all_scores:
            score, model = _evaluate_k(k, X, feature_names, texts, config)
            all_scores[k] = score
            _update_candidates(k, score, model, candidate_ks, candidate_scores, config.n_candidates)
            search_history.append({"iteration": it, "k": k, "score": score, "phase": "final_sweep"})
            if score > best_score:
                best_score, best_k = score, k
                if model is not None:
                    best_model = model

    if best_model is None:
        _, best_model = _evaluate_k(best_k, X, feature_names, texts, config)
    return _make_result(best_k, best_score, all_scores, best_model, config,
                        vectorizer, feature_names, search_history,
                        candidate_ks, candidate_scores)


def _grid_search(X, feature_names, texts, config, vectorizer):
    all_scores: Dict[int, float] = {}
    search_history: List[Dict[str, Any]] = []
    best_k = config.k_min
    best_score = -np.inf
    best_model = None
    candidate_ks: List[int] = []
    candidate_scores: List[float] = []

    for k in range(config.k_min, config.k_max + 1):
        score, model = _evaluate_k(k, X, feature_names, texts, config)
        all_scores[k] = score
        search_history.append({"iteration": k - config.k_min, "k": k, "score": score, "phase": "grid"})
        _update_candidates(k, score, model, candidate_ks, candidate_scores, config.n_candidates)
        if score > best_score:
            best_score, best_k = score, k
            best_model = model

    return _make_result(best_k, best_score, all_scores, best_model, config,
                        vectorizer, feature_names, search_history,
                        candidate_ks, candidate_scores)


def print_topics(result: ATCNDResult, top_n: int = 10):
    print(f"\n{'='*60}")
    print(f"ATCND Results ({result.model_type.upper()})")
    print(f"{'='*60}")
    print(f"Optimal K: {result.optimal_k}")
    print(f"Best score: {result.optimal_score:.4f}")
    if result.candidate_ks:
        print(f"Top candidates: {result.candidate_ks}")
        print(f"Top scores: {[f'{s:.4f}' for s in result.candidate_scores]}")
    print(f"\nSearch history ({len(result.search_history)} evaluations):")
    for entry in result.search_history:
        print(f"  K={entry['k']:3d}, score={entry['score']:.6f}, phase={entry['phase']}")
    print(f"\nTop {top_n} keywords per topic/cluster:")
    topics = _extract_topics(result.model, result.feature_names, result.model_type, top_n)
    for i, topic in enumerate(topics):
        print(f"  Topic {i}: {', '.join(topic)}")


def plot_search_curve(result: ATCNDResult, save_path: Optional[str] = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = sorted(result.all_scores.keys())
    scores = [result.all_scores[k] for k in ks]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ks, scores, "o-", color="#2196F3", linewidth=2, markersize=6, label="Score")
    ax.axvline(x=result.optimal_k, color="#F44336", linestyle="--",
               linewidth=2, label=f"Optimal K={result.optimal_k}")

    phases_colors = {"binary_search": "#4CAF50", "golden_section": "#FF9800",
                     "ternary_search": "#9C27B0", "grid": "#607D8B",
                     "refinement": "#E91E63", "boundary": "#795548",
                     "final_sweep": "#00BCD4"}
    for entry in result.search_history:
        phase = entry.get("phase", "unknown")
        color = phases_colors.get(phase, "#9E9E9E")
        ax.plot(entry["k"], entry["score"], "s", color=color, markersize=8, zorder=5)

    ax.set_xlabel("Number of Topics/Clusters (K)", fontsize=14)
    ax.set_ylabel(f"Score ({result.config.metric if result.config else 'metric'})", fontsize=14)
    ax.set_title(
        f"ATCND Search Curve ({result.model_type.upper()}, "
        f"strategy={result.config.search_strategy if result.config else '?'})\n"
        f"Optimal K = {result.optimal_k}",
        fontsize=16,
    )
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = save_path or "atcnd_search_curve.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()