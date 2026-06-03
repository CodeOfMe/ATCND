"""
ATCND: Adaptive Topic and Cluster Number Determination

Model-specific front-end that builds objective functions and delegates
to the pure search algorithms in search.py.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Union
from dataclasses import dataclass, field
from .search import search as pure_search, SearchResult, predictive_search, estimate_k_n_clusters, estimate_k_n_topics


@dataclass
class ATCNDConfig:
    k_min: int = 2
    k_max: int = 50
    metric: str = "silhouette"
    coherence_type: str = "c_v"
    max_iter: int = 20
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
        vec = TfidfVectorizer(max_df=0.95, min_df=2, stop_words="english", max_features=5000)
    else:
        vec = CountVectorizer(max_df=0.95, min_df=2, stop_words="english", max_features=5000)
    dtm = vec.fit_transform(texts)
    feature_names = vec.get_feature_names_out()
    return dtm, feature_names, vec


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
        comp = np.asarray(components[idx]).ravel()
        top_indices = comp.argsort()[-top_n:][::-1]
        topics.append([feature_names[i] for i in top_indices if i < len(feature_names)])
    return topics


def _compute_coherence(model, feature_names, texts, coherence_type="c_v", model_type="lda"):
    try:
        from gensim.models import CoherenceModel
        from gensim.corpora import Dictionary
        topics = _extract_topics(model, feature_names, model_type, top_n=20)
        tokenized = [t.lower().split() for t in texts]
        dictionary = Dictionary(tokenized)
        dictionary.filter_extremes(no_below=1, no_above=1.0)
        cm = CoherenceModel(topics=topics, texts=tokenized, dictionary=dictionary, coherence=coherence_type)
        return cm.get_coherence()
    except Exception:
        return _compute_coherence_fast(model, feature_names, model_type)


def _compute_coherence_fast(model, feature_names, model_type="lda", top_n=10):
    topics = _extract_topics(model, feature_names, model_type, top_n)
    if not topics:
        return -1.0
    topic_dists = [len(set(t)) / max(len(t), 1) for t in topics]
    diversity = np.mean(topic_dists) if topic_dists else 0.0
    n_topics = len(topics)
    all_words = set()
    for topic in topics:
        all_words.update(topic)
    uniqueness = len(all_words) / (n_topics * top_n) if n_topics > 0 else 0
    return 0.5 * diversity + 0.5 * uniqueness


def _compute_silhouette(model, X, model_type="kmeans"):
    from sklearn.metrics import silhouette_score
    X_dense = X.toarray() if hasattr(X, 'toarray') else np.asarray(X)
    if model_type == "kmeans":
        labels = model.predict(X) if hasattr(model, 'predict') else model.labels_
    elif model_type == "lda":
        labels = np.argmax(model.transform(X), axis=1)
    elif model_type == "nmf":
        labels = np.argmax(model.transform(X), axis=1)
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


def _build_objective(X, feature_names, texts, config):
    from sklearn.decomposition import LatentDirichletAllocation, NMF
    from sklearn.cluster import KMeans

    model_cache = {}

    def f(k):
        if config.model_type == "lda":
            model = LatentDirichletAllocation(
                n_components=k, max_iter=config.lda_max_iter,
                learning_method="online", random_state=config.random_state,
                doc_topic_prior=config.lda_doc_topic_prior,
                topic_word_prior=config.lda_topic_word_prior,
            )
        elif config.model_type == "nmf":
            model = NMF(
                n_components=k, init=config.nmf_init, solver=config.nmf_solver,
                max_iter=config.nmf_max_iter, beta_loss=config.nmf_beta_loss,
                random_state=config.random_state,
            )
        elif config.model_type == "kmeans":
            X_dense = X.toarray() if hasattr(X, 'toarray') else np.asarray(X)
            model = KMeans(
                n_clusters=k, n_init=config.kmeans_n_init,
                max_iter=config.kmeans_max_iter, random_state=config.random_state,
                algorithm=config.kmeans_algorithm,
            )
            model.fit(X_dense)
        else:
            raise ValueError(f"Unknown model_type: {config.model_type}")

        if config.model_type != "kmeans":
            model.fit(X)

        model_cache[k] = model

        if config.metric == "silhouette":
            score = _compute_silhouette(model, X, config.model_type)
        elif config.metric == "coherence":
            if texts is None:
                score = _compute_coherence_fast(model, feature_names, config.model_type)
            else:
                score = _compute_coherence(model, feature_names, texts, config.coherence_type, config.model_type)
        elif config.metric == "perplexity":
            if config.model_type == "lda":
                score = -model.perplexity(X)
            else:
                W = model.transform(X)
                H = model.components_
                recon = np.maximum(W @ H, 1e-10)
                X_dense = X.toarray() if hasattr(X, 'toarray') else X
                score = -np.sum((X_dense - recon) ** 2) / (X.shape[0] * X.shape[1])
        elif config.metric == "reconstruction":
            if config.model_type == "nmf":
                W = model.transform(X)
                H = model.components_
                X_dense = X.toarray() if hasattr(X, 'toarray') else X
                score = -np.sum((X_dense - W @ H) ** 2)
            elif config.model_type == "lda":
                score = model.score(X)
            else:
                score = -model.inertia_
        elif config.metric == "combined":
            sil = _compute_silhouette(model, X, config.model_type)
            if texts is not None:
                coh = _compute_coherence(model, feature_names, texts, config.coherence_type, config.model_type)
            else:
                coh = _compute_coherence_fast(model, feature_names, config.model_type)
            score = 0.5 * sil + 0.5 * coh
        else:
            raise ValueError(f"Unknown metric: {config.metric}")

        if config.verbose:
            print(f"  K={k}, metric={config.metric}, score={score:.6f}")
        return score

    return f, model_cache


def _validate_config(config: ATCNDConfig):
    if config.k_min < 2:
        raise ValueError("k_min must be >= 2")
    if config.k_max <= config.k_min:
        raise ValueError("k_max must be > k_min")
    if config.model_type not in ("lda", "nmf", "kmeans"):
        raise ValueError(f"model_type must be lda, nmf, or kmeans, got {config.model_type}")
    if config.search_strategy not in ("binary", "golden_section", "ternary", "fibonacci", "interpolation", "exponential", "predictive", "grid"):
        raise ValueError(f"search_strategy must be binary, golden_section, ternary, or grid")
    valid_metrics = ("coherence", "perplexity", "silhouette", "reconstruction", "combined")
    if config.metric not in valid_metrics:
        raise ValueError(f"metric must be one of {valid_metrics}, got {config.metric}")


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
        feature_names = np.arange(X.shape[1]).astype(str) if hasattr(X, 'shape') else np.arange(X.shape[1]).astype(str)
    else:
        raise ValueError("Either texts or X must be provided")

    f, model_cache = _build_objective(X, feature_names, texts, config)

    hot_start = None
    if config.search_strategy == "predictive":
        try:
            if config.model_type == "kmeans":
                hot_start = estimate_k_n_clusters(X, config.k_min, config.k_max)
            else:
                hot_start = estimate_k_n_topics(X, config.k_min, config.k_max)
        except Exception:
            pass

    if config.search_strategy == "predictive":
        sr = predictive_search(f, k_min=config.k_min, k_max=config.k_max,
                               hot_start=hot_start, max_iter=config.max_iter,
                               n_candidates=config.n_candidates)
    else:
        sr = pure_search(f, k_min=config.k_min, k_max=config.k_max,
                         strategy=config.search_strategy, max_iter=config.max_iter,
                         n_candidates=config.n_candidates)

    best_model = model_cache.get(sr.optimal_k)
    if best_model is None:
        best_model = model_cache.get(sorted(model_cache.keys())[-1]) if model_cache else None

    return ATCNDResult(
        optimal_k=sr.optimal_k,
        optimal_score=sr.optimal_score,
        all_scores=sr.all_scores,
        model=best_model,
        model_type=config.model_type,
        search_history=sr.search_history,
        vectorizer=vectorizer,
        feature_names=feature_names,
        config=config,
        candidate_ks=sr.candidate_ks,
        candidate_scores=sr.candidate_scores,
    )


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
    topics = _extract_topics(result.model, result.feature_names, result.model_type, top_n)
    if topics:
        print(f"\nTop {top_n} keywords per topic/cluster:")
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
    ax.axvline(x=result.optimal_k, color="#F44336", linestyle="--", linewidth=2, label=f"Optimal K={result.optimal_k}")

    phases_colors = {"binary_search": "#4CAF50", "golden_section": "#FF9800",
                     "ternary_search": "#9C27B0", "grid": "#607D8B",
                     "refinement": "#E91E63", "boundary": "#795548", "final_sweep": "#00BCD4"}
    for entry in result.search_history:
        phase = entry.get("phase", "unknown")
        color = phases_colors.get(phase, "#9E9E9E")
        ax.plot(entry["k"], entry["score"], "s", color=color, markersize=8, zorder=5)

    ax.set_xlabel("Number of Topics/Clusters (K)", fontsize=14)
    ax.set_ylabel(f"Score ({result.config.metric if result.config else 'metric'})", fontsize=14)
    ax.set_title(f"ATCND Search Curve ({result.model_type.upper()}, strategy={result.config.search_strategy if result.config else '?'})\nOptimal K = {result.optimal_k}", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = save_path or "atcnd_search_curve.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()