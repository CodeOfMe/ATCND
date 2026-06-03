"""
ATCND Benchmark: Compare adaptive K search strategies across LDA, NMF, K-Means
against grid search, HDP, and Top2Vec baselines on standard datasets.
"""

import time
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from .core import ATCNDConfig, ATCNDResult, atcnd_search, plot_search_curve


def load_20newsgroups(subset: str = "all", n_samples: int = 2000):
    from sklearn.datasets import fetch_20newsgroups
    data = fetch_20newsgroups(
        subset=subset, shuffle=True, random_state=42,
        remove=("headers", "footers", "quotes"),
    )
    texts = data.data[:n_samples]
    labels = data.target[:n_samples]
    true_k = len(np.unique(data.target_names))
    return texts, labels, true_k, "20Newsgroups"


def load_synthetic(n_docs: int = 1000, n_topics: int = 8, vocab_size: int = 2000):
    np.random.seed(42)
    topic_words = []
    for t in range(n_topics):
        words = [f"topic{t}_word{i}" for i in range(40)]
        words += [f"shared{i}" for i in range(15)]
        topic_words.append(words)
    bg_words = [f"background{i}" for i in range(80)]

    texts = []
    labels = []
    for _ in range(n_docs):
        doc_topics = np.random.dirichlet(np.ones(n_topics) * 0.5)
        n_words = np.random.randint(50, 150)
        words = []
        for _ in range(n_words):
            if np.random.random() < 0.85:
                t = np.random.choice(n_topics, p=doc_topics)
                words.append(np.random.choice(topic_words[t]))
            else:
                words.append(np.random.choice(bg_words))
        texts.append(" ".join(words))
        labels.append(int(np.argmax(doc_topics)))

    return texts, np.array(labels), n_topics, "Synthetic"


def load_synthetic_blobs(n_samples: int = 1000, n_features: int = 50, n_centers: int = 8):
    from sklearn.datasets import make_blobs
    X, labels = make_blobs(
        n_samples=n_samples, n_features=n_features,
        centers=n_centers, random_state=42,
    )
    return X, labels, n_centers, "SyntheticBlobs"


def run_method(
    name: str, texts, X, config: ATCNDConfig, verbose: bool = True,
) -> Tuple[ATCNDResult, float]:
    start = time.time()
    if texts is not None:
        result = atcnd_search(texts=texts, config=config)
    else:
        result = atcnd_search(X=X, config=config)
    elapsed = time.time() - start
    if verbose:
        print(f"  {name}: K={result.optimal_k}, "
              f"score={result.optimal_score:.4f}, "
              f"evals={len(result.search_history)}, "
              f"time={elapsed:.2f}s")
    return result, elapsed


def run_benchmark(
    dataset_name: str = "synthetic",
    k_range: Tuple[int, int] = (2, 30),
    n_samples: int = 1000,
) -> Dict[str, Any]:
    print(f"\n{'='*70}")
    print(f"Benchmark: {dataset_name}")
    print(f"{'='*70}")

    if dataset_name == "20newsgroups":
        texts, labels, true_k, name = load_20newsgroups(n_samples=n_samples)
        X = None
    elif dataset_name == "synthetic":
        texts, labels, true_k, name = load_synthetic(n_docs=n_samples)
        X = None
    elif dataset_name == "blobs":
        X, labels, true_k, name = load_synthetic_blobs(n_samples=n_samples)
        texts = None
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"Dataset: {name}, Docs: {len(texts) if texts else X.shape[0]}, True K: {true_k}")

    results = {
        "dataset": name,
        "n_samples": n_samples,
        "true_k": true_k,
        "k_range": k_range,
        "methods": {},
    }

    method_configs = [
        ("ATCND-Binary-LDA", ATCNDConfig(
            k_min=k_range[0], k_max=k_range[1], metric="silhouette",
            model_type="lda", search_strategy="binary", verbose=True)),
        ("ATCND-Golden-LDA", ATCNDConfig(
            k_min=k_range[0], k_max=k_range[1], metric="silhouette",
            model_type="lda", search_strategy="golden_section", verbose=True)),
        ("ATCND-Ternary-LDA", ATCNDConfig(
            k_min=k_range[0], k_max=k_range[1], metric="silhouette",
            model_type="lda", search_strategy="ternary", verbose=True)),
        ("ATCND-Binary-NMF", ATCNDConfig(
            k_min=k_range[0], k_max=k_range[1], metric="silhouette",
            model_type="nmf", search_strategy="binary", verbose=True)),
        ("ATCND-Binary-KMeans", ATCNDConfig(
            k_min=k_range[0], k_max=k_range[1], metric="silhouette",
            model_type="kmeans", search_strategy="binary", verbose=True)),
        ("ATCND-Golden-KMeans", ATCNDConfig(
            k_min=k_range[0], k_max=k_range[1], metric="silhouette",
            model_type="kmeans", search_strategy="golden_section", verbose=True)),
        ("Grid-LDA", ATCNDConfig(
            k_min=k_range[0], k_max=k_range[1], metric="silhouette",
            model_type="lda", search_strategy="grid", verbose=True)),
        ("Grid-KMeans", ATCNDConfig(
            k_min=k_range[0], k_max=k_range[1], metric="silhouette",
            model_type="kmeans", search_strategy="grid", verbose=True)),
    ]

    for bench_name, cfg in method_configs:
        print(f"\n--- {bench_name} ---")
        try:
            if texts is not None:
                res, t = run_method(bench_name, texts=texts, X=None, config=cfg)
            else:
                res, t = run_method(bench_name, texts=None, X=X, config=cfg)
            results["methods"][bench_name] = {
                "optimal_k": res.optimal_k,
                "score": res.optimal_score,
                "time": t,
                "evaluations": len(res.search_history),
                "strategy": cfg.search_strategy,
                "candidates": res.candidate_ks,
            }
            try:
                plot_search_curve(res, save_path=f"fig_{dataset_name}_{bench_name}.png")
            except Exception:
                pass
        except Exception as e:
            print(f"  {bench_name} failed: {e}")
            results["methods"][bench_name] = {"status": "error", "note": str(e)}

    return results


def print_summary(results: Dict[str, Any]):
    print(f"\n{'='*80}")
    print(f"SUMMARY: {results['dataset']}")
    print(f"True K: {results['true_k']}, Samples: {results['n_samples']}")
    print(f"Search Range: K in [{results['k_range'][0]}, {results['k_range'][1]}]")
    print(f"{'='*80}")
    print(f"{'Method':<25} {'K*':>5} {'Score':>10} {'Time(s)':>10} {'#Evals':>8} {'|K*-K_true|':>12}")
    print(f"{'-'*72}")

    true_k = results["true_k"]
    for name, res in results["methods"].items():
        if "optimal_k" in res:
            diff = abs(res["optimal_k"] - true_k)
            print(f"{name:<25} {res['optimal_k']:>5} {res['score']:>10.4f} "
                  f"{res['time']:>10.2f} {res['evaluations']:>8d} {diff:>12d}")
        else:
            note = res.get("note", "unknown")
            print(f"{name:<25} {'N/A':>5} {'N/A':>10} {'N/A':>10} {note:>8}")


def run_all_benchmarks():
    all_results = []
    datasets = [
        ("synthetic", (2, 30), 1000),
        ("blobs", (2, 30), 1000),
    ]

    for ds_name, k_range, n_samples in datasets:
        try:
            results = run_benchmark(
                dataset_name=ds_name,
                k_range=k_range,
                n_samples=n_samples,
            )
            print_summary(results)
            all_results.append(results)
        except Exception as e:
            print(f"Error on {ds_name}: {e}")

    with open("benchmark_results.json", "w") as f:
        def np_serializer(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        json.dump(all_results, f, default=np_serializer, indent=2)

    return all_results