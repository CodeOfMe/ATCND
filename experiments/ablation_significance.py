#!/usr/bin/env python3
"""
ATCND Ablation Study, Statistical Significance, and Extended Baselines.

Generates:
- Ablation results (refinement, Δ sensitivity, hot-start)
- Statistical significance (10 runs × 8 strategies, mean ± std)
- Extended baselines comparison (X-Means, G-Means, Kneedle, Gap)

Run: python experiments/ablation_significance.py
Output: experiments/results/ablation_*.pdf, significance_*.pdf
"""

import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(FIGDIR, exist_ok=True)


def save_fig(fig, name):
    for ext in ("svg", "pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_dataset(n=500, d=50, k_true=8, seed=42):
    return make_blobs(n_samples=n, n_features=d, centers=k_true, random_state=seed)


# ════════════════════════════════════════════════════════════════
# 1. ABLATION STUDY
# ════════════════════════════════════════════════════════════════

def ablation_refinement():
    """Test: does the refinement phase matter?"""
    from atcnd.search import binary_search, SearchResult
    X, y_true = make_dataset()
    def f(k):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        return silhouette_score(X, km.fit_predict(X))

    results = {}
    for n_refine in [0, 1, 2, 4, 8, None]:
        sr = binary_search(f, k_min=2, k_max=30, max_iter=30, n_candidates=3)
        if n_refine is None:
            key = "Full (Δ=N/8)"
            # Full includes refinement
            sr_full = binary_search(f, k_min=2, k_max=30, max_iter=30, n_candidates=3)
            results[key] = {"K*": sr_full.optimal_k, "score": sr_full.optimal_score, "evals": len(sr_full.all_scores)}
        elif n_refine == 0:
            key = "No refinement"
            results[key] = {"K*": sr.optimal_k, "score": sr.optimal_score, "evals": len(sr.all_scores)}
        else:
            key = f"Δ={n_refine}"
            # Manual refinement
            k_star = sr.optimal_k
            s_star = sr.optimal_score
            for k in range(max(2, k_star - n_refine), min(30, k_star + n_refine) + 1):
                if k not in sr.all_scores:
                    sr.all_scores[k] = f(k)
                    if sr.all_scores[k] > s_star:
                        s_star = sr.all_scores[k]
                        k_star = k
            results[key] = {"K*": k_star, "score": s_star, "evals": len(sr.all_scores)}

    print("\n=== ABLATION: Refinement Phase ===")
    print(f"  {'Config':<20} {'K*':>4} {'Score':>8} {'Evals':>6}")
    print(f"  {'-'*42}")
    for k, v in results.items():
        print(f"  {k:<20} {v['K*']:>4} {v['score']:>8.4f} {v['evals']:>6}")

    fig, ax = plt.subplots(figsize=(8, 5))
    configs = list(results.keys())
    scores = [results[c]["score"] for c in configs]
    evals = [results[c]["evals"] for c in configs]
    x = np.arange(len(configs))
    ax.bar(x - 0.2, scores, 0.35, color="#2196F3", label="Silhouette")
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, evals, 0.35, color="#FF9800", label="Evaluations")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Silhouette Score")
    ax2.set_ylabel("Evaluations")
    ax.set_title("Ablation: Refinement Phase Impact", fontsize=12)
    fig.legend(loc="upper right", bbox_to_anchor=(0.95, 0.95))
    fig.tight_layout()
    save_fig(fig, "ablation_refinement")
    return results


def ablation_delta():
    """Test: sensitivity to Δ parameter."""
    from atcnd.search import binary_search
    X, y_true = make_dataset()
    def f(k):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        return silhouette_score(X, km.fit_predict(X))

    deltas = [0, 1, 2, 3, 4, 5, 6, 7, 8, "auto"]
    results = {}
    for d in deltas:
        sr = binary_search(f, k_min=2, k_max=30, max_iter=30, n_candidates=3)
        k_star = sr.optimal_k
        s_star = sr.optimal_score
        if d == "auto":
            delta = max(1, (30 - 2) // 8)
            label = f"Δ={delta} (auto)"
        else:
            delta = d
            label = f"Δ={d}"
        for k in range(max(2, k_star - delta), min(30, k_star + delta) + 1):
            if k not in sr.all_scores:
                sr.all_scores[k] = f(k)
                if sr.all_scores[k] > s_star:
                    s_star = sr.all_scores[k]
                    k_star = k
        results[label] = {"K*": k_star, "score": s_star, "evals": len(sr.all_scores)}

    print("\n=== ABLATION: Δ Sensitivity ===")
    print(f"  {'Config':<20} {'K*':>4} {'Score':>8} {'Evals':>6}")
    print(f"  {'-'*42}")
    for k, v in results.items():
        print(f"  {k:<20} {v['K*']:>4} {v['score']:>8.4f} {v['evals']:>6}")

    fig, ax = plt.subplots(figsize=(8, 5))
    configs = list(results.keys())
    scores = [results[c]["score"] for c in configs]
    evals = [results[c]["evals"] for c in configs]
    x = np.arange(len(configs))
    ax.plot(x, scores, "o-", color="#2196F3", lw=2, ms=6, label="Silhouette")
    ax2 = ax.twinx()
    ax2.plot(x, evals, "s-", color="#FF9800", lw=2, ms=6, label="Evaluations")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Silhouette Score")
    ax2.set_ylabel("Evaluations")
    ax.set_title("Ablation: Δ Parameter Sensitivity", fontsize=12)
    fig.legend(loc="upper right", bbox_to_anchor=(0.95, 0.95))
    fig.tight_layout()
    save_fig(fig, "ablation_delta")
    return results


def ablation_hot_start():
    """Test: impact of PCA hot-start on predictive search."""
    from atcnd.search import predictive_search, estimate_k_n_clusters
    X, y_true = make_dataset()
    def f(k):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        return silhouette_score(X, km.fit_predict(X))

    hot_k = estimate_k_n_clusters(X, k_min=2, k_max=30)
    print(f"\n  PCA hot-start estimate: K̂={hot_k} (true=8)")

    configs = {
        "No hot-start": {"hot_start": None},
        "Hot-start ±1": {"hot_start": hot_k},
        "Hot-start +2": {"hot_start": hot_k + 2},
        "Hot-start -2": {"hot_start": max(2, hot_k - 2)},
        "Oracle (K=8)": {"hot_start": 8},
    }

    results = {}
    for label, kw in configs.items():
        sr = predictive_search(f, k_min=2, k_max=30, max_iter=30, n_candidates=3, **kw)
        results[label] = {"K*": sr.optimal_k, "score": sr.optimal_score, "evals": len(sr.all_scores)}

    print("\n=== ABLATION: Hot-Start Impact ===")
    print(f"  {'Config':<20} {'K*':>4} {'Score':>8} {'Evals':>6}")
    print(f"  {'-'*42}")
    for k, v in results.items():
        print(f"  {k:<20} {v['K*']:>4} {v['score']:>8.4f} {v['evals']:>6}")

    fig, ax = plt.subplots(figsize=(8, 5))
    configs = list(results.keys())
    scores = [results[c]["score"] for c in configs]
    evals = [results[c]["evals"] for c in configs]
    x = np.arange(len(configs))
    ax.bar(x - 0.2, scores, 0.35, color="#2196F3", label="Silhouette")
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, evals, 0.35, color="#FF9800", label="Evaluations")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Silhouette Score")
    ax2.set_ylabel("Evaluations")
    ax.set_title("Ablation: PCA Hot-Start Impact on Predictive Search", fontsize=12)
    fig.legend(loc="upper right", bbox_to_anchor=(0.95, 0.95))
    fig.tight_layout()
    save_fig(fig, "ablation_hot_start")
    return results


# ════════════════════════════════════════════════════════════════
# 2. STATISTICAL SIGNIFICANCE (10 runs × 8 strategies)
# ════════════════════════════════════════════════════════════════

def statistical_significance(n_runs=10):
    """Run each strategy n_runs times with different seeds, report mean ± std."""
    from atcnd.search import search
    strategies = ["binary", "golden_section", "ternary", "fibonacci",
                  "interpolation", "exponential", "predictive", "grid"]

    all_results = {s: {"K*": [], "score": [], "evals": []} for s in strategies}

    for run in range(n_runs):
        seed = 42 + run * 7
        X, y_true = make_dataset(seed=seed)
        def f(k, X=X):
            km = KMeans(n_clusters=k, random_state=seed, n_init=10)
            return silhouette_score(X, km.fit_predict(X))

        for s in strategies:
            kw = {}
            if s == "predictive":
                from atcnd.search import estimate_k_n_clusters
                kw["hot_start"] = estimate_k_n_clusters(X, k_min=2, k_max=30)
            sr = search(f, k_min=2, k_max=30, strategy=s, **kw)
            all_results[s]["K*"].append(sr.optimal_k)
            all_results[s]["score"].append(sr.optimal_score)
            all_results[s]["evals"].append(len(sr.all_scores))

    print("\n=== STATISTICAL SIGNIFICANCE (10 runs) ===")
    print(f"  {'Strategy':<18} {'K* (mean±std)':>15} {'Score (mean±std)':>18} {'Evals (mean±std)':>18}")
    print(f"  {'-'*75}")
    for s in strategies:
        k_mean = np.mean(all_results[s]["K*"])
        k_std = np.std(all_results[s]["K*"])
        s_mean = np.mean(all_results[s]["score"])
        s_std = np.std(all_results[s]["score"])
        e_mean = np.mean(all_results[s]["evals"])
        e_std = np.std(all_results[s]["evals"])
        print(f"  {s:<18} {k_mean:>5.1f}±{k_std:<4.1f} {s_mean:>8.4f}±{s_std:<7.4f} {e_mean:>6.1f}±{e_std:<5.1f}")

    # Plot: evaluation counts with error bars
    fig, ax = plt.subplots(figsize=(10, 5))
    names = [s.replace("_", " ").title() for s in strategies]
    eval_means = [np.mean(all_results[s]["evals"]) for s in strategies]
    eval_stds = [np.std(all_results[s]["evals"]) for s in strategies]
    x = np.arange(len(strategies))
    bars = ax.barh(x, eval_means, xerr=eval_stds, color="#2196F3", capsize=4, alpha=0.8)
    ax.set_yticks(x)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Mean Evaluations (±1 std)")
    ax.set_title(f"Statistical Significance: Evaluation Counts ({n_runs} runs)", fontsize=12)
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    save_fig(fig, "significance_evals")

    # Plot: score stability
    fig, ax = plt.subplots(figsize=(10, 5))
    score_means = [np.mean(all_results[s]["score"]) for s in strategies]
    score_stds = [np.std(all_results[s]["score"]) for s in strategies]
    ax.barh(x, score_means, xerr=score_stds, color="#4CAF50", capsize=4, alpha=0.8)
    ax.set_yticks(x)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Mean Silhouette Score (±1 std)")
    ax.set_title(f"Statistical Significance: Score Stability ({n_runs} runs)", fontsize=12)
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    save_fig(fig, "significance_scores")

    return all_results


# ════════════════════════════════════════════════════════════════
# 3. EXTENDED BASELINES
# ════════════════════════════════════════════════════════════════

def extended_baselines():
    """Compare with X-Means, G-Means, Kneedle, Gap Statistic."""
    from atcnd.comparison import (
        baseline_kneedle, baseline_bic_gmm, baseline_gap_statistic,
        baseline_xmeans, baseline_gmeans, baseline_eigengap
    )

    X, y_true = make_dataset()

    baselines = {
        "X-Means": lambda: baseline_xmeans(X, k_min=2, k_max=20),
        "G-Means": lambda: baseline_gmeans(X, k_min=2, k_max=20),
        "Kneedle": lambda: baseline_kneedle(X, k_min=2, k_max=20),
        "Gap Statistic": lambda: baseline_gap_statistic(X, k_min=2, k_max=20),
        "BIC-GMM": lambda: baseline_bic_gmm(X, k_min=2, k_max=20),
        "EigenGap": lambda: baseline_eigengap(X, k_min=2, k_max=20),
    }

    results = {}
    for name, fn in baselines.items():
        try:
            r = fn()
            k = r.get("k", r.get("K", r.get("k_est", 0)))
            results[name] = {"K*": k, "evals": r.get("evals", "N/A")}
        except Exception as e:
            results[name] = {"K*": "ERR", "evals": "ERR", "error": str(e)[:50]}

    print("\n=== EXTENDED BASELINES ===")
    print(f"  {'Method':<20} {'K*':>6} {'Evals':>8}")
    print(f"  {'-'*36}")
    for name, v in results.items():
        print(f"  {name:<20} {str(v['K*']):>6} {str(v['evals']):>8}")

    return results


if __name__ == "__main__":
    print("ATCND Ablation Study, Statistical Significance, Extended Baselines")
    print("=" * 70)

    ablation_refinement()
    ablation_delta()
    ablation_hot_start()
    statistical_significance(n_runs=10)
    extended_baselines()

    print("\n" + "=" * 70)
    print(f"Results saved to: {FIGDIR}/")
    print("Done!")
