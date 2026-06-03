"""
ATCND Comprehensive Demo: Adapters for numpy, pandas, scipy, sklearn, torch, gensim.

Each section demonstrates a real use case with visualization.
Run: python examples/demo_all.py
Output: examples/figures/*.png
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGDIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGDIR, exist_ok=True)


def plot_result(result, title, xlabel="K", ylabel="Score", save_name=None):
    ks = sorted(result.all_scores.keys())
    scores = [result.all_scores[k] for k in ks]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, scores, "o-", color="#2196F3", linewidth=2, markersize=5, label="f(K)")
    ax.axvline(x=result.optimal_k, color="#F44336", linestyle="--", linewidth=2,
               label=f"Optimal K={result.optimal_k}")
    phase_colors = {
        "boundary": "#795548", "binary_search": "#4CAF50",
        "golden_section": "#FF9800", "ternary_search": "#9C27B0",
        "grid": "#607D8B", "refinement": "#E91E63", "final_sweep": "#00BCD4",
    }
    for entry in result.search_history:
        c = phase_colors.get(entry.get("phase", ""), "#9E9E9E")
        ax.plot(entry["k"], entry["score"], "s", color=c, markersize=7, zorder=5)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_name:
        fig.savefig(os.path.join(FIGDIR, save_name), dpi=150, bbox_inches="tight")
    plt.close(fig)


def demo_numpy_bins():
    print("\n" + "=" * 60)
    print("1. NUMPY: Optimal histogram bin count")
    print("=" * 60)
    from atcnd import search_bins

    np.random.seed(42)
    data = np.concatenate([np.random.normal(0, 1, 500), np.random.normal(5, 0.8, 300)])
    result = search_bins(data, k_min=3, k_max=80, strategy="binary", method="freedman")
    print(f"  Optimal bins: {result.optimal_k}")
    print(f"  Evaluations: {len(result.search_history)} (grid would need {80-3+1}=78)")
    print(f"  Reduction: {100*(1 - len(result.search_history)/78):.0f}%")

    counts, edges = np.histogram(data, bins=result.optimal_k)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(data, bins=result.optimal_k, color="#4CAF50", edgecolor="white", alpha=0.8)
    axes[0].set_title(f"Optimal {result.optimal_k} bins (ATCND binary search)", fontsize=13)
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Count")
    axes[1].hist(data, bins=10, color="#9E9E9E", edgecolor="white", alpha=0.8)
    axes[1].set_title("Default 10 bins (Sturges rule)", fontsize=13)
    axes[1].set_xlabel("Value")
    axes[1].set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "numpy_bins.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    plot_result(result, "NumPy: Optimal Histogram Bins\n(binary search, entropy metric)",
                xlabel="Number of bins", ylabel="Entropy", save_name="numpy_bins_curve.png")


def demo_scipy_knots():
    print("\n" + "=" * 60)
    print("2. SCIPY: Optimal spline knot count")
    print("=" * 60)
    from atcnd import search_knots

    np.random.seed(42)
    x = np.linspace(0, 2 * np.pi, 200)
    y = np.sin(x) + 0.1 * np.random.randn(len(x))
    result = search_knots(x, y, k_min=3, k_max=30, strategy="binary", degree=3)
    print(f"  Optimal knots: {result.optimal_k}")
    print(f"  Evaluations: {len(result.search_history)} (grid would need {30-3+1}=28)")

    from scipy.interpolate import make_interp_spline
    order = np.argsort(x)
    idx = np.linspace(0, len(x) - 1, result.optimal_k, dtype=int)
    knots = x[order][idx]
    spl = make_interp_spline(x[order], y[order], k=3, t=knots)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(x, y, s=5, alpha=0.3, color="#9E9E9E", label="Noisy data")
    ax.plot(x, spl(x), color="#F44336", linewidth=2, label=f"Spline ({result.optimal_k} knots)")
    ax.set_title(f"SciPy: Optimal Spline Knots (ATCND binary search, {result.optimal_k} knots)", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "scipy_knots.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    plot_result(result, "SciPy: Optimal Spline Knot Count\n(binary search, -MSE metric)",
                xlabel="Number of knots", ylabel="-MSE", save_name="scipy_knots_curve.png")


def demo_scipy_gmm():
    print("\n" + "=" * 60)
    print("3. SCIPY/SKLEARN: Optimal GMM components via BIC")
    print("=" * 60)
    from atcnd import search_gmm_components

    np.random.seed(42)
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=800, n_features=2, centers=5, random_state=42)
    result = search_gmm_components(X, k_min=2, k_max=20, strategy="binary")
    print(f"  Optimal components: {result.optimal_k} (true=5)")
    print(f"  Evaluations: {len(result.search_history)} (grid would need {20-2+1}=19)")
    print(f"  BIC: {result.optimal_score:.0f}")

    from sklearn.mixture import GaussianMixture
    best_model = GaussianMixture(n_components=result.optimal_k, random_state=42).fit(X)
    labels = best_model.predict(X)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", s=8, alpha=0.6)
    ax.set_title(f"GMM with {result.optimal_k} components (ATCND binary search, BIC)", fontsize=13)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    plt.colorbar(scatter, label="Component")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "scipy_gmm.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    ks = sorted(result.all_scores.keys())
    scores = [result.all_scores[k] for k in ks]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, scores, "o-", color="#2196F3", linewidth=2, markersize=5)
    ax.axvline(x=result.optimal_k, color="#F44336", linestyle="--", linewidth=2,
               label=f"Optimal K={result.optimal_k}")
    ax.set_xlabel("Number of components", fontsize=13)
    ax.set_ylabel("BIC (lower is better)", fontsize=13)
    ax.set_title("Scipy/sklearn: GMM Component Selection (BIC)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "scipy_gmm_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def demo_pandas_rolling():
    print("\n" + "=" * 60)
    print("4. PANDAS: Optimal rolling window size")
    print("=" * 60)
    import pandas as pd
    from atcnd import search_rolling_window

    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=365)
    trend = np.linspace(10, 20, 365)
    noise = 3 * np.random.randn(365)
    season = 5 * np.sin(np.arange(365) * 2 * np.pi / 30)
    series = pd.Series(trend + season + noise, index=dates)

    result = search_rolling_window(series, k_min=2, k_max=60, strategy="binary", method="smoothness")
    print(f"  Optimal window: {result.optimal_k}")
    print(f"  Evaluations: {len(result.search_history)} (grid would need {60-2+1}=59)")
    print(f"  Reduction: {100*(1 - len(result.search_history)/59):.0f}%")

    smoothed = series.rolling(window=result.optimal_k, center=True).mean()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(series, alpha=0.3, color="#9E9E9E", linewidth=0.8, label="Raw")
    ax.plot(smoothed, color="#F44336", linewidth=2,
            label=f"Rolling mean (window={result.optimal_k})")
    ax.set_title(f"Pandas: Optimal Rolling Window (ATCND binary search, window={result.optimal_k})", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "pandas_rolling.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    plot_result(result, "Pandas: Optimal Rolling Window Size\n(binary search, smoothness metric)",
                xlabel="Window size", ylabel="-Var(diff2)", save_name="pandas_rolling_curve.png")


def demo_pandas_bins():
    print("\n" + "=" * 60)
    print("5. PANDAS: Optimal bins for DataFrame column")
    print("=" * 60)
    import pandas as pd
    from atcnd import search_dataframe_bins

    np.random.seed(42)
    df = pd.DataFrame({
        "price": np.exp(np.random.randn(2000) * 0.5 + 3),
        "volume": np.random.exponential(100, 2000),
    })

    result = search_dataframe_bins(df, "price", k_min=3, k_max=80, strategy="binary", method="entropy")
    print(f"  Optimal bins for 'price': {result.optimal_k}")
    print(f"  Evaluations: {len(result.search_history)} (grid: 78)")
    print(f"  Reduction: {100*(1 - len(result.search_history)/78):.0f}%")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["price"], bins=result.optimal_k, color="#4CAF50", edgecolor="white", alpha=0.8)
    ax.set_title(f"Pandas: Optimal bins for 'price' column ({result.optimal_k} bins)", fontsize=13)
    ax.set_xlabel("Price")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "pandas_bins.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def demo_sklearn_kmeans():
    print("\n" + "=" * 60)
    print("6. SKLEARN: Optimal K-Means cluster count")
    print("=" * 60)
    from atcnd import search_model
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans

    np.random.seed(42)
    X, _ = make_blobs(n_samples=1000, n_features=50, centers=8, random_state=42)
    result = search_model(KMeans, X, param_name="n_clusters", k_min=2, k_max=30,
                          strategy="binary", metric="silhouette")
    print(f"  Optimal clusters: {result.optimal_k} (true=8)")
    print(f"  Evaluations: {len(result.search_history)} (grid: 29)")
    print(f"  Reduction: {100*(1 - len(result.search_history)/29):.0f}%")
    print(f"  Silhouette: {result.optimal_score:.3f}")

    plot_result(result, "sklearn: K-Means Cluster Selection\n(binary search, silhouette metric)",
                xlabel="K (clusters)", ylabel="Silhouette", save_name="sklearn_kmeans.png")


def demo_sklearn_knn():
    print("\n" + "=" * 60)
    print("7. SKLEARN: Optimal KNN neighbors")
    print("=" * 60)
    from atcnd import search_neighbors
    from sklearn.datasets import load_iris

    data = load_iris()
    result = search_neighbors(data.data, k_min=1, k_max=30, strategy="binary",
                              metric="accuracy", y=data.target)
    print(f"  Optimal K: {result.optimal_k}")
    print(f"  Evaluations: {len(result.search_history)} (grid: 30)")
    print(f"  Reduction: {100*(1 - len(result.search_history)/30):.0f}%")
    print(f"  CV accuracy: {result.optimal_score:.3f}")

    plot_result(result, "sklearn: KNN Neighbor Selection\n(binary search, 5-fold CV accuracy)",
                xlabel="K (neighbors)", ylabel="CV Accuracy", save_name="sklearn_knn.png")


def demo_sklearn_pca():
    print("\n" + "=" * 60)
    print("8. SKLEARN: Optimal PCA components")
    print("=" * 60)
    from atcnd import search_components
    from sklearn.datasets import load_digits

    data = load_digits()
    result = search_components(data.data, k_min=1, k_max=64, strategy="binary")
    print(f"  Optimal components: {result.optimal_k} (for 95% variance)")
    print(f"  Cumulative variance: {result.optimal_score:.3f}")
    print(f"  Evaluations: {len(result.search_history)} (grid: 64)")
    print(f"  Reduction: {100*(1 - len(result.search_history)/64):.0f}%")

    ks = sorted(result.all_scores.keys())
    scores = [result.all_scores[k] for k in ks]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, scores, "o-", color="#2196F3", linewidth=2, markersize=4)
    ax.axhline(y=0.95, color="#9E9E9E", linestyle=":", linewidth=1, label="95% threshold")
    ax.axvline(x=result.optimal_k, color="#F44336", linestyle="--", linewidth=2,
               label=f"Optimal={result.optimal_k}")
    ax.set_xlabel("Number of components", fontsize=13)
    ax.set_ylabel("Cumulative variance", fontsize=13)
    ax.set_title("sklearn: PCA Component Selection (ATCND binary search)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "sklearn_pca.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def demo_sklearn_trees():
    print("\n" + "=" * 60)
    print("9. SKLEARN: Optimal Random Forest tree count")
    print("=" * 60)
    from atcnd import search_trees
    from sklearn.datasets import load_wine

    data = load_wine()
    result = search_trees(data.data, data.target, k_min=10, k_max=300, strategy="binary", cv=5)
    print(f"  Optimal trees: {result.optimal_k}")
    print(f"  Evaluations: {len(result.search_history)} (grid: {300-10+1}=291)")
    print(f"  Reduction: {100*(1 - len(result.search_history)/291):.0f}%")
    print(f"  CV accuracy: {result.optimal_score:.3f}")

    plot_result(result, "sklearn: Random Forest Tree Count Selection\n(binary search, 5-fold CV)",
                xlabel="Number of trees", ylabel="CV Accuracy", save_name="sklearn_trees.png")


def demo_sklearn_dbscan():
    print("\n" + "=" * 60)
    print("10. SKLEARN: Optimal DBSCAN eps (discretized)")
    print("=" * 60)
    from atcnd import search_dbscan_eps
    from sklearn.datasets import make_moons

    np.random.seed(42)
    X, y_true = make_moons(n_samples=500, noise=0.05, random_state=42)
    result = search_dbscan_eps(X, eps_min=1, eps_max=50, strategy="binary")
    best_eps = result.optimal_k / 10.0
    print(f"  Optimal eps: {best_eps:.1f} (scaled K={result.optimal_k})")
    print(f"  Evaluations: {len(result.search_history)} (grid: {50-1+1}=50)")

    from sklearn.cluster import DBSCAN
    labels = DBSCAN(eps=best_eps, min_samples=5).fit_predict(X)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", s=10, alpha=0.7)
    ax.set_title(f"sklearn: DBSCAN with eps={best_eps:.1f} (ATCND binary search)", fontsize=13)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    plt.colorbar(scatter, label="Cluster")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "sklearn_dbscan.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def demo_gensim_nmf():
    print("\n" + "=" * 60)
    print("11. GENSIM+SKLEARN: Optimal NMF topic count")
    print("=" * 60)
    from atcnd import search_nmf_topics

    np.random.seed(42)
    texts = []
    templates = [
        ["machine learning algorithm model neural network deep training data feature"],
        ["climate change carbon emission temperature global warming atmosphere sea level"],
        ["stock market trading investment portfolio financial risk return analysis"],
        ["protein dna gene cell biology research medical health molecule"],
        ["quantum computing particle physics theory experiment energy research"],
    ]
    for t_words in templates * 40:
        texts.append(" ".join(t_words) + " " + " ".join(np.random.choice(t_words[0].split(), 3)))

    result = search_nmf_topics(texts, k_min=2, k_max=10, strategy="binary", metric="coherence")
    print(f"  Optimal topics: {result.optimal_k} (true=5)")
    print(f"  Evaluations: {len(result.search_history)} (grid: {10-2+1}=9)")
    print(f"  Coherence: {result.optimal_score:.3f}")

    plot_result(result, "Gensim+sklearn: NMF Topic Count Selection\n(binary search, c_v coherence)",
                xlabel="K (topics)", ylabel="Coherence (c_v)", save_name="gensim_nmf.png")


def demo_torch_hidden():
    print("\n" + "=" * 60)
    print("12. TORCH: Optimal hidden layer size")
    print("=" * 60)
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("  SKIPPED: PyTorch not installed. Install with: pip install atcnd[torch]")
        return

    from atcnd import search_hidden

    np.random.seed(42)
    torch.manual_seed(42)
    X_t = torch.randn(800, 20)
    y_t = (X_t[:, 0] + X_t[:, 1] > 0).long()
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=64, shuffle=True)

    result = search_hidden(loader, input_dim=20, output_dim=2,
                           k_min=8, k_max=256, strategy="binary",
                           epochs=5, lr=1e-3)
    print(f"  Optimal hidden dim: {result.optimal_k}")
    print(f"  Evaluations: {len(result.search_history)} (grid: {256-8+1}=249)")
    print(f"  Reduction: {100*(1 - len(result.search_history)/249):.0f}%")
    print(f"  Best -loss: {result.optimal_score:.4f}")

    plot_result(result, "PyTorch: Optimal Hidden Layer Size\n(binary search, -cross-entropy loss)",
                xlabel="Hidden dim", ylabel="-Loss", save_name="torch_hidden.png")


def demo_torch_layers():
    print("\n" + "=" * 60)
    print("13. TORCH: Optimal number of hidden layers")
    print("=" * 60)
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("  SKIPPED: PyTorch not installed.")
        return

    from atcnd import search_layers

    np.random.seed(42)
    torch.manual_seed(42)
    X_t = torch.randn(800, 20)
    y_t = (X_t[:, 0] + X_t[:, 1] > 0).long()
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=64, shuffle=True)

    result = search_layers(loader, input_dim=20, output_dim=2, hidden_dim=64,
                           k_min=1, k_max=8, strategy="binary", epochs=5, lr=1e-3)
    print(f"  Optimal layers: {result.optimal_k}")
    print(f"  Evaluations: {len(result.search_history)} (grid: {8-1+1}=8)")
    print(f"  Best -loss: {result.optimal_score:.4f}")

    plot_result(result, "PyTorch: Optimal Hidden Layer Count\n(binary search, -cross-entropy loss)",
                xlabel="Number of layers", ylabel="-Loss", save_name="torch_layers.png")


def demo_comparison():
    print("\n" + "=" * 60)
    print("14. COMPARISON: Binary vs Golden vs Ternary vs Grid on K-Means")
    print("=" * 60)
    from atcnd import search
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    np.random.seed(42)
    X, _ = make_blobs(n_samples=1000, n_features=50, centers=8, random_state=42)

    def f(k):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        if len(set(labels)) < 2:
            return -1.0
        return silhouette_score(X, labels)

    strategies = ["binary", "golden_section", "ternary", "grid"]
    results = {}
    for s in strategies:
        results[s] = search(f, k_min=2, k_max=30, strategy=s)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    colors = {"binary": "#4CAF50", "golden_section": "#FF9800",
              "ternary": "#9C27B0", "grid": "#607D8B"}
    for i, s in enumerate(strategies):
        r = results[s]
        ks = sorted(r.all_scores.keys())
        scores = [r.all_scores[k] for k in ks]
        axes[i].plot(ks, scores, "o-", color=colors[s], linewidth=1.5, markersize=3)
        axes[i].axvline(x=r.optimal_k, color="#F44336", linestyle="--", linewidth=2)
        axes[i].set_title(f"{s.replace('_', ' ').title()}\nK*={r.optimal_k}, evals={len(r.search_history)}", fontsize=12)
        axes[i].set_xlabel("K")
        if i == 0:
            axes[i].set_ylabel("Silhouette")
        axes[i].grid(True, alpha=0.3)
    plt.suptitle("ATCND: Four Search Strategies on K-Means (K_true=8)", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "comparison_strategies.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    for s in strategies:
        r = results[s]
        n = len(r.search_history)
        ref = len(results["grid"].search_history)
        pct = 100 * (1 - n / ref)
        print(f"  {s:16s}: K*={r.optimal_k}, evals={n}, vs grid: {pct:.0f}% fewer")


def demo_summary():
    print("\n" + "=" * 60)
    print("SUMMARY: All ATCND Adapter Demos")
    print("=" * 60)
    rows = [
        ("NumPy", "search_bins", "Histogram bin count", "entropy"),
        ("SciPy", "search_knots", "Spline knot count", "-MSE"),
        ("SciPy/sklearn", "search_gmm_components", "GMM components", "BIC"),
        ("Pandas", "search_rolling_window", "Rolling window size", "smoothness"),
        ("Pandas", "search_dataframe_bins", "DataFrame column bins", "entropy"),
        ("sklearn", "search_model(KMeans)", "K-Means clusters", "silhouette"),
        ("sklearn", "search_neighbors", "KNN k", "CV accuracy"),
        ("sklearn", "search_components", "PCA components", "cumulative var"),
        ("sklearn", "search_trees", "RF tree count", "CV accuracy"),
        ("sklearn", "search_dbscan_eps", "DBSCAN eps (discretized)", "silhouette"),
        ("Gensim+sklearn", "search_nmf_topics", "NMF topic count", "c_v coherence"),
        ("PyTorch", "search_hidden", "Hidden layer size", "-loss"),
        ("PyTorch", "search_layers", "Hidden layer count", "-loss"),
    ]
    print(f"  {'Library':<16} {'Adapter':<28} {'Parameter':<25} {'Metric'}")
    print(f"  {'-'*16} {'-'*28} {'-'*25} {'-'*20}")
    for lib, adapter, param, metric in rows:
        print(f"  {lib:<16} {adapter:<28} {param:<25} {metric}")
    print(f"\n  All figures saved to: {FIGDIR}/")


if __name__ == "__main__":
    print("ATCND Comprehensive Demo")
    print("=" * 60)

    demo_numpy_bins()
    demo_scipy_knots()
    demo_scipy_gmm()
    demo_pandas_rolling()
    demo_pandas_bins()
    demo_sklearn_kmeans()
    demo_sklearn_knn()
    demo_sklearn_pca()
    demo_sklearn_trees()
    demo_sklearn_dbscan()
    demo_gensim_nmf()
    demo_torch_hidden()
    demo_torch_layers()
    demo_comparison()
    demo_summary()

    print("\nDone! All figures saved to examples/figures/")