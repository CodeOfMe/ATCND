"""
ATCND Comprehensive Demo — Adapters for NumPy, Pandas, SciPy, Sklearn, PyTorch, Gensim.

Each demo: runs search → generates SVG+PDF+PNG figures → prints summary.
Run: python examples/demo_all.py
Output: examples/figures/*.{svg,pdf,png}
"""

import os, warnings, io, contextlib
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGDIR, exist_ok=True)


def save_fig(fig, name):
    for ext in ("svg", "pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_curve(result, title, xlabel="K", ylabel="Score", highlight=True):
    ks = sorted(result.all_scores.keys())
    scores = [result.all_scores[k] for k in ks]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, scores, "o-", color="#2196F3", lw=2, ms=4, label="f(K)")
    if highlight:
        ax.axvline(result.optimal_k, color="#F44336", ls="--", lw=2,
                   label=f"K*={result.optimal_k}")
    phase_c = {"boundary":"#795548", "binary_search":"#4CAF50",
               "golden_section":"#FF9800", "ternary_search":"#9C27B0",
               "grid":"#607D8B", "refinement":"#E91E63", "final_sweep":"#00BCD4",
               "fibonacci":"#009688", "interpolation":"#673AB7",
               "exponential":"#FF5722", "predictive":"#E91E63",
               "probing":"#795548", "doubling":"#FF9800"}
    for e in result.search_history:
        c = phase_c.get(e.get("phase", ""), "#9E9E9E")
        ax.plot(e["k"], e["score"], "s", color=c, ms=6, zorder=5)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    return fig


# ════════════════════════════════════════════════════════════════
# 1. NUMPY — Optimal histogram bins (AIC)
# ════════════════════════════════════════════════════════════════
def demo_numpy_bins():
    print("\n[1/14] NumPy: Optimal histogram bin count (AIC)")
    from atcnd import search_bins
    np.random.seed(42)
    data = np.concatenate([np.random.normal(0, 1, 500), np.random.normal(5, 0.8, 300)])
    r = search_bins(data, k_min=3, k_max=60, strategy="binary", method="aic")
    grid_n = 60 - 3 + 1
    print(f"  K*={r.optimal_k}  evals={len(r.search_history)}/{grid_n}  reduction={100*(1-len(r.search_history)/grid_n):.0f}%")

    fig = plot_curve(r, "NumPy: Optimal Histogram Bins (AIC, binary search)",
                     xlabel="Number of bins", ylabel="AIC")
    save_fig(fig, "01_numpy_bins_curve")

    fig2, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    a1.hist(data, bins=r.optimal_k, color="#4CAF50", edgecolor="w", alpha=.8)
    a1.set_title(f"ATCND optimal: {r.optimal_k} bins", fontsize=12)
    a1.set_xlabel("Value"); a1.set_ylabel("Count")
    default_bins = int(np.ceil(1 + 3.322 * np.log10(len(data))))
    a2.hist(data, bins=default_bins, color="#9E9E9E", edgecolor="w", alpha=.8)
    a2.set_title(f"Sturges rule: {default_bins} bins", fontsize=12)
    a2.set_xlabel("Value"); a2.set_ylabel("Count")
    plt.tight_layout()
    save_fig(fig2, "01_numpy_bins_compare")


# ════════════════════════════════════════════════════════════════
# 2. SCIPY — Optimal GMM components (BIC)
# ════════════════════════════════════════════════════════════════
def demo_scipy_gmm():
    print("[2/14] SciPy/sklearn: Optimal GMM components (BIC)")
    from atcnd import search_gmm_components
    from sklearn.datasets import make_blobs
    np.random.seed(42)
    X, _ = make_blobs(n_samples=800, n_features=2, centers=5, random_state=42)
    r = search_gmm_components(X, k_min=2, k_max=20, strategy="binary")
    grid_n = 19
    print(f"  K*={r.optimal_k} (true=5)  evals={len(r.search_history)}/{grid_n}  reduction={100*(1-len(r.search_history)/grid_n):.0f}%")

    ks = sorted(r.all_scores.keys())
    scores = [r.all_scores[k] for k in ks]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, scores, "o-", color="#2196F3", lw=2, ms=5)
    ax.axvline(r.optimal_k, color="#F44336", ls="--", lw=2, label=f"K*={r.optimal_k}")
    ax.set_xlabel("Number of components", fontsize=12)
    ax.set_ylabel("BIC (lower is better)", fontsize=12)
    ax.set_title("SciPy/sklearn: GMM Component Selection (BIC)", fontsize=13)
    ax.legend(fontsize=10); ax.grid(True, alpha=.3)
    save_fig(fig, "02_scipy_gmm_curve")

    from sklearn.mixture import GaussianMixture
    gm = GaussianMixture(n_components=r.optimal_k, random_state=42).fit(X)
    labels = gm.predict(X)
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    ax2.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", s=8, alpha=.6)
    ax2.set_title(f"GMM with {r.optimal_k} components", fontsize=13)
    ax2.set_xlabel("x1"); ax2.set_ylabel("x2")
    save_fig(fig2, "02_scipy_gmm_scatter")


# ════════════════════════════════════════════════════════════════
# 3. SCIPY — Optimal smoothing spline knots
# ════════════════════════════════════════════════════════════════
def demo_scipy_spline():
    print("[3/14] SciPy: Optimal smoothing spline internal knots")
    from atcnd import search_knots
    from scipy.interpolate import UnivariateSpline
    np.random.seed(42)
    x = np.linspace(0, 2 * np.pi, 200)
    y = np.sin(x) + 0.15 * np.random.randn(200)

    def f_spline(s):
        spl = UnivariateSpline(x, y, s=s)
        y_pred = spl(x)
        from sklearn.metrics import mean_squared_error
        return -mean_squared_error(y, y_pred)

    from atcnd.search import search
    r = search(f_spline, k_min=1, k_max=200, strategy="binary")
    print(f"  optimal s={r.optimal_k}  evals={len(r.search_history)}/200  reduction={100*(1-len(r.search_history)/200):.0f}%")

    fig = plot_curve(r, "SciPy: Optimal Smoothing Parameter (UnivariateSpline)",
                     xlabel="Smoothing parameter s", ylabel="-MSE")
    save_fig(fig, "03_scipy_spline_curve")

    best_spl = UnivariateSpline(x, y, s=r.optimal_k)
    fig2, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(x, y, s=5, alpha=.3, color="#9E9E9E", label="Noisy data")
    ax.plot(x, best_spl(x), color="#F44336", lw=2, label=f"s={r.optimal_k}")
    overspl = UnivariateSpline(x, y, s=0)
    ax.plot(x, overspl(x), color="#2196F3", lw=1, alpha=.5, ls="--", label="s=0 (overfit)")
    ax.legend(fontsize=10); ax.grid(True, alpha=.3)
    ax.set_title("SciPy: Smoothing Spline with ATCND-selected parameter", fontsize=13)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    save_fig(fig2, "03_scipy_spline_fit")


# ════════════════════════════════════════════════════════════════
# 4. PANDAS — Optimal rolling window (BIC)
# ════════════════════════════════════════════════════════════════
def demo_pandas_rolling():
    print("[4/14] Pandas: Optimal rolling window size (BIC)")
    import pandas as pd
    from atcnd import search_rolling_window
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=365)
    trend = np.linspace(10, 20, 365)
    noise = 3 * np.random.randn(365)
    season = 5 * np.sin(np.arange(365) * 2 * np.pi / 30)
    series = pd.Series(trend + season + noise, index=dates)

    r = search_rolling_window(series, k_min=2, k_max=80, strategy="binary", method="bic")
    grid_n = 79
    print(f"  K*={r.optimal_k}  evals={len(r.search_history)}/{grid_n}  reduction={100*(1-len(r.search_history)/grid_n):.0f}%")

    fig = plot_curve(r, "Pandas: Optimal Rolling Window (BIC, binary search)",
                     xlabel="Window size", ylabel="BIC")
    save_fig(fig, "04_pandas_rolling_curve")

    smoothed = series.rolling(window=r.optimal_k, center=True).mean()
    fig2, ax = plt.subplots(figsize=(12, 5))
    ax.plot(series, alpha=.3, color="#9E9E9E", lw=.8, label="Raw")
    ax.plot(smoothed, color="#F44336", lw=2, label=f"Rolling mean (w={r.optimal_k})")
    ax.set_title(f"Pandas: Optimal Rolling Window (ATCND, w={r.optimal_k})", fontsize=13)
    ax.legend(fontsize=10); ax.grid(True, alpha=.3)
    save_fig(fig2, "04_pandas_rolling_fit")


# ════════════════════════════════════════════════════════════════
# 5. PANDAS — DataFrame column optimal bins
# ════════════════════════════════════════════════════════════════
def demo_pandas_bins():
    print("[5/14] Pandas + NumPy: DataFrame column optimal bins (AIC)")
    import pandas as pd
    from atcnd import search_dataframe_bins
    np.random.seed(42)
    df = pd.DataFrame({"price": np.exp(np.random.randn(2000) * 0.5 + 3)})

    r = search_dataframe_bins(df, "price", k_min=3, k_max=60, strategy="binary", method="aic")
    grid_n = 58
    print(f"  K*={r.optimal_k}  evals={len(r.search_history)}/{grid_n}  reduction={100*(1-len(r.search_history)/grid_n):.0f}%")

    fig = plot_curve(r, "Pandas + NumPy: Optimal Bins for DataFrame Column (AIC)",
                     xlabel="Number of bins", ylabel="AIC")
    save_fig(fig, "05_pandas_bins_curve")

    fig2, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["price"], bins=r.optimal_k, color="#4CAF50", edgecolor="w", alpha=.8)
    ax.set_title(f"Optimal {r.optimal_k} bins for 'price' column", fontsize=13)
    ax.set_xlabel("Price"); ax.set_ylabel("Count"); ax.grid(True, alpha=.3)
    save_fig(fig2, "05_pandas_bins_hist")


# ════════════════════════════════════════════════════════════════
# 6. SKLEARN — K-Means cluster count
# ════════════════════════════════════════════════════════════════
def demo_sklearn_kmeans():
    print("[6/14] sklearn: K-Means cluster count (silhouette)")
    from atcnd import search_model
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs
    np.random.seed(42)
    X, _ = make_blobs(n_samples=1000, n_features=50, centers=8, random_state=42)
    r = search_model(KMeans, X, param_name="n_clusters", k_min=2, k_max=30,
                     strategy="binary", metric="silhouette")
    grid_n = 29
    print(f"  K*={r.optimal_k} (true=8)  evals={len(r.search_history)}/{grid_n}  reduction={100*(1-len(r.search_history)/grid_n):.0f}%")

    fig = plot_curve(r, "sklearn: K-Means Cluster Selection (silhouette, binary search)",
                     xlabel="K (clusters)", ylabel="Silhouette")
    save_fig(fig, "06_sklearn_kmeans_curve")


# ════════════════════════════════════════════════════════════════
# 7. SKLEARN — KNN neighbors
# ════════════════════════════════════════════════════════════════
def demo_sklearn_knn():
    print("[7/14] sklearn: KNN optimal neighbors (CV accuracy)")
    from atcnd import search_neighbors
    from sklearn.datasets import load_iris
    iris = load_iris()
    r = search_neighbors(iris.data, k_min=1, k_max=30, strategy="binary", y=iris.target)
    grid_n = 30
    print(f"  K*={r.optimal_k}  evals={len(r.search_history)}/{grid_n}  reduction={100*(1-len(r.search_history)/grid_n):.0f}%  acc={r.optimal_score:.3f}")

    fig = plot_curve(r, "sklearn: KNN Neighbor Selection (5-fold CV accuracy, binary search)",
                     xlabel="K (neighbors)", ylabel="CV Accuracy")
    save_fig(fig, "07_sklearn_knn_curve")


# ════════════════════════════════════════════════════════════════
# 8. SKLEARN — PCA components
# ════════════════════════════════════════════════════════════════
def demo_sklearn_pca():
    print("[8/14] sklearn: PCA components (cumulative variance)")
    from atcnd import search_components
    from sklearn.datasets import load_digits
    d = load_digits()
    r = search_components(d.data, k_min=1, k_max=64, strategy="binary")
    grid_n = 64
    print(f"  K*={r.optimal_k} (95% var)  evals={len(r.search_history)}/{grid_n}  reduction={100*(1-len(r.search_history)/grid_n):.0f}%  cumvar={r.optimal_score:.3f}")

    ks = sorted(r.all_scores.keys())
    scores = [r.all_scores[k] for k in ks]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, scores, "o-", color="#2196F3", lw=2, ms=3)
    ax.axhline(0.95, color="#9E9E9E", ls=":", lw=1, label="95% threshold")
    ax.axvline(r.optimal_k, color="#F44336", ls="--", lw=2, label=f"K*={r.optimal_k}")
    ax.set_xlabel("Number of components", fontsize=12)
    ax.set_ylabel("Cumulative variance", fontsize=12)
    ax.set_title("sklearn: PCA Component Selection (ATCND binary search)", fontsize=13)
    ax.legend(fontsize=10); ax.grid(True, alpha=.3)
    save_fig(fig, "08_sklearn_pca")


# ════════════════════════════════════════════════════════════════
# 9. SKLEARN — Random Forest tree count
# ════════════════════════════════════════════════════════════════
def demo_sklearn_trees():
    print("[9/14] sklearn: Random Forest tree count (CV accuracy)")
    from atcnd import search_trees
    from sklearn.datasets import load_wine
    d = load_wine()
    r = search_trees(d.data, d.target, k_min=10, k_max=300, strategy="binary", cv=5)
    grid_n = 291
    print(f"  K*={r.optimal_k}  evals={len(r.search_history)}/{grid_n}  reduction={100*(1-len(r.search_history)/grid_n):.0f}%  acc={r.optimal_score:.3f}")

    fig = plot_curve(r, "sklearn: Random Forest Tree Count (5-fold CV, binary search)",
                     xlabel="Number of trees", ylabel="CV Accuracy")
    save_fig(fig, "09_sklearn_trees")


# ════════════════════════════════════════════════════════════════
# 10. SKLEARN — DBSCAN eps (discretized)
# ════════════════════════════════════════════════════════════════
def demo_sklearn_dbscan():
    print("[10/14] sklearn: DBSCAN eps (discretized, silhouette)")
    from atcnd import search_dbscan_eps
    from sklearn.datasets import make_moons
    np.random.seed(42)
    Xm, _ = make_moons(n_samples=500, noise=0.05, random_state=42)
    r = search_dbscan_eps(Xm, eps_min=5, eps_max=50, strategy="binary")
    best_eps = r.optimal_k / 10.0
    grid_n = 46
    print(f"  eps={best_eps:.1f}  evals={len(r.search_history)}/{grid_n}  reduction={100*(1-len(r.search_history)/grid_n):.0f}%")

    from sklearn.cluster import DBSCAN
    labels = DBSCAN(eps=best_eps, min_samples=5).fit_predict(Xm)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(Xm[:, 0], Xm[:, 1], c=labels, cmap="tab10", s=10, alpha=.7)
    ax.set_title(f"sklearn: DBSCAN with eps={best_eps:.1f} (ATCND)", fontsize=13)
    ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.grid(True, alpha=.3)
    save_fig(fig, "10_sklearn_dbscan")

    fig2 = plot_curve(r, "sklearn: DBSCAN eps Selection (silhouette, binary search)",
                      xlabel="eps × 10", ylabel="Silhouette")
    save_fig(fig2, "10_sklearn_dbscan_curve")


# ════════════════════════════════════════════════════════════════
# 11. GENSIM + SKLEARN — NMF topic count
# ════════════════════════════════════════════════════════════════
def demo_gensim_nmf():
    print("[11/14] Gensim + sklearn: NMF topic count (c_v coherence)")
    from atcnd import search_nmf_topics
    np.random.seed(42)
    templates = [
        "machine learning algorithm model neural network deep training data feature",
        "climate change carbon emission temperature global warming atmosphere sea level",
        "stock market trading investment portfolio financial risk return analysis",
        "protein dna gene cell biology research medical health molecule",
        "quantum computing particle physics theory experiment energy research",
    ]
    texts = []
    for t in templates * 40:
        texts.append(t + " " + " ".join(np.random.choice(t.split(), 3)))

    r = search_nmf_topics(texts, k_min=2, k_max=15, strategy="binary", metric="coherence")
    grid_n = 14
    print(f"  K*={r.optimal_k} (true=5)  evals={len(r.search_history)}/{grid_n}  reduction={100*(1-len(r.search_history)/grid_n):.0f}%  coh={r.optimal_score:.3f}")

    fig = plot_curve(r, "Gensim+sklearn: NMF Topic Count (c_v coherence, binary search)",
                     xlabel="K (topics)", ylabel="Coherence (c_v)")
    save_fig(fig, "11_gensim_nmf")


# ════════════════════════════════════════════════════════════════
# 12. PYTORCH — Hidden layer size
# ════════════════════════════════════════════════════════════════
def demo_torch_hidden():
    print("[12/14] PyTorch: Optimal hidden layer size")
    try:
        import torch, torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("  SKIPPED: pip install atcnd[torch]"); return
    from atcnd import search_hidden
    np.random.seed(42); torch.manual_seed(42)
    X_t = torch.randn(800, 20)
    y_t = (X_t[:, 0] + X_t[:, 1] > 0).long()
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=64, shuffle=True)
    r = search_hidden(loader, input_dim=20, output_dim=2,
                      k_min=8, k_max=256, strategy="binary", epochs=5, lr=1e-3)
    grid_n = 249
    print(f"  K*={r.optimal_k}  evals={len(r.search_history)}/{grid_n}  reduction={100*(1-len(r.search_history)/grid_n):.0f}%")

    fig = plot_curve(r, "PyTorch: Optimal Hidden Layer Size (binary search, -CE loss)",
                     xlabel="Hidden dim", ylabel="-Loss")
    save_fig(fig, "12_torch_hidden")


# ════════════════════════════════════════════════════════════════
# 13. PYTORCH — Number of hidden layers
# ════════════════════════════════════════════════════════════════
def demo_torch_layers():
    print("[13/14] PyTorch: Optimal hidden layer count")
    try:
        import torch, torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("  SKIPPED: pip install atcnd[torch]"); return
    from atcnd import search_layers
    np.random.seed(42); torch.manual_seed(42)
    X_t = torch.randn(800, 20)
    y_t = (X_t[:, 0] + X_t[:, 1] > 0).long()
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=64, shuffle=True)
    r = search_layers(loader, input_dim=20, output_dim=2, hidden_dim=64,
                      k_min=1, k_max=8, strategy="binary", epochs=5, lr=1e-3)
    grid_n = 8
    print(f"  K*={r.optimal_k}  evals={len(r.search_history)}/{grid_n}")

    fig = plot_curve(r, "PyTorch: Optimal Hidden Layer Count (-CE loss)",
                     xlabel="Number of layers", ylabel="-Loss")
    save_fig(fig, "13_torch_layers")


# ════════════════════════════════════════════════════════════════
# 14. COMPARISON — All 8 strategies on K-Means
# ════════════════════════════════════════════════════════════════
def demo_comparison():
    print("[14/14] Comparison: 8 strategies on K-Means (K_true=8, [2,30])")
    from atcnd import search, estimate_k_n_clusters
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    np.random.seed(42)
    X, _ = make_blobs(n_samples=1000, n_features=50, centers=8, random_state=42)

    def f(k):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        if len(set(labels)) < 2: return -1.0
        return silhouette_score(X, labels)

    hot = estimate_k_n_clusters(X, k_min=2, k_max=30)
    print(f"  hot_start estimate: K={hot} (true=8)")

    strategies = ["grid", "binary", "golden_section", "ternary",
                  "fibonacci", "interpolation", "exponential", "predictive"]
    results = {}
    for s in strategies:
        kw = dict(hot_start=hot) if s == "predictive" else {}
        results[s] = search(f, k_min=2, k_max=30, strategy=s, **kw)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    colors = {"grid":"#607D8B", "binary":"#4CAF50", "golden_section":"#FF9800",
              "ternary":"#9C27B0", "fibonacci":"#009688", "interpolation":"#673AB7",
              "exponential":"#FF5722", "predictive":"#E91E63"}
    grid_n = len(results["grid"].search_history)
    for i, s in enumerate(strategies):
        r = results[s]
        ax = axes[i // 4][i % 4]
        ks = sorted(r.all_scores.keys())
        scores = [r.all_scores[k] for k in ks]
        ax.plot(ks, scores, "o-", color=colors[s], lw=1.5, ms=3)
        ax.axvline(r.optimal_k, color="#F44336", ls="--", lw=2)
        n = len(r.search_history)
        pct = f"{100*(1-n/grid_n):.0f}%" if s != "grid" else "-"
        label = s.replace("_", " ").title()
        ax.set_title(f"{label}\nK*={r.optimal_k}, evals={n} ({pct})", fontsize=11)
        ax.set_xlabel("K"); ax.grid(True, alpha=.3)
        if i % 4 == 0: ax.set_ylabel("Silhouette")
    plt.suptitle("ATCND: All Search Strategies on K-Means (K_true=8)", fontsize=14, y=1.01)
    plt.tight_layout()
    save_fig(fig, "14_comparison_strategies")

    print(f"\n  {'Strategy':<18} {'K*':>4} {'Evals':>6} {'vs Grid':>8}")
    print(f"  {'-'*38}")
    for s in strategies:
        r = results[s]
        n = len(r.search_history)
        pct = f"{100*(1-n/grid_n):.0f}%" if s != "grid" else "-"
        print(f"  {s:<18} {r.optimal_k:>4} {n:>6} {pct:>8}")


def demo_summary():
    print("\n" + "=" * 60)
    print("SUMMARY: ATCND Adapter Demos")
    print("=" * 60)
    rows = [
        ("NumPy", "search_bins", "Histogram bins", "AIC"),
        ("SciPy/sklearn", "search_gmm_components", "GMM components", "BIC"),
        ("SciPy", "search_knots (UnivariateSpline)", "Smoothing param", "-MSE+penalty"),
        ("Pandas", "search_rolling_window", "Rolling window", "BIC"),
        ("Pandas+NumPy", "search_dataframe_bins", "DataFrame bins", "AIC"),
        ("sklearn", "search_model(KMeans)", "K-Means clusters", "silhouette"),
        ("sklearn", "search_neighbors", "KNN k", "CV accuracy"),
        ("sklearn", "search_components", "PCA components", "cum. var"),
        ("sklearn", "search_trees", "RF tree count", "CV accuracy"),
        ("sklearn", "search_dbscan_eps", "DBSCAN eps", "silhouette"),
        ("Gensim+sklearn", "search_nmf_topics", "NMF topics", "c_v coherence"),
        ("PyTorch", "search_hidden", "Hidden dim", "-CE loss"),
        ("PyTorch", "search_layers", "Hidden layers", "-CE loss"),
    ]
    print(f"  {'Library':<16} {'Adapter':<30} {'Parameter':<18} {'Metric'}")
    print(f"  {'-'*16} {'-'*30} {'-'*18} {'-'*20}")
    for lib, adapter, param, metric in rows:
        print(f"  {lib:<16} {adapter:<30} {param:<18} {metric}")
    print(f"\n  Figures: {FIGDIR}/")
    print(f"  Formats: .svg (markdown)  .pdf (LaTeX)  .png (wechat)")


if __name__ == "__main__":
    print("ATCND Comprehensive Demo")
    print("=" * 60)
    demo_numpy_bins()
    demo_scipy_gmm()
    demo_scipy_spline()
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
    print("\nDone!")