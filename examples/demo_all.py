"""
ATCND Comprehensive Demo — Real datasets, 3D visualizations, all adapters.

Each demo: runs search → generates SVG+PDF+PNG figures → prints summary.
Run: python examples/demo_all.py
Output: examples/figures/*.{svg,pdf,png}
"""

import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGDIR, exist_ok=True)


def save_fig(fig, name):
    for ext in ("svg", "pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _annotate_eval_history(ax, history, color_map, all_scores=None, ks_range=None,
                          show_arrows=True, show_numbers=True, fontsize=7,
                          ms=7, arrow_alpha=0.35, arrow_lw=1.0):
    """Annotate evaluation history with numbered labels and direction arrows.

    Handles duplicate K values: only the first evaluation at each K gets a
    numbered label; subsequent re-evaluations get a small hollow marker.
    Label positions are adjusted to avoid overlapping text.
    """
    if not history:
        return
    visited_k_count = {}

    # First pass: compute label positions with collision avoidance
    label_positions = []
    for i, e in enumerate(history):
        k_val = e["k"]
        s_val = e["score"]
        visited_k_count[k_val] = visited_k_count.get(k_val, 0) + 1
        is_first = visited_k_count[k_val] == 1
        label_positions.append({
            "idx": i, "k": k_val, "score": s_val,
            "is_first": is_first, "phase": e.get("phase", ""),
            "visit_num": visited_k_count[k_val],
        })

    # Collision avoidance: shift labels that would overlap
    # We collect all label (x, y) positions and check pairwise distances
    min_dx = 1.5
    min_dy_frac = 0.04
    if all_scores and ks_range:
        score_min = min(all_scores.values())
        score_max = max(all_scores.values())
        dy_abs = (score_max - score_min) * min_dy_frac
    else:
        dy_abs = 0.03

    shifts = {}
    for i in range(len(label_positions)):
        if not label_positions[i]["is_first"]:
            continue
        x_i, y_i = label_positions[i]["k"], label_positions[i]["score"]
        shift_y = 0
        for j in range(i):
            if not label_positions[j]["is_first"]:
                continue
            x_j = label_positions[j]["k"] + shifts.get(j, (0, 0))[0]
            y_j = label_positions[j]["score"] + shifts.get(j, (0, 0))[1]
            if abs(x_i - x_j) < min_dx and abs(y_i - y_j) < dy_abs:
                shift_y = max(shift_y, shifts.get(j, (0, 0))[1] + dy_abs)
        shifts[i] = (0, shift_y)

    # Second pass: draw markers, labels, and arrows
    for i, lp in enumerate(label_positions):
        c = color_map.get(lp["phase"], "#9E9E9E")
        k_val, s_val = lp["k"], lp["score"]
        sy = shifts.get(i, (0, 0))[1]

        if lp["is_first"]:
            ax.plot(k_val, s_val, "o", color=c, ms=ms, zorder=5)
            if show_numbers:
                ax.annotate(str(i + 1), (k_val, s_val + sy),
                             textcoords="offset points",
                             xytext=(0, 10), ha='center', fontsize=fontsize,
                             color=c, fontweight='bold',
                             bbox=dict(boxstyle='round,pad=0.12', fc='white',
                                       ec=c, lw=0.4, alpha=0.85))
        else:
            ax.plot(k_val, s_val, "o", color=c, ms=ms - 2, zorder=5, alpha=0.4,
                    markerfacecolor='none', markeredgewidth=1.5)

    # Arrows between consecutive evaluations
    if show_arrows and len(history) > 1:
        for j in range(len(history) - 1):
            e1, e2 = history[j], history[j + 1]
            c = color_map.get(e1.get("phase", ""), "#9E9E9E")
            # If same K, no arrow needed (just re-evaluation)
            if e1["k"] == e2["k"]:
                continue
            ax.annotate('', xy=(e2["k"], e2["score"]),
                         xytext=(e1["k"], e1["score"]),
                         arrowprops=dict(arrowstyle='->', color=c,
                                         lw=arrow_lw, alpha=arrow_alpha,
                                         connectionstyle='arc3,rad=0'))


def plot_curve(result, title, xlabel="K", ylabel="Score", highlight=True):
    ks = sorted(result.all_scores.keys())
    scores = [result.all_scores[k] for k in ks]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, scores, "o-", color="#2196F3", lw=2, ms=4, label="f(K)")
    if highlight:
        ax.axvline(result.optimal_k, color="#F44336", ls="--", lw=2, label=f"K*={result.optimal_k}")
    phase_c = {"boundary":"#795548", "binary_search":"#4CAF50",
               "golden_section":"#FF9800", "ternary_search":"#9C27B0",
               "grid":"#607D8B", "refinement":"#E91E63", "final_sweep":"#00BCD4",
               "fibonacci":"#009688", "interpolation":"#673AB7",
               "exponential":"#FF5722", "predictive":"#E91E63",
               "probing":"#795548", "doubling":"#FF9800"}
    _annotate_eval_history(ax, result.search_history, phase_c,
                          all_scores=result.all_scores, ks_range=ks)
    ax.set_xlabel(xlabel, fontsize=12); ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13); ax.legend(fontsize=10); ax.grid(True, alpha=.3)
    return fig


# ════════════════════════════════════════════════════════════════
# 1. SKLEARN — Iris K-Means (real dataset, 3D scatter)
# ════════════════════════════════════════════════════════════════
def demo_sklearn_iris():
    print("[1/14] sklearn: K-Means on Iris (real data, 3D)")
    from atcnd import search_model
    from sklearn.datasets import load_iris
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    iris = load_iris()
    r = search_model(KMeans, iris.data, param_name="n_clusters", k_min=2, k_max=15,
                     strategy="binary", metric="silhouette")
    print(f"  K*={r.optimal_k} (true=3)  evals={len(r.search_history)}/{14}  reduction={100*(1-len(r.search_history)/14):.0f}%")

    fig = plot_curve(r, "sklearn: K-Means on Iris (silhouette, binary search)",
                     xlabel="K (clusters)", ylabel="Silhouette")
    save_fig(fig, "01_iris_kmeans_curve")

    km = KMeans(n_clusters=r.optimal_k, random_state=42, n_init=10).fit(iris.data)
    labels = km.predict(iris.data)
    X3 = PCA(n_components=3, random_state=42).fit_transform(iris.data)
    fig3 = plt.figure(figsize=(9, 7))
    ax3 = fig3.add_subplot(111, projection="3d")
    ax3.scatter(X3[:, 0], X3[:, 1], X3[:, 2], c=labels, cmap="Set1", s=40, alpha=.8, edgecolors="k", linewidths=.3)
    ax3.set_xlabel("PC1"); ax3.set_ylabel("PC2"); ax3.set_zlabel("PC3")
    ax3.set_title(f"Iris: K-Means K*={r.optimal_k} (ATCND)", fontsize=13)
    save_fig(fig3, "01_iris_kmeans_3d")


# ════════════════════════════════════════════════════════════════
# 2. SKLEARN — Wine GMM (real dataset, 3D)
# ════════════════════════════════════════════════════════════════
def demo_sklearn_wine_gmm():
    print("[2/14] sklearn: GMM on Wine (real data, 3D)")
    from atcnd import search_gmm_components
    from sklearn.datasets import load_wine
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    wine = load_wine()
    r = search_gmm_components(wine.data, k_min=2, k_max=15, strategy="binary")
    print(f"  K*={r.optimal_k}  evals={len(r.search_history)}/{14}  reduction={100*(1-len(r.search_history)/14):.0f}%")

    ks = sorted(r.all_scores.keys())
    scores = [r.all_scores[k] for k in ks]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, scores, "o-", color="#2196F3", lw=2, ms=5)
    ax.axvline(r.optimal_k, color="#F44336", ls="--", lw=2, label=f"K*={r.optimal_k}")
    ax.set_xlabel("Number of components", fontsize=12); ax.set_ylabel("BIC (lower=better)", fontsize=12)
    ax.set_title("sklearn: GMM on Wine (BIC, binary search)", fontsize=13)
    ax.legend(fontsize=10); ax.grid(True, alpha=.3)
    save_fig(fig, "02_wine_gmm_curve")

    gm = GaussianMixture(n_components=r.optimal_k, random_state=42).fit(wine.data)
    labels = gm.predict(wine.data)
    X3 = PCA(n_components=3, random_state=42).fit_transform(wine.data)
    fig3 = plt.figure(figsize=(9, 7))
    ax3 = fig3.add_subplot(111, projection="3d")
    ax3.scatter(X3[:, 0], X3[:, 1], X3[:, 2], c=labels, cmap="Set1", s=40, alpha=.8, edgecolors="k", linewidths=.3)
    ax3.set_xlabel("PC1"); ax3.set_ylabel("PC2"); ax3.set_zlabel("PC3")
    ax3.set_title(f"Wine: GMM K*={r.optimal_k} (ATCND)", fontsize=13)
    save_fig(fig3, "02_wine_gmm_3d")


# ════════════════════════════════════════════════════════════════
# 3. SKLEARN — Digits PCA (real dataset, 3D)
# ════════════════════════════════════════════════════════════════
def demo_sklearn_digits_pca():
    print("[3/14] sklearn: PCA on Digits (real data, 3D)")
    from atcnd import search_components
    from sklearn.datasets import load_digits
    from sklearn.decomposition import PCA
    digits = load_digits()
    r = search_components(digits.data, k_min=1, k_max=64, strategy="binary")
    print(f"  K*={r.optimal_k} (95% var)  evals={len(r.search_history)}/64  reduction={100*(1-len(r.search_history)/64):.0f}%")

    ks = sorted(r.all_scores.keys())
    scores = [r.all_scores[k] for k in ks]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, scores, "o-", color="#2196F3", lw=2, ms=3)
    ax.axhline(0.95, color="#9E9E9E", ls=":", lw=1, label="95% threshold")
    ax.axvline(r.optimal_k, color="#F44336", ls="--", lw=2, label=f"K*={r.optimal_k}")
    ax.set_xlabel("Number of components", fontsize=12); ax.set_ylabel("Cumulative variance", fontsize=12)
    ax.set_title("sklearn: PCA on Digits (binary search)", fontsize=13)
    ax.legend(fontsize=10); ax.grid(True, alpha=.3)
    save_fig(fig, "03_digits_pca_curve")

    X3 = PCA(n_components=3, random_state=42).fit_transform(digits.data)
    fig3 = plt.figure(figsize=(9, 7))
    ax3 = fig3.add_subplot(111, projection="3d")
    scatter = ax3.scatter(X3[:, 0], X3[:, 1], X3[:, 2], c=digits.target, cmap="tab10", s=15, alpha=.7)
    ax3.set_xlabel("PC1"); ax3.set_ylabel("PC2"); ax3.set_zlabel("PC3")
    ax3.set_title(f"Digits: 3D PCA (10 classes, K*={r.optimal_k} comps for 95%)", fontsize=13)
    fig3.colorbar(scatter, ax=ax3, shrink=.6, label="Digit")
    save_fig(fig3, "03_digits_pca_3d")


# ════════════════════════════════════════════════════════════════
# 4. SKLEARN — DBSCAN on Two-Moons (3D with 3rd feature)
# ════════════════════════════════════════════════════════════════
def demo_sklearn_moons_dbscan():
    print("[4/14] sklearn: DBSCAN on Two-Moons")
    from atcnd import search_dbscan_eps
    from sklearn.datasets import make_moons
    np.random.seed(42)
    Xm, ym = make_moons(n_samples=500, noise=0.05, random_state=42)
    r = search_dbscan_eps(Xm, eps_min=3, eps_max=30, strategy="binary")
    best_eps = r.optimal_k / 10.0
    print(f"  eps={best_eps:.1f}  evals={len(r.search_history)}/{28}  reduction={100*(1-len(r.search_history)/28):.0f}%")

    from sklearn.cluster import DBSCAN
    labels = DBSCAN(eps=best_eps, min_samples=5).fit_predict(Xm)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(Xm[:, 0], Xm[:, 1], c=labels, cmap="tab10", s=12, alpha=.7)
    ax.set_title(f"Two-Moons: DBSCAN eps={best_eps:.1f} (ATCND)", fontsize=13)
    ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.grid(True, alpha=.3)
    save_fig(fig, "04_moons_dbscan")

    fig2 = plot_curve(r, "sklearn: DBSCAN eps Selection (silhouette, binary search)",
                      xlabel="eps × 10", ylabel="Silhouette")
    save_fig(fig2, "04_moons_dbscan_curve")


# ════════════════════════════════════════════════════════════════
# 5. SKLEARN — Random Forest on Wine
# ════════════════════════════════════════════════════════════════
def demo_sklearn_wine_trees():
    print("[5/14] sklearn: Random Forest on Wine")
    from atcnd import search_trees
    from sklearn.datasets import load_wine
    wine = load_wine()
    r = search_trees(wine.data, wine.target, k_min=10, k_max=300, strategy="binary", cv=5)
    grid_n = 291
    print(f"  K*={r.optimal_k}  evals={len(r.search_history)}/{grid_n}  reduction={100*(1-len(r.search_history)/grid_n):.0f}%  acc={r.optimal_score:.3f}")

    fig = plot_curve(r, "sklearn: Random Forest Tree Count on Wine (5-fold CV)",
                     xlabel="Number of trees", ylabel="CV Accuracy")
    save_fig(fig, "05_wine_rf_trees")


# ════════════════════════════════════════════════════════════════
# 6. SKLEARN — KNN on Iris
# ════════════════════════════════════════════════════════════════
def demo_sklearn_iris_knn():
    print("[6/14] sklearn: KNN on Iris")
    from atcnd import search_neighbors
    from sklearn.datasets import load_iris
    iris = load_iris()
    r = search_neighbors(iris.data, k_min=1, k_max=30, strategy="binary", y=iris.target)
    grid_n = 30
    print(f"  K*={r.optimal_k}  evals={len(r.search_history)}/{grid_n}  reduction={100*(1-len(r.search_history)/grid_n):.0f}%  acc={r.optimal_score:.3f}")

    fig = plot_curve(r, "sklearn: KNN on Iris (5-fold CV accuracy, binary search)",
                     xlabel="K (neighbors)", ylabel="CV Accuracy")
    save_fig(fig, "06_iris_knn_curve")


# ════════════════════════════════════════════════════════════════
# 7. NUMPY — Optimal histogram bins (AIC)
# ════════════════════════════════════════════════════════════════
def demo_numpy_bins():
    print("[7/14] NumPy: Optimal histogram bins (AIC)")
    from atcnd import search_bins
    np.random.seed(42)
    data = np.concatenate([np.random.normal(0, 1, 500), np.random.normal(5, 0.8, 300)])
    r = search_bins(data, k_min=3, k_max=60, strategy="binary", method="aic")
    grid_n = 58
    print(f"  K*={r.optimal_k}  evals={len(r.search_history)}/{grid_n}  reduction={100*(1-len(r.search_history)/grid_n):.0f}%")

    fig2, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    a1.hist(data, bins=r.optimal_k, color="#4CAF50", edgecolor="w", alpha=.8)
    a1.set_title(f"ATCND optimal: {r.optimal_k} bins (AIC)", fontsize=12)
    a1.set_xlabel("Value"); a1.set_ylabel("Count")
    default_bins = int(np.ceil(1 + 3.322 * np.log10(len(data))))
    a2.hist(data, bins=default_bins, color="#9E9E9E", edgecolor="w", alpha=.8)
    a2.set_title(f"Sturges rule: {default_bins} bins", fontsize=12)
    a2.set_xlabel("Value"); a2.set_ylabel("Count")
    plt.tight_layout()
    save_fig(fig2, "07_numpy_bins_compare")
    save_fig(plot_curve(r, "NumPy: Optimal Bins (AIC, binary search)",
                        xlabel="Number of bins", ylabel="AIC"), "07_numpy_bins_curve")


# ════════════════════════════════════════════════════════════════
# 8. SCIPY — Smoothing parameter for UnivariateSpline
# ════════════════════════════════════════════════════════════════
def demo_scipy_spline():
    print("[8/14] SciPy: Optimal smoothing parameter (UnivariateSpline)")
    from atcnd import search as pure_search
    from scipy.interpolate import UnivariateSpline
    from sklearn.metrics import mean_squared_error
    np.random.seed(42)
    x = np.linspace(0, 2 * np.pi, 200)
    y = np.sin(x) + 0.15 * np.random.randn(200)

    def f_spline(s):
        try:
            spl = UnivariateSpline(x, y, s=s)
            return -mean_squared_error(y, spl(x))
        except Exception:
            return -1e10

    r = pure_search(f_spline, k_min=1, k_max=200, strategy="binary")
    print(f"  optimal s={r.optimal_k}  evals={len(r.search_history)}/200  reduction={100*(1-len(r.search_history)/200):.0f}%")

    best_spl = UnivariateSpline(x, y, s=r.optimal_k)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(x, y, s=5, alpha=.3, color="#9E9E9E", label="Noisy data")
    ax.plot(x, best_spl(x), color="#F44336", lw=2, label=f"s={r.optimal_k}")
    overspl = UnivariateSpline(x, y, s=0)
    ax.plot(x, overspl(x), color="#2196F3", lw=1, alpha=.5, ls="--", label="s=0 (overfit)")
    ax.plot(x, np.sin(x), color="#4CAF50", lw=1.5, alpha=.7, label="True sin(x)")
    ax.legend(fontsize=10); ax.grid(True, alpha=.3)
    ax.set_title("SciPy: Smoothing Spline (ATCND-selected parameter)", fontsize=13)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    save_fig(fig, "08_scipy_spline_fit")
    save_fig(plot_curve(r, "SciPy: Smoothing Parameter (binary search)",
                        xlabel="s (smoothing)", ylabel="-MSE"), "08_scipy_spline_curve")


# ════════════════════════════════════════════════════════════════
# 9. PANDAS — Rolling window (BIC) on synthetic time series
# ════════════════════════════════════════════════════════════════
def demo_pandas_rolling():
    print("[9/14] Pandas: Optimal rolling window (BIC)")
    import pandas as pd
    from atcnd import search_rolling_window
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=365)
    trend = np.linspace(10, 20, 365)
    noise = 3 * np.random.randn(365)
    season = 5 * np.sin(np.arange(365) * 2 * np.pi / 30)
    series = pd.Series(trend + season + noise, index=dates)
    r = search_rolling_window(series, k_min=3, k_max=50, strategy="binary", method="bic")
    grid_n = 48
    print(f"  K*={r.optimal_k}  evals={len(r.search_history)}/{grid_n}  reduction={100*(1-len(r.search_history)/grid_n):.0f}%")

    smoothed = series.rolling(window=r.optimal_k, center=True).mean()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(series, alpha=.3, color="#9E9E9E", lw=.8, label="Raw")
    ax.plot(smoothed, color="#F44336", lw=2, label=f"Rolling mean (w={r.optimal_k})")
    ax.set_title(f"Pandas: Optimal Rolling Window (ATCND, w={r.optimal_k})", fontsize=13)
    ax.legend(fontsize=10); ax.grid(True, alpha=.3)
    save_fig(fig, "09_pandas_rolling_fit")
    save_fig(plot_curve(r, "Pandas: Rolling Window (BIC, binary search)",
                        xlabel="Window size", ylabel="BIC"), "09_pandas_rolling_curve")


# ════════════════════════════════════════════════════════════════
# 10. PANDAS — DataFrame column bins (AIC)
# ════════════════════════════════════════════════════════════════
def demo_pandas_bins():
    print("[10/14] Pandas+NumPy: DataFrame column bins (AIC)")
    import pandas as pd
    from atcnd import search_dataframe_bins
    np.random.seed(42)
    df = pd.DataFrame({"price": np.exp(np.random.randn(2000) * 0.5 + 3),
                        "volume": np.random.exponential(100, 2000)})
    r = search_dataframe_bins(df, "price", k_min=3, k_max=60, strategy="binary", method="aic")
    grid_n = 58
    print(f"  K*={r.optimal_k}  evals={len(r.search_history)}/{grid_n}  reduction={100*(1-len(r.search_history)/grid_n):.0f}%")

    save_fig(plot_curve(r, "Pandas: DataFrame Column Bins (AIC, binary search)",
                        xlabel="Number of bins", ylabel="AIC"), "10_pandas_bins_curve")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["price"], bins=r.optimal_k, color="#4CAF50", edgecolor="w", alpha=.8)
    ax.set_title(f"Optimal {r.optimal_k} bins for 'price' (ATCND, AIC)", fontsize=13)
    ax.set_xlabel("Price"); ax.set_ylabel("Count"); ax.grid(True, alpha=.3)
    save_fig(fig, "10_pandas_bins_hist")


# ════════════════════════════════════════════════════════════════
# 11. GENSIM+SKLEARN — NMF topic count
# ════════════════════════════════════════════════════════════════
def demo_gensim_nmf():
    print("[11/14] Gensim+sklearn: NMF topic count (c_v)")
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
    for t in templates * 50:
        texts.append(t + " " + " ".join(np.random.choice(t.split(), 3)))
    r = search_nmf_topics(texts, k_min=2, k_max=15, strategy="binary", metric="coherence")
    grid_n = 14
    print(f"  K*={r.optimal_k} (true=5)  evals={len(r.search_history)}/{grid_n}  reduction={100*(1-len(r.search_history)/grid_n):.0f}%")

    save_fig(plot_curve(r, "Gensim+sklearn: NMF Topics (c_v coherence, binary search)",
                        xlabel="K (topics)", ylabel="Coherence (c_v)"), "11_gensim_nmf")


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
    X_t = torch.randn(800, 20); y_t = (X_t[:, 0] + X_t[:, 1] > 0).long()
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=64, shuffle=True)
    r = search_hidden(loader, input_dim=20, output_dim=2,
                      k_min=8, k_max=256, strategy="binary", epochs=5, lr=1e-3)
    grid_n = 249
    print(f"  K*={r.optimal_k}  evals={len(r.search_history)}/{grid_n}  reduction={100*(1-len(r.search_history)/grid_n):.0f}%")
    save_fig(plot_curve(r, "PyTorch: Hidden Layer Size (binary search, -CE loss)",
                        xlabel="Hidden dim", ylabel="-Loss"), "12_torch_hidden")


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
    X_t = torch.randn(800, 20); y_t = (X_t[:, 0] + X_t[:, 1] > 0).long()
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=64, shuffle=True)
    r = search_layers(loader, input_dim=20, output_dim=2, hidden_dim=64,
                      k_min=1, k_max=8, strategy="binary", epochs=5, lr=1e-3)
    grid_n = 8
    print(f"  K*={r.optimal_k}  evals={len(r.search_history)}/{grid_n}")
    save_fig(plot_curve(r, "PyTorch: Hidden Layer Count (-CE loss)",
                        xlabel="Number of layers", ylabel="-Loss"), "13_torch_layers")


# ════════════════════════════════════════════════════════════════
# 14. COMPARISON — All 8 strategies on K-Means (real+synthetic 3D)
# ════════════════════════════════════════════════════════════════
def demo_comparison():
    print("[14/14] Comparison: 8 strategies on K-Means (Iris 3D + SyntheticBlobs)")
    from atcnd import search, estimate_k_n_clusters
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    np.random.seed(42)
    X, y_true = make_blobs(n_samples=1000, n_features=50, centers=8, random_state=42)

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

    grid_n = len(results["grid"].search_history)
    print(f"\n  {'Strategy':<18} {'K*':>4} {'Evals':>6} {'vs Grid':>8} {'Complexity'}")
    print(f"  {'-'*58}")
    complexities = {'grid':'O(N)', 'binary':'O(log N)', 'golden_section':'O(log_φ N)',
                   'ternary':'O(log_{1.5} N)', 'fibonacci':'O(log_φ N)',
                   'interpolation':'O(log log N)*', 'exponential':'O(log K*)',
                   'predictive':'O(1)+O(log Δ)'}
    for s in strategies:
        r = results[s]
        n = len(r.search_history)
        pct = f"{100*(1-n/grid_n):.0f}%" if s != "grid" else "-"
        print(f"  {s:<18} {r.optimal_k:>4} {n:>6} {pct:>8} {complexities.get(s,'')}")

    # 3D scatter of K-Means result
    from sklearn.decomposition import PCA
    km = KMeans(n_clusters=8, random_state=42, n_init=10).fit(X)
    labels = km.predict(X)
    X3 = PCA(n_components=3, random_state=42).fit_transform(X)
    fig3 = plt.figure(figsize=(10, 8))
    ax3 = fig3.add_subplot(111, projection="3d")
    ax3.scatter(X3[:, 0], X3[:, 1], X3[:, 2], c=labels, cmap="tab10", s=8, alpha=.6)
    ax3.set_xlabel("PC1"); ax3.set_ylabel("PC2"); ax3.set_zlabel("PC3")
    ax3.set_title("SyntheticBlobs: K-Means K=8 (ATCND)", fontsize=13)
    save_fig(fig3, "14_blobs_3d")

    # 8-strategy comparison figure
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    colors = {"grid":"#607D8B", "binary":"#4CAF50", "golden_section":"#FF9800",
              "ternary":"#9C27B0", "fibonacci":"#009688", "interpolation":"#673AB7",
              "exponential":"#FF5722", "predictive":"#E91E63"}
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
        is_grid = (s == "grid")
        show_nums = not is_grid
        phase_c = {"boundary":"#795548", "binary_search":"#4CAF50",
                   "golden_section":"#FF9800", "ternary_search":"#9C27B0",
                   "grid":"#607D8B", "refinement":"#E91E63", "final_sweep":"#00BCD4",
                   "fibonacci":"#009688", "interpolation":"#673AB7",
                   "exponential":"#FF5722", "predictive":"#E91E63",
                   "probing":"#795548", "doubling":"#FF9800"}
        _annotate_eval_history(ax, r.search_history,
                              {**phase_c, **{s: colors[s]}},
                              all_scores=r.all_scores, ks_range=ks,
                              show_numbers=show_nums, show_arrows=not is_grid,
                              ms=6 if not is_grid else 3, fontsize=6.5,
                              arrow_alpha=0.3, arrow_lw=0.8)
    plt.suptitle("ATCND: All Search Strategies on K-Means (K_true=8)", fontsize=14, y=1.01)
    plt.tight_layout()
    save_fig(fig, "14_comparison_strategies")


def demo_summary():
    print("\n" + "=" * 60)
    print("SUMMARY: ATCND Adapter Demos on Real Datasets")
    print("=" * 60)
    rows = [
        ("sklearn", "K-Means on Iris", "K*", "3D scatter", "silhouette"),
        ("sklearn", "GMM on Wine", "components", "3D scatter", "BIC"),
        ("sklearn", "PCA on Digits", "components", "3D scatter", "cum. var"),
        ("sklearn", "DBSCAN on Moons", "eps", "2D scatter", "silhouette"),
        ("sklearn", "RF on Wine", "trees", "curve", "CV accuracy"),
        ("sklearn", "KNN on Iris", "k", "curve", "CV accuracy"),
        ("NumPy", "Histogram bins", "bins", "compare", "AIC"),
        ("SciPy", "UnivariateSpline", "s", "curve+fit", "-MSE"),
        ("Pandas", "Rolling window", "w", "time series", "BIC"),
        ("Pandas", "DataFrame bins", "bins", "histogram", "AIC"),
        ("Gensim", "NMF topics", "K", "curve", "c_v"),
        ("PyTorch", "Hidden dim", "dim", "curve", "-CE"),
        ("PyTorch", "Hidden layers", "layers", "curve", "-CE"),
        ("All", "8 strategies", "K", "3D+8-panel", "silhouette"),
    ]
    print(f"  {'Library':<10} {'Demo':<22} {'Param':<10} {'Viz':<12} {'Metric'}")
    print(f"  {'-'*66}")
    for lib, demo, param, viz, metric in rows:
        print(f"  {lib:<10} {demo:<22} {param:<10} {viz:<12} {metric}")
    print(f"\n  Figures: {FIGDIR}/")
    print(f"  Formats: .svg (markdown)  .pdf (LaTeX)  .png (WeChat)")


if __name__ == "__main__":
    print("ATCND Comprehensive Demo (Real Datasets + 3D)")
    print("=" * 60)
    demo_sklearn_iris()
    demo_sklearn_wine_gmm()
    demo_sklearn_digits_pca()
    demo_sklearn_moons_dbscan()
    demo_sklearn_wine_trees()
    demo_sklearn_iris_knn()
    demo_numpy_bins()
    demo_scipy_spline()
    demo_pandas_rolling()
    demo_pandas_bins()
    demo_gensim_nmf()
    demo_torch_hidden()
    demo_torch_layers()
    demo_comparison()
    demo_summary()
    print("\nDone!")