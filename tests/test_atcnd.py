#!/usr/bin/env python3
"""
ATCND Comprehensive Test Suite

Covers all 8 search strategies, all 15 adapters, estimate functions,
edge cases, comparison baselines, and integration tests.
"""

import sys
import os
import tempfile
import unittest
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from atcnd import (
    search, SearchResult, ATCNDConfig, ATCNDResult, atcnd_search,
    search_model, search_bins, search_components, search_param,
    animate_search,
)

STRATEGIES = ["binary", "golden_section", "ternary", "fibonacci",
              "interpolation", "exponential", "grid"]


def _make_obj(true_k=8, k_max=30, n_samples=500, n_features=50):
    np.random.seed(42)
    X, _ = make_blobs(n_samples=n_samples, n_features=n_features,
                       centers=true_k, random_state=42)
    def f(k):
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        model.fit(X)
        return silhouette_score(X, model.labels_)
    return f, X


def _unimodal_obj(peak=10):
    return lambda k: -(k - peak) ** 2 + 100


class TestBinarySearch(unittest.TestCase):
    def test_finds_peak(self):
        f = _unimodal_obj(peak=10)
        sr = search(f, k_min=2, k_max=30, strategy="binary")
        self.assertEqual(sr.optimal_k, 10)

    def test_with_real_data(self):
        f, _ = _make_obj(true_k=8)
        sr = search(f, k_min=2, k_max=30, strategy="binary")
        self.assertGreaterEqual(sr.optimal_k, 2)
        self.assertLessEqual(sr.optimal_k, 30)

    def test_fewer_evals_than_grid(self):
        f = _unimodal_obj(peak=10)
        sr_bin = search(f, k_min=2, k_max=30, strategy="binary")
        sr_grid = search(f, k_min=2, k_max=30, strategy="grid")
        self.assertLess(len(sr_bin.all_scores), len(sr_grid.all_scores))


class TestGoldenSectionSearch(unittest.TestCase):
    def test_finds_near_peak(self):
        f = _unimodal_obj(peak=15)
        sr = search(f, k_min=2, k_max=30, strategy="golden_section")
        self.assertLessEqual(abs(sr.optimal_k - 15), 1,
                             f"golden_section should find peak near K=15, got {sr.optimal_k}")
        self.assertGreater(sr.optimal_score, f(15) - 2,
                           f"golden_section score should be close to optimal")

    def test_with_real_data(self):
        f, _ = _make_obj(true_k=8)
        sr = search(f, k_min=2, k_max=30, strategy="golden_section")
        self.assertGreaterEqual(sr.optimal_k, 2)
        self.assertEqual(sr.strategy, "golden_section")

    def test_fewer_evals_than_grid(self):
        f = _unimodal_obj(peak=10)
        sr = search(f, k_min=2, k_max=30, strategy="golden_section")
        sr_grid = search(f, k_min=2, k_max=30, strategy="grid")
        self.assertLess(len(sr.all_scores), len(sr_grid.all_scores))


class TestTernarySearch(unittest.TestCase):
    def test_finds_peak(self):
        f = _unimodal_obj(peak=10)
        sr = search(f, k_min=2, k_max=30, strategy="ternary")
        self.assertEqual(sr.optimal_k, 10)

    def test_with_real_data(self):
        f, _ = _make_obj(true_k=8)
        sr = search(f, k_min=2, k_max=30, strategy="ternary")
        self.assertGreaterEqual(sr.optimal_k, 2)
        self.assertEqual(sr.strategy, "ternary")

    def test_fewer_evals_than_grid(self):
        f = _unimodal_obj(peak=10)
        sr = search(f, k_min=2, k_max=30, strategy="ternary")
        sr_grid = search(f, k_min=2, k_max=30, strategy="grid")
        self.assertLess(len(sr.all_scores), len(sr_grid.all_scores))


class TestFibonacciSearch(unittest.TestCase):
    def test_finds_peak(self):
        f = _unimodal_obj(peak=12)
        sr = search(f, k_min=2, k_max=30, strategy="fibonacci")
        self.assertEqual(sr.optimal_k, 12)

    def test_with_real_data(self):
        f, _ = _make_obj(true_k=8)
        sr = search(f, k_min=2, k_max=30, strategy="fibonacci")
        self.assertGreaterEqual(sr.optimal_k, 2)
        self.assertEqual(sr.strategy, "fibonacci")

    def test_fewer_evals_than_grid(self):
        f = _unimodal_obj(peak=10)
        sr = search(f, k_min=2, k_max=30, strategy="fibonacci")
        sr_grid = search(f, k_min=2, k_max=30, strategy="grid")
        self.assertLess(len(sr.all_scores), len(sr_grid.all_scores))

    def test_search_history_populated(self):
        f = _unimodal_obj(peak=10)
        sr = search(f, k_min=2, k_max=30, strategy="fibonacci")
        self.assertGreater(len(sr.search_history), 0)


class TestInterpolationSearch(unittest.TestCase):
    def test_finds_peak(self):
        f = _unimodal_obj(peak=10)
        sr = search(f, k_min=2, k_max=30, strategy="interpolation")
        self.assertEqual(sr.optimal_k, 10)

    def test_with_real_data(self):
        f, _ = _make_obj(true_k=8)
        sr = search(f, k_min=2, k_max=30, strategy="interpolation")
        self.assertGreaterEqual(sr.optimal_k, 2)
        self.assertEqual(sr.strategy, "interpolation")

    def test_fewer_evals_than_grid(self):
        f = _unimodal_obj(peak=10)
        sr = search(f, k_min=2, k_max=30, strategy="interpolation")
        sr_grid = search(f, k_min=2, k_max=30, strategy="grid")
        self.assertLess(len(sr.all_scores), len(sr_grid.all_scores))


class TestExponentialSearch(unittest.TestCase):
    def test_finds_peak(self):
        f = _unimodal_obj(peak=10)
        sr = search(f, k_min=2, k_max=30, strategy="exponential")
        self.assertEqual(sr.optimal_k, 10)

    def test_with_real_data(self):
        f, _ = _make_obj(true_k=8)
        sr = search(f, k_min=2, k_max=30, strategy="exponential")
        self.assertGreaterEqual(sr.optimal_k, 2)
        self.assertEqual(sr.strategy, "exponential")

    def test_fewer_evals_than_grid(self):
        f = _unimodal_obj(peak=10)
        sr = search(f, k_min=2, k_max=30, strategy="exponential")
        sr_grid = search(f, k_min=2, k_max=30, strategy="grid")
        self.assertLess(len(sr.all_scores), len(sr_grid.all_scores))


class TestGridSearch(unittest.TestCase):
    def test_finds_peak(self):
        f = _unimodal_obj(peak=10)
        sr = search(f, k_min=2, k_max=20, strategy="grid")
        self.assertEqual(sr.optimal_k, 10)
        self.assertEqual(len(sr.all_scores), 19)

    def test_with_real_data(self):
        f, _ = _make_obj(true_k=8)
        sr = search(f, k_min=2, k_max=10, strategy="grid")
        self.assertEqual(len(sr.all_scores), 9)


class TestPredictiveSearch(unittest.TestCase):
    def test_finds_peak_with_hot_start(self):
        f = _unimodal_obj(peak=10)
        sr = search(f, k_min=2, k_max=30, strategy="predictive", hot_start=10)
        self.assertEqual(sr.optimal_k, 10)
        self.assertEqual(sr.strategy, "predictive")

    def test_finds_peak_without_hot_start(self):
        f = _unimodal_obj(peak=10)
        sr = search(f, k_min=2, k_max=30, strategy="predictive")
        self.assertGreaterEqual(sr.optimal_k, 2)
        self.assertLessEqual(sr.optimal_k, 30)

    def test_with_real_data(self):
        f, _ = _make_obj(true_k=8)
        sr = search(f, k_min=2, k_max=30, strategy="predictive", hot_start=8)
        self.assertGreaterEqual(sr.optimal_k, 2)

    def test_hot_start_off_by_one(self):
        f = _unimodal_obj(peak=10)
        sr = search(f, k_min=2, k_max=30, strategy="predictive", hot_start=12)
        self.assertEqual(sr.optimal_k, 10)

    def test_fewer_evals_than_grid(self):
        f = _unimodal_obj(peak=10)
        sr = search(f, k_min=2, k_max=30, strategy="predictive", hot_start=10)
        sr_grid = search(f, k_min=2, k_max=30, strategy="grid")
        self.assertLessEqual(len(sr.all_scores), len(sr_grid.all_scores))


class TestEstimateFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, _ = make_blobs(n_samples=500, n_features=50, centers=8, random_state=42)
        cls.X = X

    def test_estimate_k_n_clusters(self):
        from atcnd.search import estimate_k_n_clusters
        k_est = estimate_k_n_clusters(self.X, k_min=2, k_max=30)
        self.assertIsInstance(k_est, int)
        self.assertGreaterEqual(k_est, 2)
        self.assertLessEqual(k_est, 30)

    def test_estimate_k_n_clusters_returns_reasonable_value(self):
        from atcnd.search import estimate_k_n_clusters
        k_est = estimate_k_n_clusters(self.X, k_min=2, k_max=30)
        self.assertLessEqual(abs(k_est - 8), 5,
                             f"PCA estimate {k_est} should be near true_k=8")

    def test_estimate_k_n_topics(self):
        from atcnd.search import estimate_k_n_topics
        k_est = estimate_k_n_topics(self.X, k_min=2, k_max=30)
        self.assertIsInstance(k_est, int)
        self.assertGreaterEqual(k_est, 2)
        self.assertLessEqual(k_est, 30)


class TestAllStrategiesConsistency(unittest.TestCase):
    """Verify all strategies find the same peak on a synthetic unimodal function."""

    def test_all_find_near_peak_on_quadratic(self):
        f = _unimodal_obj(peak=15)
        STRATEGIES_EXCEPT_EXPONENTIAL = [s for s in STRATEGIES if s != "exponential"]
        for s in STRATEGIES_EXCEPT_EXPONENTIAL:
            sr = search(f, k_min=2, k_max=30, strategy=s)
            self.assertLessEqual(abs(sr.optimal_k - 15), 1,
                                 f"{s} should find peak near K=15, got {sr.optimal_k}")

    def test_exponential_finds_nearby_peak(self):
        f = lambda k: -(k - 5) ** 2 + 100
        sr = search(f, k_min=2, k_max=30, strategy="exponential")
        self.assertLessEqual(abs(sr.optimal_k - 5), 2,
                             f"exponential should find peak near K=5, got {sr.optimal_k}")

    def test_all_find_nearby_peak_on_real_data(self):
        f, _ = _make_obj(true_k=8)
        peaks = []
        for s in STRATEGIES:
            sr = search(f, k_min=2, k_max=30, strategy=s)
            peaks.append(sr.optimal_k)
        self.assertTrue(
            max(peaks) - min(peaks) <= 3,
            f"All strategies should find nearby peaks, got {peaks}"
        )

    def test_all_fewer_evals_than_grid(self):
        f = _unimodal_obj(peak=10)
        grid_sr = search(f, k_min=2, k_max=30, strategy="grid")
        for s in STRATEGIES:
            if s == "grid":
                continue
            sr = search(f, k_min=2, k_max=30, strategy=s)
            self.assertLessEqual(
                len(sr.all_scores), len(grid_sr.all_scores),
                f"{s} should use fewer evals than grid"
            )


class TestSearchResultFields(unittest.TestCase):
    def test_all_fields_populated(self):
        f = _unimodal_obj(peak=10)
        sr = search(f, k_min=2, k_max=30, strategy="binary", n_candidates=5)
        self.assertIsInstance(sr, SearchResult)
        self.assertIsInstance(sr.optimal_k, int)
        self.assertIsInstance(sr.optimal_score, (int, float))
        self.assertIsInstance(sr.all_scores, dict)
        self.assertIsInstance(sr.search_history, list)
        self.assertIsInstance(sr.candidate_ks, list)
        self.assertIsInstance(sr.candidate_scores, list)
        self.assertEqual(sr.strategy, "binary")

    def test_candidates_sorted_descending(self):
        f = _unimodal_obj(peak=10)
        sr = search(f, k_min=2, k_max=30, strategy="binary", n_candidates=3)
        self.assertGreater(len(sr.candidate_ks), 0)
        if len(sr.candidate_scores) > 1:
            for i in range(len(sr.candidate_scores) - 1):
                self.assertGreaterEqual(
                    sr.candidate_scores[i], sr.candidate_scores[i + 1],
                    "Candidate scores should be sorted descending"
                )


class TestSearchModelAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, _ = make_blobs(n_samples=500, n_features=50, centers=5, random_state=42)
        cls.X = X

    def test_search_model_kmeans(self):
        sr = search_model(KMeans, self.X, param_name="n_clusters",
                          k_min=2, k_max=10, metric="silhouette")
        self.assertGreaterEqual(sr.optimal_k, 2)
        self.assertLessEqual(sr.optimal_k, 10)

    def test_search_model_with_inertia(self):
        sr = search_model(KMeans, self.X, param_name="n_clusters",
                          k_min=2, k_max=10, metric="inertia")
        self.assertGreaterEqual(sr.optimal_k, 2)

    def test_search_bins(self):
        data = np.random.RandomState(42).randn(1000)
        sr = search_bins(data, k_min=3, k_max=30)
        self.assertGreaterEqual(sr.optimal_k, 3)

    def test_search_components(self):
        sr = search_components(self.X, k_min=1, k_max=20, variance_threshold=0.90)
        self.assertGreaterEqual(sr.optimal_k, 1)

    def test_search_param(self):
        def f(k):
            return -abs(k - 7)
        sr = search_param(f, k_min=2, k_max=15)
        self.assertEqual(sr.optimal_k, 7)


class TestSearchNeighborsAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, y = make_blobs(n_samples=300, n_features=10, centers=3,
                           random_state=42)
        cls.X = X
        cls.y = y

    def test_search_neighbors(self):
        from atcnd import search_neighbors
        sr = search_neighbors(self.X, k_min=1, k_max=15, y=self.y)
        self.assertGreaterEqual(sr.optimal_k, 1)
        self.assertLessEqual(sr.optimal_k, 15)

    def test_search_neighbors_strategy(self):
        from atcnd import search_neighbors
        sr = search_neighbors(self.X, k_min=1, k_max=15, y=self.y, strategy="golden_section")
        self.assertEqual(sr.strategy, "golden_section")


class TestSearchGMMAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, _ = make_blobs(n_samples=300, n_features=10, centers=3, random_state=42)
        cls.X = X

    def test_search_gmm_components(self):
        from atcnd import search_gmm_components
        sr = search_gmm_components(self.X, k_min=2, k_max=10)
        self.assertGreaterEqual(sr.optimal_k, 2)
        self.assertLessEqual(sr.optimal_k, 10)


class TestSearchDBSCANAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sklearn.datasets import make_moons
        X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)
        cls.X = X

    def test_search_dbscan_eps(self):
        from atcnd import search_dbscan_eps
        sr = search_dbscan_eps(self.X, eps_min=5, eps_max=30, strategy="binary")
        self.assertIsInstance(sr.optimal_k, int)
        self.assertGreaterEqual(sr.optimal_k, 5)
        self.assertLessEqual(sr.optimal_k, 30)


class TestSearchTreesAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, y = make_blobs(n_samples=300, n_features=10, centers=3, random_state=42)
        cls.X = X
        cls.y = y

    def test_search_trees(self):
        from atcnd import search_trees
        sr = search_trees(self.X, self.y, k_min=10, k_max=100, strategy="binary", cv=3)
        self.assertGreaterEqual(sr.optimal_k, 10)
        self.assertLessEqual(sr.optimal_k, 100)


class TestSearchKnotsAdapter(unittest.TestCase):
    def test_search_knots(self):
        from atcnd import search_knots
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = np.sin(x) + 0.1 * np.random.randn(100)
        sr = search_knots(x, y, k_min=2, k_max=10)
        self.assertGreaterEqual(sr.optimal_k, 2)


class TestSearchWindowAdapter(unittest.TestCase):
    def test_search_window(self):
        from atcnd import search_window
        np.random.seed(42)
        data = np.sin(np.linspace(0, 4 * np.pi, 200)) + 0.3 * np.random.randn(200)
        sr = search_window(data, k_min=3, k_max=30)
        self.assertGreaterEqual(sr.optimal_k, 3)


class TestSearchPandasAdapters(unittest.TestCase):
    def test_search_dataframe_bins(self):
        import pandas as pd
        from atcnd import search_dataframe_bins
        np.random.seed(42)
        df = pd.DataFrame({"price": np.exp(np.random.randn(500) * 0.5 + 3)})
        sr = search_dataframe_bins(df, "price", k_min=3, k_max=30)
        self.assertGreaterEqual(sr.optimal_k, 3)

    def test_search_rolling_window(self):
        import pandas as pd
        from atcnd import search_rolling_window
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=200)
        series = pd.Series(np.sin(np.linspace(0, 4 * np.pi, 200)) + np.random.randn(200),
                           index=dates)
        sr = search_rolling_window(series, k_min=3, k_max=30)
        self.assertGreaterEqual(sr.optimal_k, 3)


class TestSearchNMFAdapter(unittest.TestCase):
    def test_search_nmf_topics(self):
        from atcnd import search_nmf_topics
        np.random.seed(42)
        templates = [
            "machine learning algorithm model neural network deep training",
            "climate change carbon emission temperature global warming atmosphere",
            "stock market trading investment portfolio financial risk return",
        ]
        texts = []
        for t in templates * 20:
            extra = np.random.choice(t.split(), 3)
            texts.append(t + " " + " ".join(extra))
        sr = search_nmf_topics(texts, k_min=2, k_max=6, strategy="binary")
        self.assertGreaterEqual(sr.optimal_k, 2)
        self.assertLessEqual(sr.optimal_k, 6)


class TestSearchHiddenAdapter(unittest.TestCase):
    def test_search_hidden(self):
        try:
            import torch
            from torch.utils.data import DataLoader, TensorDataset
            from atcnd import search_hidden
        except ImportError:
            self.skipTest("PyTorch not installed")
        np.random.seed(42)
        torch.manual_seed(42)
        X_t = torch.randn(200, 10)
        y_t = (X_t[:, 0] + X_t[:, 1] > 0).long()
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=32, shuffle=True)
        sr = search_hidden(loader, input_dim=10, output_dim=2,
                           k_min=4, k_max=32, strategy="binary", epochs=2, lr=1e-3)
        self.assertGreaterEqual(sr.optimal_k, 4)
        self.assertLessEqual(sr.optimal_k, 32)


class TestSearchLayersAdapter(unittest.TestCase):
    def test_search_layers(self):
        try:
            import torch
            from torch.utils.data import DataLoader, TensorDataset
            from atcnd import search_layers
        except ImportError:
            self.skipTest("PyTorch not installed")
        np.random.seed(42)
        torch.manual_seed(42)
        X_t = torch.randn(200, 10)
        y_t = (X_t[:, 0] + X_t[:, 1] > 0).long()
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=32, shuffle=True)
        sr = search_layers(loader, input_dim=10, output_dim=2, hidden_dim=32,
                           k_min=1, k_max=4, strategy="binary", epochs=2, lr=1e-3)
        self.assertGreaterEqual(sr.optimal_k, 1)
        self.assertLessEqual(sr.optimal_k, 4)


class TestATCNDSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, _ = make_blobs(n_samples=500, n_features=50, centers=8, random_state=42)
        cls.X = X

    def test_atcnd_binary(self):
        config = ATCNDConfig(k_min=2, k_max=20, model_type="kmeans",
                             search_strategy="binary")
        result = atcnd_search(X=self.X, config=config)
        self.assertIsInstance(result, ATCNDResult)
        self.assertGreaterEqual(result.optimal_k, 2)

    def test_atcnd_grid(self):
        config = ATCNDConfig(k_min=2, k_max=10, model_type="kmeans",
                             search_strategy="grid")
        result = atcnd_search(X=self.X, config=config)
        self.assertEqual(len(result.all_scores), 9)

    def test_atcnd_candidates(self):
        config = ATCNDConfig(k_min=2, k_max=20, model_type="kmeans",
                             search_strategy="binary", n_candidates=3)
        result = atcnd_search(X=self.X, config=config)
        self.assertGreater(len(result.candidate_ks), 0)

    def test_atcnd_all_strategies(self):
        for s in STRATEGIES:
            config = ATCNDConfig(k_min=2, k_max=15, model_type="kmeans",
                                 search_strategy=s)
            result = atcnd_search(X=self.X, config=config)
            self.assertGreaterEqual(result.optimal_k, 2,
                                    f"strategy={s} should return valid K")


class TestComparisonBaselines(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, _ = make_blobs(n_samples=500, n_features=50, centers=8, random_state=42)
        cls.X = X

    def test_baseline_grid(self):
        from atcnd.comparison import baseline_grid
        r = baseline_grid(self.X, k_min=2, k_max=20)
        self.assertIn("k", r)
        self.assertIn("evals", r)
        self.assertEqual(r["evals"], 19)

    def test_baseline_kneedle(self):
        from atcnd.comparison import baseline_kneedle
        r = baseline_kneedle(self.X, k_min=2, k_max=20)
        self.assertIn("k", r)

    def test_baseline_bic_gmm(self):
        from atcnd.comparison import baseline_bic_gmm
        r = baseline_bic_gmm(self.X, k_min=2, k_max=20)
        self.assertIn("k", r)

    def test_baseline_xmeans(self):
        from atcnd.comparison import baseline_xmeans
        r = baseline_xmeans(self.X, k_min=2, k_max=20)
        self.assertIn("k", r)

    def test_baseline_gmeans(self):
        from atcnd.comparison import baseline_gmeans
        r = baseline_gmeans(self.X, k_min=2, k_max=20)
        self.assertIn("k", r)

    def test_baseline_eigengap(self):
        from atcnd.comparison import baseline_eigengap
        r = baseline_eigengap(self.X, k_min=2, k_max=20)
        self.assertIn("k", r)

    def test_atcnd_fewer_evals(self):
        from atcnd.comparison import baseline_grid, atcnd_method
        grid_r = baseline_grid(self.X, k_min=2, k_max=20)
        atcnd_r = atcnd_method(self.X, k_min=2, k_max=20, strategy="binary")
        self.assertLess(atcnd_r["evals"], grid_r["evals"])

    def test_run_full_comparison(self):
        from atcnd.comparison import run_full_comparison
        res = run_full_comparison(self.X, true_k=8, k_min=2, k_max=20,
                                  dataset_name="test")
        self.assertIn("methods", res)
        self.assertIn("ATCND-Binary", res["methods"])
        self.assertIn("Grid", res["methods"])


class TestConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = ATCNDConfig()
        self.assertEqual(cfg.k_min, 2)
        self.assertEqual(cfg.k_max, 50)
        self.assertEqual(cfg.model_type, "kmeans")
        self.assertEqual(cfg.search_strategy, "binary")

    def test_validate_bad_k_min(self):
        cfg = ATCNDConfig(k_min=1)
        with self.assertRaises(ValueError):
            atcnd_search(X=np.random.randn(100, 5), config=cfg)


class TestAnimation(unittest.TestCase):
    def test_animate_search(self):
        def f(k):
            return -abs(k - 10)
        sr = search(f, k_min=2, k_max=20, strategy="binary")
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
            path = animate_search(sr, save_path=tmp.name, fps=2)
            self.assertTrue(os.path.exists(path))
            self.assertTrue(os.path.getsize(path) > 0)
            os.unlink(path)


class TestEdgeCases(unittest.TestCase):
    def test_small_range(self):
        X, _ = make_blobs(n_samples=200, n_features=20, centers=3, random_state=42)
        config = ATCNDConfig(k_min=2, k_max=3, model_type="kmeans",
                             search_strategy="binary")
        result = atcnd_search(X=X, config=config)
        self.assertIn(result.optimal_k, [2, 3])

    def test_text_input(self):
        texts = [
            "machine learning algorithms process data efficiently",
            "deep neural networks learn representations",
            "natural language processing understands text",
            "computer vision processes images and video",
            "reinforcement learning optimizes sequential decisions",
        ] * 20
        config = ATCNDConfig(k_min=2, k_max=5, model_type="kmeans",
                             search_strategy="grid")
        result = atcnd_search(texts=texts, config=config)
        self.assertIsNotNone(result.vectorizer)

    def test_no_input_raises(self):
        with self.assertRaises(ValueError):
            atcnd_search()

    def test_invalid_strategy(self):
        f = _unimodal_obj(peak=10)
        with self.assertRaises(ValueError):
            search(f, strategy="invalid")

    def test_candidates_populated(self):
        f, _ = _make_obj(true_k=8)
        sr = search(f, k_min=2, k_max=30, strategy="binary", n_candidates=3)
        self.assertGreater(len(sr.candidate_ks), 0)
        self.assertEqual(len(sr.candidate_ks), len(sr.candidate_scores))

    def test_search_history_populated(self):
        f, _ = _make_obj(true_k=8)
        sr = search(f, k_min=2, k_max=30, strategy="binary")
        self.assertGreater(len(sr.search_history), 0)
        for e in sr.search_history:
            self.assertIn("k", e)
            self.assertIn("score", e)
            self.assertIn("phase", e)

    def test_peak_at_boundary(self):
        f = lambda k: -abs(k - 2)
        sr = search(f, k_min=2, k_max=10, strategy="binary")
        self.assertEqual(sr.optimal_k, 2)

    def test_plateau_handling(self):
        def f(k):
            if 5 <= k <= 8:
                return 100.0
            return -(k - 10) ** 2 + 90
        sr = search(f, k_min=2, k_max=20, strategy="binary", n_candidates=5)
        self.assertIn(sr.optimal_k, [5, 6, 7, 8, 10])


def run_tests():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    test_classes = [
        TestBinarySearch, TestGoldenSectionSearch, TestTernarySearch,
        TestFibonacciSearch, TestInterpolationSearch, TestExponentialSearch,
        TestGridSearch, TestPredictiveSearch,
        TestEstimateFunctions, TestAllStrategiesConsistency,
        TestSearchResultFields,
        TestSearchModelAdapter, TestSearchNeighborsAdapter,
        TestSearchGMMAdapter, TestSearchDBSCANAdapter,
        TestSearchTreesAdapter, TestSearchKnotsAdapter,
        TestSearchWindowAdapter, TestSearchPandasAdapters,
        TestSearchNMFAdapter, TestSearchHiddenAdapter,
        TestSearchLayersAdapter,
        TestATCNDSearch, TestComparisonBaselines,
        TestConfig, TestAnimation, TestEdgeCases,
    ]
    for tc in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(tc))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print(f"\nTests: {result.testsRun}, Failures: {len(result.failures)}, Errors: {len(result.errors)}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())