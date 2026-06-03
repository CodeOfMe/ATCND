#!/usr/bin/env python3
"""
ATCND Comprehensive Test Suite
"""

import sys
import os
import json
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


class TestPureSearch(unittest.TestCase):
    def _make_obj(self, true_k=8, k_max=30):
        np.random.seed(42)
        X, _ = make_blobs(n_samples=500, n_features=50, centers=true_k, random_state=42)
        def f(k):
            model = KMeans(n_clusters=k, n_init=10, random_state=42)
            model.fit(X)
            return silhouette_score(X, model.labels_)
        return f

    def test_binary_finds_k(self):
        f = self._make_obj(true_k=8)
        sr = search(f, k_min=2, k_max=30, strategy="binary")
        self.assertIsInstance(sr, SearchResult)
        self.assertGreaterEqual(sr.optimal_k, 2)
        self.assertLessEqual(sr.optimal_k, 30)
        self.assertIn(8, sr.all_scores)

    def test_golden_section_finds_k(self):
        f = self._make_obj(true_k=8)
        sr = search(f, k_min=2, k_max=30, strategy="golden_section")
        self.assertGreaterEqual(sr.optimal_k, 2)

    def test_ternary_finds_k(self):
        f = self._make_obj(true_k=8)
        sr = search(f, k_min=2, k_max=30, strategy="ternary")
        self.assertGreaterEqual(sr.optimal_k, 2)

    def test_grid_finds_k(self):
        f = self._make_obj(true_k=8)
        sr = search(f, k_min=2, k_max=10, strategy="grid")
        self.assertEqual(len(sr.all_scores), 9)

    def test_binary_fewer_evals_than_grid(self):
        f = self._make_obj(true_k=8)
        sr_bin = search(f, k_min=2, k_max=30, strategy="binary")
        sr_grid = search(f, k_min=2, k_max=30, strategy="grid")
        self.assertLess(len(sr_bin.all_scores), len(sr_grid.all_scores))

    def test_candidates_populated(self):
        f = self._make_obj(true_k=8)
        sr = search(f, k_min=2, k_max=30, strategy="binary", n_candidates=3)
        self.assertGreater(len(sr.candidate_ks), 0)
        self.assertEqual(len(sr.candidate_ks), len(sr.candidate_scores))

    def test_search_history(self):
        f = self._make_obj(true_k=8)
        sr = search(f, k_min=2, k_max=30, strategy="binary")
        self.assertGreater(len(sr.search_history), 0)
        for e in sr.search_history:
            self.assertIn("k", e)
            self.assertIn("score", e)
            self.assertIn("phase", e)

    def test_invalid_strategy(self):
        f = self._make_obj(true_k=8)
        with self.assertRaises(ValueError):
            search(f, strategy="invalid")

    def test_generic_callable(self):
        def f(k):
            return -abs(k - 15)
        sr = search(f, k_min=2, k_max=30, strategy="binary")
        self.assertEqual(sr.optimal_k, 15)

    def test_unimodal_synthetic(self):
        def f(k):
            return -(k - 10) ** 2 + 100
        sr = search(f, k_min=2, k_max=20, strategy="binary")
        self.assertEqual(sr.optimal_k, 10)

    def test_strategy_field(self):
        f = self._make_obj(true_k=8)
        sr = search(f, k_min=2, k_max=30, strategy="golden_section")
        self.assertEqual(sr.strategy, "golden_section")


class TestModelSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, _ = make_blobs(n_samples=500, n_features=50, centers=5, random_state=42)
        cls.X = X

    def test_search_model_kmeans(self):
        sr = search_model(KMeans, self.X, param_name="n_clusters",
                          k_min=2, k_max=10, metric="silhouette")
        self.assertGreaterEqual(sr.optimal_k, 2)
        self.assertLessEqual(sr.optimal_k, 10)

    def test_search_bins(self):
        data = np.random.randn(1000)
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


class TestATCNDSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, _ = make_blobs(n_samples=500, n_features=50, centers=8, random_state=42)
        cls.X = X

    def test_atcnd_binary(self):
        config = ATCNDConfig(k_min=2, k_max=20, model_type="kmeans", search_strategy="binary")
        result = atcnd_search(X=self.X, config=config)
        self.assertIsInstance(result, ATCNDResult)
        self.assertGreaterEqual(result.optimal_k, 2)

    def test_atcnd_grid(self):
        config = ATCNDConfig(k_min=2, k_max=10, model_type="kmeans", search_strategy="grid")
        result = atcnd_search(X=self.X, config=config)
        self.assertEqual(len(result.all_scores), 9)

    def test_atcnd_candidates(self):
        config = ATCNDConfig(k_min=2, k_max=20, model_type="kmeans", search_strategy="binary", n_candidates=3)
        result = atcnd_search(X=self.X, config=config)
        self.assertGreater(len(result.candidate_ks), 0)


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
        from atcnd.comparison import run_full_comparison, print_comparison_table
        res = run_full_comparison(self.X, true_k=8, k_min=2, k_max=20, dataset_name="test")
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
        import tempfile
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
        config = ATCNDConfig(k_min=2, k_max=3, model_type="kmeans", search_strategy="binary")
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
        config = ATCNDConfig(k_min=2, k_max=5, model_type="kmeans", search_strategy="grid")
        result = atcnd_search(texts=texts, config=config)
        self.assertIsNotNone(result.vectorizer)

    def test_no_input_raises(self):
        with self.assertRaises(ValueError):
            atcnd_search()


def run_tests():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    test_classes = [
        TestPureSearch, TestModelSearch, TestATCNDSearch,
        TestComparisonBaselines, TestConfig, TestAnimation, TestEdgeCases,
    ]
    for tc in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(tc))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print(f"\nTests: {result.testsRun}, Failures: {len(result.failures)}, Errors: {len(result.errors)}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())