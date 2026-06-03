#!/usr/bin/env python3
"""
ATCND Comprehensive Test Suite
Tests all components: config, search strategies, metrics, candidates, CLI
"""

import sys
import os
import json
import unittest
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from atcnd import ATCNDConfig, ATCNDResult, atcnd_search, print_topics
from atcnd.core import (
    _validate_config,
    _extract_topics,
    _compute_silhouette,
    _compute_coherence_fast,
    _update_candidates,
    _binary_search,
    _golden_section_search,
    _ternary_search,
    _grid_search,
)
from atcnd.benchmark import load_synthetic, load_synthetic_blobs


class TestATCNDConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = ATCNDConfig()
        self.assertEqual(cfg.k_min, 2)
        self.assertEqual(cfg.k_max, 50)
        self.assertEqual(cfg.model_type, "kmeans")
        self.assertEqual(cfg.search_strategy, "binary")
        self.assertEqual(cfg.metric, "silhouette")
        self.assertEqual(cfg.n_candidates, 3)

    def test_custom(self):
        cfg = ATCNDConfig(k_min=5, k_max=20, model_type="lda", search_strategy="grid", metric="coherence")
        self.assertEqual(cfg.k_min, 5)
        self.assertEqual(cfg.k_max, 20)
        self.assertEqual(cfg.model_type, "lda")

    def test_validate_ok(self):
        cfg = ATCNDConfig(k_min=2, k_max=10, model_type="kmeans", search_strategy="binary")
        _validate_config(cfg)

    def test_validate_k_min_lt_2(self):
        cfg = ATCNDConfig(k_min=1)
        with self.assertRaises(ValueError):
            _validate_config(cfg)

    def test_validate_k_max_le_k_min(self):
        cfg = ATCNDConfig(k_min=10, k_max=10)
        with self.assertRaises(ValueError):
            _validate_config(cfg)

    def test_validate_bad_model_type(self):
        cfg = ATCNDConfig(model_type="invalid")
        with self.assertRaises(ValueError):
            _validate_config(cfg)

    def test_validate_bad_strategy(self):
        cfg = ATCNDConfig(search_strategy="invalid")
        with self.assertRaises(ValueError):
            _validate_config(cfg)

    def test_validate_bad_metric(self):
        cfg = ATCNDConfig(metric="invalid")
        with self.assertRaises(ValueError):
            _validate_config(cfg)


class TestBinarySearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, _ = make_blobs(n_samples=500, n_features=50, centers=8, random_state=42)
        cls.X = X
        cls.config = ATCNDConfig(
            k_min=2, k_max=20, model_type="kmeans",
            search_strategy="binary", metric="silhouette",
            random_state=42, verbose=False,
        )

    def test_finds_reasonable_k(self):
        result = atcnd_search(X=self.X, config=self.config)
        self.assertIsInstance(result, ATCNDResult)
        self.assertGreaterEqual(result.optimal_k, 2)
        self.assertLessEqual(result.optimal_k, 20)
        self.assertGreater(result.optimal_score, -1.0)

    def test_fewer_evals_than_grid(self):
        result_binary = atcnd_search(X=self.X, config=self.config)
        config_grid = ATCNDConfig(
            k_min=2, k_max=20, model_type="kmeans",
            search_strategy="grid", metric="silhouette",
            random_state=42, verbose=False,
        )
        result_grid = atcnd_search(X=self.X, config=config_grid)
        self.assertLess(len(result_binary.all_scores), len(result_grid.all_scores))

    def test_binary_recover_true_k(self):
        result = atcnd_search(X=self.X, config=self.config)
        self.assertIn(8, result.all_scores)

    def test_candidates_populated(self):
        result = atcnd_search(X=self.X, config=self.config)
        self.assertGreater(len(result.candidate_ks), 0)
        self.assertEqual(len(result.candidate_ks), len(result.candidate_scores))

    def test_search_history_present(self):
        result = atcnd_search(X=self.X, config=self.config)
        self.assertGreater(len(result.search_history), 0)
        for entry in result.search_history:
            self.assertIn("k", entry)
            self.assertIn("score", entry)
            self.assertIn("phase", entry)

    def test_refinement_or_boundary_covers_range(self):
        result = atcnd_search(X=self.X, config=self.config)
        phases = set(e["phase"] for e in result.search_history)
        core_phases = {"binary_search", "boundary"}
        self.assertTrue(phases & core_phases, "Expected binary_search or boundary phase")


class TestGoldenSectionSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, _ = make_blobs(n_samples=500, n_features=50, centers=8, random_state=42)
        cls.X = X
        cls.config = ATCNDConfig(
            k_min=2, k_max=20, model_type="kmeans",
            search_strategy="golden_section", metric="silhouette",
            random_state=42, verbose=False,
        )

    def test_finds_reasonable_k(self):
        result = atcnd_search(X=self.X, config=self.config)
        self.assertGreaterEqual(result.optimal_k, 2)
        self.assertLessEqual(result.optimal_k, 20)

    def test_search_history_has_golden(self):
        result = atcnd_search(X=self.X, config=self.config)
        phases = [e["phase"] for e in result.search_history]
        self.assertIn("golden_section", phases)


class TestTernarySearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, _ = make_blobs(n_samples=500, n_features=50, centers=8, random_state=42)
        cls.X = X
        cls.config = ATCNDConfig(
            k_min=2, k_max=20, model_type="kmeans",
            search_strategy="ternary", metric="silhouette",
            random_state=42, verbose=False,
        )

    def test_finds_reasonable_k(self):
        result = atcnd_search(X=self.X, config=self.config)
        self.assertGreaterEqual(result.optimal_k, 2)
        self.assertLessEqual(result.optimal_k, 20)

    def test_search_history_has_ternary(self):
        result = atcnd_search(X=self.X, config=self.config)
        phases = [e["phase"] for e in result.search_history]
        self.assertIn("ternary_search", phases)


class TestGridSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, _ = make_blobs(n_samples=500, n_features=50, centers=8, random_state=42)
        cls.X = X
        cls.config = ATCNDConfig(
            k_min=2, k_max=10, model_type="kmeans",
            search_strategy="grid", metric="silhouette",
            random_state=42, verbose=False,
        )

    def test_evaluates_all_k(self):
        result = atcnd_search(X=self.X, config=self.config)
        self.assertEqual(len(result.all_scores), 9)

    def test_grid_finds_best(self):
        result = atcnd_search(X=self.X, config=self.config)
        scores = result.all_scores
        best_k = max(scores, key=scores.get)
        self.assertEqual(result.optimal_k, best_k)


class TestNMFSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        texts, _, _, _ = load_synthetic(n_docs=200, n_topics=5)
        cls.texts = texts

    def test_nmf_binary(self):
        config = ATCNDConfig(
            k_min=2, k_max=10, model_type="nmf",
            search_strategy="binary", metric="silhouette",
            random_state=42, nmf_max_iter=200,
        )
        result = atcnd_search(texts=self.texts, config=config)
        self.assertGreaterEqual(result.optimal_k, 2)
        self.assertLessEqual(result.optimal_k, 10)

    def test_nmf_grid(self):
        config = ATCNDConfig(
            k_min=2, k_max=8, model_type="nmf",
            search_strategy="grid", metric="silhouette",
            random_state=42, nmf_max_iter=200,
        )
        result = atcnd_search(texts=self.texts, config=config)
        self.assertEqual(len(result.all_scores), 7)


class TestLDASearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        texts, _, _, _ = load_synthetic(n_docs=200, n_topics=5)
        cls.texts = texts

    def test_lda_binary(self):
        config = ATCNDConfig(
            k_min=2, k_max=10, model_type="lda",
            search_strategy="binary", metric="silhouette",
            random_state=42, lda_max_iter=50,
        )
        result = atcnd_search(texts=self.texts, config=config)
        self.assertGreaterEqual(result.optimal_k, 2)
        self.assertLessEqual(result.optimal_k, 10)


class TestMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, _ = make_blobs(n_samples=300, n_features=30, centers=5, random_state=42)
        cls.X = X

    def test_silhouette_metric(self):
        config = ATCNDConfig(
            k_min=2, k_max=8, model_type="kmeans",
            search_strategy="grid", metric="silhouette",
        )
        result = atcnd_search(X=self.X, config=config)
        self.assertGreater(result.optimal_score, -1.0)

    def test_reconstruction_metric(self):
        config = ATCNDConfig(
            k_min=2, k_max=8, model_type="kmeans",
            search_strategy="grid", metric="reconstruction",
        )
        result = atcnd_search(X=self.X, config=config)
        self.assertIsInstance(result.optimal_score, float)


class TestCandidatesMechanism(unittest.TestCase):
    def test_update_candidates_empty(self):
        ks, scores = [], []
        _update_candidates(5, 0.8, None, ks, scores, 3)
        self.assertEqual(ks, [5])
        self.assertEqual(scores, [0.8])

    def test_update_candidates_sorted(self):
        ks, scores = [], []
        _update_candidates(5, 0.8, None, ks, scores, 3)
        _update_candidates(3, 0.9, None, ks, scores, 3)
        self.assertEqual(ks, [3, 5])
        self.assertEqual(scores, [0.9, 0.8])

    def test_update_candidates_eviction(self):
        ks, scores = [], []
        _update_candidates(2, 0.5, None, ks, scores, 3)
        _update_candidates(3, 0.6, None, ks, scores, 3)
        _update_candidates(4, 0.7, None, ks, scores, 3)
        _update_candidates(8, 0.55, None, ks, scores, 3)
        self.assertIn(8, ks)
        self.assertNotIn(2, ks)

    def test_n_candidates_limit(self):
        ks, scores = [], []
        for k in range(10):
            _update_candidates(k, float(k) / 10, None, ks, scores, 3)
        self.assertLessEqual(len(ks), 3)


class TestHelperFunctions(unittest.TestCase):
    def test_extract_topics_kmeans(self):
        from sklearn.cluster import KMeans
        X, _ = make_blobs(n_samples=100, n_features=20, centers=3, random_state=42)
        model = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X)
        feature_names = [f"f{i}" for i in range(20)]
        topics = _extract_topics(model, feature_names, "kmeans", top_n=5)
        self.assertEqual(len(topics), 3)
        for topic in topics:
            self.assertEqual(len(topic), 5)

    def test_compute_coherence_fast(self):
        from sklearn.cluster import KMeans
        X, _ = make_blobs(n_samples=100, n_features=20, centers=3, random_state=42)
        model = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X)
        feature_names = [f"f{i}" for i in range(20)]
        score = _compute_coherence_fast(model, feature_names, "kmeans")
        self.assertIsInstance(score, float)


class TestTextInput(unittest.TestCase):
    def test_text_input_text_vectorization(self):
        texts = [
            "machine learning algorithms process data efficiently",
            "deep neural networks learn representations automatically",
            "natural language processing understands human text",
            "computer vision processes images and videos",
            "reinforcement learning optimizes sequential decisions",
        ] * 20
        config = ATCNDConfig(
            k_min=2, k_max=5, model_type="kmeans",
            search_strategy="grid", metric="silhouette",
        )
        result = atcnd_search(texts=texts, config=config)
        self.assertIsNotNone(result.vectorizer)
        self.assertGreater(len(result.all_scores), 0)

    def test_no_input_raises(self):
        with self.assertRaises(ValueError):
            atcnd_search(config=ATCNDConfig())


class BenchmarkLoadersTest(unittest.TestCase):
    def test_load_synthetic_blobs(self):
        X, labels, true_k, name = load_synthetic_blobs(n_samples=200, n_centers=5)
        self.assertEqual(X.shape[0], 200)
        self.assertEqual(true_k, 5)
        self.assertEqual(name, "SyntheticBlobs")

    def test_load_synthetic_text(self):
        texts, labels, true_k, name = load_synthetic(n_docs=100, n_topics=4)
        self.assertEqual(len(texts), 100)
        self.assertEqual(true_k, 4)


class TestPrintTopics(unittest.TestCase):
    def test_print_topics_no_crash(self):
        X, _ = make_blobs(n_samples=200, n_features=20, centers=3, random_state=42)
        config = ATCNDConfig(
            k_min=2, k_max=5, model_type="kmeans",
            search_strategy="grid", metric="silhouette",
        )
        result = atcnd_search(X=X, config=config)
        try:
            print_topics(result, top_n=5)
        except Exception as e:
            self.fail(f"print_topics raised {e}")


class TestEdgeCases(unittest.TestCase):
    def test_small_range(self):
        X, _ = make_blobs(n_samples=200, n_features=20, centers=3, random_state=42)
        config = ATCNDConfig(
            k_min=2, k_max=3, model_type="kmeans",
            search_strategy="binary", metric="silhouette",
        )
        result = atcnd_search(X=X, config=config)
        self.assertIn(result.optimal_k, [2, 3])

    def test_single_model_eval(self):
        X, _ = make_blobs(n_samples=100, n_features=10, centers=2, random_state=42)
        from atcnd.core import _evaluate_k
        config = ATCNDConfig(model_type="kmeans", metric="silhouette")
        score, model = _evaluate_k(2, X, np.arange(10).astype(str), None, config)
        self.assertIsInstance(score, float)
        self.assertIsNotNone(model)


def run_tests():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestATCNDConfig,
        TestBinarySearch,
        TestGoldenSectionSearch,
        TestTernarySearch,
        TestGridSearch,
        TestNMFSearch,
        TestLDASearch,
        TestMetrics,
        TestCandidatesMechanism,
        TestHelperFunctions,
        TestTextInput,
        BenchmarkLoadersTest,
        TestPrintTopics,
        TestEdgeCases,
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    if result.wasSuccessful():
        print("\nAll tests passed!")
        return 0
    else:
        print("\nSome tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())