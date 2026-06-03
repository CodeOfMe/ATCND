"""ATCND CLI entry point."""

import argparse
import sys
import json
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        prog="atcnd",
        description="ATCND: Adaptive Topic and Cluster Number Determination",
    )
    parser.add_argument("-V", "--version", action="version", version="atcnd 0.5.0")

    sub = parser.add_subparsers(dest="command")

    sp = sub.add_parser("search", help="Run ATCND search")
    sp.add_argument("--model", choices=["lda", "nmf", "kmeans"], default="kmeans")
    sp.add_argument("--strategy", choices=["binary", "golden_section", "ternary", "grid"], default="binary")
    sp.add_argument("--metric", choices=["silhouette", "coherence", "perplexity", "reconstruction", "combined"], default="silhouette")
    sp.add_argument("--k-min", type=int, default=2)
    sp.add_argument("--k-max", type=int, default=50)
    sp.add_argument("--n-candidates", type=int, default=3)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("-v", "--verbose", action="store_true")
    sp.add_argument("-o", "--output", help="Save plot to file")
    sp.add_argument("--json", action="store_true")
    sp.add_argument("-q", "--quiet", action="store_true")

    cp = sub.add_parser("compare", help="Run comparison against literature baselines")
    cp.add_argument("--dataset", choices=["blobs", "moons", "circles", "synthetic"], default="blobs")
    cp.add_argument("--k-min", type=int, default=2)
    cp.add_argument("--k-max", type=int, default=30)
    cp.add_argument("--n-samples", type=int, default=1000)
    cp.add_argument("--true-k", type=int, default=8)

    ap = sub.add_parser("animate", help="Animate search process as GIF")
    ap.add_argument("--strategy", choices=["binary", "golden_section", "ternary", "grid"], default="binary")
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=30)
    ap.add_argument("--fps", type=int, default=3)
    ap.add_argument("-o", "--output", default="atcnd_search.gif")

    bp = sub.add_parser("benchmark", help="Run internal benchmarks")
    bp.add_argument("--dataset", choices=["synthetic", "blobs"], default="blobs")
    bp.add_argument("--k-min", type=int, default=2)
    bp.add_argument("--k-max", type=int, default=30)
    bp.add_argument("--n-samples", type=int, default=1000)

    adp = sub.add_parser("adaptive", help="Adaptive strategy selection based on data characteristics")
    adp.add_argument("--k-min", type=int, default=2)
    adp.add_argument("--k-max", type=int, default=50)
    adp.add_argument("--seed", type=int, default=42)
    adp.add_argument("--run", action="store_true", help="Run search with recommended strategy")
    adp.add_argument("--json", action="store_true")

    mop = sub.add_parser("multi", help="Multi-objective optimization across metrics")
    mop.add_argument("--metrics", nargs="+", default=["silhouette", "bic"],
                     choices=["silhouette", "silhouette_knee", "bic", "combined", "silhouette_drop"])
    mop.add_argument("--weights", nargs="+", type=float, default=None, help="Weights per metric (sum to 1)")
    mop.add_argument("--k-min", type=int, default=2)
    mop.add_argument("--k-max", type=int, default=50)
    mop.add_argument("--seed", type=int, default=42)
    mop.add_argument("--json", action="store_true")

    args = parser.parse_args()

    if args.command == "search":
        _cmd_search(args)
    elif args.command == "compare":
        _cmd_compare(args)
    elif args.command == "animate":
        _cmd_animate(args)
    elif args.command == "benchmark":
        _cmd_benchmark(args)
    elif args.command == "adaptive":
        _cmd_adaptive(args)
    elif args.command == "multi":
        _cmd_multi(args)
    else:
        parser.print_help()


def _cmd_search(args):
    from atcnd import ATCNDConfig, atcnd_search, print_topics, plot_search_curve
    config = ATCNDConfig(
        k_min=args.k_min, k_max=args.k_max, metric=args.metric,
        model_type=args.model, search_strategy=args.strategy,
        random_state=args.seed, n_candidates=args.n_candidates, verbose=args.verbose,
    )
    if args.model == "kmeans":
        from sklearn.datasets import make_blobs
        X, _ = make_blobs(n_samples=500, n_features=50, centers=8, random_state=args.seed)
        result = atcnd_search(X=X, config=config)
    else:
        from atcnd.benchmark import load_synthetic
        texts, _, _, _ = load_synthetic(n_docs=500, n_topics=8)
        result = atcnd_search(texts=texts, config=config)

    if args.json:
        print(json.dumps({
            "optimal_k": result.optimal_k, "optimal_score": result.optimal_score,
            "model_type": result.model_type, "evaluations": len(result.search_history),
            "candidate_ks": result.candidate_ks, "candidate_scores": result.candidate_scores,
        }, indent=2))
    elif not args.quiet:
        print_topics(result)
    if args.output:
        plot_search_curve(result, save_path=args.output)


def _cmd_compare(args):
    from atcnd.comparison import run_full_comparison, print_comparison_table
    if args.dataset == "blobs":
        from sklearn.datasets import make_blobs
        X, _ = make_blobs(n_samples=args.n_samples, n_features=50,
                          centers=args.true_k, random_state=42)
    elif args.dataset == "moons":
        from sklearn.datasets import make_moons
        X, _ = make_moons(n_samples=args.n_samples, noise=0.1, random_state=42)
    elif args.dataset == "circles":
        from sklearn.datasets import make_circles
        X, _ = make_circles(n_samples=args.n_samples, noise=0.05, factor=0.5, random_state=42)
    else:
        from atcnd.benchmark import load_synthetic
        texts, _, _, _ = load_synthetic(n_docs=args.n_samples, n_topics=args.true_k)
        from atcnd import ATCNDConfig, atcnd_search
        config = ATCNDConfig(k_min=args.k_min, k_max=args.k_max, model_type="nmf", metric="silhouette")
        result = atcnd_search(texts=texts, config=config)
        print(f"NMF on synthetic text: K*={result.optimal_k}")
        return

    res = run_full_comparison(X, true_k=args.true_k, k_min=args.k_min,
                              k_max=args.k_max, dataset_name=args.dataset)
    print_comparison_table([res])


def _cmd_animate(args):
    from atcnd.search import search
    from atcnd.animate import animate_search
    from sklearn.datasets import make_blobs
    from sklearn.metrics import silhouette_score
    from sklearn.cluster import KMeans

    X, _ = make_blobs(n_samples=1000, n_features=50, centers=8, random_state=42)

    def f(k):
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        model.fit(X)
        return silhouette_score(X, model.labels_)

    sr = search(f, k_min=args.k_min, k_max=args.k_max, strategy=args.strategy)
    path = animate_search(sr, save_path=args.output, fps=args.fps)
    print(f"Animation saved to {path}")
    print(f"Optimal K={sr.optimal_k}, score={sr.optimal_score:.4f}, evals={len(sr.all_scores)}")


def _cmd_benchmark(args):
    from atcnd.benchmark import run_benchmark, print_summary
    results = run_benchmark(dataset_name=args.dataset, k_range=(args.k_min, args.k_max), n_samples=args.n_samples)
    print_summary(results)


def _cmd_adaptive(args):
    from atcnd import adaptive_select, adaptive_search
    from sklearn.datasets import make_blobs
    np.random.seed(args.seed)
    X, _ = make_blobs(n_samples=500, n_features=50, centers=8, random_state=args.seed)
    rec = adaptive_select(X, k_min=args.k_min, k_max=args.k_max)
    p = rec.profile
    if args.json:
        print(json.dumps({
            "strategy": rec.strategy,
            "metric": rec.metric,
            "confidence": round(rec.confidence, 3),
            "profile": {"n_samples": p.n_samples, "n_features": p.n_features,
                        "sparsity": round(p.sparsity, 3), "intrinsic_dim": p.intrinsic_dim,
                        "separation_ratio": round(p.separation_ratio, 2)},
            "top5": [{"strategy": r["strategy"], "metric": r["metric"], "score": round(r["score"], 3)}
                     for r in rec.all_recommendations[:5]],
        }, indent=2))
    else:
        print(f"Data profile: n={p.n_samples}, d={p.n_features}, sparsity={p.sparsity:.3f}, "
              f"intrinsic_dim={p.intrinsic_dim}, separation={p.separation_ratio:.2f}")
        print(f"\nRecommended: {rec.strategy} + {rec.metric} (confidence: {rec.confidence:.2f})")
        print(f"\nTop 5 combinations:")
        for r in rec.all_recommendations[:5]:
            print(f"  {r['strategy']:18s} + {r['metric']:18s} = {r['score']:.3f}")

    if args.run:
        result = adaptive_search(X, k_min=args.k_min, k_max=args.k_max)
        if args.json:
            print(json.dumps({"optimal_k": result.optimal_k, "strategy": result.strategy}, indent=2))
        else:
            print(f"\nAdaptive search result: K*={result.optimal_k}, strategy={result.strategy}")


def _cmd_multi(args):
    from atcnd import multi_objective_search
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs
    np.random.seed(args.seed)
    X, _ = make_blobs(n_samples=500, n_features=50, centers=8, random_state=args.seed)
    weights = None
    if args.weights:
        weights = dict(zip(args.metrics, args.weights))
    mo = multi_objective_search(KMeans, X, metrics=args.metrics, weights=weights,
                                 k_min=args.k_min, k_max=args.k_max)
    if args.json:
        print(json.dumps({
            "optimal_k": mo.optimal_k,
            "optimal_score": round(mo.optimal_score, 4),
            "pareto_ks": mo.pareto_ks,
            "weights": mo.weights,
            "per_metric_best": {m: max(s.keys(), key=s.get) for m, s in mo.per_metric_scores.items()},
        }, indent=2))
    else:
        print(f"Multi-objective result: K*={mo.optimal_k} (combined={mo.optimal_score:.4f})")
        print(f"  Pareto frontier: {mo.pareto_ks}")
        print(f"  Weights: {mo.weights}")
        print(f"  Per-metric best:")
        for metric, scores in mo.per_metric_scores.items():
            best_k = max(scores, key=scores.get)
            print(f"    {metric}: K*={best_k} (score={scores[best_k]:.4f})")


if __name__ == "__main__":
    main()