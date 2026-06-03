"""ATCND CLI entry point."""

import argparse
import sys
import json
import numpy as np
from atcnd import ATCNDConfig, atcnd_search, print_topics, plot_search_curve


def main():
    parser = argparse.ArgumentParser(
        prog="atcnd",
        description="ATCND: Adaptive Topic and Cluster Number Determination",
    )
    parser.add_argument("-V", "--version", action="version", version=f"%(prog)s {atcnd_search.__module__.split('.')[0] if '.' in atcnd_search.__module__ else 'atcnd'}")

    sub = parser.add_subparsers(dest="command")

    search_parser = sub.add_parser("search", help="Run ATCND search on data")
    search_parser.add_argument("--model", choices=["lda", "nmf", "kmeans"], default="kmeans")
    search_parser.add_argument("--strategy", choices=["binary", "golden_section", "ternary", "grid"], default="binary")
    search_parser.add_argument("--metric", choices=["silhouette", "coherence", "perplexity", "reconstruction", "combined"], default="silhouette")
    search_parser.add_argument("--k-min", type=int, default=2)
    search_parser.add_argument("--k-max", type=int, default=50)
    search_parser.add_argument("--n-candidates", type=int, default=3)
    search_parser.add_argument("--seed", type=int, default=42)
    search_parser.add_argument("-v", "--verbose", action="store_true")
    search_parser.add_argument("-o", "--output", help="Save plot to file")
    search_parser.add_argument("--json", action="store_true", help="Output results as JSON")
    search_parser.add_argument("-q", "--quiet", action="store_true")

    bench_parser = sub.add_parser("benchmark", help="Run benchmarks")
    bench_parser.add_argument("--dataset", choices=["synthetic", "blobs"], default="blobs")
    bench_parser.add_argument("--k-min", type=int, default=2)
    bench_parser.add_argument("--k-max", type=int, default=30)
    bench_parser.add_argument("--n-samples", type=int, default=1000)

    args = parser.parse_args()

    if args.command == "search":
        _cmd_search(args)
    elif args.command == "benchmark":
        _cmd_benchmark(args)
    else:
        parser.print_help()


def _cmd_search(args):
    config = ATCNDConfig(
        k_min=args.k_min,
        k_max=args.k_max,
        metric=args.metric,
        model_type=args.model,
        search_strategy=args.strategy,
        random_state=args.seed,
        n_candidates=args.n_candidates,
        verbose=args.verbose,
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
        output = {
            "optimal_k": result.optimal_k,
            "optimal_score": result.optimal_score,
            "model_type": result.model_type,
            "search_strategy": result.config.search_strategy,
            "evaluations": len(result.search_history),
            "candidate_ks": result.candidate_ks,
            "candidate_scores": result.candidate_scores,
            "all_scores": {str(k): v for k, v in result.all_scores.items()},
        }
        print(json.dumps(output, indent=2))
    elif not args.quiet:
        print_topics(result)

    if args.output:
        plot_search_curve(result, save_path=args.output)


def _cmd_benchmark(args):
    from atcnd.benchmark import run_benchmark, print_summary
    results = run_benchmark(
        dataset_name=args.dataset,
        k_range=(args.k_min, args.k_max),
        n_samples=args.n_samples,
    )
    print_summary(results)


if __name__ == "__main__":
    main()