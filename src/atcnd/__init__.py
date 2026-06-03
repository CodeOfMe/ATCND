"""ATCND - Adaptive Topic and Cluster Number Determination."""

from .core import (
    ATCNDConfig,
    ATCNDResult,
    atcnd_search,
    print_topics,
    plot_search_curve,
)
from .benchmark import (
    load_synthetic,
    load_synthetic_blobs,
    run_benchmark,
    run_all_benchmarks,
    print_summary,
)

__version__ = "0.1.0"

__all__ = [
    "ATCNDConfig",
    "ATCNDResult",
    "atcnd_search",
    "print_topics",
    "plot_search_curve",
    "load_synthetic",
    "load_synthetic_blobs",
    "run_benchmark",
    "run_all_benchmarks",
    "print_summary",
]