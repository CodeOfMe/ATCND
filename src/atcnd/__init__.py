"""ATCND - Adaptive Topic and Cluster Number Determination."""

from .search import search, SearchResult
from .core import (
    ATCNDConfig,
    ATCNDResult,
    atcnd_search,
    print_topics,
    plot_search_curve,
)
from .adapters import (
    search_model,
    search_neighbors,
    search_bins,
    search_components,
    search_knots,
    search_window,
    search_param,
    search_hidden,
    search_layers,
    search_trees,
    search_dbscan_eps,
    search_gmm_components,
    search_dataframe_bins,
    search_rolling_window,
    search_nmf_topics,
)
from .search import estimate_k_n_clusters, estimate_k_n_topics
from .animate import animate_search, animate_search_frames
from .comparison import (
    baseline_grid,
    baseline_kneedle,
    baseline_gap_statistic,
    baseline_bic_gmm,
    baseline_xmeans,
    baseline_gmeans,
    baseline_hdbscan,
    baseline_eigengap,
    baseline_hdp,
    run_full_comparison,
    print_comparison_table,
)
from .adaptive import (
    adaptive_select,
    adaptive_search,
    profile_data,
    AdaptiveRecommendation,
    DataProfile,
)
from .multi_objective import (
    multi_objective_search,
    MultiObjectiveResult,
)

__version__ = "0.5.1"

__all__ = [
    "search",
    "SearchResult",
    "ATCNDConfig",
    "ATCNDResult",
    "atcnd_search",
    "print_topics",
    "plot_search_curve",
    "search_model",
    "search_neighbors",
    "search_bins",
    "search_components",
    "search_knots",
    "search_window",
    "search_param",
    "search_hidden",
    "search_layers",
    "search_trees",
    "search_dbscan_eps",
    "search_gmm_components",
    "search_dataframe_bins",
    "search_rolling_window",
    "search_nmf_topics",
    "estimate_k_n_clusters",
    "estimate_k_n_topics",
    "animate_search",
    "animate_search_frames",
    "baseline_grid",
    "baseline_kneedle",
    "baseline_gap_statistic",
    "baseline_bic_gmm",
    "baseline_xmeans",
    "baseline_gmeans",
    "baseline_hdbscan",
    "baseline_eigengap",
    "baseline_hdp",
    "run_full_comparison",
    "print_comparison_table",
    "adaptive_select",
    "adaptive_search",
    "profile_data",
    "AdaptiveRecommendation",
    "DataProfile",
    "multi_objective_search",
    "MultiObjectiveResult",
]