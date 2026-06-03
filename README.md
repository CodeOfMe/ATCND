# ATCND

Adaptive Topic and Cluster Number Determination via Structured Search over Sliding Ranges.

ATCND is a model-agnostic framework that determines the optimal number of topics (for LDA/NMF) or clusters (for K-Means) by treating K selection as a structured search problem over a user-specified integer range. Instead of exhaustive grid search requiring O(K_max - K_min) model evaluations, ATCND applies binary search, golden section search, or ternary search to achieve O(log(K_max - K_min)) evaluations while matching or exceeding the quality of grid search.

## Key Features

- Four search strategies: binary, golden section, ternary, grid
- Three model families: LDA, NMF, K-Means
- Five quality metrics: silhouette, coherence, perplexity, reconstruction error, combined
- Returns a ranked set of top-K candidates (handles plateaus, ties, multiple optima)
- 60--80% fewer model evaluations than grid search
- CLI and Python API

## Installation

```bash
pip install atcnd
```

For gensim-based coherence metrics:

```bash
pip install atcnd[gensim]
```

From source:

```bash
git clone https://github.com/CodeOfMe/ATCND.git
cd ATCND
pip install .
```

## Quick Start

### Python API

```python
from atcnd import ATCNDConfig, atcnd_search, print_topics

# K-Means on numeric data
from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=1000, n_features=50, centers=8, random_state=42)

config = ATCNDConfig(
    k_min=2, k_max=30,
    model_type="kmeans",
    search_strategy="binary",
    metric="silhouette",
    n_candidates=3,
)
result = atcnd_search(X=X, config=config)

print(f"Optimal K: {result.optimal_k}")
print(f"Best score: {result.optimal_score:.4f}")
print(f"Top candidates: {result.candidate_ks}")
print(f"Evaluations: {len(result.search_history)}")
```

### Text data (LDA/NMF)

```python
from atcnd import ATCNDConfig, atcnd_search

texts = ["your document texts here"] * 100

# NMF with coherence metric
config = ATCNDConfig(
    k_min=2, k_max=20,
    model_type="nmf",
    search_strategy="binary",
    metric="coherence",
)
result = atcnd_search(texts=texts, config=config)
```

### CLI

```bash
# Search with K-Means on synthetic data
atcnd search --model kmeans --strategy binary --k-min 2 --k-max 30

# Search with NMF
atcnd search --model nmf --strategy golden_section --metric silhouette

# JSON output
atcnd search --model kmeans --json

# Run benchmarks
atcnd benchmark --dataset blobs --k-min 2 --k-max 30
```

## Search Strategies

| Strategy | Complexity | Best for | Evaluations (K in [2,30]) |
|----------|-----------|----------|--------------------------|
| Grid | O(N) | Baseline comparison | 29 |
| Binary | O(log N) | Unimodal objectives | ~6 |
| Golden Section | O(log_phi N) | General objectives | ~7 |
| Ternary | O(log_{1.5} N) | Multi-step objectives | ~8 |

## Quality Metrics

| Metric | LDA | NMF | K-Means | Description |
|--------|-----|-----|---------|-------------|
| Silhouette | Yes | Yes | Yes | Inter-cluster separation vs intra-cluster cohesion |
| Coherence (c_v) | Yes | Yes | No | Semantic coherence of top topic words |
| Perplexity | Yes | No | No | Negative log-likelihood per word |
| Reconstruction | No | Yes | Yes | Frobenius norm / inertia |
| Combined | Yes | Yes | No | 0.5 * silhouette + 0.5 * coherence |

## Multiple Optima

K is a discrete integer parameter. Multiple values of K may achieve the same or nearly the same quality score due to:

- Exact equality in f(K) at different K values
- Plateaus where nearby K values produce identical scores
- Multiple local maxima from different data partitioning granularities

ATCND returns `candidate_ks` (a ranked list) and `candidate_scores` alongside the single best `optimal_k`, enabling users to make informed decisions based on domain knowledge.

## Development

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
python -m pytest tests/

# Format code
black src/atcnd/ tests/

# Lint code
ruff check src/atcnd/ tests/
```

## License

GNU General Public License v3.0 (GPLv3)