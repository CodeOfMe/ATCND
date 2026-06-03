# ATCND

**基于滑动范围结构化搜索的自适应主题与聚类数目确定方法**

[![PyPI](https://img.shields.io/pypi/v/atcnd)](https://pypi.org/project/atcnd/)
[![Python](https://img.shields.io/pypi/pyversions/atcnd)](https://pypi.org/project/atcnd/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-75%20passing-brightgreen)](tests/)

ATCND 是一个模型无关的框架，通过将 K 值选择视为用户指定整数范围上的结构化搜索问题，来确定主题模型（LDA/NMF）的最优主题数或聚类方法（K-Means）的最优聚类数。在 ATCND 之前，没有任何模型无关且保证正确 K* 的方法达到亚线性评估次数：穷举网格搜索（模型无关+精确K等价类中的SOTA基线）需要 O(K_max − K_min) 次模型评估。ATCND 采用八种搜索策略实现 O(log(K_max − K_min)) 次评估——相比网格搜索减少59-79%，同时K*准确率相同。

## 核心特性

- **8种搜索策略**：网格、二分、黄金分割、三分、斐波那契、插值、指数、预测
- **15个适配器**：覆盖 NumPy、SciPy、scikit-learn、PyTorch、Gensim、Pandas
- **3种模型族**：LDA、NMF、K-Means（加上 GMM、PCA、DBSCAN 等）
- **5种质量指标**：轮廓系数、一致性(c_v)、困惑度、重构误差、组合指标
- **排序候选集**：返回多个最优K值（处理平台区、并列分数、多极值情况）
- **模型无关+精确K等价类中首个O(log N)方法**：相比网格搜索减少59-79%的模型评估次数
- **预测搜索**配合PCA热启动：减少76%（7次评估 vs 网格的29次）
- **提供 CLI 和 Python API**

## 安装

```bash
pip install atcnd
```

PyTorch适配器：

```bash
pip install atcnd[torch]
```

从源码安装：

```bash
git clone https://github.com/CodeOfMe/ATCND.git
cd ATCND
pip install -e .
```

## 快速入门

### Python API

```python
from atcnd import ATCNDConfig, atcnd_search

# 对数值型数据使用 K-Means
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

print(f"最优 K: {result.optimal_k}")
print(f"最佳分数: {result.optimal_score:.4f}")
print(f"候选 K 值: {result.candidate_ks}")
print(f"评估次数: {len(result.search_history)}")
```

### 低层 API（任意可调用函数）

```python
from atcnd import search
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def f(k):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    return silhouette_score(X, labels)

result = search(f, k_min=2, k_max=30, strategy="binary")
print(f"K* = {result.optimal_k}, 评估次数 = {len(result.search_history)}")
```

### 适配器 API（scikit-learn、PyTorch、Gensim、...）

```python
from atcnd import search_model, search_gmm_components, search_nmf_topics
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# K-Means
r = search_model(KMeans, X, param_name="n_clusters", k_min=2, k_max=30, strategy="binary")

# GMM with BIC
r = search_gmm_components(X, k_min=2, k_max=15, strategy="binary")

# NMF 主题数 with 一致性
r = search_nmf_topics(texts, k_min=2, k_max=20, strategy="binary", metric="coherence")

# 预测搜索配合PCA热启动
from atcnd import estimate_k_n_clusters
hot_k = estimate_k_n_clusters(X, k_min=2, k_max=30)
r = search(f, k_min=2, k_max=30, strategy="predictive", hot_start=hot_k)
```

### 文本数据（LDA/NMF）

```python
from atcnd import ATCNDConfig, atcnd_search

texts = ["你的文档文本"] * 100

# 使用一致性指标的 NMF
config = ATCNDConfig(
    k_min=2, k_max=20,
    model_type="nmf",
    search_strategy="binary",
    metric="coherence",
)
result = atcnd_search(texts=texts, config=config)
```

### 命令行

```bash
# 对合成数据使用 K-Means 搜索
atcnd search --model kmeans --strategy binary --k-min 2 --k-max 30

# 使用 NMF 搜索
atcnd search --model nmf --strategy golden_section --metric silhouette

# JSON 输出
atcnd search --model kmeans --json

# 运行基准测试
atcnd benchmark --dataset blobs --k-min 2 --k-max 30

# 8策略对比
atcnd benchmark --dataset blobs --k-min 2 --k-max 30 --all-strategies
```

## 搜索策略

| 策略 | 复杂度 | 适用场景 | 评估次数（K∈[2,30]） | 比网格减少 |
|------|--------|---------|----------------------|-----------|
| 网格搜索 | O(N) | 基线对比 | 29 | — |
| 二分搜索 | O(log N) | 单峰目标函数 | 9 | 69% |
| 黄金分割 | O(log\_φ N) | 一般目标函数 | 12 | 59% |
| 三分搜索 | O(log\_{1.5} N) | 多步目标函数 | 14 | 52% |
| 斐波那契 | O(log\_φ N) | 离散单峰（最优） | 10 | 66% |
| 插值搜索 | O(log log N)\* | 平滑目标函数 | 11 | 62% |
| 指数搜索 | O(log K\*) | K范围未知 | 10 | 66% |
| 预测搜索 | O(1)探测+O(log Δ) | 有数据驱动的热启动 | 7 | **76%** |

\*最坏情况对高度非均匀目标退化为 O(N)。

基准测试：K-Means 在 SyntheticBlobs（K\_true=8，范围[2,30]）。所有策略均恢复K\*=8。

### 预测搜索

预测搜索使用基于PCA的热启动在模型评估前估计K\*，然后应用抛物线峰值拟合：

```python
from atcnd import estimate_k_n_clusters, search

# PCA热启动从特征值肘部估计K
hot_k = estimate_k_n_clusters(X, k_min=2, k_max=30)

# 预测搜索使用热启动减少评估次数
result = search(f, k_min=2, k_max=30, strategy="predictive", hot_start=hot_k)
```

## 适配器

15个适配器将纯搜索策略与适当的质量度量和模型训练封装：

| 库 | 适配器 | 参数 | 度量 |
|----|--------|------|------|
| sklearn | `search_model` | n\_clusters | 轮廓系数 |
| sklearn | `search_neighbors` | n\_neighbors | CV准确率 |
| NumPy | `search_bins` | bins | AIC |
| sklearn | `search_components` | n\_components | 累积方差 |
| SciPy | `search_knots` | 内部节点 | MSE |
| signal | `search_window` | 窗口 | BIC |
| 任意 | `search_param` | 任意 | 用户自定义 |
| PyTorch | `search_hidden` | hidden\_dim | −CE |
| PyTorch | `search_layers` | n\_layers | −CE |
| sklearn | `search_trees` | n\_estimators | CV准确率 |
| sklearn | `search_dbscan_eps` | eps | 轮廓系数 |
| sklearn | `search_gmm_components` | n\_components | BIC |
| Pandas | `search_dataframe_bins` | bins | AIC |
| Pandas | `search_rolling_window` | window | BIC |
| Gensim | `search_nmf_topics` | n\_topics | c\_v一致性 |

运行所有演示并生成图片（SVG + PDF + PNG）：

```bash
python examples/demo_all.py
# 输出: examples/figures/*.{svg,pdf,png}
```

## 质量指标

| 指标 | LDA | NMF | K-Means | 描述 |
|------|-----|-----|---------|------|
| 轮廓系数 | 是 | 是 | 是 | 簇间分离度与簇内凝聚度之比 |
| 一致性 (c\_v) | 是 | 是 | 否 | 主题高频词的语义一致性 |
| 困惑度 | 是 | 否 | 否 | 每词负对数似然 |
| 重构误差 | 否 | 是 | 是 | Frobenius 范数 / 惯性 |
| 组合指标 | 是 | 是 | 否 | 0.5 × 轮廓系数 + 0.5 × 一致性 |

## 多解问题

K 是离散整数参数。多个 K 值可能达到相同或接近的质量分数。ATCND 除了返回单一最优 `optimal_k` 外，还返回 `candidate_ks`（排序列表），处理平台区、并列分数和多极值情况。

## 与基线方法对比

ATCND 是模型无关+精确K等价类中首个实现 O(log N) 评估次数的方法：

| 方法 | 模型无关？ | 精确K\*？ | 评估次数 |
|------|-----------|----------|---------|
| **ATCND（全部）** | **是** | **是** | **O(log N)** |
| 网格搜索 | 是 | 是 | O(N) |
| HDP | 否（仅LDA） | 否 | 1（代价高） |
| Top2Vec | 否 | 否 | 1 |
| 黑箱优化 | 否（仅LDA） | 是 | 可变 |

## 开发

```bash
# 开发模式安装
pip install -e ".[dev]"

# 运行测试（75个测试）
python -m pytest tests/ -v

# 格式化代码
black src/atcnd/ tests/

# 代码检查
ruff check src/atcnd/ tests/
```

## 授权协议

GNU 通用公共许可证 v3.0 (GPLv3)