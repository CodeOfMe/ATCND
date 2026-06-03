# ATCND

**基于滑动范围结构化搜索的自适应主题与聚类数目确定方法**

[![PyPI](https://img.shields.io/pypi/v/atcnd)](https://pypi.org/project/atcnd/)
[![Python](https://img.shields.io/pypi/pyversions/atcnd)](https://pypi.org/project/atcnd/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-75%20passing-brightgreen)](tests/)

ATCND 是一个模型无关的框架，通过将 K 值选择视为用户指定整数范围上的结构化搜索问题，来确定主题模型（LDA/NMF）的最优主题数或聚类方法（K-Means）的最优聚类数。在 ATCND 之前，没有任何模型无关且保证正确 K* 的方法达到亚线性评估次数：穷举网格搜索（模型无关+精确K等价类中的SOTA基线）需要 O(K_max − K_min) 次模型评估。ATCND 采用八种搜索策略实现 O(log(K_max − K_min)) 次评估——相比网格搜索减少59-79%，同时K*准确率相同。

## 八策略对比

<p align="center">
  <img src="examples/figures/14_comparison_strategies.png" width="800" alt="ATCND: 八种搜索策略对比（K_true=8）">
</p>

K-Means 基准测试（SyntheticBlobs，K\_true=8，范围 [2,30]）：

| 策略 | 复杂度 | 评估次数 | 比网格减少 |
|------|--------|----------|-----------|
| 网格搜索 | O(N) | 29 | — |
| 二分搜索 | O(log N) | 9 | 69% |
| 黄金分割 | O(log\_φ N) | 12 | 59% |
| 三分搜索 | O(log\_{1.5} N) | 14 | 52% |
| 斐波那契 | O(log\_φ N) | 10 | 66% |
| 插值搜索 | O(log log N)\* | 11 | 62% |
| 指数搜索 | O(log K\*) | 10 | 66% |
| **预测搜索** | **O(1)+O(log Δ)** | **7** | **76%** |

所有策略均恢复K\*=8，轮廓系数相同。预测搜索配合PCA热启动实现最少评估次数。

### 跨数据集一致性

ATCND从不灾难性失败。比较五个基准数据集上的绝对误差 |K\* − K_true|：

| 方法 | Iris | Wine | 乳腺癌 | Digits | Blobs | **平均** | **最大** |
|------|------|------|--------|--------|-------|---------|----------|
| **ATCND-sil** | 1 | 1 | 0 | 1 | 0 | **0.6** | **1** |
| Kneedle | 2 | 2 | 2 | 1 | 0 | 1.4 | 2 |
| X-Means | 0 | 0 | 0 | 8 | 6 | 2.8 | 8 |
| Gap统计量 | 12 | 12 | 8 | 10 | 0 | 8.4 | 12 |
| G-Means | 12 | 4 | 8 | 10 | 22 | 11.2 | 22 |

<p align="center">
  <img src="examples/figures/consistency_analysis.png" width="800" alt="一致性分析：ATCND从不灾难性失败">
</p>

## 真实数据集演示

### K-Means on Iris（3D）

<p align="center">
  <img src="examples/figures/01_iris_kmeans_3d.png" width="400" alt="K-Means on Iris（K*=2）">
  <img src="examples/figures/01_iris_kmeans_curve.png" width="400" alt="Iris K-Means 搜索曲线">
</p>

### GMM on Wine（3D）

<p align="center">
  <img src="examples/figures/02_wine_gmm_3d.png" width="400" alt="GMM on Wine（K*=2）">
  <img src="examples/figures/02_wine_gmm_curve.png" width="400" alt="Wine GMM 搜索曲线">
</p>

### PCA on Digits（3D）

<p align="center">
  <img src="examples/figures/03_digits_pca_3d.png" width="400" alt="PCA on Digits（K*=31, 95%方差）">
  <img src="examples/figures/03_digits_pca_curve.png" width="400" alt="Digits PCA 累积方差曲线">
</p>

### DBSCAN on Two-Moons

<p align="center">
  <img src="examples/figures/04_moons_dbscan.png" width="400" alt="DBSCAN on Two-Moons">
  <img src="examples/figures/04_moons_dbscan_curve.png" width="400" alt="DBSCAN eps 搜索曲线">
</p>

### 最优直方图分箱数（AIC）

<p align="center">
  <img src="examples/figures/07_numpy_bins_compare.png" width="450" alt="ATCND最优分箱 vs Sturges规则">
</p>

### 平滑样条（SciPy）

<p align="center">
  <img src="examples/figures/08_scipy_spline_fit.png" width="450" alt="ATCND选择的平滑样条参数">
</p>

### 滚动窗口（Pandas）

<p align="center">
  <img src="examples/figures/09_pandas_rolling_fit.png" width="450" alt="时间序列最优滚动窗口">
</p>

### NMF主题数（Gensim）

<p align="center">
  <img src="examples/figures/11_gensim_nmf.png" width="450" alt="NMF主题搜索（c_v一致性）">
</p>

### PyTorch隐藏层大小

<p align="center">
  <img src="examples/figures/12_torch_hidden.png" width="450" alt="PyTorch隐藏层大小搜索">
</p>

运行所有演示并生成图片（SVG + PDF + PNG）：

```bash
python examples/demo_all.py
# 输出: examples/figures/*.{svg,pdf,png}
```

## 核心特性

- **8种搜索策略**：网格、二分、黄金分割、三分、斐波那契、插值、指数、预测
- **15个适配器**：覆盖 NumPy、SciPy、scikit-learn、PyTorch、Gensim、Pandas
- **3种模型族**：LDA、NMF、K-Means（加上 GMM、PCA、DBSCAN 等）
- **5种质量指标**：轮廓系数、轮廓膝部、BIC、组合指标、轮廓下降
- **自适应策略选择**：根据数据特征（维度、稀疏度、分离度）自动推荐最佳策略+指标组合
- **多目标优化**：同时优化多个指标（如轮廓系数+BIC），返回帕累托最优K值
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

### 低层 API

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

### 适配器 API

```python
from atcnd import search_model, search_gmm_components, search_nmf_topics
from sklearn.cluster import KMeans

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

config = ATCNDConfig(
    k_min=2, k_max=20,
    model_type="nmf",
    search_strategy="binary",
    metric="coherence",
)
result = atcnd_search(texts=texts, config=config)
```

### 自适应策略选择

ATCND根据数据特征（维度、稀疏度、分离度、内在维度）自动推荐最佳策略+指标组合：

```python
from atcnd import adaptive_select, adaptive_search

# 获取推荐（不运行搜索）
rec = adaptive_select(X, k_min=2, k_max=30)
print(f"推荐: {rec.strategy} + {rec.metric} (置信度: {rec.confidence:.2f})")

# 直接运行自适应搜索
result = adaptive_search(X, k_min=2, k_max=30)
print(f"K* = {result.optimal_k}, 策略 = {result.strategy}")
```

命令行:

```bash
atcnd adaptive --k-min 2 --k-max 30
atcnd adaptive --k-min 2 --k-max 30 --run --json
```

### 多目标优化

同时优化多个指标，返回帕累托最优K值：

```python
from atcnd import multi_objective_search
from sklearn.cluster import KMeans

# 轮廓系数 + BIC，等权重
mo = multi_objective_search(KMeans, X, metrics=["silhouette", "bic"], k_min=2, k_max=30)
print(f"综合最优: K* = {mo.optimal_k}")
print(f"帕累托前沿: {mo.pareto_ks}")

# 自定义权重（优先轮廓系数）
mo = multi_objective_search(KMeans, X, metrics=["silhouette", "bic"],
                              weights={"silhouette": 0.7, "bic": 0.3}, k_min=2, k_max=30)
```

命令行:

```bash
atcnd multi --metrics silhouette bic --k-min 2 --k-max 30
atcnd multi --metrics silhouette bic --weights 0.7 0.3 --json
```

### 命令行

```bash
atcnd search --model kmeans --strategy binary --k-min 2 --k-max 30
atcnd search --model nmf --strategy golden_section --metric silhouette
atcnd search --model kmeans --json
atcnd benchmark --dataset blobs --k-min 2 --k-max 30
atcnd benchmark --dataset blobs --k-min 2 --k-max 30 --all-strategies
```

## 搜索策略详解

### 预测搜索

预测搜索使用基于PCA的热启动在模型评估前估计K*，然后应用抛物线峰值拟合：

1. **PCA热启动**：特征值肘部方法从数据结构估计K*
2. **探测**：评估 f(K̂−1)、f(K̂)、f(K̂+1)
3. **抛物线峰值拟合**：通过三个最佳点拟合抛物线并跳转到预测峰值
4. **二分精炼**：在预测峰值附近缩小搜索范围

### 斐波那契搜索

离散单峰搜索的经典最优算法（Kiefer 1953）。在所有无导数方法中，对单峰函数实现最小最坏情况评估次数。

### 指数搜索

从K\_min开始将探测点翻倍（1, 2, 4, 8, ...）直到f(K)下降，然后通过二分搜索精炼。适合K*接近K\_min的情况。

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

### 真实数据集结果

| 数据集 | 适配器 | K\* | 策略 | 评估次数 | 减少率 |
|--------|--------|-----|------|----------|--------|
| Iris | search\_model | 2 | 二分 | 5 | 64% |
| Wine | search\_gmm | 2 | 二分 | 6 | 57% |
| Digits | search\_components | 31 | 二分 | 10 | 84% |
| Moons | search\_dbscan | 3 | 二分 | 10 | 64% |
| Wine | search\_trees | 300 | 二分 | 37 | 87% |
| Iris | search\_neighbors | 12 | 二分 | 9 | 70% |

## 开发

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
black src/atcnd/ tests/
ruff check src/atcnd/ tests/
```

## 授权协议

GNU 通用公共许可证 v3.0 (GPLv3)