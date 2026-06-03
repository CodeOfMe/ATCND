# ATCND

基于滑动范围结构化搜索的自适应主题与聚类数目确定方法。

ATCND 是一个模型无关的框架，通过将 K 值选择视为用户指定整数范围上的结构化搜索问题，来确定主题模型（LDA/NMF）的最优主题数或聚类方法（K-Means）的最优聚类数。在 ATCND 之前，没有任何模型无关且保证正确 K* 的方法达到亚线性评估次数：穷举网格搜索（模型无关+精确K等价类中的SOTA基线）需要 O(K_max - K_min) 次模型评估。ATCND 采用二分搜索、黄金分割搜索或三分搜索实现 O(log(K_max - K_min)) 次评估——相比网格搜索减少59-79%，同时K*准确率相同。

## 核心特性

- 四种搜索策略：二分搜索、黄金分割搜索、三分搜索、网格搜索
- 三种模型族：LDA、NMF、K-Means
- 五种质量指标：轮廓系数、一致性、困惑度、重构误差、组合指标
- 返回排序后的最优 K 候选集（处理平台区、并列分数、多极值情况）
- 模型无关+精确K等价类中首个O(log N)方法：相比网格搜索减少59-79%的模型评估次数
- 提供 CLI 和 Python API

## 安装

```bash
pip install atcnd
```

从源码安装：

```bash
git clone https://github.com/CodeOfMe/ATCND.git
cd ATCND
pip install .
```

## 快速入门

### Python API

```python
from atcnd import ATCNDConfig, atcnd_search, print_topics

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
```

## 搜索策略

| 策略 | 复杂度 | 适用场景 | 评估次数（K 在 [2,30]）| 比Grid减少 |
|------|--------|---------|----------------------|-----------|
| 网格搜索 | O(N) | 基线对比 | 29 | - |
| 二分搜索 | O(log N) | 单峰目标函数 | 9 | 69% |
| 黄金分割 | O(log_phi N) | 一般目标函数 | 12 | 59% |
| 三分搜索 | O(log_{1.5} N) | 多步目标函数 | 14 | 52% |
| Fibonacci | O(log_phi N) | 离散单峰（最优） | 10 | 66% |
| 插值搜索 | O(log log N)* | 平滑目标函数 | 11 | 62% |
| 指数搜索 | O(log K*) | K范围未知 | 10 | 66% |
| 预测搜索 | O(1)探测+O(log N) | 有数据驱动的热启动 | 7 | 76% |

## 适配器（NumPy / Pandas / SciPy / Sklearn / PyTorch）

| 库 | 适配器 | 参数 | 度量 |
|----|--------|------|------|
| NumPy | `search_bins` | 直方图分箱数 | AIC |
| SciPy/sklearn | `search_gmm_components` | GMM分量数 | BIC |
| SciPy | `search_knots` | 平滑参数 | -MSE+惩罚 |
| Pandas | `search_rolling_window` | 滚动窗口 | BIC |
| Pandas+NumPy | `search_dataframe_bins` | DataFrame分箱 | AIC |
| sklearn | `search_model(KMeans)` | K-Means聚类数 | 轮廓系数 |
| sklearn | `search_neighbors` | KNN k | CV准确率 |
| sklearn | `search_components` | PCA分量数 | 累积方差 |
| sklearn | `search_trees` | 随机森林树数 | CV准确率 |
| sklearn | `search_dbscan_eps` | DBSCAN eps | 轮廓系数 |
| Gensim+sklearn | `search_nmf_topics` | NMF主题数 | c_v一致性 |
| PyTorch | `search_hidden` | 隐藏层大小 | -loss |
| PyTorch | `search_layers` | 隐藏层数 | -loss |

运行所有演示并生成图片（SVG + PDF + PNG）：

```bash
python examples/demo_all.py
# 输出: examples/figures/*.{svg,pdf,png}
```

## 质量指标

| 指标 | LDA | NMF | K-Means | 描述 |
|------|-----|-----|---------|------|
| 轮廓系数 | 是 | 是 | 是 | 簇间分离度与簇内凝聚度之比 |
| 一致性 (c_v) | 是 | 是 | 否 | 主题高频词的语义一致性 |
| 困惑度 | 是 | 否 | 否 | 每词负对数似然 |
| 重构误差 | 否 | 是 | 是 | Frobenius 范数 / 惯性 |
| 组合指标 | 是 | 是 | 否 | 0.5 * 轮廓系数 + 0.5 * 一致性 |

## 多解问题

K 是离散整数参数。多个 K 值可能达到相同或接近的质量分数，原因包括：

- 不同 K 值的 f(K) 精确相等
- 相邻 K 值产生相同分数的平台区
- 不同数据划分粒度产生多个局部极大值

ATCND 除了返回单一最优 `optimal_k` 外，还返回 `candidate_ks`（排序列表）和 `candidate_scores`，使用户能够基于领域知识做出知情决策。

## 开发

```bash
# 开发模式安装
pip install -e ".[dev]"

# 运行测试
python -m pytest tests/

# 格式化代码
black src/atcnd/ tests/

# 代码检查
ruff check src/atcnd/ tests/
```

## 授权协议

GNU 通用公共许可证 v3.0 (GPLv3)