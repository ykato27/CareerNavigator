# 機械学習推薦システム - 技術詳細ドキュメント

## 目次

1. [概要](#概要)
2. [アーキテクチャ](#アーキテクチャ)
3. [Matrix Factorization](#matrix-factorization)
4. [多様性再ランキング](#多様性再ランキング)
5. [評価メトリクス](#評価メトリクス)
6. [実装詳細](#実装詳細)
7. [パラメータチューニング](#パラメータチューニング)
8. [パフォーマンス最適化](#パフォーマンス最適化)

---

## 概要

本システムは、**協調フィルタリング**に基づく機械学習推薦システムです。Non-negative Matrix Factorization (NMF) を用いてメンバーの習得パターンをモデル化し、多様性再ランキングにより推薦の質を向上させます。

> **関連ドキュメント**: より詳細なモデル実装については、[MODELS_TECHNICAL_GUIDE.md](MODELS_TECHNICAL_GUIDE.md)を参照してください。

### 主な特徴

- ✅ **NMFベースの協調フィルタリング**
- ✅ **4種類の多様性戦略**
- ✅ **説明可能な推薦**
- ✅ **コールドスタート対応**
- ✅ **リアルタイム推薦**

---

## アーキテクチャ

### システム構成

```
skillnote_recommendation/ml/
├── matrix_factorization.py    # NMFモデル
├── diversity.py                # 多様性再ランキング
└── ml_recommender.py           # 統合インターフェース
```

### データフロー

```
1. 生データ読み込み
   ↓
2. データ変換（skill_matrix作成）
   ↓
3. NMFモデル学習
   ↓
4. 予測スコア計算
   ↓
5. 多様性再ランキング
   ↓
6. 推薦結果出力
```

> **詳細**: データ前処理とハイパーパラメータチューニングについては、[MODELS_TECHNICAL_GUIDE.md](MODELS_TECHNICAL_GUIDE.md)を参照してください。

---

## Matrix Factorization

### アルゴリズム: NMF (Non-negative Matrix Factorization)

#### 数式

メンバー×力量マトリクス $V$ を2つの低ランク行列の積に分解：

$$
V \approx WH
$$

where:
- $V \in \mathbb{R}^{m \times n}$ : メンバー×力量マトリクス (mメンバー, n力量)
- $W \in \mathbb{R}^{m \times k}$ : メンバーの潜在因子行列
- $H \in \mathbb{R}^{k \times n}$ : 力量の潜在因子行列
- $k$ : 潜在因子数（デフォルト: 20）

#### 目的関数

$$
\min_{W,H} ||V - WH||_F^2
$$

subject to: $W \geq 0, H \geq 0$

where $||\cdot||_F$ はFrobeniusノルム

### 実装

#### クラス: `MatrixFactorizationModel`

```python
class MatrixFactorizationModel:
    def __init__(
        self,
        n_components: int = 20,
        random_state: int = 42,
        **nmf_params
    ):
        """
        Args:
            n_components: 潜在因子数
            random_state: 乱数シード
            **nmf_params: NMFパラメータ
                - init: 初期化方法 ('nndsvda'推奨)
                - max_iter: 最大反復回数 (500)
                - tol: 収束判定閾値 (1e-4)
        """
```

#### 主要メソッド

**1. fit(skill_matrix)**

```python
def fit(self, skill_matrix: pd.DataFrame) -> 'MatrixFactorizationModel':
    """NMFモデルを学習

    Args:
        skill_matrix: メンバー×力量のDataFrame (行: メンバー, 列: 力量)

    Returns:
        self
    """
```

**処理内容:**
1. DataFrameからNumPy配列に変換
2. NMFモデルの学習
3. W (メンバー因子) と H (力量因子) を保存
4. インデックスマッピングを作成

**2. predict(member_code, competence_codes)**

```python
def predict(
    self,
    member_code: str,
    competence_codes: Optional[List[str]] = None
) -> Dict[str, float]:
    """力量スコアを予測

    Args:
        member_code: メンバーコード
        competence_codes: 予測対象力量コード（Noneの場合全力量）

    Returns:
        {力量コード: 予測スコア}
    """
```

**予測式:**

$$
\text{score}(m, c) = W_m \cdot H_c
$$

where:
- $W_m$ : メンバーmの潜在因子ベクトル (k次元)
- $H_c$ : 力量cの潜在因子ベクトル (k次元)

**3. predict_top_k(member_code, k, exclude_acquired)**

```python
def predict_top_k(
    self,
    member_code: str,
    k: int = 10,
    exclude_acquired: bool = True
) -> List[Tuple[str, float]]:
    """Top-K推薦

    Args:
        member_code: メンバーコード
        k: 推薦件数
        exclude_acquired: 習得済み力量を除外するか

    Returns:
        [(力量コード, スコア), ...]
    """
```

### パラメータ詳細

#### n_components (潜在因子数)

**推奨値:** 10-50

| 値 | 特徴 | 適用ケース |
|----|------|----------|
| 5-10 | 高速、シンプル | 小規模データ（メンバー<100） |
| **10-20** | **バランス** | **中規模データ（メンバー100-1000）** ⭐ |
| 20-50 | 高精度、遅い | 大規模データ（メンバー>1000） |

**選択基準:**
```python
if メンバー数 < 100:
    n_components = 10
elif メンバー数 < 1000:
    n_components = 20  # デフォルト
else:
    n_components = min(50, int(sqrt(メンバー数)))
```

#### init (初期化方法)

**推奨:** `'nndsvda'`

| 方法 | 説明 | 特徴 |
|------|------|------|
| `random` | ランダム初期化 | 遅い、不安定 |
| `nndsvd` | SVDベース | 速い、決定的 |
| **`nndsvda`** | **SVD + 平均値** | **速い、安定** ⭐ |
| `nndsvdar` | SVD + ランダム | 中程度 |

#### max_iter (最大反復回数)

**推奨:** 200-1000

```python
max_iter = 500  # デフォルト
```

- 少ない（<200）: 高速だが収束しない可能性
- **適切（200-500）**: バランス良し
- 多い（>1000）: 過学習のリスク

### 潜在因子の解釈

#### 因子の意味

NMFは非負制約により、因子が**加算的**に解釈できます。

**例: k=5の場合**

```python
メンバーA = 0.8 × 因子1 + 0.2 × 因子2 + 0.5 × 因子3 + 0.0 × 因子4 + 0.3 × 因子5
```

各因子は以下のような「テーマ」を表す可能性：
- 因子1: 技術的スキル（プログラミング、DB等）
- 因子2: ビジネススキル（営業、企画等）
- 因子3: マネジメント（PM、リーダーシップ等）
- 因子4: ドメイン知識（製造、金融等）
- 因子5: 資格・認定

#### 因子の抽出

```python
# メンバー因子の取得
member_factors = model.get_member_factors('m001')
# → array([0.8, 0.2, 0.5, 0.0, 0.3])

# 力量因子の取得
competence_factors = model.get_competence_factors('s001')
# → array([0.9, 0.1, 0.3, 0.2, 0.0])
```

---

## 多様性再ランキング

### 4つの戦略

#### 1. MMR (Maximal Marginal Relevance)

**目的:** 関連度と多様性のバランスを取る

**数式:**

$$
\text{MMR} = \arg\max_{c \in C \setminus S} [\lambda \cdot \text{Sim}_1(c, Q) - (1-\lambda) \cdot \max_{s \in S} \text{Sim}_2(c, s)]
$$

where:
- $C$ : 候補力量集合
- $S$ : 既に選択された力量集合
- $Q$ : クエリ（メンバー）
- $\lambda$ : 関連度と多様性のバランスパラメータ (0.7推奨)
- $\text{Sim}_1$ : メンバーと力量の類似度（MLスコア）
- $\text{Sim}_2$ : 力量間の類似度（カテゴリの一致度）

**実装:**

```python
def rerank_mmr(
    self,
    candidates: List[Dict],
    competence_info: pd.DataFrame,
    k: int = 10,
    lambda_param: float = 0.7
) -> List[Dict]:
    """MMR再ランキング"""
```

**アルゴリズム:**

```python
selected = []
remaining = candidates.copy()

while len(selected) < k and remaining:
    best_score = -float('inf')
    best_idx = -1

    for idx, candidate in enumerate(remaining):
        # 関連度スコア
        relevance = candidate['MLスコア']

        # 既選択力量との最大類似度
        max_similarity = 0
        if selected:
            for sel in selected:
                similarity = category_similarity(candidate, sel)
                max_similarity = max(max_similarity, similarity)

        # MMRスコア
        mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity

        if mmr_score > best_score:
            best_score = mmr_score
            best_idx = idx

    selected.append(remaining.pop(best_idx))

return selected
```

#### 2. Category Diversity (カテゴリ多様性)

**目的:** 異なるカテゴリから均等に推薦

**アルゴリズム:**

```python
def rerank_category_diversity(
    self,
    candidates: List[Dict],
    competence_info: pd.DataFrame,
    k: int = 10,
    max_per_category: Optional[int] = None
) -> List[Dict]:
    """カテゴリ多様性再ランキング"""

    if max_per_category is None:
        max_per_category = max(1, k // num_categories)

    category_counts = defaultdict(int)
    selected = []

    for candidate in sorted(candidates, key=lambda x: x['MLスコア'], reverse=True):
        category = candidate['カテゴリ名']

        if category_counts[category] < max_per_category:
            selected.append(candidate)
            category_counts[category] += 1

            if len(selected) >= k:
                break

    return selected
```

**特徴:**
- カテゴリごとの上限を設定
- スコア順に選択しつつカテゴリを分散

#### 3. Type Diversity (タイプ多様性)

**目的:** SKILL/EDUCATION/LICENSEをバランス良く

**デフォルト比率:**

```python
default_type_ratios = {
    'SKILL': 0.6,      # 60%
    'EDUCATION': 0.25,  # 25%
    'LICENSE': 0.15     # 15%
}
```

**アルゴリズム:**

```python
def rerank_type_diversity(
    self,
    candidates: List[Dict],
    competence_info: pd.DataFrame,
    k: int = 10,
    type_ratios: Optional[Dict[str, float]] = None
) -> List[Dict]:
    """タイプ多様性再ランキング"""

    # 目標件数を計算
    target_counts = {
        type_name: int(k * ratio)
        for type_name, ratio in type_ratios.items()
    }

    # タイプごとに分類
    by_type = defaultdict(list)
    for c in candidates:
        by_type[c['力量種別']].append(c)

    # 各タイプから目標件数取得
    selected = []
    for type_name, target_count in target_counts.items():
        type_candidates = sorted(
            by_type[type_name],
            key=lambda x: x['MLスコア'],
            reverse=True
        )
        selected.extend(type_candidates[:target_count])

    # スコア順にソート
    return sorted(selected, key=lambda x: x['MLスコア'], reverse=True)[:k]
```

#### 4. Hybrid (ハイブリッド)

**目的:** MMR、カテゴリ、タイプを全て組み合わせ

**アルゴリズム:**

```python
def rerank_hybrid(
    self,
    candidates: List[Dict],
    competence_info: pd.DataFrame,
    k: int = 10
) -> List[Dict]:
    """ハイブリッド再ランキング"""

    # ステップ1: タイプ多様性で粗選択 (k × 1.5件)
    type_diverse = self.rerank_type_diversity(
        candidates,
        competence_info,
        k=int(k * 1.5)
    )

    # ステップ2: カテゴリ多様性でさらに絞る (k × 1.2件)
    category_diverse = self.rerank_category_diversity(
        type_diverse,
        competence_info,
        k=int(k * 1.2)
    )

    # ステップ3: MMRで最終選択 (k件)
    final = self.rerank_mmr(
        category_diverse,
        competence_info,
        k=k,
        lambda_param=0.7
    )

    return final
```

### 多様性メトリクス

#### 1. Category Diversity (カテゴリ多様性)

$$
\text{CategoryDiv} = \frac{\text{推薦に含まれる異なるカテゴリ数}}{\text{推薦件数}}
$$

**範囲:** [0, 1]
**良い値:** > 0.7

#### 2. Type Diversity (タイプ多様性)

Shannon Entropy を使用:

$$
H = -\sum_{t \in \{S, E, L\}} p_t \log_2(p_t)
$$

正規化:

$$
\text{TypeDiv} = \frac{H}{\log_2(3)}
$$

where:
- $p_t$ : タイプtの割合
- $S, E, L$ : SKILL, EDUCATION, LICENSE

**範囲:** [0, 1]
**良い値:** > 0.5

#### 3. Coverage (カバレッジ)

$$
\text{Coverage} = \frac{\text{推薦に含まれるカテゴリ数}}{\text{全カテゴリ数}}
$$

**範囲:** [0, 1]
**良い値:** > 0.6

#### 4. Intra-list Diversity (リスト内多様性)

力量間の平均非類似度:

$$
\text{ILD} = \frac{2}{n(n-1)} \sum_{i=1}^{n-1} \sum_{j=i+1}^{n} (1 - \text{sim}(c_i, c_j))
$$

where:
- $n$ : 推薦件数
- $\text{sim}(c_i, c_j)$ : 力量$c_i$と$c_j$の類似度（カテゴリ一致度）

**範囲:** [0, 1]
**良い値:** > 0.4

---

## 評価メトリクス

### 精度メトリクス

#### 1. Precision@K

$$
\text{Precision@K} = \frac{|\text{推薦}  \cap \text{実際に習得}|}{K}
$$

#### 2. Recall@K

$$
\text{Recall@K} = \frac{|\text{推薦} \cap \text{実際に習得}|}{|\text{実際に習得}|}
$$

#### 3. NDCG@K (Normalized Discounted Cumulative Gain)

$$
\text{NDCG@K} = \frac{DCG@K}{IDCG@K}
$$

$$
DCG@K = \sum_{i=1}^{K} \frac{rel_i}{\log_2(i+1)}
$$

### 評価プロセス

#### 時系列分割

```python
from skillnote_recommendation.core.evaluator import RecommendationEvaluator
from skillnote_recommendation.ml.ml_recommender import MLRecommender

evaluator = RecommendationEvaluator()

# 時系列分割 (80%学習、20%テスト)
train_data, test_data = evaluator.temporal_train_test_split(
    member_competence=member_competence,
    train_ratio=0.8
)

# MLモデル学習（n_componentsパラメータを指定可能）
ml_recommender = MLRecommender.build(
    member_competence=train_data,
    competence_master=competence_master,
    member_master=member_master,
    use_preprocessing=False,
    use_tuning=False,
    n_components=20  # 潜在因子数を明示的に指定
)

# 評価
metrics = evaluator.evaluate_with_diversity(
    train_data=train_data,
    test_data=test_data,
    competence_master=competence_master,
    top_k=10
)

# 結果表示
evaluator.print_evaluation_results(metrics)
```

---

## 実装詳細

### ファイル構成

```
skillnote_recommendation/ml/
│
├── __init__.py                 # パッケージ初期化
│   └── exports: MLRecommender, MatrixFactorizationModel, DiversityReranker
│
├── matrix_factorization.py    # 253行
│   └── class MatrixFactorizationModel
│       ├── __init__(n_components=20, **nmf_params)
│       ├── fit(skill_matrix)
│       ├── predict(member_code, competence_codes)
│       ├── predict_top_k(member_code, k, exclude_acquired)
│       ├── get_member_factors(member_code)
│       ├── get_competence_factors(competence_code)
│       ├── save(filepath)
│       └── load(filepath) [classmethod]
│
├── diversity.py                # 282行
│   └── class DiversityReranker
│       ├── __init__(lambda_param=0.7)
│       ├── rerank_mmr(candidates, competence_info, k, lambda_param)
│       ├── rerank_category_diversity(candidates, competence_info, k, max_per_category)
│       ├── rerank_type_diversity(candidates, competence_info, k, type_ratios)
│       ├── rerank_hybrid(candidates, competence_info, k)
│       └── calculate_diversity_metrics(recommendations, competence_info)
│
└── ml_recommender.py           # 204行
    └── class MLRecommender
        ├── __init__(data, n_components=20, **nmf_params)
        ├── recommend(member_code, top_n, competence_type, category_filter,
        │             use_diversity, diversity_strategy)
        └── calculate_diversity_metrics(recommendations, competence_master)
```

### クラス関係図

```
MLRecommender
    │
    ├── MatrixFactorizationModel (has-a)
    │   └── sklearn.decomposition.NMF
    │
    ├── DiversityReranker (has-a)
    │
    └── DataLoader (uses)
```

### データ構造

#### スキルマトリクス (skill_matrix)

```python
# DataFrame形式
"""
          s001  s002  s015  e001  e005  l002  ...
m001       1     1     0     1     0     0   ...
m002       1     0     1     0     1     0   ...
m003       0     1     1     1     0     1   ...
...
"""

# 型: pd.DataFrame
# 行: メンバーコード (index)
# 列: 力量コード (columns)
# 値: 0 (未習得) or 1 (習得済み)
```

#### 推薦結果 (recommendations)

```python
# DataFrame形式
[
    {
        '力量コード': 's020',
        '力量名': 'プロジェクトマネジメント',
        '力量種別': 'SKILL',
        'カテゴリ名': '技術管理',
        'MLスコア': 0.85,
        '推薦理由': 'ML推薦 (類似メンバーのパターンから学習)'
    },
    ...
]

# 型: List[Dict] または pd.DataFrame
```

---

## パラメータチューニング

### NMFパラメータ

#### n_components (潜在因子数)

**グリッドサーチ例:**

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_components': [10, 15, 20, 30, 50]
}

best_components = None
best_score = -float('inf')

for n_comp in param_grid['n_components']:
    model = MatrixFactorizationModel(n_components=n_comp)
    model.fit(train_skill_matrix)

    # 評価
    score = evaluate_model(model, test_data)

    if score > best_score:
        best_score = score
        best_components = n_comp

print(f"Best n_components: {best_components}")
```

#### max_iter (最大反復回数)

**収束確認:**

```python
from sklearn.decomposition import NMF

nmf = NMF(n_components=20, max_iter=500, verbose=1)
W = nmf.fit_transform(skill_matrix)
H = nmf.components_

# 収束したか確認
print(f"Converged: {nmf.n_iter_} iterations")
print(f"Reconstruction error: {nmf.reconstruction_err_:.4f}")
```

### 多様性パラメータ

#### lambda_param (MMRバランス)

**推奨値:** 0.5 - 0.9

```python
# 関連度重視 (λ=0.9)
reranker.rerank_mmr(candidates, competence_info, k=10, lambda_param=0.9)

# バランス (λ=0.7) ⭐ 推奨
reranker.rerank_mmr(candidates, competence_info, k=10, lambda_param=0.7)

# 多様性重視 (λ=0.5)
reranker.rerank_mmr(candidates, competence_info, k=10, lambda_param=0.5)
```

#### type_ratios (タイプ比率)

**カスタマイズ例:**

```python
# 技術職向け (SKILL重視)
tech_ratios = {
    'SKILL': 0.7,
    'EDUCATION': 0.2,
    'LICENSE': 0.1
}

# マネージャー向け (EDUCATION重視)
manager_ratios = {
    'SKILL': 0.3,
    'EDUCATION': 0.5,
    'LICENSE': 0.2
}

reranker.rerank_type_diversity(candidates, competence_info, k=10, type_ratios=tech_ratios)
```

---

## パフォーマンス最適化

### 学習時間の最適化

#### 1. データサイズの削減

```python
# スパースなデータの除去
min_acquisitions = 3  # 最低習得力量数

filtered_members = member_competence.groupby('メンバーコード').filter(
    lambda x: len(x) >= min_acquisitions
)
```

#### 2. 並列処理

```python
# scikit-learnのn_jobsパラメータは使用できないため、
# 複数メンバーの予測を並列化

from concurrent.futures import ThreadPoolExecutor

def predict_parallel(model, member_codes, k=10):
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(model.predict_top_k, member_code, k)
            for member_code in member_codes
        ]
        results = [f.result() for f in futures]
    return results
```

#### 3. モデルキャッシング (Streamlit)

```python
import streamlit as st

@st.cache_resource
def load_ml_model(data):
    """MLモデルをキャッシュ"""
    return MLRecommender(data)

# 使用例
ml_recommender = load_ml_model(st.session_state.raw_data)
```

### メモリ最適化

#### スパース行列の使用

```python
from scipy.sparse import csr_matrix

# 密行列 → スパース行列
sparse_matrix = csr_matrix(skill_matrix.values)

# NMFでスパース行列を使用
from sklearn.decomposition import NMF

nmf = NMF(n_components=20)
W = nmf.fit_transform(sparse_matrix)
H = nmf.components_
```

### 予測速度の最適化

#### バッチ予測

```python
# 悪い例: 1つずつ予測
for member_code in member_codes:
    scores = model.predict(member_code)

# 良い例: バッチ予測
member_indices = [model.member_to_idx[mc] for mc in member_codes]
W_batch = model.W[member_indices]
all_scores = W_batch @ model.H  # 行列積で一括計算
```

---

## トラブルシューティング

### よくある問題と解決策

#### 1. 収束しない

**症状:**
```
ConvergenceWarning: Maximum number of iterations reached before convergence.
```

**解決策:**
```python
# max_iterを増やす
model = MatrixFactorizationModel(n_components=20, max_iter=1000)

# または、tolを緩和
model = MatrixFactorizationModel(n_components=20, tol=1e-3)
```

#### 2. メモリ不足

**症状:**
```
MemoryError: Unable to allocate array
```

**解決策:**
```python
# n_componentsを減らす
model = MatrixFactorizationModel(n_components=10)

# またはスパース行列を使用
from scipy.sparse import csr_matrix
sparse_skill_matrix = csr_matrix(skill_matrix.values)
```

#### 3. 推薦結果が全て同じ

**症状:**
全メンバーに同じ力量が推薦される

**原因:**
- データが少なすぎる
- 潜在因子数が多すぎる

**解決策:**
```python
# データを増やす、またはn_componentsを減らす
model = MatrixFactorizationModel(n_components=5)

# 多様性戦略を使用
ml_recommender.recommend(
    member_code='m001',
    top_n=10,
    diversity_strategy='hybrid'  # 多様性を強制
)
```

---

## ベストプラクティス

### 1. モデル更新頻度

**推奨:** 月次または四半期ごと

```python
# 定期的な再学習スクリプト例
import schedule
from skillnote_recommendation.core.data_loader import DataLoader
from skillnote_recommendation.core.data_transformer import DataTransformer
from skillnote_recommendation.ml.ml_recommender import MLRecommender

def retrain_model():
    loader = DataLoader()
    data = loader.load_all_data()

    transformer = DataTransformer()
    competence_master = transformer.create_competence_master(data)
    member_competence, _ = transformer.create_member_competence(data, competence_master)

    ml_recommender = MLRecommender.build(
        member_competence=member_competence,
        competence_master=competence_master,
        member_master=data['members'],
        use_preprocessing=True,
        use_tuning=False,
        n_components=20
    )

    # モデルの保存（必要に応じて）
    ml_recommender.mf_model.save('models/ml_model_latest.pkl')

# 毎月1日に実行
schedule.every().month.at("01 00:00").do(retrain_model)
```

### 2. A/Bテスト

```python
# 異なる多様性戦略の比較
import random

def get_recommendation(member_code):
    if random.random() < 0.5:
        # グループA: ハイブリッド戦略
        return ml_recommender.recommend(
            member_code=member_code,
            top_n=10,
            use_diversity=True,
            diversity_strategy='hybrid'
        )
    else:
        # グループB: MMR戦略
        return ml_recommender.recommend(
            member_code=member_code,
            top_n=10,
            use_diversity=True,
            diversity_strategy='mmr'
        )

# 効果測定
# - 力量習得率
# - ユーザー満足度
# - 多様性メトリクス
```

### 3. ログ記録

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 推薦実行時のログ
logger.info(f"ML recommendation for {member_code}: {recommendations}")
logger.info(f"Diversity metrics: {diversity_metrics}")

# 学習時のログ
logger.info(f"Model trained with {len(skill_matrix)} members")
logger.info(f"Reconstruction error: {model.model.reconstruction_err_:.4f}")
```

---

## 参考文献

### 論文・書籍

1. **Matrix Factorization Techniques for Recommender Systems**
   - Y. Koren, R. Bell, C. Volinsky (2009)
   - IEEE Computer, 42(8)

2. **Non-negative Matrix Factorization**
   - D. D. Lee, H. S. Seung (1999)
   - Nature, 401

3. **Improving Recommendation Lists Through Topic Diversification**
   - C. Ziegler et al. (2005)
   - WWW Conference

4. **The Use of MMR, Diversity-Based Reranking for Reordering Documents**
   - J. Carbonell, J. Goldstein (1998)
   - SIGIR Conference

### オンラインリソース

- [scikit-learn NMF Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html)
- [Netflix Prize](https://www.netflixprize.com/)
- [Recommender Systems Handbook](https://www.springer.com/gp/book/9780387858203)

---

## まとめ

本ドキュメントでは、スキルノート推薦システムの機械学習実装について詳細に説明しました。

**キーポイント:**

✅ **NMFによる協調フィルタリング**
- 非負制約により解釈可能
- 潜在因子数 k=20 が推奨
- `MLRecommender.build(n_components=20)` で明示的に指定可能

✅ **4つの多様性戦略**
- Hybrid が最もバランスが良い
- 用途に応じて選択可能

✅ **評価メトリクス**
- 精度 (Precision, Recall, NDCG)
- 多様性 (Category, Type, Coverage, ILD)

✅ **パフォーマンス最適化**
- スパース行列、バッチ予測
- モデルキャッシング

✅ **ベストプラクティス**
- 定期的な再学習
- A/Bテスト
- ログ記録

---

## 関連ドキュメント

- [MODELS_TECHNICAL_GUIDE.md](MODELS_TECHNICAL_GUIDE.md) - モデル実装の詳細、データ前処理、ハイパーパラメータチューニング
- [EVALUATION.md](EVALUATION.md) - 推薦システムの評価方法
- [QUICKSTART.md](QUICKSTART.md) - クイックスタートガイド
- [CODE_STRUCTURE.md](CODE_STRUCTURE.md) - コード構造とモジュール設計

---

**更新履歴:**
- 2025-11-07: MLRecommender.build()のn_componentsパラメータ、関連ドキュメントへのリンク追加
- 2025-10-24: 初版作成
