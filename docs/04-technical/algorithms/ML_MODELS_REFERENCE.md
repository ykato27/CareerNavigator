# 機械学習/統計モデル - 参考資料

CareerNavigatorの基盤となる機械学習および統計モデルの技術詳細を解説します。これらのモデルは、StreamlitアプリおよびWebUIの推薦機能で使用されています。

## 概要

本システムでは複数の機械学習・統計手法を組み合わせて、高精度なスキル推薦を実現しています：

- **Matrix Factorization (NMF)**: 協調フィルタリング
- **グラフベース推薦**: Random Walk with Restart
- **因果推論 (LiNGAM)**: スキル依存関係の抽出
- **ベイジアンネットワーク**: 習得確率の予測
- **SEM (構造方程式モデリング)**: キャリアパス分析

---

## Matrix Factorization (NMF)

### 概要

Non-negative Matrix Factorization（非負値行列因子分解）を用いた協調フィルタリング手法。

### 手法の詳細

**入力**:
- メンバー × 力量マトリックス（習得=1, 未習得=0）

**処理**:
```python
from sklearn.decomposition import NMF

# NMFモデル
model = NMF(
    n_components=20,     # 潜在因子数
    init='nndsvda',       # 初期化法
    random_state=42,
    max_iter=200
)

# 行列分解: X ≈ W × H
W = model.fit_transform(member_competence_matrix)  # メンバー × 潜在因子
H = model.components_                               # 潜在因子 × 力量
```

**出力**:
- 未習得力量に対する予測スコア

### パラメータ

| パラメータ | デフォルト値 | 説明 |
|-----------|-------------|------|
| `n_components` | 20 | 潜在因子の数。多いほど表現力が高いが、過学習のリスク |
| `max_iter` | 200 | 最大イテレーション数 |
| `random_state` | 42 | 乱数シード（再現性のため） |

### 実装例

```python
from skillnote_recommendation.ml import MLRecommender

# ML推薦システム初期化
ml_recommender = MLRecommender(data)

# メンバーへの推薦
recommendations = ml_recommender.recommend(
    member_code='m48',
    top_n=10,
    use_diversity=True
)
```

### メリット・デメリット

**メリット**:
- メンバーの習得パターンから類似メンバーの傾向を学習
- 新しいメンバーにも適用可能（コールドスタート問題の軽減）
- 計算効率が良い

**デメリット**:
- スキル間の因果関係を考慮しない
- 説明可能性が低い
- スパースデータに弱い

---

## グラフベース推薦

### Random Walk with Restart (RWR)

知識グラフ上でのランダムウォークにより、関連性の高いスキルを推薦。

### アルゴリズム

1. **知識グラフの構築**
```python
from skillnote_recommendation.graph import KnowledgeGraph

kg = KnowledgeGraph(
    member_competence_df,
    competence_master_df
)
kg.build_graph()
```

2. **RWRの実行**
```python
restart_prob = 0.15  # リスタート確率
max_iterations = 100

# 初期確率ベクトル（現在保有スキルから開始）
p0 = initial_probability_vector(owned_skills)

# 反復計算
p = p0
for i in range(max_iterations):
    p_new = (1 - restart_prob) * transition_matrix @ p + restart_prob * p0
    if converged(p, p_new):
        break
    p = p_new

# スコア上位のスキルを推薦
recommendations = top_k_skills(p, k=10)
```

### パラメータ

| パラメータ | デフォルト値 | 説明 |
|-----------|-------------|------|
| `restart_prob` | 0.15 | リスタート確率。高いほど保有スキル近傍を重視 |
| `max_iterations` | 100 | 最大イテレーション数 |
| `convergence_threshold` | 1e-6 | 収束判定閾値 |

### メリット・デメリット

**メリット**:
- グラフ構造を活用した推薦
- スキル間の関連性を考慮
- 解釈性が高い

**デメリット**:
- グラフ構築のコスト
- スケーラビリティの課題
- 孤立ノードへの対応が難しい

---

## 因果推論 (LiNGAM)

### 概要

Linear Non-Gaussian Acyclic Model（LiNGAM）を用いて、スキル間の因果構造を学習。

### アルゴリズム

**DirectLiNGAM**:
```python
from lingam import DirectLiNGAM

# スキル頻度行列の準備
skill_frequency_matrix = compute_skill_frequency()

# 因果構造学習
model = DirectLiNGAM()
model.fit(skill_frequency_matrix)

# 因果効果行列の取得
causal_effects = model.adjacency_matrix_
```

**因果効果の解釈**:
- `causal_effects[i][j] > 0`: スキルiがスキルjの習得を促進
- `causal_effects[i][j] < 0`: スキルiがスキルjの習得を阻害
- `causal_effects[i][j] == 0`: 因果関係なし

### 3軸スコアリング

LiNGAMの結果を基に、3つの観点からスキルを評価：

#### 1. 準備完了度 (Readiness Score)

現在保有スキルからの因果効果の総和。

```python
readiness_score[skill_j] = sum(
    causal_effects[owned_skill_i][skill_j]
    for owned_skill_i in owned_skills
    if causal_effects[owned_skill_i][skill_j] > threshold
)
```

#### 2. 確率スコア (Probability Score)

ベイジアンネットワークによる習得確率。

```python
from pgmpy.models import BayesianNetwork

# 因果グラフからベイジアンネットワークを構築
bn = build_bayesian_network(causal_effects)

# 条件付き確率の計算
prob_score[skill_j] = P(skill_j = 1 | owned_skills)
```

#### 3. 有用性スコア (Utility Score)

将来のスキル習得への貢献度。

```python
utility_score[skill_j] = sum(
    causal_effects[skill_j][future_skill_k]
    for future_skill_k in target_skills
    if causal_effects[skill_j][future_skill_k] > threshold
)
```

**総合スコア**:
```python
total_score = (
    readiness_score^0.3 *
    probability_score^0.3 *
    utility_score^0.4
)
```

### 実装例

```python
from skillnote_recommendation.graph import CausalGraphRecommender

# Causal Recommender初期化
recommender = CausalGraphRecommender(
    member_competence=member_competence_df,
    competence_master=competence_master_df
)

# 因果グラフ学習
recommender.fit()

# 推薦生成
recommendations = recommender.recommend(
    member_code='m48',
    min_total_score=0.02,
    min_readiness_score=0.0
)
```

### メリット・デメリット

**メリット**:
- スキル依存関係の明示的なモデル化
- 高い説明可能性
- 学習の順序を考慮した推薦

**デメリット**:
- 線形性と非ガウス性の仮定
- サンプルサイズの要求
- 計算コストが高い

---

## ベイジアンネットワーク

### 概要

因果グラフを基にベイジアンネットワークを構築し、スキル習得確率を予測。

### 階層的ベイジアン推薦

3層アーキテクチャによる推薦システム：

**Layer 1: ベイジアンネットワーク（大カテゴリ）**
```python
from pgmpy.estimators import HillClimbSearch, BicScore

# カテゴリレベルでのネットワーク学習
hc = HillClimbSearch(category_data)
best_model = hc.estimate(
    scoring_method=BicScore(category_data),
    max_indegree=3  # 過学習防止
)
```

**Layer 2: 条件付き確率（中カテゴリ）**
```python
# P(中カテゴリ | 大カテゴリ)の学習
conditional_probs = estimate_conditional_probability(
    l2_categories, l1_categories
)
```

**Layer 3: カテゴリ別行列分解（スキルレベル）**
```python
# 各L2カテゴリごとに独立したMF
category_models = {}
for category in l2_categories:
    category_data = filter_by_category(data, category)
    category_models[category] = NMF(n_components=10).fit(category_data)
```

**スコア統合**:
```python
final_score = (
    l1_readiness ** 0.3 *
    l2_probability ** 0.3 *
    l3_skill_score ** 0.4
)
```

### 実装例

```python
from skillnote_recommendation.ml import HierarchicalBayesianRecommender

recommender = HierarchicalBayesianRecommender(
    member_competence=member_competence_df,
    competence_master=competence_master_df,
    category_csv_path='data/categories/competence_category.csv',
    max_indegree=3,
    n_components=10
)

# 学習
recommender.fit()

# 推薦生成（階層的説明付き）
recommendations = recommender.recommend(member_code='m48', top_n=10)
```

---

## SEM (構造方程式モデリング)

### 概要

Structural Equation Modeling（SEM）を用いたキャリアパス分析。

### モデル構造

**潜在変数**:
- キャリアステージ（初級/中級/上級）
- スキルカテゴリ（技術/マネジメント/ビジネス）

**観測変数**:
- 個別スキル

**パス図**:
```
[初級ステージ] → [技術スキル群] → [個別スキルA, B, C]
       ↓
[中級ステージ] → [マネジメントスキル群] → [個別スキルD, E, F]
       ↓
[上級ステージ] → [ビジネススキル群] → [個別スキルG, H, I]
```

### 実装

```python
import semopy

# SEMモデルの定義
model_spec = """
# 測定モデル
TechSkills =~ skill_a + skill_b + skill_c
MgmtSkills =~ skill_d + skill_e + skill_f
BizSkills =~ skill_g + skill_h + skill_i

# 構造モデル
MgmtSkills ~ TechSkills
BizSkills ~ MgmtSkills
"""

# モデル推定
model = semopy.Model(model_spec)
result = model.fit(data)

# 適合度指標
print(f"CFI: {result.cfi}")
print(f"RMSEA: {result.rmsea}")
```

### 適合度指標

| 指標 | 基準値 | 説明 |
|------|-------|------|
| CFI | \> 0.95 | Comparative Fit Index（比較適合度指標） |
| RMSEA | \< 0.06 | Root Mean Square Error of Approximation |
| SRMR | \< 0.08 | Standardized Root Mean Square Residual |

### メリット・デメリット

**メリット**:
- キャリアパスの構造化
- 潜在変数の導入による抽象化
- 理論的枠組みの構築

**デメリット**:
- モデル構築の複雑性
- 大規模データの要求
- 計算コストが高い

---

## ハイブリッド推薦システム

### 概要

複数の推薦手法を組み合わせて、各手法の長所を活かす。

### スコア統合方法

**重み付き線形結合**:
```python
hybrid_score = (
    w_mf * mf_score +
    w_graph * graph_score +
    w_causal * causal_score
)

# デフォルト重み
w_mf = 0.3
w_graph = 0.3
w_causal = 0.4
```

**ランクベース統合**:
```python
# 各手法のランキングを統合
mf_ranks = rank(mf_scores)
graph_ranks = rank(graph_scores)
causal_ranks = rank(causal_scores)

# Borda Count
hybrid_ranks = mf_ranks + graph_ranks + causal_ranks

# 最終推薦リスト
recommendations = sort_by_rank(hybrid_ranks)
```

### 実装例

```python
from skillnote_recommendation.graph import HybridRecommender

recommender = HybridRecommender(
    member_competence=member_competence_df,
    competence_master=competence_master_df
)

# 推薦生成（ハイブリッド）
recommendations = recommender.recommend(
    member_code='m48',
    method='hybrid',  # 'mf', 'graph', 'causal', 'hybrid'
    mf_weight=0.3,
    graph_weight=0.3,
    causal_weight=0.4
)
```

---

## 多様性再ランキング

### 概要

推薦結果の多様性を確保するための再ランキング戦略。

### 戦略

#### 1. MMR (Maximal Marginal Relevance)

関連度と多様性のバランス。

```python
def mmr_rerank(candidates, selected, lambda_param=0.7):
    scores = []
    for candidate in candidates:
        relevance = candidate.score
        diversity = min_similarity(candidate, selected)
        mmr_score = lambda_param * relevance + (1 - lambda_param) * diversity
        scores.append(mmr_score)
    return top_k(scores)
```

#### 2. カテゴリ多様性

異なるカテゴリから推薦。

```python
def category_diversity(recommendations, max_per_category=3):
    result = []
    category_counts = defaultdict(int)
    
    for rec in sorted_recommendations:
        category = rec.category
        if category_counts[category] < max_per_category:
            result.append(rec)
            category_counts[category] += 1
    
    return result
```  

#### 3. タイプ多様性

SKILL / EDUCATION / LICENSE をバランス良く。

```python
def type_diversity(recommendations, type_ratio={'SKILL': 0.7, 'EDUCATION': 0.2, 'LICENSE': 0.1}):
    result = []
    type_counts = defaultdict(int)
    target_counts = {k: int(v * len(recommendations)) for k, v in type_ratio.items()}
    
    for rec in sorted_recommendations:
        rec_type = rec.competence_type
        if type_counts[rec_type] < target_counts[rec_type]:
            result.append(rec)
            type_counts[rec_type] += 1
    
    return result
```

### 多様性メトリクス

```python
from skillnote_recommendation.ml import calculate_diversity_metrics

metrics = calculate_diversity_metrics(recommendations, competence_master)

print(f"カテゴリ多様性: {metrics['category_diversity']:.3f}")
print(f"タイプ多様性: {metrics['type_diversity']:.3f}")
print(f"カバレッジ: {metrics['coverage']:.3f}")
print(f"リスト内多様性: {metrics['intra_list_diversity']:.3f}")
```

---

## モデル選択ガイド

| 目的 | 推奨モデル | 理由 |
|------|----------|------|
| 高精度な推薦 | ハイブリッド | 複数手法の統合 |
| 説明可能性重視 | 因果推論 (LiNGAM) | 明示的な因果関係 |
| 計算効率重視 | Matrix Factorization | 高速計算 |
| グラフ構造活用 | RWR | 関連性の伝播 |
| キャリアパス分析 | SEM | 構造的モデリング |

---

## 関連ドキュメント

- [因果推論推薦の詳細](CAUSAL_RECOMMENDATION.md)
- [機械学習技術詳細](ML_TECHNICAL_DETAILS.md)
- [ハイブリッド推薦システム](HYBRID_RECOMMENDATION_SYSTEM.md)
- [評価ガイド](EVALUATION.md)
- [SEM実装サマリー](SEM_IMPLEMENTATION_SUMMARY.md)

---

## 参考文献

1. Shimizu, S., et al. (2006). "A Linear Non-Gaussian Acyclic Model for Causal Discovery"
2. Tong, H., et al. (2006). "Fast Random Walk with Restart"
3. Lee, D. D., & Seung, H. S. (1999). "Learning the parts of objects by non-negative matrix factorization"
4. Pearl, J. (2009). "Causality: Models, Reasoning, and Inference"
5. Koren, Y., et al. (2009). "Matrix factorization techniques for recommender systems"
