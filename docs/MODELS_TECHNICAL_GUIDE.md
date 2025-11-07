# モデル技術ガイド（初級者向け）

本ドキュメントでは、CareerNavigatorシステムで使用されている各種推薦モデルについて、初級者向けにわかりやすく解説します。

## 目次

1. [Matrix Factorization（行列分解）](#1-matrix-factorization行列分解)
2. [多様性再ランキング](#2-多様性再ランキング)
3. [グラフベース推薦](#3-グラフベース推薦)
4. [ハイブリッド推薦](#4-ハイブリッド推薦)

---

## 1. Matrix Factorization（行列分解）

### 概要

Matrix Factorization（行列分解）は、協調フィルタリングの一種で、メンバーと力量の関係を潜在因子（見えない特徴）で表現します。

### 仕組み

```
メンバー×力量マトリクス = メンバー因子行列 × 力量因子行列
```

- **入力**: メンバーが習得している力量の情報（習得=1, 未習得=0）
- **出力**: 未習得力量に対する予測スコア（習得しそうな度合い）

### 使用アルゴリズム

**NMF (Non-negative Matrix Factorization: 非負値行列因子分解)**

- 全ての値を0以上に制約することで、解釈しやすい結果を得られます
- scikit-learnのNMFを使用しています

### パラメータ

| パラメータ | 説明 | デフォルト値 |
|-----------|------|-------------|
| `n_components` | 潜在因子数（特徴の数） | 20 |
| `random_state` | 乱数シード（再現性のため） | 42 |
| `max_iter` | 最大イテレーション数 | 200 |

### 使い方

```python
from skillnote_recommendation.ml import MLRecommender, MatrixFactorizationModel
from skillnote_recommendation.core.data_loader import DataLoader

# データ読み込み
loader = DataLoader()
data = loader.load_all_data()

# MLRecommenderをビルド（自動的にMatrix Factorizationを学習）
recommender = MLRecommender.build(
    member_competence=data['member_competence'],
    competence_master=data['competence_master'],
    member_master=data['members'],
    use_preprocessing=False,  # データ前処理なし
    use_tuning=False,  # ハイパーパラメータチューニングなし
    n_components=20  # 潜在因子数
)

# メンバーm48への推薦
recommendations = recommender.recommend(
    member_code='m48',
    top_n=10
)
```

### メリット・デメリット

**メリット**:
- メンバー間の類似性を自動的に学習
- 計算が高速
- スケーラビリティが高い

**デメリット**:
- 新規メンバー・新規力量に対応できない（コールドスタート問題）
- 習得データが少ないと精度が低い

---

## 2. 多様性再ランキング

### 概要

Matrix Factorizationで得られた推薦結果を、多様性の観点から並べ替えることで、バランスの良い推薦を実現します。

### 戦略

#### 2.1. MMR (Maximal Marginal Relevance)

関連度と多様性のバランスを取る手法です。

**式**:
```
score(item) = λ × 関連度 - (1-λ) × (既選択アイテムとの類似度)
```

- `λ = 0.7` がデフォルト（関連度重視）
- `λ = 0.5` で完全なバランス
- `λ = 0.3` で多様性重視

#### 2.2. カテゴリ多様性

異なるカテゴリから均等に推薦します。

- 例: プログラミング、データベース、クラウドなど、様々なカテゴリから推薦

#### 2.3. タイプ多様性

SKILL、EDUCATION、LICENSEのバランスを取ります。

- デフォルト比率: SKILL 60%, EDUCATION 30%, LICENSE 10%

#### 2.4. ハイブリッド戦略

上記3つの戦略を組み合わせた総合的な多様性確保手法です。

### 使い方

```python
# 多様性戦略を指定して推薦
recommendations = recommender.recommend(
    member_code='m48',
    top_n=10,
    use_diversity=True,  # 多様性を有効化
    diversity_strategy='hybrid'  # hybrid / mmr / category / type
)
```

### 多様性メトリクス

推薦結果の多様性を評価する指標：

- **カテゴリ多様性**: ユニークなカテゴリ数 / 推薦数
- **タイプ多様性**: ユニークなタイプ数 / 推薦数
- **カタログカバレッジ**: 推薦に含まれた力量の種類 / 全力量数

```python
# 多様性メトリクスを計算
metrics = recommender.calculate_diversity_metrics(recommendations)
print(f"カテゴリ多様性: {metrics['category_diversity']:.3f}")
print(f"タイプ多様性: {metrics['type_diversity']:.3f}")
```

---

## 3. グラフベース推薦

### 概要

力量間の関係を「グラフ（ネットワーク）」として表現し、関連する力量を推薦します。

### 仕組み

1. **知識グラフの構築**
   - ノード: 力量、メンバー、カテゴリ
   - エッジ: 関係（習得、所属、類似など）

2. **グラフ探索**
   - メンバーが習得している力量から出発
   - グラフを辿って関連する力量を発見

3. **スコアリング**
   - 経路の長さ
   - 共起頻度
   - カテゴリの近さ

### 使い方

```python
from skillnote_recommendation.graph import KnowledgeGraph

# 知識グラフを構築
kg = KnowledgeGraph()
kg.build_from_data(
    member_competence=data['member_competence'],
    competence_master=data['competence_master']
)

# グラフベース推薦
recommendations = kg.recommend(
    member_code='m48',
    top_n=10
)
```

### メリット・デメリット

**メリット**:
- 力量間の関係を明示的にモデル化
- 解釈しやすい
- 新規力量への対応が容易

**デメリット**:
- グラフ構築に時間がかかる
- メモリ使用量が大きい

---

## 4. ハイブリッド推薦

### 概要

複数の推薦手法を組み合わせて、それぞれの長所を活かします。

### 組み合わせ方法

#### 4.1. スコア加重平均

```
最終スコア = α × MLスコア + β × グラフスコア
```

- デフォルト: α=0.7, β=0.3（ML重視）

#### 4.2. カスケード方式

1. MLで候補を生成（100件）
2. グラフで再ランキング
3. 多様性で最終調整

#### 4.3. スイッチング方式

- データが豊富: ML推薦
- データが少ない: グラフ推薦

### 使い方

```python
from skillnote_recommendation.ml import MLRecommender
from skillnote_recommendation.graph import KnowledgeGraph

# ハイブリッド推薦
ml_recs = ml_recommender.recommend(member_code='m48', top_n=50)
graph_recs = kg.recommend(member_code='m48', top_n=50)

# スコアを組み合わせ
hybrid_recs = combine_recommendations(
    ml_recs, graph_recs,
    ml_weight=0.7,
    graph_weight=0.3
)
```

---

## モデル選択ガイド

### どのモデルを使うべきか？

| 状況 | 推奨モデル |
|------|----------|
| データが豊富（数千件以上） | Matrix Factorization |
| データが少ない（数百件以下） | グラフベース |
| 多様性を重視したい | Matrix Factorization + 多様性再ランキング |
| 解釈性を重視したい | グラフベース |
| 全般的に高精度を目指す | ハイブリッド推薦 |

### パフォーマンス比較

| モデル | 学習時間 | 推論時間 | 精度 | 多様性 |
|--------|---------|---------|------|--------|
| Matrix Factorization | ⚡ 高速 | ⚡⚡ 最速 | 🎯🎯🎯 高 | 😐 中 |
| + 多様性再ランキング | ⚡ 高速 | ⚡ 高速 | 🎯🎯🎯 高 | 🌈🌈🌈 高 |
| グラフベース | 😐 中速 | 😐 中速 | 🎯🎯 中高 | 🌈🌈 中高 |
| ハイブリッド | 😐 中速 | 😓 やや遅 | 🎯🎯🎯🎯 最高 | 🌈🌈🌈 高 |

---

## よくある質問

### Q1. n_componentsはどう決めれば良いですか？

**A**: 以下の目安に従ってください：
- データ量が少ない（〜100件）: `n_components=5〜10`
- データ量が中程度（100〜1000件）: `n_components=10〜20`（デフォルト）
- データ量が多い（1000件〜）: `n_components=20〜50`

注意: `n_components` はメンバー数と力量数の最小値以下にする必要があります。

### Q2. 推薦結果が空になるのはなぜですか？

**A**: 以下の原因が考えられます：
1. 既に全ての力量を習得している
2. フィルタ条件が厳しすぎる
3. コールドスタート問題（新規メンバー）

### Q3. 多様性戦略はどれを選べば良いですか？

**A**: 目的に応じて選択してください：
- **バランス重視**: `hybrid`（デフォルト、推奨）
- **関連度重視**: `mmr` with `λ=0.8`
- **カテゴリバランス重視**: `category`
- **タイプバランス重視**: `type`

### Q4. 学習にどのくらい時間がかかりますか？

**A**: データ量に依存します：
- 100メンバー × 500力量: 数秒
- 1000メンバー × 1000力量: 数十秒
- 10000メンバー × 5000力量: 数分

---

## 参考資料

### 論文・書籍

1. **Matrix Factorization**
   - Koren, Y., Bell, R., & Volinsky, C. (2009). "Matrix factorization techniques for recommender systems."

2. **多様性**
   - Carbonell, J., & Goldstein, J. (1998). "The use of MMR, diversity-based reranking for reordering documents and producing summaries."

3. **グラフベース推薦**
   - Sun, Y., et al. (2011). "PathSim: Meta path-based top-k similarity search in heterogeneous information networks."

### 関連ドキュメント

- [機械学習技術詳細 (ML_TECHNICAL_DETAILS.md)](ML_TECHNICAL_DETAILS.md) - より詳細な技術情報
- [評価ガイド (EVALUATION.md)](EVALUATION.md) - 推薦システムの評価方法
- [クイックスタート (QUICKSTART.md)](QUICKSTART.md) - 使い方ガイド

---

## まとめ

CareerNavigatorでは、以下の推薦モデルを提供しています：

1. **Matrix Factorization**: 高速・高精度な協調フィルタリング
2. **多様性再ランキング**: バランスの良い推薦を実現
3. **グラフベース推薦**: 関係性を重視した推薦
4. **ハイブリッド推薦**: 複数手法の組み合わせ

初心者の方は、まず **Matrix Factorization + 多様性再ランキング（hybridモード）** から始めることをお勧めします。これがデフォルト設定で、最もバランスの取れた推薦結果が得られます。
