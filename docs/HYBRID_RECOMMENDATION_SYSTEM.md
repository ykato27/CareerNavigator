# ハイブリッド推薦システム

## 概要

CareerNavigatorのハイブリッド推薦システムは、以下の3つのアプローチを統合した高度な推薦システムです：

1. **グラフベース推薦（RWR）**: ナレッジグラフ上のRandom Walk with Restartによる推薦
2. **協調フィルタリング（NMF）**: 行列分解による類似メンバーパターンの学習
3. **コンテンツベース**: 職種・等級などのメンバー属性と力量属性の親和性分析

これらを適切な重み付けで統合することで、より精度の高い推薦を実現します。

## アーキテクチャ

```
┌─────────────────────────────────────────────────────────────┐
│              ハイブリッド推薦システム                         │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ グラフベース │  │ 協調フィルタ │  │ コンテンツ   │       │
│  │   (RWR)      │  │  リング(NMF) │  │   ベース     │       │
│  │              │  │              │  │              │       │
│  │ 重み: 0.4    │  │ 重み: 0.3    │  │ 重み: 0.3    │       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │
│         │                 │                 │               │
│         └─────────────────┼─────────────────┘               │
│                           ▼                                 │
│                    スコア融合・統合                          │
│                           │                                 │
│                           ▼                                 │
│                    最終推薦結果                              │
└─────────────────────────────────────────────────────────────┘
```

## 特徴量エンジニアリング

### 1. 職種・等級のワンホットエンコーディング

メンバーの職種と等級をワンホットエンコーディングして、属性ベースの推薦に活用します。

```python
from skillnote_recommendation.ml.feature_engineering import FeatureEngineer

fe = FeatureEngineer(
    member_master=member_master,
    competence_master=competence_master,
    member_competence=member_competence
)

# メンバー属性のエンコーディング
member_vec = fe.encode_member_attributes('M001')
```

### 2. 習得力量の時系列パターン抽出

メンバーの力量習得履歴から、以下のパターンを抽出します：

- **acquisition_rate**: 過去6ヶ月の力量習得率
- **recent_activity**: 直近30日の活動度
- **skill_variety**: スキルの多様性（カテゴリ数/全力量数）
- **category_focus**: カテゴリの集中度
- **learning_velocity**: 学習速度

```python
temporal_features = fe.extract_temporal_patterns('M001')
print(temporal_features)
# {
#     'acquisition_rate': 0.05,
#     'recent_activity': 0.1,
#     'skill_variety': 0.6,
#     'category_focus': 0.3,
#     'learning_velocity': 0.02
# }
```

### 3. 力量間の共起関係の活用

メンバーが同時に習得する傾向のある力量を分析し、推薦に活用します。

```python
# 力量の共起スコアを取得
cooccurrence_score = fe.get_competence_cooccurrence_score('C001', 'C002')

# 関連する力量を取得
related_comps = fe.get_related_competences('C001', top_k=10)
```

### 4. カテゴリ階層の埋め込み表現

カテゴリごとの統計情報から埋め込みベクトルを生成し、カテゴリ間の関係性を学習します。

```python
# カテゴリの埋め込みベクトルを取得
category_embedding = fe.get_category_embedding('プログラミング')
```

## 使用方法

### 基本的な使用方法

```python
from skillnote_recommendation.graph.hybrid_builder import build_hybrid_recommender

# ハイブリッド推薦システムを構築
hybrid_recommender = build_hybrid_recommender(
    member_competence=member_competence_df,
    competence_master=competence_master_df,
    member_master=member_master_df,
    graph_weight=0.4,      # グラフベースの重み
    cf_weight=0.3,         # 協調フィルタリングの重み
    content_weight=0.3     # コンテンツベースの重み
)

# 推薦を実行
recommendations = hybrid_recommender.recommend(
    member_code='M001',
    top_n=10,
    competence_type=['SKILL'],  # オプション: フィルタリング
    use_diversity=True           # オプション: 多様性を考慮
)

# 結果を表示
for rec in recommendations:
    print(f"力量: {rec.competence_info['力量名']}")
    print(f"  総合スコア: {rec.score:.3f}")
    print(f"  グラフスコア: {rec.graph_score:.3f}")
    print(f"  協調フィルタリングスコア: {rec.cf_score:.3f}")
    print(f"  コンテンツベーススコア: {rec.content_score:.3f}")
    print(f"  理由: {', '.join(rec.reasons)}")
    print()
```

### クイック推薦

```python
from skillnote_recommendation.graph.hybrid_builder import quick_recommend

# 1行で推薦を実行
recommendations = quick_recommend(
    member_code='M001',
    member_competence=member_competence_df,
    competence_master=competence_master_df,
    member_master=member_master_df,
    top_n=10
)
```

## 評価指標

### 精度指標

以下の精度指標がサポートされています（既存のevaluator.pyで実装済み）：

- **Precision@K**: 推薦した力量のうち実際に習得した割合
- **Recall@K**: 習得した力量のうち推薦に含まれた割合
- **NDCG@K**: 推薦順位の質を評価（正規化割引累積利得）

### 多様性指標

```python
from skillnote_recommendation.ml.diversity import DiversityReranker

reranker = DiversityReranker()

# 多様性指標を計算
diversity_metrics = reranker.calculate_diversity_metrics(
    recommendations=[(rec.competence_code, rec.score) for rec in recommendations],
    competence_info=competence_master,
    all_competences=competence_master['competence_code'].tolist()
)

print(diversity_metrics)
# {
#     'category_diversity': 0.8,        # カテゴリの多様性
#     'type_diversity': 0.6,            # タイプの多様性
#     'intra_list_diversity': 0.7,      # リスト内多様性
#     'intra_list_similarity': 0.3,     # リスト内類似度
#     'coverage': 0.05,                 # カバレッジ
#     'unique_categories': 8,
#     'unique_types': 3
# }
```

- **カバレッジ**: 全力量中、推薦された力量の割合
- **Intra-List Similarity**: 推薦リスト内の類似度（低いほど多様）

### 新規性指標

```python
# セレンディピティ（意外だが有用な推薦）を計算
serendipity = reranker.calculate_serendipity(
    recommendations=[(rec.competence_code, rec.score) for rec in recommendations],
    member_competence=member_competence_df,
    member_code='M001',
    competence_info=competence_master,
    popularity_threshold=0.3
)

print(f"セレンディピティスコア: {serendipity:.3f}")
```

- **セレンディピティ**: 意外だが有用な推薦の割合
  - ユーザーの既習得力量と異なるカテゴリ・タイプ（意外性）
  - 推薦スコアが高い（有用性）
  - 適度な人気度（実績がある）

## 重みの調整

推薦手法ごとの重みは、プロジェクトの特性に応じて調整できます：

### パターン1: グラフ重視（デフォルト）

```python
hybrid_recommender = build_hybrid_recommender(
    ...,
    graph_weight=0.4,     # グラフ構造を重視
    cf_weight=0.3,
    content_weight=0.3
)
```

- **適用場面**: ネットワーク効果が重要な場合
- **特徴**: 類似メンバーやカテゴリのつながりを重視

### パターン2: 協調フィルタリング重視

```python
hybrid_recommender = build_hybrid_recommender(
    ...,
    graph_weight=0.2,
    cf_weight=0.5,        # 協調フィルタリングを重視
    content_weight=0.3
)
```

- **適用場面**: メンバー間の習得パターンが重要な場合
- **特徴**: 類似メンバーの行動から学習

### パターン3: コンテンツベース重視

```python
hybrid_recommender = build_hybrid_recommender(
    ...,
    graph_weight=0.2,
    cf_weight=0.2,
    content_weight=0.6    # コンテンツベースを重視
)
```

- **適用場面**: 職種・等級による推薦が重要な場合
- **特徴**: メンバー属性と力量属性の親和性を重視

## 推薦アプローチの選択

### 1. ハイブリッド推薦（推奨）

3つのアプローチを統合した最も精度の高い推薦です。

```python
from skillnote_recommendation.graph.hybrid_builder import build_hybrid_recommender

hybrid_recommender = build_hybrid_recommender(...)
recommendations = hybrid_recommender.recommend(member_code='M001', top_n=10)
```

### 2. グラフ構造ベース推薦（RWR単独）

グラフ構造のみを使用したシンプルな推薦です。

```python
from skillnote_recommendation.graph.random_walk import RandomWalkRecommender
from skillnote_recommendation.graph.knowledge_graph import CompetenceKnowledgeGraph

# 知識グラフを構築（コンストラクタで自動構築）
kg = CompetenceKnowledgeGraph(
    member_competence=member_competence,
    member_master=member_master,
    competence_master=competence_master,
    use_category_hierarchy=True
)

# RWR推薦エンジン
rwr = RandomWalkRecommender(knowledge_graph=kg, restart_prob=0.15)
recommendations = rwr.recommend(member_code='M001', top_n=10, return_paths=True)

# 推薦結果を表示
for comp_code, score, paths in recommendations:
    print(f"力量: {comp_code}, スコア: {score:.3f}")
    if paths:
        print(f"  推薦パス: {len(paths)}件")
```

**注意**: 協調フィルタリング単独の推薦は廃止されました。ハイブリッド推薦またはグラフ構造ベース推薦を使用してください。

## パフォーマンス最適化

### キャッシュの有効化

```python
hybrid_recommender = build_hybrid_recommender(
    ...,
    enable_cache=True  # PageRankキャッシュを有効化
)
```

### ハイパーパラメータチューニング

```python
hybrid_recommender = build_hybrid_recommender(
    ...,
    use_tuning=True  # NMFのハイパーパラメータを自動調整
)
```

## トラブルシューティング

### エラー: メンバーが見つからない

```python
# メンバーマスタにメンバーが存在するか確認
if member_code not in member_master['member_code'].values:
    print(f"エラー: メンバー {member_code} が見つかりません")
```

### 推薦結果が少ない

- フィルタ条件（competence_type, category_filter）を緩和
- top_nの値を増やす
- 未習得力量が少ない場合は、データを確認

### メモリ不足

- enable_cache=Falseにしてキャッシュを無効化
- データを分割して処理

## 参考資料

- [ML推薦システムの詳細](../skillnote_recommendation/ml/README.md)
- [グラフ推薦システムの詳細](../skillnote_recommendation/graph/README.md)
- [評価方法の完全ガイド](./EVALUATION.md)
- [ML技術の詳細](./ML_TECHNICAL_DETAILS.md)
