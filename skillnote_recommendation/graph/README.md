# Graph モジュール

このディレクトリには、**グラフベースの推薦システム**が含まれています。

## 📋 このモジュールの役割

- ✅ 知識グラフの構築と活用
- ✅ スキル間の関係性を考慮した推薦
- ✅ キャリアパスの可視化
- ✅ ハイブリッド推薦（グラフ + 機械学習）
- ✅ スキルの依存関係分析

## 🕸️ グラフベース推薦とは？

グラフベース推薦は、**力量（スキル）間のつながり**を考慮して推薦を行います。

**例:**
```
Python → Django → Webアプリ開発
  ↓
NumPy → データ分析
```

**推薦アプローチの違い:**
- **機械学習ベース**: 過去のパターンから学習 (ml モジュール)
- **グラフベース**: 力量間の関係性を活用 (このモジュール)
- **ハイブリッド**: 両方を組み合わせて最適な推薦を実現

## 📂 ファイル分類

### 🎯 主要なクラス（重要）

| ファイル | クラス名 | 役割 |
|---------|---------|------|
| **knowledge_graph.py** | `CompetenceKnowledgeGraph` | **知識グラフ** - 力量間の関係性をグラフ構造で表現 |
| **hybrid_recommender.py** | `HybridGraphRecommender` | **ハイブリッド推薦** - RWR + NMF + Content-Basedを組み合わせた推薦システム |
| **career_path.py** | `CareerPath` | **キャリアパス** - メンバーのキャリア経路を表現 |

### 🧭 グラフアルゴリズム

| ファイル | クラス名 | 役割 |
|---------|---------|------|
| **random_walk.py** | `RandomWalkRecommender` | **Random Walk with Restart (RWR)** - PageRankベースのグラフ推薦 |
| **category_hierarchy.py** | `CategoryHierarchy` | **カテゴリ階層** - 力量カテゴリの階層構造を管理 |

### 📊 可視化

| ファイル | 関数 | 役割 |
|---------|-----|------|
| **career_path_visualizer.py** | 可視化関数 | キャリアパスをグラフで可視化 |
| **path_visualizer.py** | パス可視化関数 | 推薦経路を可視化 |
| **visualization_utils.py** | ヘルパー関数 | 可視化の補助関数 |

### 🧪 その他

| ファイル | 内容 | 役割 |
|---------|-----|------|
| **test_knowledge_graph.py** | テストコード | 知識グラフの統合テスト |

## 🚀 使い方

### 知識グラフの構築

```python
from skillnote_recommendation.graph.knowledge_graph import CompetenceKnowledgeGraph
from skillnote_recommendation.core.data_loader import DataLoader

# データ読み込み
loader = DataLoader()
data = loader.load_all_data()

# 知識グラフを構築
kg = CompetenceKnowledgeGraph(
    member_competence=data['member_competence'],
    member_master=data['member_master'],
    competence_master=data['competence_master'],
    use_category_hierarchy=True
)

# グラフの統計情報
print(f"ノード数: {kg.G.number_of_nodes()}")
print(f"エッジ数: {kg.G.number_of_edges()}")
```

### ハイブリッド推薦

```python
from skillnote_recommendation.graph.hybrid_recommender import HybridGraphRecommender
from skillnote_recommendation.ml.ml_recommender import MLRecommender
from skillnote_recommendation.ml.content_based_recommender import ContentBasedRecommender
from skillnote_recommendation.ml.feature_engineering import FeatureEngineer

# 各推薦エンジンを準備
ml_recommender = MLRecommender.build(...)
content_recommender = ContentBasedRecommender(...)
feature_engineer = FeatureEngineer(...)

# ハイブリッド推薦システムを初期化
hybrid = HybridGraphRecommender(
    knowledge_graph=kg,
    ml_recommender=ml_recommender,
    content_recommender=content_recommender,
    feature_engineer=feature_engineer,
    graph_weight=0.4,    # RWRの重み
    cf_weight=0.3,       # NMFの重み
    content_weight=0.3   # コンテンツベースの重み
)

# メンバーへの推薦
recommendations = hybrid.recommend(
    member_code='M001',
    top_n=10,
    competence_type=['SKILL', 'EDUCATION']  # オプション: 力量タイプフィルタ
)

# 推薦結果を表示
for rec in recommendations:
    print(f"{rec.competence_info['力量名']}: スコア {rec.score:.3f}")
    print(f"  - RWRスコア: {rec.graph_score:.3f}")
    print(f"  - NMFスコア: {rec.cf_score:.3f}")
    print(f"  - コンテンツスコア: {rec.content_score:.3f}")
    print(f"  - 推薦理由:")
    for reason in rec.reasons:
        print(f"    {reason}")
```

### キャリアパスの可視化

```python
from skillnote_recommendation.graph import CareerPath
from skillnote_recommendation.graph.career_path_visualizer import visualize_career_path

# キャリアパスを構築
career_path = CareerPath(
    member_code='m48',
    member_competence=data['member_competence'],
    competence_master=data['competence_master']
)

# キャリアパスを可視化
fig = visualize_career_path(career_path)
fig.show()  # または fig.savefig('career_path.png')
```

### Random Walk with Restart (RWR) による推薦

```python
from skillnote_recommendation.graph.random_walk import RandomWalkRecommender

# RWR推薦エンジンを初期化
rwr = RandomWalkRecommender(
    knowledge_graph=kg,
    restart_prob=0.15,      # 再スタート確率
    max_path_length=10,     # 推薦パスの最大長
    max_paths=10,           # 各力量の推薦パス数
    enable_cache=True       # PageRankキャッシュを有効化
)

# メンバーへの推薦（パス付き）
recommendations = rwr.recommend(
    member_code='M001',
    top_n=10,
    return_paths=True,
    competence_type=['SKILL']  # オプション: 力量タイプフィルタ
)

# 推薦結果を表示
for comp_code, score, paths in recommendations:
    comp_info = kg.get_node_info(f"competence_{comp_code}")
    print(f"{comp_info['name']}: スコア {score:.5f}")
    print(f"  推薦パス数: {len(paths)}個")

    # 最初のパスを表示
    if paths:
        path_names = [kg.get_node_info(node)['name'] for node in paths[0]]
        print(f"  パス例: {' → '.join(path_names)}")
```

## 📊 グラフ構造の例

### 力量グラフ

```
         [Python基礎]
            /    \
           /      \
    [NumPy]      [Django]
       |            |
  [Pandas]    [Webアプリ開発]
       |
  [データ分析]
```

### キャリアパス

```
2020年 → [Excel] → [SQL]
                      |
2021年 →          [Python] → [データ分析]
                      |
2022年 →          [機械学習] → [AI開発]
```

## 🔬 技術詳細

### 知識グラフの構築方法

1. **共起ベース**: 同じメンバーが保有している力量を結びつける
   ```
   メンバーAが Python と NumPy を保有
   → Python ←→ NumPy の関係を追加
   ```

2. **カテゴリベース**: 同じカテゴリの力量を結びつける
   ```
   Python と Django が「プログラミング」カテゴリ
   → Python ←→ Django の関係を追加
   ```

3. **時系列ベース**: 習得順序で結びつける
   ```
   メンバーBが 2020年にPython、2021年にDjangoを習得
   → Python → Django の有向関係を追加
   ```

### ハイブリッドスコア計算

```
ハイブリッドスコア = (グラフスコア × graph_weight) + (MLスコア × ml_weight)

ここで、graph_weight + ml_weight = 1.0
```

## 📖 詳しく知りたい方へ

- **初心者向け**: [初心者向けガイド](../../docs/BEGINNER_GUIDE.md)
- **グラフ理論の基礎**: [グラフ理論 (Wikipedia)](https://ja.wikipedia.org/wiki/グラフ理論)
- **知識グラフ**: [Knowledge Graph (Wikipedia)](https://en.wikipedia.org/wiki/Knowledge_graph)

## 🔗 関連モジュール

- **[core/](../core/)** - コアビジネスロジック、データ構造
- **[ml/](../ml/)** - 機械学習ベースの推薦システム
- **[utils/](../utils/)** - ユーティリティ関数

## ⚠️ 注意事項

- **グラフ構築**: 十分なデータ量がないと、有意義なグラフが構築できません（メンバー数 100人以上、力量数 50以上を推奨）。
- **計算量**: グラフアルゴリズムは、ノード数・エッジ数が多いと計算時間がかかります。
- **実験的機能**: このモジュールの一部機能は実験的です。本番環境での使用前に十分なテストを行ってください。

## 🎓 学習リソース

- [NetworkXドキュメント](https://networkx.org/) - Pythonグラフライブラリ
- [グラフ理論入門](https://ja.wikipedia.org/wiki/グラフ理論)
- [PageRankアルゴリズム](https://ja.wikipedia.org/wiki/PageRank)

## 💡 活用例

### 1. スキル依存関係の発見

「Djangoを習得している人は、その前にPythonを習得している」といった依存関係を自動発見できます。

### 2. キャリアパスの推薦

過去の成功例から、「Aさんと同じような経路をたどれば、同じような成長が期待できる」という推薦が可能です。

### 3. スキルギャップ分析

目標とする力量セットに到達するために、どの力量を習得すべきか、グラフ上の最短経路を計算できます。
