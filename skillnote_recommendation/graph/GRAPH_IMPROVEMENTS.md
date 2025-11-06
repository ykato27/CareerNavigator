# Graph-based Skill Recommendation: 改善内容詳細

## 📋 概要

スキル遷移パターンから学習パスを推薦するGraph-basedシステムの描画方法とロジックを大幅に改善しました。

---

## 🔴 問題点の分析

### 1. ロジック面の問題

#### ❌ 従来の問題

**graph_recommender.py**
- **O(n²)の非効率な全ペア探索**: 各メンバーのスキル履歴全ペアを探索（145-160行目）
- **時間減衰の欠如**: 180日以内の遷移を全て同等に扱う（最近の遷移と古い遷移を区別しない）
- **スケールが異なるスコアの単純加算**: 遷移人数（整数）と類似度（0-1）を任意の係数10で加算
- **Node2Vecパラメータの固定**: p=1.0, q=2.0がハードコード

**hybrid_recommender.py**
- **Min-Max正規化の脆弱性**: 外れ値に弱い
- **固定的な重み配分**: 全ユーザーに同じ重み（graph=0.4, cf=0.3, content=0.3）

**random_walk.py**
- **最短パス偏重**: 最短パス≠最適なパス
- **フォールバックパスの質**: [source, target]のような意味のないパスになることがある

### 2. 可視化面の問題

#### ❌ 従来の問題

**path_visualizer.py**
- **単純な階層レイアウト**: ノードが重なって見づらい
- **パス色の区別困難**: 3色以上で区別しづらい
- **静的な可視化**: インタラクティブ機能が限定的
- **スケーラビリティの欠如**: パス数が多いと見づらい
- **時間情報の欠如**: スキル遷移の時間が可視化されていない
- **統計情報の不足**: 遷移人数、平均習得期間などが表示されない

---

## ✅ 改善内容

### A. ロジックの改善

#### 1. **時間減衰重み付け** (`enhanced_graph_recommender.py`)

```python
# 指数減衰で最近の遷移を重視
days_ago = (now - target_date).days
decay_weight = np.exp(-time_decay_factor * days_ago / 365)
```

**効果:**
- 最近の遷移パターンを重視
- 古いデータの影響を適切に減衰
- 時代遅れの推薦を防止

#### 2. **効率的なパス探索**

```python
# 連続するスキルペアのみを抽出（O(n)）
for i in range(len(skills) - 1):
    source_skill, source_date = skills[i]
    target_skill, target_date = skills[i + 1]
```

**効果:**
- O(n²) → O(n)に改善
- 大規模データでも高速処理

#### 3. **パス品質スコアリング**

```python
quality = (
    0.25 * length_score +      # パス長（短いほど良い）
    0.30 * strength_score +    # 遷移強度（多いほど良い）
    0.25 * stability_score +   # 安定性（標準偏差が小さいほど良い）
    0.20 * recency_score       # 新規性（最近の遷移があるほど良い）
)
```

**考慮要素:**
- パス長: 2-5ステップが最適
- 遷移強度: 多くの人が辿ったパス
- 安定性: 習得期間のばらつきが小さい
- 新規性: 最近の遷移実績がある

#### 4. **Robust Scaling（外れ値に強い正規化）**

```python
# Min-Max正規化 → Robust Scaling
median = np.median(scores)
q75, q25 = np.percentile(scores, [75, 25])
iqr = q75 - q25
scaled = ((score - median) / iqr) * 5 + 5
```

**効果:**
- 外れ値の影響を軽減
- スコア分布の歪みを補正

#### 5. **詳細な統計情報の保存**

```python
self.edge_details[(source, target)] = {
    'count': count,                    # 遷移人数
    'weighted_count': weighted_count,  # 時間減衰重み付きカウント
    'avg_days': avg_time_diff,         # 平均日数
    'median_days': median_time_diff,   # 中央値日数
    'std_days': std_time_diff,         # 標準偏差
    'recency_days': recency_days,      # 最新の遷移からの日数
    'transitions': transitions          # 個別の遷移データ
}
```

---

### B. 可視化の大幅改善

#### 1. **高度なインタラクティブ可視化** (`enhanced_path_visualizer.py`)

##### 特徴:

✅ **力学的レイアウト（Fruchterman-Reingold）**
- ノードの重なりを自動解消
- 見やすい配置を自動計算

```python
pos = nx.spring_layout(
    G,
    k=1.0/np.sqrt(len(G.nodes())),  # 最適距離
    iterations=50,
    scale=2.0
)
```

✅ **パスごとのトグル表示**
- 凡例クリックでパスの表示/非表示
- 「全パス表示」「Top3のみ」ボタン

✅ **詳細なホバー情報**

ノード:
- ノード名
- タイプ（メンバー/力量/カテゴリー）
- 経由パス数

エッジ:
- 遷移元 → 遷移先
- 遷移人数
- 平均期間・中央値
- 成功率（オプション）

✅ **パス品質スコアリング**
- 各パスに品質スコア（0-1）を計算
- 品質スコアでフィルタリング可能

✅ **色覚多様性対応**
- Okabe-Itoカラーパレット使用
- 色覚多様性に配慮した色選択

```python
COLORBLIND_SAFE_PALETTE = [
    '#0173B2',  # Blue
    '#DE8F05',  # Orange
    '#029E73',  # Green
    '#CC78BC',  # Purple
    # ...
]
```

##### 使用例:

```python
from skillnote_recommendation.graph.enhanced_path_visualizer import (
    EnhancedPathVisualizer,
    EdgeStatistics
)

# 初期化
visualizer = EnhancedPathVisualizer(
    layout_algorithm='fruchterman_reingold',
    colorblind_safe=True,
    show_edge_statistics=True
)

# エッジ統計情報を準備
edge_stats = {
    ('skill_A', 'skill_B'): EdgeStatistics(
        source_name='スキルA',
        target_name='スキルB',
        transition_count=10,
        avg_days=45.0,
        median_days=40.0,
        success_rate=0.85
    )
}

# 可視化
fig = visualizer.visualize_paths(
    paths=recommendation_paths,
    target_member_name='山田太郎',
    target_competence_name='Python上級',
    edge_statistics=edge_stats,
    path_scores=[0.9, 0.8, 0.7],
    min_quality_score=0.5
)

fig.show()
```

#### 2. **サンキーダイアグラム可視化** (`sankey_visualizer.py`)

##### 特徴:

✅ **遷移フローの直感的表現**
- スキル遷移の流れを視覚化
- フローの太さで遷移人数を表現

✅ **時間情報の統合**
- エッジの色で平均習得期間を表現
  - 🟢 緑: 0-30日（速い）
  - 🟡 黄: 30-90日（普通）
  - 🟠 オレンジ: 90-180日（遅い）
  - 🔴 赤: 180日+（とても遅い）

✅ **ヒートマップ表示**
- スキル間の遷移強度をマトリクスで可視化
- 全体のパターンを俯瞰

##### 使用例:

```python
from skillnote_recommendation.graph.sankey_visualizer import (
    SkillTransitionSankeyVisualizer,
    TimeBasedSankeyVisualizer
)

# 標準的なサンキーダイアグラム
sankey_vis = SkillTransitionSankeyVisualizer(
    show_percentages=True,
    color_by_category=True
)

fig1 = sankey_vis.visualize_transition_flow(
    paths=recommendation_paths,
    target_member_name='山田太郎',
    target_competence_name='Python上級',
    transition_counts=transition_count_dict
)

# 時間情報を含むサンキーダイアグラム
time_sankey = TimeBasedSankeyVisualizer()

fig2 = time_sankey.visualize_with_time_info(
    paths=recommendation_paths,
    target_member_name='山田太郎',
    target_competence_name='Python上級',
    edge_time_info={
        ('skill_A', 'skill_B'): {
            'avg_days': 45.0,
            'median_days': 40.0,
            'count': 10
        }
    }
)

# ヒートマップ
fig3 = sankey_vis.visualize_skill_matrix_heatmap(
    transition_matrix=transition_matrix,
    skill_names=skill_name_dict
)
```

#### 3. **比較ビュー**

複数の可視化を並べて比較:

```python
from skillnote_recommendation.graph.enhanced_path_visualizer import (
    create_comparison_view
)

fig = create_comparison_view([
    ('従来の可視化', fig_old),
    ('改善版の可視化', fig_new),
    ('サンキーダイアグラム', fig_sankey),
])

fig.show()
```

---

## 📊 改善効果の比較

### ロジック面

| 項目 | 従来 | 改善後 | 効果 |
|------|------|--------|------|
| パス探索の計算量 | O(n²) | O(n) | **大幅な高速化** |
| 時間考慮 | ❌ なし | ✅ 指数減衰 | **最新パターンを重視** |
| パス品質評価 | ❌ なし | ✅ 4要素で評価 | **推薦精度向上** |
| 正規化手法 | Min-Max | Robust Scaling | **外れ値に強い** |
| 統計情報 | 基本のみ | 詳細（7項目） | **解釈性向上** |

### 可視化面

| 項目 | 従来 | 改善後 | 効果 |
|------|------|--------|------|
| レイアウト | 階層のみ | 力学的/階層 | **見やすさ向上** |
| インタラクション | 限定的 | 豊富 | **探索性向上** |
| 時間情報 | ❌ なし | ✅ 色・太さで表現 | **理解度向上** |
| 色覚多様性対応 | ❌ なし | ✅ Okabe-Itoパレット | **アクセシビリティ向上** |
| パスフィルタリング | ❌ なし | ✅ 品質スコアでフィルタ | **ノイズ削減** |
| 可視化形式 | ネットワーク図のみ | ネットワーク図<br>サンキーダイアグラム<br>ヒートマップ | **多角的分析** |

---

## 🎯 使い分けガイド

### 可視化の選択

#### 1. **Enhanced Path Visualizer（拡張パス可視化）**

**使うべき場面:**
- 少数（1-10個）のパスを詳細に分析したい
- パスの構造を理解したい
- ノード・エッジの詳細情報を見たい

**設定のポイント:**
```python
visualizer = EnhancedPathVisualizer(
    layout_algorithm='fruchterman_reingold',  # 力学的レイアウト
    colorblind_safe=True,                     # 色覚多様性対応
    show_edge_statistics=True                 # エッジ統計表示
)
```

#### 2. **Sankey Diagram（サンキーダイアグラム）**

**使うべき場面:**
- 遷移フローを直感的に理解したい
- 複数のパスを同時に比較したい
- 遷移人数の大小を視覚的に把握したい

**設定のポイント:**
```python
sankey_vis = SkillTransitionSankeyVisualizer(
    show_percentages=True,      # パーセンテージ表示
    color_by_category=True,     # カテゴリーごとに色分け
    min_flow_threshold=3        # 最小フロー数（ノイズ削減）
)
```

#### 3. **Time-based Sankey（時間考慮サンキー）**

**使うべき場面:**
- 習得期間を重視したい
- 速く習得できるパスを見つけたい
- 遷移期間のばらつきを確認したい

#### 4. **Heatmap（ヒートマップ）**

**使うべき場面:**
- 全体のスキル遷移パターンを俯瞰したい
- スキル間の関連性を発見したい
- データ分析のための全体像把握

---

## 🚀 パフォーマンス最適化

### 1. **計算量の削減**

従来:
```python
# O(n²) の全ペア探索
for i in range(len(skills)):
    for j in range(i + 1, len(skills)):
        # 処理
```

改善後:
```python
# O(n) の連続ペア探索
for i in range(len(skills) - 1):
    # 処理
```

### 2. **キャッシング**

エッジ統計情報をキャッシュ:
```python
# 一度計算した統計情報を再利用
edge_statistics = recommender.get_edge_statistics()
```

### 3. **フィルタリング**

品質スコアでフィルタリングしてノイズを削減:
```python
fig = visualizer.visualize_paths(
    paths=paths,
    min_quality_score=0.5  # 品質スコア0.5以上のみ表示
)
```

---

## 📈 今後の拡張可能性

### 1. **機械学習による品質スコア予測**
- 過去の推薦結果をフィードバック
- より精度の高い品質スコアを自動学習

### 2. **3D可視化**
- Plotly 3Dグラフで立体的に表示
- 複雑なパスを分かりやすく表現

### 3. **アニメーション**
- 学習パスを時系列で動的に表示
- 習得の流れを視覚的に表現

### 4. **A/Bテスト機能**
- 従来版と改善版の推薦結果を比較
- 効果測定を自動化

---

## 🎓 参考文献

### レイアウトアルゴリズム
- Fruchterman, T. M., & Reingold, E. M. (1991). "Graph drawing by force-directed placement"
- NetworkX Documentation: https://networkx.org/documentation/stable/reference/generated/networkx.drawing.layout.spring_layout.html

### 色覚多様性
- Okabe, M., & Ito, K. (2008). "Color Universal Design (CUD)"
- https://jfly.uni-koeln.de/color/

### グラフ推薦システム
- Random Walk with Restart (RWR): Tong, H., et al. (2006). "Fast Random Walk with Restart"
- Node2Vec: Grover, A., & Leskovec, J. (2016). "node2vec: Scalable Feature Learning for Networks"

---

## 💡 まとめ

この改善により、Graph-basedスキル推薦システムは以下を実現しました:

✅ **ロジックの改善**
- 時間減衰重み付けで最新パターンを重視
- パス品質評価で推薦精度を向上
- Robust Scalingで外れ値に強い正規化

✅ **可視化の大幅改善**
- 力学的レイアウトで見やすい配置
- リッチなインタラクション機能
- 時間情報の統合
- 色覚多様性対応
- 複数の可視化形式（ネットワーク図、サンキー、ヒートマップ）

✅ **パフォーマンス最適化**
- O(n²) → O(n)の計算量削減
- キャッシング機構
- フィルタリング機能

これらの改善により、より精度が高く、解釈しやすく、ユーザーフレンドリーな推薦システムが実現されました。
