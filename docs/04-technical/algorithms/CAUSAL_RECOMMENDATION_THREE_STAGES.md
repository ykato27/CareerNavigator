# Causal Recommendation が三段階で必要な理由

## 📚 目次

1. [概要](#概要)
2. [なぜ三段階なのか？](#なぜ三段階なのか)
3. [第一段階: LiNGAM（因果構造学習）](#第一段階-lingam因果構造学習)
4. [第二段階: Bayesian Network（ベイジアンネットワーク）](#第二段階-bayesian-networkベイジアンネットワーク)
5. [第三段階: 3軸スコアリング](#第三段階-3軸スコアリング)
6. [三段階を統合する意義](#三段階を統合する意義)
7. [実装の詳細](#実装の詳細)
8. [まとめ](#まとめ)

---

## 概要

**Causal Recommendation（因果推論ベース推薦システム）**は、従来の推薦システムとは異なり、「**なぜそのスキルを学ぶべきか**」を科学的に説明できる推薦システムです。

このシステムは**三段階のアプローチ**を採用しています：

```
第一段階: LiNGAM
    ↓ （因果グラフを生成）
第二段階: Bayesian Network
    ↓ （確率推論を実行）
第三段階: 3軸スコアリング
    ↓ （最終的な推薦スコアを計算）
推薦結果
```

なぜ三段階必要なのか？それは、**各段階が異なる役割を持ち、それぞれが補完し合うことで、精度が高く説明可能な推薦を実現する**ためです。

---

## なぜ三段階なのか？

### 従来の推薦システムの問題点

従来の推薦システム（協調フィルタリングなど）には以下の問題があります：

| 問題点 | 説明 | 例 |
|--------|------|-----|
| **ブラックボックス** | なぜ推薦されたのか説明できない | 「似た人が学んでいるから」だけでは納得できない |
| **因果関係の無視** | 相関しか見ていない | 「AとBを同時に学ぶ人が多い」≠「AがBの習得に役立つ」 |
| **個別最適化が弱い** | 今のあなたの状況を考慮しない | 前提知識なしに難しいスキルを推薦してしまう |
| **将来性を考慮しない** | 次の次まで見据えていない | 今すぐ役立つが、キャリアに繋がらないスキルを推薦 |

### 三段階アプローチの解決策

これらの問題を解決するために、三段階のアプローチが必要です：

| 段階 | 役割 | 解決する問題 |
|------|------|-------------|
| **第一段階: LiNGAM** | 因果関係の発見 | 「相関」ではなく「原因→結果」を特定 |
| **第二段階: Bayesian Network** | 習得確率の推論 | 「あなたなら」習得できる確率を計算 |
| **第三段階: 3軸スコアリング** | 総合的な評価 | 準備度・確率・将来性を統合評価 |

**これら三段階は独立して機能するのではなく、段階的にデータを洗練させていくパイプラインです。**

---

## 第一段階: LiNGAM（因果構造学習）

### 🎯 目的

**スキル間の「原因→結果」の関係を発見する**

例：
- 「Python基礎」を学ぶと → 「機械学習」が習得しやすくなる
- 「統計学」を学ぶと → 「データ分析」が習得しやすくなる

これは**因果関係**であり、単なる相関ではありません。

### 🔬 技術：LiNGAM (Linear Non-Gaussian Acyclic Model)

#### LiNGAMとは？

LiNGAMは、**観測データのみから因果の向き（A→B なのか B→A なのか）を特定できる**画期的な手法です。

#### なぜ因果の向きが分かるのか？

通常、相関からは因果の向きは分かりません。しかし、LiNGAMは以下の仮定を使います：

1. **線形性**: 因果関係は線形（例: Y = a×X + ノイズ）
2. **非ガウス性**: ノイズが正規分布ではない
3. **非循環性**: 循環（A→B→A）はない

この仮定の下では、**データの統計的性質（独立成分分析）から因果の向きが一意に決まる**のです。

#### 素人向けの説明

**例え話：ドミノ倒しを観察する**

- ドミノA、B、Cが倒れた順番の記録が100回分ある
- 記録を分析すると、「Aが倒れた後に必ずBが倒れる」ことが分かる
- これは相関だが、LiNGAMはさらに「Aが原因でBが倒れる」（A→B）という**向き**まで特定できる

#### 実装の工夫：クラスタリング

スキルが300個以上ある場合、全スキルを一度に計算すると**計算コストが膨大**になります。

そこで、**相関の高いスキル同士をグループ化**（コミュニティ検出）し、グループごとに並列でLiNGAMを実行します。

```python
# 実装例（簡略版）
# 1. 相関行列を計算
corr_matrix = data.corr().abs()

# 2. 相関が高いスキル同士をグラフで接続
G = nx.Graph()
for i, j in combinations(skills, 2):
    if corr_matrix[i, j] > threshold:
        G.add_edge(i, j)

# 3. コミュニティ検出（Louvain法）
clusters = community_detection(G)

# 4. 各クラスタでLiNGAMを実行
for cluster in clusters:
    model = lingam.DirectLiNGAM()
    model.fit(data[cluster])
```

### 📊 出力結果

**因果グラフ（Causal Graph）**

- ノード：スキル
- エッジ：因果関係（矢印の向き = 因果の向き）
- エッジの重み：因果効果の強さ

例：
```
Python基礎 --[0.45]--> 機械学習
統計学     --[0.38]--> データ分析
機械学習   --[0.52]--> ディープラーニング
```

### ❓ なぜこの段階が必要か？

**因果グラフがないと、次の段階（Bayesian NetworkとScoring）ができない**からです。

- 因果グラフは、Bayesian Networkの構造（どのノード間にエッジを張るか）を決定します
- 因果グラフは、Readiness Score（準備度）とUtility Score（有用性）の計算に直接使われます

---

## 第二段階: Bayesian Network（ベイジアンネットワーク）

### 🎯 目的

**「あなたが」そのスキルを習得できる確率を計算する**

第一段階で因果グラフは分かりましたが、それだけでは**確率的な推論**ができません。

例：
- 「Python基礎 → 機械学習」という因果関係は分かった
- しかし、「Python基礎を持っている人が機械学習を習得する確率は何％か？」は分からない

この**条件付き確率**を計算するのが、Bayesian Networkの役割です。

### 🔬 技術：Bayesian Network

#### Bayesian Networkとは？

Bayesian Networkは、**確率変数間の依存関係をグラフで表現し、条件付き確率を効率的に計算する**手法です。

#### 構造

Bayesian Networkは以下の2つから構成されます：

1. **グラフ構造**: どの変数がどの変数に影響を与えるか（第一段階のLiNGAMから取得）
2. **条件付き確率表（CPT）**: 各変数の確率分布

例：
```
Python基礎 → 機械学習

CPT (条件付き確率表):
P(機械学習=1 | Python基礎=1) = 0.75
P(機械学習=1 | Python基礎=0) = 0.10
```

#### 推論方法：MCMC（マルコフ連鎖モンテカルロ法）

複雑なBayesian Networkでは、確率を厳密に計算するのが困難です。

そこで、**MCMC**という近似手法を使います：

1. ランダムに状態をサンプリング
2. サンプルを繰り返すことで、真の確率分布に収束
3. 十分なサンプル数（例: 10,000回）で精度の高い推定が可能

#### 素人向けの説明

**例え話：占い師のカード**

- 占い師が「あなたのスキルカード」を見る（保有スキル）
- 「このカードの組み合わせを持つ人は、75%の確率で機械学習を習得している」と教えてくれる
- これが条件付き確率の推論

### 📊 出力結果

**各スキルの習得確率**

メンバーm48の場合：
```
機械学習: P(習得可能) = 0.72
ディープラーニング: P(習得可能) = 0.45
データ分析: P(習得可能) = 0.88
```

### ❓ なぜこの段階が必要か？

**確率的な推論がないと、推薦の信頼性が分からない**からです。

- 因果グラフだけでは、「習得しやすい」ことは分かっても「どのくらいの確率で習得できるか」は分からない
- Bayesian Scoreは、第三段階の3軸スコアリングの1つとして使われます

---

## 第三段階: 3軸スコアリング

### 🎯 目的

**第一段階と第二段階の結果を統合し、最終的な推薦スコアを計算する**

### 🔬 技術：3軸スコアリング

第三段階では、以下の**3つの観点**からスキルを評価します：

#### 1️⃣ Readiness Score（準備完了度）

**「今すぐ学べる準備ができているか？」**

計算式：
```
Readiness(B) = Σ 因果効果(A → B)
               A ∈ 保有スキル
```

- 第一段階で学習した因果グラフを使用
- 保有スキルから対象スキルへの因果効果を合計

例：
```
保有スキル: Python基礎、統計学
対象スキル: 機械学習

Readiness(機械学習) = 因果効果(Python基礎 → 機械学習) + 因果効果(統計学 → 機械学習)
                     = 0.45 + 0.38
                     = 0.83
```

**意味**：
- Readinessが高い = 前提スキルが揃っている = 今すぐ学べる
- Readinessが低い = 前提スキルが不足 = まだ早い

#### 2️⃣ Bayesian Score（習得確率）

**「あなたなら習得できる確率は？」**

計算式：
```
Bayesian(B) = P(B=1 | 保有スキル)
```

- 第二段階で学習したBayesian Networkを使用
- 保有スキルを条件として、対象スキルの習得確率を計算

例：
```
保有スキル: Python基礎、統計学
対象スキル: 機械学習

Bayesian(機械学習) = P(機械学習=1 | Python基礎=1, 統計学=1)
                   = 0.75
```

**意味**：
- Bayesian Scoreが高い = 統計的に習得しやすい
- Bayesian Scoreが低い = 習得が困難かもしれない

#### 3️⃣ Utility Score（将来性）

**「このスキルを学ぶと、将来何の役に立つか？」**

計算式：
```
Utility(B) = Σ 因果効果(B → C)
             C ∈ 未習得スキル
```

- 第一段階で学習した因果グラフを使用
- 対象スキルから未習得スキルへの因果効果を合計

例：
```
対象スキル: 機械学習
未習得スキル: ディープラーニング、強化学習、自然言語処理

Utility(機械学習) = 因果効果(機械学習 → ディープラーニング)
                  + 因果効果(機械学習 → 強化学習)
                  + 因果効果(機械学習 → 自然言語処理)
                  = 0.52 + 0.41 + 0.38
                  = 1.31
```

**意味**：
- Utilityが高い = 将来のキャリアに役立つ「土台」スキル
- Utilityが低い = 末端のスキル（他に繋がらない）

### 📊 スコアの統合

3つのスコアを重み付けして統合します：

```
総合スコア = Readiness × w₁ + Bayesian × w₂ + Utility × w₃

デフォルト重み:
- w₁ (Readiness) = 0.6 (60%)
- w₂ (Bayesian)  = 0.3 (30%)
- w₃ (Utility)   = 0.1 (10%)
```

#### なぜこの重み？

| スコア | 重み | 理由 |
|--------|------|------|
| Readiness | 60% | **最重要**。前提スキルがないと学習効率が悪い |
| Bayesian | 30% | **重要**。統計的な実現可能性を保証 |
| Utility | 10% | **補助的**。将来性も考慮するが、今学べることを優先 |

この重みは**ベイズ最適化**で自動調整することもできます。

### 📊 正規化

各スコアは範囲が異なるため、**0〜1に正規化**してから統合します：

```python
# 全候補スキルの中での相対的な位置
normalized_readiness = readiness / max(all_readiness_scores)
normalized_bayesian = bayesian  # 既に0〜1
normalized_utility = utility / max(all_utility_scores)

# 統合
total_score = (
    normalized_readiness * 0.6 +
    normalized_bayesian * 0.3 +
    normalized_utility * 0.1
)
```

### ❓ なぜこの段階が必要か？

**第一段階と第二段階だけでは、総合的な判断ができない**からです。

| ケース | Readiness | Bayesian | Utility | 判断 |
|--------|-----------|----------|---------|------|
| ケースA | 高 | 高 | 低 | 今すぐ学べるが、将来性は低い → **学ぶべき**（短期目標） |
| ケースB | 低 | 高 | 高 | 前提不足だが、重要 → **まだ早い**（後で学ぶ） |
| ケースC | 高 | 低 | 高 | 準備はOKだが、習得困難 → **慎重に検討** |
| ケースD | 高 | 高 | 高 | 全て高い → **最優先で推薦！** |

このように、3つの軸を総合的に見ることで、**状況に応じた適切な推薦**ができます。

---

## 三段階を統合する意義

### 🔗 各段階の依存関係

三段階は**パイプライン**として機能します：

```
┌─────────────────────────────────────────────────────────┐
│ 第一段階: LiNGAM                                          │
│ - 入力: メンバー×スキルの習得データ                        │
│ - 出力: 因果グラフ（隣接行列）                             │
└─────────────────┬───────────────────────────────────────┘
                  │ 因果グラフを渡す
                  ↓
┌─────────────────────────────────────────────────────────┐
│ 第二段階: Bayesian Network                               │
│ - 入力: 因果グラフ + 習得データ                           │
│ - 出力: 条件付き確率表（CPT）                             │
└─────────────────┬───────────────────────────────────────┘
                  │ 因果グラフ + CPTを渡す
                  ↓
┌─────────────────────────────────────────────────────────┐
│ 第三段階: 3軸スコアリング                                 │
│ - 入力: 因果グラフ + CPT + メンバーの保有スキル           │
│ - 出力: 各スキルの総合スコア                              │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ↓
            【推薦結果】
```

### 🎯 各段階の相乗効果

| 段階の組み合わせ | 効果 |
|------------------|------|
| **LiNGAM + Bayesian** | 因果関係に基づく確率推論 → 説明可能性UP |
| **LiNGAM + Scoring** | Readiness・Utilityの計算が可能 → 個別最適化UP |
| **Bayesian + Scoring** | 統計的信頼性を考慮 → 推薦精度UP |
| **三段階統合** | 説明可能 + 個別最適 + 高精度 → 最強の推薦システム |

### 📈 三段階vs一段階の比較

もし一段階だけで推薦システムを作るとどうなるか？

#### パターンA: LiNGAMのみ

```
✅ 因果関係は分かる
❌ 確率が分からない（習得可能性を保証できない）
❌ 総合評価ができない（どのスキルを優先すべきか不明）
```

#### パターンB: Bayesian Networkのみ

```
✅ 習得確率は分かる
❌ 因果の向きが不明（なぜ習得しやすいのか説明できない）
❌ 将来性を考慮できない（Utilityが計算できない）
```

#### パターンC: スコアリングのみ

```
✅ スコアの統合はできる
❌ 因果グラフがないので、ReadinessとUtilityが計算できない
❌ Bayesian Networkがないので、Bayesian Scoreが計算できない
```

**結論：三段階全てが揃って初めて、完全な推薦システムが完成します。**

---

## 実装の詳細

### 📁 ファイル構成

```
skillnote_recommendation/
├── ml/
│   ├── causal_structure_learner.py     # 第一段階: LiNGAM
│   ├── bayesian_network_recommender.py # 第二段階: Bayesian Network
│   └── causal_graph_recommender.py     # 第三段階: 3軸スコアリング
└── graph/
    ├── causal_career_path.py           # 推薦結果の可視化
    └── causal_graph_visualizer.py      # 因果グラフの可視化
```

### 🔧 使用ライブラリ

| 段階 | ライブラリ | 用途 |
|------|-----------|------|
| 第一段階 | `lingam` | LiNGAMアルゴリズムの実装 |
| 第一段階 | `networkx` | グラフ構造の操作 |
| 第一段階 | `python-louvain` | コミュニティ検出 |
| 第二段階 | `pgmpy` | Bayesian Networkの実装 |
| 第三段階 | `numpy`, `pandas` | 数値計算 |

### 💻 実装例

#### 完全な推薦フロー

```python
from skillnote_recommendation.ml.causal_graph_recommender import CausalGraphRecommender

# 1. 推薦システムを初期化
recommender = CausalGraphRecommender(
    member_competence=member_competence_df,
    competence_master=competence_master_df,
    learner_params={
        "correlation_threshold": 0.2,  # クラスタリング閾値
        "min_cluster_size": 3
    },
    weights={
        'readiness': 0.6,
        'bayesian': 0.3,
        'utility': 0.1
    }
)

# 2. 三段階の学習を実行
recommender.fit(min_members_per_skill=5)

# 内部では以下が実行される：
# - 第一段階: self.learner.fit() → 因果グラフを学習
# - 第二段階: self.bn_recommender.fit() → Bayesian Networkを学習
# - 第三段階の準備完了（推薦時に使用）

# 3. メンバーへの推薦
recommendations = recommender.recommend(member_code='m48', top_n=10)

# 4. 結果の表示
for rec in recommendations:
    print(f"【推薦】 {rec['competence_name']}")
    print(f"  総合スコア: {rec['score']:.3f}")
    print(f"  説明: {rec['explanation']}")
    print(f"  詳細:")
    print(f"    - Readiness: {rec['details']['readiness_score_normalized']:.3f}")
    print(f"    - Bayesian: {rec['details']['bayesian_score_normalized']:.3f}")
    print(f"    - Utility: {rec['details']['utility_score_normalized']:.3f}")
    print()
```

#### 各段階の詳細

**第一段階: LiNGAM**

```python
# causal_structure_learner.py
class CausalStructureLearner:
    def fit(self, skill_matrix: pd.DataFrame):
        # 1. コミュニティ検出でクラスタリング
        self.clusters_ = self._detect_communities(skill_matrix)

        # 2. 各クラスタでLiNGAMを実行
        for cluster_features in self.clusters_:
            cluster_data = skill_matrix[cluster_features]
            model = lingam.DirectLiNGAM(random_state=self.random_state)
            model.fit(cluster_data)

            # 因果グラフの隣接行列を保存
            self.adjacency_matrix_.loc[cluster_features, cluster_features] = model.adjacency_matrix_.T
```

**第二段階: Bayesian Network**

```python
# bayesian_network_recommender.py
class BayesianNetworkRecommender:
    def fit(self, data: pd.DataFrame):
        # 1. LiNGAMの結果からBayesian Networkを構築
        model = BayesianNetwork(self.causal_graph_edges)

        # 2. 条件付き確率表を学習
        model.fit(data, estimator=MaximumLikelihoodEstimator)

        # 3. MCMCサンプリングの準備
        self.inference = BayesianModelSampling(model)
```

**第三段階: 3軸スコアリング**

```python
# causal_graph_recommender.py
class CausalGraphRecommender:
    def recommend(self, member_code: str, top_n: int):
        # 保有スキルと未習得スキルを取得
        owned_skills = self.get_owned_skills(member_code)
        unowned_skills = self.get_unowned_skills(member_code)

        scores = []
        for target_skill in unowned_skills:
            # 1. Readiness Score
            readiness = sum(
                self.get_effect(owned, target_skill)
                for owned in owned_skills
            )

            # 2. Bayesian Score
            bayesian = self.bn_recommender.predict_probability(
                owned_skills, target_skill
            )

            # 3. Utility Score
            utility = sum(
                self.get_effect(target_skill, future)
                for future in unowned_skills
            )

            # 正規化して統合
            total_score = (
                normalize(readiness) * self.weights['readiness'] +
                normalize(bayesian) * self.weights['bayesian'] +
                normalize(utility) * self.weights['utility']
            )

            scores.append({
                'skill': target_skill,
                'total_score': total_score,
                'readiness': readiness,
                'bayesian': bayesian,
                'utility': utility
            })

        # スコア順にソートして上位N件を返す
        return sorted(scores, key=lambda x: x['total_score'], reverse=True)[:top_n]
```

---

## まとめ

### 三段階の必要性

| 段階 | 役割 | 必要な理由 |
|------|------|-----------|
| **第一段階: LiNGAM** | 因果関係の発見 | 相関ではなく因果を特定。次の段階の基盤となる |
| **第二段階: Bayesian Network** | 習得確率の推論 | 統計的な実現可能性を保証。信頼性を向上 |
| **第三段階: 3軸スコアリング** | 総合評価 | 準備度・確率・将来性を統合。最終的な推薦を決定 |

### 各段階の相互依存

```
因果グラフ（LiNGAM）がないと...
  → Bayesian Networkの構造が決まらない
  → Readiness/Utilityが計算できない

Bayesian Networkがないと...
  → 習得確率が分からない
  → 推薦の信頼性が低い

3軸スコアリングがないと...
  → 総合的な判断ができない
  → どのスキルを優先すべきか不明
```

### 三段階の効果

従来の推薦システムと比較して：

| 項目 | 従来の推薦 | Causal Recommendation（三段階） |
|------|-----------|-------------------------------|
| **説明可能性** | ❌ ブラックボックス | ✅ 因果グラフで可視化 |
| **個別最適化** | ⚠️ 弱い | ✅ Readinessで前提スキルを考慮 |
| **統計的信頼性** | ⚠️ 低い | ✅ Bayesian Networkで確率推論 |
| **将来性考慮** | ❌ なし | ✅ Utilityでキャリアパスを考慮 |
| **推薦精度** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

### 最後に

**Causal Recommendationの三段階アプローチは、単なる技術的な選択ではなく、「科学的で説明可能な推薦システム」を実現するための必然的な設計**です。

各段階が相互に補完し合い、初めて完全な推薦システムが完成します。一段階でも欠けると、システムの価値は大きく損なわれます。

この三段階のアプローチにより、CareerNavigatorは**「なぜそのスキルを学ぶべきか」を明確に説明できる、唯一無二の推薦システム**となっています。

---

## 参考文献

- Shimizu, S., et al. (2006). "A Linear Non-Gaussian Acyclic Model for Causal Discovery". Journal of Machine Learning Research.
- Pearl, J. (2009). "Causality: Models, Reasoning, and Inference". Cambridge University Press.
- Koller, D., & Friedman, N. (2009). "Probabilistic Graphical Models: Principles and Techniques". MIT Press.

---

**作成日**: 2025-11-24
**対象バージョン**: CareerNavigator v1.3.0+
