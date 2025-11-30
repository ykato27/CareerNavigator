# 因果推論ベース推薦 - 技術詳細

因果推論ベースの推薦システムは、CareerNavigatorの中核機能であり、スキル間の因果関係を明示的にモデル化して最適な学習経路を提案します。

## 概要

従来の協調フィルタリングでは「どのスキルが一緒に習得されやすいか」という**相関関係**を学習しますが、因果推論では「どのスキルを習得すると他のスキルの習得が容易になるか」という**因果関係**を学習します。

### 従来手法との比較

| 観点 | 協調フィルタリング | 因果推論ベース推薦 |
|------|-------------------|-------------------|
| 関係性 | 相関 | 因果 |
| 説明可能性 | 低い | 高い |
| 学習順序 | 考慮しない | 明示的に考慮 |
| 依存関係 | 暗黙的 | 明示的 |
| 計算コスト | 低い | 高い |

---

## 因果グラフ構築

### LiNGAMアルゴリズム

**Linear Non-Gaussian Acyclic Model (LiNGAM)** を使用して、スキル間の因果構造を学習します。

#### アルゴリズムの前提

1. **線形性**: スキル間の関係は線形
2. **非ガウス性**: 外生変数は非ガウス分布
3. **非巡回性**: 因果グラフに循環はない

#### DirectLiNGAMの処理フロー

```python
from lingam import DirectLiNGAM

# ステップ1: データ準備
# スキル頻度行列: 各スキルの保有率を行列化
skill_frequency_matrix = compute_skill_frequency_matrix(
    member_competence, competence_master
)

# ステップ2: 因果構造学習
model = DirectLiNGAM(random_state=42)
model.fit(skill_frequency_matrix)

# ステップ3: 因果効果行列の取得
# causal_effects[i][j]: スキルiがスキルjに与える因果効果
causal_effects = model.adjacency_matrix_

# ステップ4: 因果順序の取得
# 学習すべき順序（トポロジカル順序）
causal_order = model.causal_order_
```

#### 因果効果行列の解釈

```python
# 例: causal_effects[skill_A][skill_B] = 0.25
# 解釈: スキルAを習得すると、スキルBの習得が25%促進される

# 閾値フィルタリング
min_effect = 0.03
filtered_edges = [
    (i, j, causal_effects[i][j])
    for i in range(n_skills)
    for j in range(n_skills)
    if causal_effects[i][j] > min_effect
]
```

### 実装例

```python
from skillnote_recommendation.graph import CausalGraphRecommender

# 初期化
recommender = CausalGraphRecommender(
    member_competence=member_competence_df,
    competence_master=competence_master_df
)

# 因果グラフ学習
recommender.fit()

# 因果効果行列の取得
causal_effects = recommender.causal_effects_matrix

# スキルA -> スキルBの因果効果
effect = causal_effects[skill_A_index][skill_B_index]
print(f"Skill A → Skill B: {effect:.3f}")
```

---

## 3軸スコアリングシステム

因果グラフを基に、3つの観点からスキルを評価します。

### 1. 準備完了度 (Readiness Score)

**定義**: 現在保有しているスキルから対象スキルへの因果効果の総和。

**計算式**:
```python
readiness_score[j] = sum(
    causal_effects[i][j]
    for i in owned_skills
    if causal_effects[i][j] > min_effect_threshold
)
```

**意味**:
- 高い → 既に保有スキルからの前提条件が揃っている
- 低い → まだ学習の準備ができていない

**正規化**:
```python
# 0-1の範囲にスケーリング
readiness_normalized = (
    readiness_score - readiness_min
) / (readiness_max - readiness_min + 1e-10)
```

#### 準備完了度の詳細理由

推薦結果には、どのスキルがどれだけ貢献しているかを表示：

```python
readiness_reasons = [
    (prerequisite_skill_name, causal_effect)
    for prerequisite_skill, causal_effect in sorted_prerequisites
]

# 例:
# [('Python基礎', 0.15), ('データ構造', 0.12), ('アルゴリズム', 0.08)]
```

### 2. 確率スコア (Probability Score)

**定義**: ベイジアンネットワークによる習得確率。

**処理フロー**:

```python
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator

# ステップ1: 因果グラフからベイジアンネットワークを構築
edges = extract_edges_from_causal_graph(causal_effects)
bn = BayesianNetwork(edges)

# ステップ2: 条件付き確率の学習
bn.fit(
    data=member_competence_binary,
    estimator=MaximumLikelihoodEstimator
)

# ステップ3: 習得確率の推論
from pgmpy.inference import VariableElimination
inference = VariableElimination(bn)

probability_score[j] = inference.query(
    variables=[skill_j],
    evidence={skill_i: 1 for skill_i in owned_skills}
).values[1]  # P(skill_j = 1 | owned_skills)
```

**意味**:
- 高い → 現在の保有スキルから見て習得しやすい
- 低い → 習得が難しい

### 3. 有用性スコア (Utility Score)

**定義**: 対象スキルから目標スキルへの因果効果の総和。

**計算式**:
```python
utility_score[j] = sum(
    causal_effects[j][k]
    for k in target_skills
    if causal_effects[j][k] > min_effect_threshold
)
```

**意味**:
- 高い → 将来の目標達成に大きく貢献する
- 低い → 目標達成への貢献度が低い

#### 有用性の詳細理由

どのターゲットスキルへの貢献が大きいかを表示：

```python
utility_reasons = [
    (target_skill_name, causal_effect)
    for target_skill, causal_effect in sorted_targets
]

# 例:
# [('機械学習', 0.20), ('深層学習', 0.18), ('データ分析', 0.10)]
```

### 総合スコア

3つのスコアを統合した最終スコア。

**計算式**:
```python
total_score = (
    readiness_score ** 0.3 *
    probability_score ** 0.3 *
    utility_score ** 0.4
)
```

**重みの意味**:
- 準備完了度: 0.3（学習の準備状況）
- 確率スコア: 0.3（習得可能性）
- 有用性: 0.4（目標達成への貢献度を最重視）

**調整可能**:
```python
# カスタム重み設定
custom_weights = {
    'readiness': 0.4,   # 準備完了度を重視
    'probability': 0.3,
    'utility': 0.3
}

total_score = (
    readiness_score ** custom_weights['readiness'] *
    probability_score ** custom_weights['probability'] *
    utility_score ** custom_weights['utility']
)
```

---

## Causalフィルタリング

ギャップ分析で抽出された不足スキルを、因果推論に基づいてフィルタリングします。

### 処理フロー

```python
def generate_filtered_path(gap_analysis, member_code, min_total_score=0.02):
    # ステップ1: 現在の保有スキル取得
    owned_skills = get_owned_skills(member_code)
    
    # ステップ2: 不足スキル取得
    missing_skills = gap_analysis['missing_competences']
    
    # ステップ3: 各不足スキルのスコア計算
    scored_skills = []
    for skill in missing_skills:
        # 3軸スコア計算
        readiness = compute_readiness(skill, owned_skills)
        probability = compute_probability(skill, owned_skills)
        utility = compute_utility(skill, target_skills)
        
        total = (readiness ** 0.3) * (probability ** 0.3) * (utility ** 0.4)
        
        scored_skills.append({
            'skill': skill,
            'total_score': total,
            'readiness_score': readiness,
            'probability_score': probability,
            'utility_score': utility,
            'readiness_reasons': get_readiness_reasons(skill, owned_skills),
            'utility_reasons': get_utility_reasons(skill, target_skills)
        })
    
    # ステップ4: 閾値フィルタリング
    filtered_skills = [
        s for s in scored_skills
        if s['total_score'] >= min_total_score
    ]
    
    # ステップ5: スコア順にソート
    recommended_skills = sorted(
        filtered_skills,
        key=lambda x: x['total_score'],
        reverse=True
    )
    
    return recommended_skills
```

### 統合例（Streamlitアプリ）

```python
# pages/2_Employee_Career_Dashboard.py

# ==== ギャップ分析 ====
gap_result = analyze_career_gap(
    source_member=selected_member,
    target_member=target_member  # またはtarget_role
)

# ==== Causalフィルタリング ====
recommended_skills = causal_path_generator.generate_filtered_path(
    gap_analysis=gap_result,
    member_code=selected_member,
    min_total_score=min_total_score,
    min_readiness_score=min_readiness
)

# ==== 推薦結果表示 ====
for skill in recommended_skills:
    st.write(f"**{skill['competence_name']}**")
    st.write(f"総合スコア: {skill['total_score']:.3f}")
    st.write(f"準備完了度: {skill['readiness_score']:.3f}")
    st.write(f"確率: {skill['probability_score']:.3f}")
    st.write(f"有用性: {skill['utility_score']:.3f}")
```

---

## 依存関係の抽出

推薦されたスキル間の依存関係を抽出し、学習の順序を決定します。

### アルゴリズム

```python
def extract_dependencies(recommended_skills, competence_master, min_effect=0.03):
    dependencies = {}
    
    for skill in recommended_skills:
        skill_code = skill['competence_code']
        prerequisites = []
        
        # 推薦スキル内での前提スキルを探す
        for other_skill in recommended_skills:
            if other_skill['competence_code'] == skill_code:
                continue
            
            effect = causal_effects[other_skill_index][skill_index]
            if effect > min_effect:
                prerequisites.append({
                    'skill': other_skill['competence_code'],
                    'effect': effect
                })
        
        dependencies[skill_code] = sorted(
            prerequisites,
            key=lambda x: x['effect'],
            reverse=True
        )
    
    return dependencies
```

### トポロジカルソート

依存関係を考慮した学習順序を決定。

```python
from collections import defaultdict, deque

def topological_sort(skills, dependencies):
    # ステップ1: 入次数の計算
    indegree = defaultdict(int)
    graph = defaultdict(list)
    
    for skill in skills:
        skill_code = skill['competence_code']
        for dep in dependencies.get(skill_code, []):
            graph[dep['skill']].append(skill_code)
            indegree[skill_code] += 1
    
    # ステップ2: トポロジカルソート（Kahn's Algorithm）
    queue = deque([s['competence_code'] for s in skills if indegree[s['competence_code']] == 0])
    sorted_order = []
    
    while queue:
        current = queue.popleft()
        sorted_order.append(current)
        
        for neighbor in graph[current]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)
    
    return sorted_order
```

---

## スマートロードマップ生成

依存関係を考慮したガントチャート形式の学習計画を生成。

### レイヤー分類

依存関係に基づいてスキルをレイヤー（階層）に分類：

```python
def compute_layers(skills, dependencies):
    layers = {}
    
    # レイヤー0: 前提条件なし（入次数0）
    layer_0 = [s for s in skills if not dependencies.get(s['competence_code'], [])]
    for skill in layer_0:
        layers[skill['competence_code']] = 0
    
    # それ以降のレイヤー: 前提スキルの最大レイヤー + 1
    remaining = [s for s in skills if s not in layer_0]
    while remaining:
        for skill in remaining[:]:
            prereqs = dependencies.get(skill['competence_code'], [])
            prereq_layers = [layers.get(p['skill']) for p in prereqs]
            
            if all(l is not None for l in prereq_layers):
                layers[skill['competence_code']] = max(prereq_layers) + 1
                remaining.remove(skill)
    
    return layers
```

### ガントチャート生成

```python
import plotly.figure_factory as ff

def create_gantt_chart(skills, layers, avg_learning_time=30):
    tasks = []
    current_date = datetime.now()
    
    for skill in skills:
        skill_code = skill['competence_code']
        layer = layers[skill_code]
        
        start_date = current_date + timedelta(days=layer * avg_learning_time)
        finish_date = start_date + timedelta(days=avg_learning_time)
        
        tasks.append({
            'Task': skill['competence_name'],
            'Start': start_date,
            'Finish': finish_date,
            'Resource': f"Layer {layer}"
        })
    
    # Plotlyガントチャート
    fig = ff.create_gantt(
        tasks,
        index_col='Resource',
        show_colorbar=True,
        group_tasks=True
    )
    
    return fig
```

### Streamlit表示

```python
# ロードマップ可視化
gantt_fig = create_gantt_chart(
    recommended_skills,
    layers,
    avg_learning_time=30
)

st.plotly_chart(gantt_fig, use_container_width=True)
```

---

## インタラクティブ因果グラフ可視化

### Pyvisによる可視化

```python
from pyvis.network import Network

def create_causal_graph_visualization(
    recommended_skills,
    causal_effects,
    competence_master,
    min_effect=0.05
):
    # ネットワーク初期化
    net = Network(
        height='600px',
        width='100%',
        bgcolor='#222222',
        font_color='white',
        directed=True
    )
    
    # ノード追加
    for skill in recommended_skills:
        skill_code = skill['competence_code']
        skill_name = skill['competence_name']
        total_score = skill['total_score']
        
        # ノードサイズをスコアに応じて調整
        size = 10 + total_score * 50
        
        net.add_node(
            skill_code,
            label=skill_name,
            title=f"{skill_name}\\nスコア: {total_score:.3f}",
            size=size,
            color='#FF6B6B'
        )
    
    # エッジ追加
    for i, skill_i in enumerate(recommended_skills):
        for j, skill_j in enumerate(recommended_skills):
            if i == j:
                continue
            
            effect = causal_effects[i][j]
            if effect > min_effect:
                net.add_edge(
                    skill_i['competence_code'],
                    skill_j['competence_code'],
                    value=effect,
                    title=f"因果効果: {effect:.3f}",
                    arrows='to'
                )
    
    # 物理エンジン設定
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -8000,
          "centralGravity": 0.3,
          "springLength": 95
        }
      }
    }
    """)
    
    # HTML生成
    net.save_graph('causal_graph.html')
    
    return 'causal_graph.html'
```

### Streamlit統合

```python
# 因果グラフ可視化
graph_html = create_causal_graph_visualization(
    recommended_skills,
    causal_effects,
    competence_master,
    min_effect=graph_display_threshold
)

# HTMLコンポーネントとして表示
with open(graph_html, 'r', encoding='utf-8') as f:
    html_content = f.read()

st.components.v1.html(html_content, height=650)
```

---

## パラメータチューニング

### 重要なパラメータ

| パラメータ | デフォルト | 範囲 | 説明 |
|-----------|----------|------|------|
| `min_total_score` | 0.02 | 0.0 ~ 1.0 | 総合スコアの最小閾値 |
| `min_readiness_score` | 0.0 | 0.0 ~ 1.0 | 準備完了度の最小閾値 |
| `min_effect_threshold` | 0.03 | 0.0 ~ 0.5 | 因果効果の最小閾値 |
| `graph_display_threshold` | 0.05 | 0.01 ~ 1.0 | グラフ表示用閾値 |

### チューニングガイド

**厳しい推薦（高品質）**:
```python
params = {
    'min_total_score': 0.05,
    'min_readiness_score': 0.1,
    'min_effect_threshold': 0.05
}
```

**緩い推薦（多様性重視）**:
```python
params = {
    'min_total_score': 0.01,
    'min_readiness_score': 0.0,
    'min_effect_threshold': 0.02
}
```

**バランス型（推奨）**:
```python
params = {
    'min_total_score': 0.02,
    'min_readiness_score': 0.0,
    'min_effect_threshold': 0.03
}
```

---

## メリット・デメリット

### メリット

1. **高い説明可能性**: 因果関係が明示的
2. **学習順序の考慮**: 依存関係を考慮した推薦
3. **理論的裏付け**: 因果推論の理論に基づく
4. **直感的な理解**: 「Aを学ぶとBが習得しやすい」という説明

### デメリット

1. **計算コストが高い**: 因果構造学習にコストがかかる
2. **仮定の制約**: 線形性・非ガウス性・非巡回性の仮定
3. **データ要求**: 十分なサンプルサイズが必要
4. **スケーラビリティ**: スキル数が多いと計算時間が増加

---

## 関連ドキュメント

- [Streamlitアプリガイド](STREAMLIT_APPS.md)
- [機械学習モデル参考資料](ML_MODELS_REFERENCE.md)
- [アーキテクチャドキュメント](ARCHITECTURE.md)
- [コード構造ガイド](CODE_STRUCTURE.md)

---

## 参考文献

1. Shimizu, S., et al. (2006). "A Linear Non-Gaussian Acyclic Model for Causal Discovery." Journal of Machine Learning Research, 7, 2003-2030.
2. Pearl, J. (2009). "Causality: Models, Reasoning, and Inference." Cambridge University Press.
3. Peters, J., et al. (2017). "Elements of Causal Inference: Foundations and Learning Algorithms." MIT Press.
