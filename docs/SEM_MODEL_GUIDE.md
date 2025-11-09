# スキル領域潜在変数SEMモデル 実装ガイド

## 概要

**SkillDomainSEMModel**は、構造方程式モデリング（SEM: Structural Equation Modeling）の理論に基づいたスキル推薦モデルです。

スキルを領域別に分類し、各領域内で「初級→中級→上級」の段階的な潜在変数を設定します。メンバーのスキル習得レベルから潜在変数を推定し、スキル間の構造的な因果関係を把握することで、より説明可能な推薦を実現します。

### SEM理論の構成

1. **測定モデル（Measurement Model）**: スキル → 潜在変数
   - 観測可能なスキルから潜在段階変数を推定
   - ファクターローディングを実データから計算

2. **構造モデル（Structural Model）**: 潜在変数 → 潜在変数
   - 初級 → 中級 → 上級への因果関係
   - パス係数を相関係数から統計的に推定

3. **統計的有意性検定**
   - t値とp値で因果効果の有意性を判定
   - 信頼区間でパス係数の精度を示す

## 主要クラス

### 1. SkillDomainSEMModel

**位置**: `skillnote_recommendation/ml/skill_domain_sem_model.py`

#### 初期化

```python
from skillnote_recommendation.ml.skill_domain_sem_model import SkillDomainSEMModel

model = SkillDomainSEMModel(
    member_competence_df=member_competence_df,  # メンバー習得力量データ
    competence_master_df=competence_master_df,  # 力量マスタ
    num_domain_categories=8,  # スキル領域の分類数（5～10推奨）
    confidence_level=0.95  # 信頼区間レベル（デフォルト95%）
)
```

#### 主要メソッド

##### `calculate_sem_score(member_code, skill_code)`

**説明**: メンバーがスキルを習得する確率を推定（0-1）

```python
sem_score = model.calculate_sem_score("M001", "C001")
# 返り値: 0.48（習得確率48%）
```

**内部処理**:
1. スキルが属する領域を検索
2. メンバーのその領域での現在レベル（-1/0/1/2）を推定
3. 次のレベルへのパス係数を取得
4. 現在のレベルスコア × パス係数で習得確率を計算

##### `get_member_domain_profile(member_code)`

**説明**: メンバーの領域別プロファイル（全潜在変数のスコア）を取得

```python
profile = model.get_member_domain_profile("M001")
# 返り値: {
#   'プログラミング': {
#     'プログラミング_初級': 0.85,
#     'プログラミング_中級': 0.72,
#     'プログラミング_上級': 0.20
#   },
#   'データベース': {
#     'データベース_初級': 0.30,
#     'データベース_中級': 0.10,
#     'データベース_上級': 0.00
#   }
# }
```

**用途**: メンバーの強み・弱みの可視化、キャリア開発相談

##### `get_all_domains()`

**説明**: すべてのスキル領域名を取得

```python
domains = model.get_all_domains()
# 返り値: ['プログラミング', 'データベース', 'インフラ', ...]
```

##### `get_domain_info(domain_name)`

**説明**: 領域の詳細情報（潜在変数、パス係数、統計量）を取得

```python
info = model.get_domain_info("プログラミング")
# 返り値: {
#   'domain_name': 'プログラミング',
#   'num_latent_factors': 3,
#   'latent_factors': [
#     {
#       'name': 'プログラミング_初級',
#       'level': 0,
#       'num_skills': 5,
#       'factor_loadings': {'C001': 0.78, 'C002': 0.65, ...}
#     },
#     ...
#   ],
#   'path_coefficients': [
#     {
#       'from': 'プログラミング_初級',
#       'to': 'プログラミング_中級',
#       'coefficient': 0.68,
#       'p_value': 0.0234,
#       't_value': 2.43,
#       'is_significant': True,
#       'ci': (0.15, 1.21)  # 95%信頼区間
#     },
#     ...
#   ]
# }
```

## 理論的背景

### 測定モデル（スキル → 潜在変数）

スキルをレベル帯別に分類し、各潜在変数に対応させます：

```
低平均習得レベル（≤2） → プログラミング_初級
中平均習得レベル（2-4） → プログラミング_中級
高平均習得レベル（>4）  → プログラミング_上級
```

**ファクターローディング計算**:
```
loading = std(skill_levels)  # スキル分散の平方根
loading = clip(loading, 0.3, 0.95)  # 0.3-0.95に正規化
```

### 構造モデル（潜在変数 → 潜在変数）

初級から中級、中級から上級への因果効果を推定：

**パス係数計算**:
```
coefficient = correlation(from_scores, to_scores)
t_value = coefficient × √(n-2) / √(1-coefficient²)
p_value = 2 × (1 - t.cdf(|t_value|, n-2))
```

**信頼区間**（Fisherのz変換）:
```
z = 0.5 × ln((1+r)/(1-r))
se_z = 1 / √(n-3)
CI = tanh(z ± z_critical × se_z)
```

## データ要件

### 必須カラム

**member_competence_df**:
- `メンバーコード`: メンバーID
- `力量コード`: スキルID
- `正規化レベル`: スキルレベル（0-5）

**competence_master_df**:
- `力量コード`: スキルID
- `力量名`: スキル名
- `力量カテゴリー名`: カテゴリー（階層形式推奨）
  - 例: `プログラミング > Python`

### データ品質の目安

- メンバー数: 50名以上推奨
- スキル数: 50以上推奨
- 領域当たりスキル数: 平均3～10個

## 設計パラメータ

### スキル領域の分類数（num_domain_categories）

- **推奨値**: 5～10
- **デフォルト**: 8
- **効果**:
  - 少ない（5）: より粗い分類、計算が高速
  - 多い（10）: より詳細な分類、計算がやや遅い

### 信頼区間レベル（confidence_level）

- **推奨値**: 0.95（95%）
- **デフォルト**: 0.95
- **効果**: パス係数の信頼区間の幅に影響

## 統計量の解釈

### パス係数（coefficient）

-1～1の範囲。潜在変数間の標準化された因果効果の大きさ。

- `0.0-0.3`: 弱い効果
- `0.3-0.7`: 中程度の効果
- `0.7-1.0`: 強い効果

### p値（p_value）

有意性検定の結果（両側検定）。

- `p < 0.05`: 統計的に有意（95%信頼度）
- `p ≥ 0.05`: 統計的に有意でない

### 信頼区間（CI）

パス係数の不確実性を示す区間。区間が0を含まなければ有意。

### t値（t_value）

統計検定用の統計量。絶対値が大きいほど有意性が高い。

## トラブルシューティング

### 問題: SEMスコアが常に0または1

**原因**: スキルが領域に含まれていない、またはメンバーがそのスキルを習得していない

**解決**:
1. `get_all_domains()`で領域を確認
2. `_find_skill_domain(skill_code)`でスキルの領域を確認
3. member_competence_dfにスキルが含まれているか確認

### 問題: すべてのパス係数が有意でない

**原因**: メンバー数が少ないか、潜在変数間の因果関係が弱い

**解決**:
1. メンバー数を確認（50名以上推奨）
2. `get_domain_info(domain_name)`でパス係数を確認
3. 領域の構造が妥当か検討

### 問題: パフォーマンスが低い

**原因**: メンバー/スキルが多すぎるか、計算が複雑

**解決**:
1. num_domain_categoriesを5に減らす
2. メンバー数を絞る
3. ログレベルをWARNING以上に設定

## 参考資料

- **理論背景**: 構造方程式モデリング（SEM）
- **参考文献**:
  - Kline, R. B. (2015). Principles and Practice of Structural Equation Modeling
  - Bollen, K. A. (1989). Structural Equations with Latent Variables
- **既存統合先**: MLSEMRecommender（skillnote_recommendation/ml/ml_sem_recommender.py）
