# スキル領域潜在変数SEMモデル 実装ガイド

## 概要

**SkillDomainSEMModel**は、スキルを領域別に分類し、各領域内で「初級→中級→上級」の段階的な潜在変数を設定するモデルです。メンバーのスキル習得レベルから潜在変数を推定し、スキル間の構造的な依存関係を把握して、より説明可能な推薦を実現します。

## 主要クラス

### 1. SkillDomainSEMModel
**位置**: `skillnote_recommendation/ml/skill_domain_sem_model.py`

スキル領域潜在変数モデルの核心クラスです。

#### 初期化

```python
from skillnote_recommendation.ml.skill_domain_sem_model import SkillDomainSEMModel

model = SkillDomainSEMModel(
    member_competence_df=member_competence_df,  # メンバー習得力量データ
    competence_master_df=competence_master_df,  # 力量マスタ
    num_domain_categories=8  # スキル領域の分類数（5～10推奨）
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
2. メンバーのその領域での現在レベルを判定
3. 次のレベルへの潜在変数スコアを計算
4. 領域の信頼度を乗じる

##### `get_direct_effect_skills(member_code, domain_category, top_n=5)`
**説明**: 特定領域で潜在変数を獲得したときの、その領域内の次レベルスキル推薦

```python
direct_recs = model.get_direct_effect_skills(
    member_code="M001",
    domain_category="プログラミング",
    top_n=5
)
# 返り値: [
#   {
#     'skill_code': 'C003',
#     'skill_name': 'Webアプリ開発',
#     'direct_effect_score': 0.64,
#     'next_level': 'プログラミング_中級',
#     'reason': 'プログラミングの初級スキル習得後の推奨です'
#   },
#   ...
# ]
```

**適用場面**: 「Pythonを習得したからJavaを学ぶ」ではなく、「プログラミング領域の初級段階を完了したから中級スキルへ進もう」という推奨

##### `get_indirect_support_skills(member_code, target_skill, top_n=5)`
**説明**: ターゲットスキル習得を支援する、他領域のスキル推薦

```python
indirect_recs = model.get_indirect_support_skills(
    member_code="M001",
    target_skill="C010",  # システム設計
    top_n=5
)
# 返り値: [
#   {
#     'skill_code': 'C005',
#     'skill_name': 'DB設計',
#     'indirect_support_score': 0.32,
#     'reason': 'DB設計スキルを習得することで、システム設計の理解が深まります'
#   },
#   ...
# ]
```

**適用場面**: 「システム設計を学ぶ際、データベース領域の知識があると役立つ」という補完的な推薦

##### `get_member_domain_profile(member_code)`
**説明**: メンバーの領域別プロファイル（全潜在変数のスコア）を取得

```python
profile = model.get_member_domain_profile("M001")
# 返り値: {
#   'プログラミング': {
#     'プログラミング_初級': 0.80,
#     'プログラミング_中級': 0.80,
#     'プログラミング_上級': 0.20
#   },
#   'データベース': {
#     'データベース_初級': 0.00,
#     'データベース_中級': 0.00,
#     'データベース_上級': 0.00
#   }
# }
```

**適用場面**: メンバーの強み・弱み分析、キャリア開発の相談

##### `get_all_domains()`
**説明**: すべてのスキル領域名を取得

```python
domains = model.get_all_domains()
# 返り値: ['プログラミング', 'データベース', 'インフラ', ...]
```

##### `get_domain_info(domain_name)`
**説明**: 領域の詳細情報を取得

```python
info = model.get_domain_info("プログラミング")
# 返り値: {
#   'domain_name': 'プログラミング',
#   'num_latent_factors': 3,
#   'latent_factors': [
#     {'name': 'プログラミング_初級', 'level': 0, 'num_skills': 5},
#     {'name': 'プログラミング_中級', 'level': 1, 'num_skills': 5},
#     {'name': 'プログラミング_上級', 'level': 2, 'num_skills': 5}
#   ],
#   'path_coefficients': [
#     {
#       'from': 'プログラミング_初級',
#       'to': 'プログラミング_中級',
#       'coefficient': 0.75,
#       'is_significant': True
#     },
#     ...
#   ],
#   'domain_reliability': 0.95
# }
```

### 2. MLSEMRecommender
**位置**: `skillnote_recommendation/ml/ml_sem_recommender.py`

MLRecommenderを拡張し、SEMモデルを統合した推薦エンジンです。

#### 初期化

```python
from skillnote_recommendation.ml.ml_sem_recommender import MLSEMRecommender

# ビルド（SEM統合）
recommender = MLSEMRecommender.build(
    member_competence=member_competence_df,
    competence_master=competence_master_df,
    member_master=member_master_df,
    use_sem=True,
    sem_weight=0.2,  # SEM重み20%（他の方法の合計重みは80%）
    num_domain_categories=8
)
```

#### 主要メソッド

##### `recommend(member_code, top_n=10, use_sem=True, return_explanation=False)`
**説明**: SEMスコアを適用した推薦を実施

```python
recommendations = recommender.recommend(
    member_code="M001",
    top_n=10,
    use_sem=True,
    return_explanation=True
)

# 返り値: [Recommendation(...), ...]
# recommendation.reason には SEM 分析結果が含まれます
```

**スコア計算式**:
```
最終スコア = 基本スコア × (1 - SEM重み) + SEMスコア × SEM重み
```

##### `get_direct_effect_recommendations(member_code, domain_category, top_n=5)`
**説明**: 直接効果に基づく推薦

```python
recs = recommender.get_direct_effect_recommendations(
    member_code="M001",
    domain_category="プログラミング",
    top_n=5
)
```

##### `get_indirect_support_recommendations(member_code, target_skill, top_n=5)`
**説明**: 間接効果に基づく推薦

```python
recs = recommender.get_indirect_support_recommendations(
    member_code="M001",
    target_skill="C010",
    top_n=5
)
```

##### `get_member_domain_profile(member_code)`
**説明**: メンバーの領域別プロファイルを取得

```python
profile = recommender.get_member_domain_profile("M001")
```

##### `get_all_domains()`
**説明**: すべてのスキル領域を取得

```python
domains = recommender.get_all_domains()
```

## 設計パラメータ

### スキル領域の分類数（num_domain_categories）

- **推奨値**: 5～10
- **デフォルト**: 8
- **効果**:
  - 少ない（5）: より粗い分類、計算が高速
  - 多い（10）: より詳細な分類、計算がやや遅い

### SEMスコアの重み（sem_weight）

- **推奨値**: 0.15～0.25
- **デフォルト**: 0.20
- **効果**:
  - 0.1: SEMをマイナー要素として使用（NMF/グラフを重視）
  - 0.2: バランス型（推奨）
  - 0.3: SEMを重視

### 潜在変数の段階（初級/中級/上級）

現在は固定で3段階ですが、これは設計によって変更可能です。

## データ要件

### 必須カラム

**member_competence_df**:
- `メンバーコード`: メンバーID
- `力量コード`: スキルID
- `正規化レベル`: スキルレベル（0-5）

**competence_master_df**:
- `力量コード`: スキルID
- `力量名`: スキル名
- `力量カテゴリー名`: カテゴリー（階層形式推奨、例：「プログラミング > Python」）

### データ品質の目安

- メンバー数: 50名以上推奨
- スキル数: 50以上推奨
- 領域当たりスキル数: 平均3～10個

## 動作確認

### ユニットテスト

```bash
cd CareerNavigator
uv run pytest tests/test_skill_domain_sem_model.py -v
```

### 簡易テスト

```python
import pandas as pd
from skillnote_recommendation.ml.skill_domain_sem_model import SkillDomainSEMModel

# サンプルデータ
member_df = pd.DataFrame({...})
competence_df = pd.DataFrame({...})

# モデル初期化
model = SkillDomainSEMModel(member_df, competence_df)

# テスト
sem_score = model.calculate_sem_score("M001", "C001")
print(f"SEM Score: {sem_score:.3f}")  # 期待値: 0-1の範囲
```

## 今後の拡張計画

### 短期（1～2ヶ月）

1. **UI可視化**
   - メンバーの領域別プロファイル表示
   - 推薦理由の詳細説明

2. **説明文自動生成の強化**
   - より自然な日本語表現
   - コンテキストに応じた推薦理由

### 中期（2～3ヶ月）

1. **キャリアパス因果構造モデル**
   - 役職別の成長パス可視化
   - 段階的な推薦

2. **統計的有意性の検定**
   - パス係数の信頼区間計算
   - p値の計算

### 長期（3～6ヶ月）

1. **マルチステップ依存チェーン**
   - A→B→Cのような複数段階の依存性検出

2. **条件付き依存性**
   - 「AかつBの場合のみC」のような条件付き推奨

3. **個人別カスタマイズ**
   - メンバー属性に基づく相互作用モデル

## トラブルシューティング

### 問題: SEMスコアが常に0

**原因**: スキルが領域に含まれていない、またはメンバーがそのスキルを習得していない

**解決**:
1. `get_all_domains()`で領域を確認
2. `_find_skill_domain(skill_code)`でスキルの領域を確認
3. member_competence_dfにスキルが含まれているか確認

### 問題: 直接効果推薦が空

**原因**: メンバーが既に最高レベルに達している、または領域内にスキルが不足している

**解決**:
1. `get_member_domain_profile(member_code)`でレベルを確認
2. `get_domain_info(domain_name)`で領域のスキル数を確認

### 問題: パフォーマンスが低い

**原因**: num_domain_categoriesが大きすぎる、またはメンバー/スキルが多すぎる

**解決**:
1. num_domain_categoriesを5に減らす
2. 計算をバッチ処理化する
3. キャッシング機構を追加する

## 参考資料

- **理論背景**: 構造方程式モデリング（SEM）
- **実装参考**: `skillnote_recommendation/core/skill_dependency_analyzer.py`
- **既存統合先**: MLRecommender（NMF推薦）
