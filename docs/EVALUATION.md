# 推薦システム評価ガイド

## 概要

CareerNavigator推薦システムは、時系列分割による評価機能を提供しています。
過去のデータで学習し、未来のデータで評価することで、推薦精度を定量的に測定できます。

## 評価の流れ

### 1. 時系列分割（Temporal Split）

習得力量データを時系列で学習データと評価データに分割します。

```python
from skillnote_recommendation.core.evaluator import RecommendationEvaluator
from skillnote_recommendation.core.data_loader import DataLoader
from skillnote_recommendation.core.data_transformer import DataTransformer

# データ読み込み
loader = DataLoader()
data = loader.load_all_data()

# メンバー習得力量データ作成
transformer = DataTransformer()
competence_master = transformer.create_competence_master(data)
member_competence, _ = transformer.create_member_competence(data, competence_master)

# 評価器を初期化
evaluator = RecommendationEvaluator()

# 時系列分割（80%を学習、20%を評価）
train_data, test_data = evaluator.temporal_train_test_split(
    member_competence=member_competence,
    train_ratio=0.8
)

print(f"学習データ: {len(train_data)}件")
print(f"評価データ: {len(test_data)}件")
```

### 2. 明示的な分割日指定

特定の日付で分割する場合:

```python
# 2024年7月1日を境に分割
train_data, test_data = evaluator.temporal_train_test_split(
    member_competence=member_competence,
    split_date='2024-07-01'
)
```

### 3. 評価実行

学習データで推薦を生成し、評価データで精度を測定します。

```python
# 評価実行
metrics = evaluator.evaluate_recommendations(
    train_data=train_data,
    test_data=test_data,
    competence_master=competence_master,
    top_k=10  # Top-10推薦の精度を評価
)

# 結果表示
evaluator.print_evaluation_results(metrics)
```

出力例:
```
================================================================================
推薦システム評価結果
================================================================================

評価対象メンバー数: 150名

【Top-10 推薦の評価】
  Precision@10: 0.3245
  Recall@10:    0.4521
  NDCG@10:      0.3876
  Hit Rate:     0.7200

================================================================================
```

### 4. 評価結果の保存

```python
# CSV出力
evaluator.export_evaluation_results(
    metrics=metrics,
    output_path='output/evaluation_results.csv'
)
```

## 評価メトリクス

### Precision@K（適合率）

推薦されたK件のうち、実際に習得した力量の割合。

**解釈**:
- 1.0に近いほど推薦精度が高い
- 例: Precision@10 = 0.3 → 推薦10件のうち3件が正解

### Recall@K（再現率）

実際に習得した力量のうち、推薦に含まれていた割合。

**解釈**:
- 1.0に近いほど推薦の網羅性が高い
- 例: Recall@10 = 0.5 → 実際に習得した力量の50%を推薦で捕捉

### NDCG@K（正規化割引累積利得）

推薦順位の質を評価（上位にあるほど重要）。

**解釈**:
- 1.0に近いほど推薦ランキングが正確
- 関連アイテムが上位にランクされるほどスコアが高い

### Hit Rate（ヒット率）

少なくとも1つの正解を含む推薦を得たメンバーの割合。

**解釈**:
- 1.0に近いほど多くのメンバーに有用な推薦を提供
- 例: Hit Rate = 0.72 → 72%のメンバーに少なくとも1つの有用な推薦

## 時系列クロスバリデーション

複数の時間窓で評価を実行し、安定性を確認:

```python
# 5分割クロスバリデーション
cv_results = evaluator.cross_validate_temporal(
    member_competence=member_competence,
    competence_master=competence_master,
    n_splits=5,
    top_k=10
)

# 各foldの結果を表示
for result in cv_results:
    print(f"Fold {result['fold']}: "
          f"Precision={result['precision@10']:.3f}, "
          f"Recall={result['recall@10']:.3f}")

# 平均メトリクスを計算
import numpy as np
avg_precision = np.mean([r['precision@10'] for r in cv_results])
avg_recall = np.mean([r['recall@10'] for r in cv_results])

print(f"\n平均 Precision@10: {avg_precision:.3f}")
print(f"平均 Recall@10: {avg_recall:.3f}")
```

## 特定メンバーのみ評価

```python
# 評価対象を特定メンバーに限定
target_members = ['m001', 'm002', 'm003']

metrics = evaluator.evaluate_recommendations(
    train_data=train_data,
    test_data=test_data,
    competence_master=competence_master,
    top_k=10,
    member_sample=target_members
)
```

## ベストプラクティス

### 1. 適切な分割比率

- **80/20分割**: 一般的な選択
- **70/30分割**: 評価データを多めに確保したい場合
- **90/10分割**: 学習データを最大化したい場合

### 2. 評価対象の選択

```python
# テストデータで実際に習得したメンバーのみ評価
# （デフォルトの動作）
metrics = evaluator.evaluate_recommendations(
    train_data=train_data,
    test_data=test_data,
    competence_master=competence_master,
    top_k=10
)
```

### 3. K値の選択

- **K=5**: 厳密な推薦精度を評価
- **K=10**: バランスの良い評価
- **K=20**: 網羅性を重視した評価

### 4. 時系列の考慮

```python
# 季節性を考慮した分割
# 例: 年度末（3月）をまたがないように分割
train_data, test_data = evaluator.temporal_train_test_split(
    member_competence=member_competence,
    split_date='2024-03-31'  # 年度末で分割
)
```

## 実装例: 完全なワークフロー

```python
from skillnote_recommendation.core.data_loader import DataLoader
from skillnote_recommendation.core.data_transformer import DataTransformer
from skillnote_recommendation.core.evaluator import RecommendationEvaluator

# 1. データ読み込み
print("データ読み込み中...")
loader = DataLoader()
data = loader.load_all_data()

# 2. データ変換
print("データ変換中...")
transformer = DataTransformer()
competence_master = transformer.create_competence_master(data)
member_competence, _ = transformer.create_member_competence(data, competence_master)

# 3. 評価器初期化
evaluator = RecommendationEvaluator()

# 4. 時系列分割
print("時系列分割実行...")
train_data, test_data = evaluator.temporal_train_test_split(
    member_competence=member_competence,
    train_ratio=0.8
)

print(f"学習データ: {len(train_data)}件")
print(f"評価データ: {len(test_data)}件")

# 5. 複数のK値で評価
for k in [5, 10, 20]:
    print(f"\n{'='*80}")
    print(f"Top-{k} 推薦の評価")
    print(f"{'='*80}")

    metrics = evaluator.evaluate_recommendations(
        train_data=train_data,
        test_data=test_data,
        competence_master=competence_master,
        top_k=k
    )

    evaluator.print_evaluation_results(metrics)

    # CSV出力
    evaluator.export_evaluation_results(
        metrics=metrics,
        output_path=f'output/evaluation_top{k}.csv'
    )

# 6. クロスバリデーション
print("\nクロスバリデーション実行...")
cv_results = evaluator.cross_validate_temporal(
    member_competence=member_competence,
    competence_master=competence_master,
    n_splits=5,
    top_k=10
)

# 結果サマリー
import pandas as pd
cv_df = pd.DataFrame(cv_results)
print("\nクロスバリデーション結果:")
print(cv_df[['fold', 'precision@10', 'recall@10', 'ndcg@10', 'hit_rate']])

# 統計情報
print("\n統計情報:")
print(cv_df[['precision@10', 'recall@10', 'ndcg@10', 'hit_rate']].describe())
```

## トラブルシューティング

### 問題: 評価対象メンバーが0になる

**原因**: テストデータに習得記録がない、または学習データに存在しないメンバー

**解決策**:
```python
# データの確認
print(f"学習データのメンバー数: {train_data['メンバーコード'].nunique()}")
print(f"評価データのメンバー数: {test_data['メンバーコード'].nunique()}")

# 共通メンバーの確認
train_members = set(train_data['メンバーコード'].unique())
test_members = set(test_data['メンバーコード'].unique())
common_members = train_members & test_members
print(f"共通メンバー数: {len(common_members)}")
```

### 問題: メトリクスが全て0.0

**原因**: 推薦が正解と全く一致していない

**解決策**:
```python
# 推薦結果を確認
from skillnote_recommendation.core.recommendation_engine import RecommendationEngine

# エンジンを作成して推薦を確認
members_data = pd.DataFrame({
    'メンバーコード': train_data['メンバーコード'].unique()
})

engine = RecommendationEngine(
    df_members=members_data,
    df_competence_master=competence_master,
    df_member_competence=train_data,
    df_similarity=pd.DataFrame(columns=['力量1', '力量2', '類似度'])
)

# サンプルメンバーで推薦を確認
sample_member = train_data['メンバーコード'].iloc[0]
recommendations = engine.recommend(sample_member, top_n=10)

print(f"メンバー {sample_member} の推薦:")
for rec in recommendations[:5]:
    print(f"  - {rec.competence_name} (スコア: {rec.priority_score:.2f})")
```

### 問題: 取得日カラムがない

**原因**: データに時系列情報が含まれていない

**解決策**:
- データローダーで取得日カラムを含むデータを読み込む
- データ加工時に取得日を保持する

```python
# 取得日カラムの確認
print("member_competenceのカラム:")
print(member_competence.columns.tolist())

# 取得日が欠損している場合は元データを確認
print("\n元データ(acquired)のカラム:")
print(data['acquired'].columns.tolist())
```

## 参考

- [TEST_DESIGN.md](../TEST_DESIGN.md): 評価器のテスト設計
- [tests/test_evaluator.py](../tests/test_evaluator.py): 評価器のテストコード
- [skillnote_recommendation/core/evaluator.py](../skillnote_recommendation/core/evaluator.py): 評価器の実装
