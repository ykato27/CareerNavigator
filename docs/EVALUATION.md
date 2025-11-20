# 推薦システム評価ガイド

## 概要

CareerNavigator推薦システムは、時系列分割による評価機能を提供しています。
過去のデータで学習し、未来のデータで評価することで、推薦精度を定量的に測定できます。

> **関連ドキュメント**: モデル評価の詳細については、[MODELS_TECHNICAL_GUIDE.md](MODELS_TECHNICAL_GUIDE.md)も参照してください。

## ⚠️ データリーケージ防止とベストプラクティス

### データリーケージとは

**データリーケージ（Data Leakage）**は、訓練データにテスト期間の情報が紛れ込むことで、評価メトリクスが20-40%過大評価される深刻な問題です。これにより、本番環境での推薦精度が期待値を大きく下回る結果となります。

### 本システムの対策

CareerNavigatorでは、以下の3つの仕組みでデータリーケージを完全に防止しています：

#### 1. **グローバル分割方式**

全メンバー共通の分割日（split_date）を使用し、以下を保証します：

```python
# ✅ 正しい実装（グローバル分割）
train_data, test_data = evaluator.temporal_train_test_split(
    member_competence,
    split_date="2023-07-01"
)
# → 全メンバーについて、2023-07-01以前のスキルを訓練、以降を予測
```

**保証される性質：**
- 各メンバーについて、訓練データの最新日 < テストデータの最古日
- 時系列の整合性が厳密に保たれる
- 同一メンバーのデータが訓練とテストに分散しない（時系列順）

#### 2. **Cold-start問題の自動処理**

訓練セットに存在しないメンバー（Cold-startメンバー）は評価データから自動的に除外されます：

```python
train_data, test_data = evaluator.temporal_train_test_split(
    member_competence,
    split_date="2023-07-01"
)

# ログ出力例:
# WARNING: Cold-start問題により3名のメンバー（15レコード）を評価データから除外しました
#          （訓練セットに存在しないメンバー）
# INFO: 時系列分割完了:
#   分割日: 2023-07-01
#   訓練データ: 1200レコード, 50名のメンバー
#   評価データ: 300レコード, 47名のメンバー  ← Cold-startメンバー3名を除外
```

#### 3. **データリーケージの検証機能**

`validate_temporal_split` メソッドで、分割の妥当性を自動検証できます：

```python
# 分割の妥当性を検証
validation = evaluator.validate_temporal_split(
    train_data,
    test_data,
    split_date="2023-07-01"
)

if not validation["is_valid"]:
    print("⚠️ データリーケージが検出されました！")
    for issue in validation["issues"]:
        print(f"  - {issue}")

    # 詳細情報
    print(f"\n【検証結果】")
    print(f"訓練期間: {validation['train_date_range'][0]} ~ {validation['train_date_range'][1]}")
    print(f"テスト期間: {validation['test_date_range'][0]} ~ {validation['test_date_range'][1]}")
    print(f"データリーケージ発生メンバー: {validation['leakage_members']}名")
    print(f"Cold-startメンバー: {validation['cold_start_members']}名")
else:
    print("✅ データリーケージなし。正しい分割です。")
```

### レベルアップデータの扱い

同一力量のレベルアップ（例: Python Lv2 → Lv3）がある場合：

**力量コードにレベルが含まれる場合（推奨）:**
```python
# 例: 力量コード = "Python_Lv2", "Python_Lv3"
# → 異なる力量コードとして扱われるため、データリーケージの心配なし
```

**同一力量コードでレベルカラムのみ変わる場合:**
```python
# 例: 力量コード = "Python", 正規化レベル = 2 → 3
# → 両レコードが別々に扱われるが、時系列順序が保たれるため問題なし
# （グローバル分割により、レベルアップ前後が正しく訓練/テストに分割される）
```

### 推奨される使用方法

```python
# 1. 時系列分割を実行
train_data, test_data = evaluator.temporal_train_test_split(
    member_competence,
    split_date="2023-07-01"
)

# 2. 分割の妥当性を確認（本番運用前に必ず実行）
validation = evaluator.validate_temporal_split(
    train_data, test_data, split_date="2023-07-01"
)

if not validation["is_valid"]:
    raise ValueError(f"データリーケージが検出されました: {validation['issues']}")

# 3. 評価実行
metrics = evaluator.evaluate_recommendations(
    train_data=train_data,
    test_data=test_data,
    competence_master=competence_master,
    top_k=10
)
```

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
# 評価実行（238テスト全体のカバレッジは30%）
metrics = evaluator.evaluate_recommendations(
    train_data=train_data,
    test_data=test_data,
    competence_master=competence_master,
    top_k=10  # Top-10推薦の精度を評価
)

# 結果表示（print()を使用して標準出力に表示）
evaluator.print_evaluation_results(metrics)
```

出力例:
```
================================================================================
推薦システム評価結果
================================================================================

評価対象メンバー数: 150名

【Top-10 推薦の精度評価】
  Precision@10:  0.3245  (推薦のうち正解の割合)
  Recall@10:     0.4521  (正解のうち推薦された割合)
  F1@10:         0.3776  (PrecisionとRecallの調和平均)
  NDCG@10:       0.3876  (ランキング品質)
  Hit Rate:      0.7200  (少なくとも1つ正解があった割合)

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

### 推薦精度メトリクス

#### Precision@K（適合率）

推薦されたK件のうち、実際に習得した力量の割合。

**解釈**:
- 1.0に近いほど推薦精度が高い
- 例: Precision@10 = 0.3 → 推薦10件のうち3件が正解

#### Recall@K（再現率）

実際に習得した力量のうち、推薦に含まれていた割合。

**解釈**:
- 1.0に近いほど推薦の網羅性が高い
- 例: Recall@10 = 0.5 → 実際に習得した力量の50%を推薦で捕捉

#### NDCG@K（正規化割引累積利得）

推薦順位の質を評価（上位にあるほど重要）。

**解釈**:
- 1.0に近いほど推薦ランキングが正確
- 関連アイテムが上位にランクされるほどスコアが高い

#### Hit Rate（ヒット率）

少なくとも1つの正解を含む推薦を得たメンバーの割合。

**解釈**:
- 1.0に近いほど多くのメンバーに有用な推薦を提供
- 例: Hit Rate = 0.72 → 72%のメンバーに少なくとも1つの有用な推薦

### NMFモデル評価メトリクス

#### 再構成誤差（Frobenius ノルム）

NMFモデルが元のデータを再構成する際の絶対的な誤差。

**計算式**: ||X - WH||_F

**解釈**:
- 値が小さいほどモデルの再現性が高い
- データのスケールに依存（異なるデータセット間での比較は困難）
- 通常は0.1～0.5の範囲が良好

#### 正規化再構成誤差（相対誤差）

データのスケールに依存しない相対的な誤差。異なるデータセット間での比較に有用。

**計算式**: error / ||X||_F

**解釈**:
- 0～1の範囲の無次元量
- 0.1以下：非常に良好
- 0.1～0.2：良好
- 0.2～0.3：許容範囲内だが改善余地あり
- 0.3以上：改善が必要

#### モデルスパース性（疎行列性）

W（メンバー因子）行列とH（力量因子）行列における0要素の割合。

**解釈**:
- スパース性が高い：モデルが解釈しやすく、計算効率が良い
- スパース性が低い：すべての潜在因子が活用されている
- 未使用の潜在因子（すべて0）が多い場合：潜在因子数を削減することで効率化可能

**改善方法**:
```python
sparsity_info = mf_model.get_model_sparsity()
print(f"W_sparsity: {sparsity_info['W_sparsity']:.2f}%")
print(f"H_sparsity: {sparsity_info['H_sparsity']:.2f}%")
print(sparsity_info['recommendation'])
```

### 汎化性能メトリクス

#### 訓練 vs テスト誤差（汎化ギャップ）

モデルの過学習（過剰適応）の程度を診断するメトリクス。

**計算式**:
- 汎化ギャップ = test_error - train_error
- 差分比 = |汎化ギャップ| / train_error

**解釈**:
- **ギャップが小さい（<10%）**: 優れた汎化性能
- **ギャップが中程度（10-30%）**: 軽度の過学習（許容範囲）
- **ギャップが大きい（>30%）**: 顕著な過学習、改善推奨

**改善方法**:
1. 正則化強度（alpha_W, alpha_H）を増加させる
2. 早期停止（Early Stopping）を有効にする
3. データ前処理を有効にする
4. より多くの訓練データを用意する

#### メンバーごとの評価

各メンバー個別のPrecision、Recall、F1、NDCGを計算し、推薦精度が低いメンバーを特定。

**用途**:
- モデルの弱点分析
- 特定グループへの推薦精度確認
- メンバー属性と推薦精度の関連性分析

**実装例**:
```python
from skillnote_recommendation.core.evaluator import RecommendationEvaluator

evaluator = RecommendationEvaluator()

# メンバーごとの評価を計算
per_member_df = evaluator.evaluate_per_member(
    train_data=train_data,
    test_data=test_data,
    competence_master=competence_master,
    top_k=10
)

# 統計サマリーを取得
summary = evaluator.get_member_performance_summary(per_member_df, top_k=10)

print(f"高精度メンバー（Precision>=70%）: {summary['high_performers']}名")
print(f"中程度メンバー（40%<=Precision<70%）: {summary['medium_performers']}名")
print(f"低精度メンバー（Precision<40%）: {summary['low_performers']}名")

# 低精度メンバーの確認
low_performers = per_member_df[per_member_df['precision@10'] < 0.4]
print(f"\n精度が低いメンバー:")
print(low_performers[['member_code', 'precision@10', 'recall@10']])
```

## 時系列クロスバリデーション

### TimeSeriesSplit方式

CareerNavigatorは、scikit-learnと同様の**TimeSeriesSplit方式**を採用しています。これにより、複数の時間窓で評価を実行し、モデルの安定性を確認できます。

**特徴：**
- ✅ **累積的な学習データ**: 各foldで訓練データを段階的に増やす
- ✅ **時系列整合性の保証**: 日付ベースの分割で、メンバー単位のデータリーケージを防止
- ✅ **Cold-start問題の自動処理**: 各foldで訓練セットに存在しないメンバーを除外
- ✅ **最小テストサイズ**: 不十分なfoldを自動スキップ

### 基本的な使用方法

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
    print(f"\nFold {result['fold']}:")
    print(f"  訓練期間: ~ {result['train_end_date']}")
    print(f"  テスト期間: {result['test_start_date']} ~ {result['test_end_date']}")
    print(f"  訓練データ: {result['train_size']}レコード, {result['train_members']}名")
    print(f"  テストデータ: {result['test_size']}レコード, {result['test_members']}名")
    print(f"  Precision@10: {result['precision@10']:.3f}")
    print(f"  Recall@10: {result['recall@10']:.3f}")

# 平均メトリクスを計算
import numpy as np
avg_precision = np.mean([r['precision@10'] for r in cv_results])
avg_recall = np.mean([r['recall@10'] for r in cv_results])
std_precision = np.std([r['precision@10'] for r in cv_results])
std_recall = np.std([r['recall@10'] for r in cv_results])

print(f"\n【クロスバリデーション結果】")
print(f"平均 Precision@10: {avg_precision:.3f} ± {std_precision:.3f}")
print(f"平均 Recall@10: {avg_recall:.3f} ± {std_recall:.3f}")
```

### 高度な設定

```python
# 最小テストサイズを指定
cv_results = evaluator.cross_validate_temporal(
    member_competence=member_competence,
    competence_master=competence_master,
    n_splits=5,
    top_k=10,
    min_test_size=200  # 各foldで最低200レコード必要
)

# ログ出力例:
# INFO: === Fold 1/5 ===
# INFO:   訓練期間: ~ 2023-03-01
# INFO:   テスト期間: 2023-03-01 ~ 2023-06-01
# WARNING: Fold 5: テストデータが少なすぎるためスキップ (train=1500, test=50, min_required=200)
```

### TimeSeriesSplit方式の仕組み

```
データ全体: [--------------------時系列データ-------------------->]

Fold 1: [train  ] [test]
Fold 2: [----train----] [test]
Fold 3: [---------train--------] [test]
Fold 4: [-------------train-------------] [test]
Fold 5: [------------------train------------------] [test]

各foldで：
- 訓練データが累積的に増加
- テストデータは次の時間窓
- 時系列の整合性を厳密に保証
```

**ランダムクロスバリデーションとの違い：**

| 方式 | 時系列整合性 | データリーケージ | 本番環境への適用 |
|------|------------|----------------|-----------------|
| **TimeSeriesSplit** | ✅ 保証 | ✅ 防止 | ✅ 高い信頼性 |
| ランダムKFold | ❌ 無視 | ❌ 発生 | ❌ 過大評価 |

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

### 問題: Cold-startメンバーが多数除外される

**原因**: 訓練セットに存在しないメンバーがテストデータに多く含まれている

**症状**:
```
WARNING: Cold-start問題により50名のメンバー（500レコード）を評価データから除外しました
```

**解決策**:

1. **分割日を調整する**:
```python
# 分割日を早める（訓練データを増やす）
train_data, test_data = evaluator.temporal_train_test_split(
    member_competence,
    train_ratio=0.9  # 90%を訓練に使用
)
```

2. **データの状況を確認する**:
```python
# 時系列でメンバー数の推移を確認
df = member_competence.copy()
df["取得日_dt"] = pd.to_datetime(df["取得日"], errors="coerce")
df["年月"] = df["取得日_dt"].dt.to_period("M")

member_trend = df.groupby("年月")["メンバーコード"].nunique()
print("月別のアクティブメンバー数:")
print(member_trend)

# 新規メンバーの流入が多い時期を避けて分割日を設定
```

3. **Cold-start専用の評価を実施する**:
```python
# Cold-startメンバーのみを対象とした評価
cold_start_members = set(test_data_raw["メンバーコード"].unique()) - train_members

if cold_start_members:
    print(f"Cold-startメンバー: {len(cold_start_members)}名")
    # ベースライン推薦（人気度ベース等）で対応を検討
```

### 問題: 評価対象メンバーが0になる

**原因**: テストデータに習得記録がない、または学習データに存在しないメンバー（Cold-start問題）

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

# Cold-startメンバーの確認
cold_start_members = test_members - train_members
if cold_start_members:
    print(f"⚠️ Cold-startメンバー: {len(cold_start_members)}名")
    print(f"  これらのメンバーは自動的に評価から除外されます")
```

### 問題: メトリクスが全て0.0

**原因**: 推薦が正解と全く一致していない

**解決策**:
```python
# 推薦結果を確認
from skillnote_recommendation.ml.ml_recommender import MLRecommender

# メンバーマスタの準備
members_data = pd.DataFrame({
    'メンバーコード': train_data['メンバーコード'].unique()
})

# MLモデルを学習（n_componentsを明示的に指定可能）
ml_recommender = MLRecommender.build(
    member_competence=train_data,
    competence_master=competence_master,
    member_master=members_data,
    use_preprocessing=False,
    use_tuning=False,
    n_components=20  # 潜在因子数を指定（省略可能）
)

# サンプルメンバーで推薦を確認
sample_member = train_data['メンバーコード'].iloc[0]
recommendations = ml_recommender.recommend(
    member_code=sample_member,
    top_n=10,
    use_diversity=False
)

print(f"メンバー {sample_member} の推薦:")
for rec in recommendations[:5]:
    print(f"  - {rec.competence_name} (スコア: {rec.priority_score:.2f})")
```

### 問題: データリーケージが検出される

**原因**: 時系列分割が正しく実装されていない、または手動で分割したデータに問題がある

**症状**:
```python
validation = evaluator.validate_temporal_split(train_data, test_data)
# is_valid: False
# issues: ['メンバー単位のデータリーケージが10名で検出されました']
```

**解決策**:

1. **必ず `temporal_train_test_split` を使用する**:
```python
# ❌ 手動で分割しない
train_data = member_competence[member_competence["メンバーコード"] < "m050"]
test_data = member_competence[member_competence["メンバーコード"] >= "m050"]

# ✅ 公式メソッドを使用
train_data, test_data = evaluator.temporal_train_test_split(
    member_competence,
    split_date="2023-07-01"
)
```

2. **検出されたリーケージの詳細を確認**:
```python
validation = evaluator.validate_temporal_split(train_data, test_data)

if not validation["is_valid"]:
    print("⚠️ データリーケージの詳細:")
    for issue in validation["issues"]:
        print(f"  - {issue}")

    # リーケージが発生しているメンバーの詳細調査
    if validation["leakage_members"] > 0:
        # メンバーごとの日付範囲を確認
        for member in test_data["メンバーコード"].unique()[:5]:
            train_dates = train_data[train_data["メンバーコード"] == member]["取得日"]
            test_dates = test_data[test_data["メンバーコード"] == member]["取得日"]
            print(f"\nメンバー {member}:")
            print(f"  訓練: {train_dates.min()} ~ {train_dates.max()}")
            print(f"  テスト: {test_dates.min()} ~ {test_dates.max()}")
```

3. **外部で作成した分割データを検証**:
```python
# 外部ツールで分割したデータを使用する場合は必ず検証
validation = evaluator.validate_temporal_split(
    train_data, test_data, split_date="2023-07-01"
)

if not validation["is_valid"]:
    raise ValueError(
        f"データリーケージが検出されました。公式の temporal_train_test_split を使用してください。"
        f"\n問題: {validation['issues']}"
    )
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

## 関連ドキュメント

- [MODELS_TECHNICAL_GUIDE.md](MODELS_TECHNICAL_GUIDE.md) - モデル実装の詳細、評価手法の解説
- [ML_TECHNICAL_DETAILS.md](ML_TECHNICAL_DETAILS.md) - 機械学習推薦システムの技術詳細
- [TEST_DESIGN.md](TEST_DESIGN.md) - 評価器のテスト設計（238テストケース、カバレッジ30%）
- [tests/test_evaluator.py](../tests/test_evaluator.py) - 評価器のテストコード
- [skillnote_recommendation/core/evaluator.py](../skillnote_recommendation/core/evaluator.py) - 評価器の実装
