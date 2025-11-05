# ML (Machine Learning) モジュール

このディレクトリには、**機械学習ベースの推薦システム**が含まれています。

## 📋 このモジュールの役割

- ✅ 協調フィルタリングによる推薦
- ✅ Matrix Factorization (行列分解) モデル
- ✅ 多様性を考慮した推薦
- ✅ ハイパーパラメータの自動調整
- ✅ モデルの評価と保存

## 🧠 機械学習とは？

機械学習ベースの推薦は、**過去のデータからパターンを学習**して推薦を行います。

- **ルールベース**: 人間が決めたルールで推薦 (core モジュール)
- **機械学習ベース**: データから自動的にパターンを学習して推薦 (このモジュール)

## 📂 ファイル分類

### 🎯 主要なクラス（重要）

| ファイル | クラス名 | 役割 |
|---------|---------|------|
| **ml_recommender.py** | `MLRecommender` | **ML推薦システム統合** - 機械学習推薦のメインクラス |
| **matrix_factorization.py** | `MatrixFactorizationModel` | **行列分解モデル** - 協調フィルタリングの実装 |
| **base_recommender.py** | `BaseRecommender` | **基底クラス** - 推薦システムの抽象インターフェース |

### 🎨 多様性制御

推薦結果の多様性を高めるためのモジュール。

| ファイル | クラス名 | 役割 |
|---------|---------|------|
| **diversity.py** | `DiversityReranker` | **多様性再ランキング** - 推薦結果を多様化する |

**多様性戦略:**
- **MMR (Maximal Marginal Relevance)**: 関連度と多様性のバランス
- **カテゴリ多様性**: 異なるカテゴリから推薦
- **タイプ多様性**: SKILL/EDUCATION/LICENSE をバランス良く推薦
- **ハイブリッド**: 上記を組み合わせた総合的な多様性

### ⚙️ モデル訓練と評価

| ファイル | クラス/関数 | 役割 |
|---------|-----------|------|
| **hyperparameter_tuning.py** | `HyperparameterTuner` | **ハイパーパラメータ調整** - Optunaで最適パラメータを探索 |
| **data_preprocessing.py** | 前処理関数 | 機械学習用のデータ前処理 |
| **ml_evaluation.py** | 評価指標関数 | モデル性能の評価（Precision, Recall, NDCG など） |
| **model_serialization.py** | 保存・読込関数 | モデルの保存と読み込み |

### 🧪 その他

| ファイル | クラス名 | 役割 |
|---------|---------|------|
| **career_pattern_classifier.py** | `CareerPatternClassifier` | キャリアパターンの分類 |
| **multi_pattern_recommender.py** | `MultiPatternRecommender` | 複数パターンに基づく推薦 |
| **graph_recommender.py** | `GraphRecommender` | グラフベースの機械学習推薦（実験的） |
| **exceptions.py** | ML固有の例外 | 機械学習モジュール専用のエラークラス |

## 🚀 使い方

### 基本的な使い方

```python
from skillnote_recommendation.ml import MLRecommender
from skillnote_recommendation.core.data_loader import DataLoader

# データ読み込み
loader = DataLoader()
data = loader.load_all_data()

# ML推薦システムを初期化・訓練
ml_recommender = MLRecommender(data)

# メンバーへの推薦
recommendations = ml_recommender.recommend(
    member_code='m48',
    top_n=10,
    use_diversity=True,
    diversity_strategy='hybrid'
)

# 推薦結果の表示
for rec in recommendations:
    print(f"{rec['力量名']}: スコア {rec['MLスコア']:.3f}")
```

### 多様性を考慮した推薦

```python
# MMR戦略で多様性を確保
recommendations = ml_recommender.recommend(
    member_code='m48',
    top_n=10,
    diversity_strategy='mmr',  # 多様性戦略を指定
    lambda_param=0.7  # 関連度と多様性のバランス（0.0～1.0）
)

# カテゴリ多様性重視
recommendations = ml_recommender.recommend(
    member_code='m48',
    top_n=10,
    diversity_strategy='category'
)

# タイプ多様性重視
recommendations = ml_recommender.recommend(
    member_code='m48',
    top_n=10,
    diversity_strategy='type'
)
```

### ハイパーパラメータ調整

```python
from skillnote_recommendation.ml.hyperparameter_tuning import HyperparameterTuner

# チューナーを初期化
tuner = HyperparameterTuner(
    skill_matrix=skill_matrix,
    member_competence=member_competence,
    competence_master=competence_master
)

# 最適なパラメータを探索
best_params = tuner.tune(n_trials=50)
print(f"最適パラメータ: {best_params}")

# 最適なパラメータでモデルを訓練
ml_recommender = MLRecommender(
    data,
    n_components=best_params['n_components'],
    max_iter=best_params['max_iter']
)
```

### モデルの保存と読み込み

```python
from skillnote_recommendation.ml.model_serialization import save_model, load_model

# モデルを保存
save_model(ml_recommender.mf_model, 'models/my_model.pkl')

# モデルを読み込み
loaded_model = load_model('models/my_model.pkl')
```

## 📊 評価指標

このモジュールでは、以下の評価指標を使用します：

| 指標 | 説明 |
|-----|------|
| **Precision@K** | 推薦した力量のうち、実際に習得した割合 |
| **Recall@K** | 習得した力量のうち、推薦できた割合 |
| **NDCG@K** | 推薦順位を考慮した評価指標 |
| **Hit Rate** | 少なくとも1つ当たった推薦の割合 |
| **多様性メトリクス** | カテゴリ多様性、タイプ多様性、カバレッジ |

```python
# 多様性メトリクスを計算
diversity_metrics = ml_recommender.calculate_diversity_metrics(
    recommendations,
    ml_recommender.competence_master
)

print(f"カテゴリ多様性: {diversity_metrics['category_diversity']:.3f}")
print(f"タイプ多様性: {diversity_metrics['type_diversity']:.3f}")
print(f"カバレッジ: {diversity_metrics['coverage']:.3f}")
```

## 🔬 技術詳細

### Matrix Factorization (行列分解)

**仕組み:**
```
メンバー×力量マトリクス (M × N)
        ↓ 分解
メンバー潜在因子 (M × K) × 力量潜在因子 (K × N)
```

- **M**: メンバー数
- **N**: 力量数
- **K**: 潜在因子数（デフォルト: 20）

**アルゴリズム:** NMF (Non-negative Matrix Factorization)
- すべての値が非負（0以上）
- 解釈しやすい潜在因子

### 多様性再ランキング

推薦スコア上位の候補から、多様性を考慮して再選択します。

```
1. MLモデルでスコア計算
2. スコア上位 N×3 の候補を取得
3. 多様性戦略に基づいて再ランキング
4. 最終的に上位 N 件を推薦
```

## 📖 詳しく知りたい方へ

- **ML技術の詳細**: [機械学習技術詳細](../../docs/ML_TECHNICAL_DETAILS.md)
- **初心者向け**: [初心者向けガイド](../../docs/BEGINNER_GUIDE.md)
- **評価方法**: [評価ガイド](../../docs/EVALUATION.md)

## 🔗 関連モジュール

- **[core/](../core/)** - コアビジネスロジック、ルールベース推薦
- **[graph/](../graph/)** - グラフベースの推薦システム
- **[utils/](../utils/)** - ユーティリティ関数

## ⚠️ 注意事項

- **データ量**: 機械学習モデルは十分なデータ量が必要です。メンバー数が少ない（50人未満）場合は、ルールベース推薦の方が精度が高い場合があります。
- **訓練時間**: ハイパーパラメータ調整は時間がかかります（50試行で10〜30分程度）。
- **メモリ使用量**: 大規模なデータセットでは、メモリ使用量が多くなることがあります。

## 🎓 学習リソース

- [協調フィルタリングとは？](https://ja.wikipedia.org/wiki/協調フィルタリング)
- [Matrix Factorization](https://en.wikipedia.org/wiki/Matrix_factorization_(recommender_systems))
- [Optuna - ハイパーパラメータ最適化](https://optuna.org/)
