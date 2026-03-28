# Core モジュール

このディレクトリには、CareerNavigatorの**基盤コンポーネント**が含まれています。

## 📋 このモジュールの役割

- ✅ データの読み込みと変換
- ✅ 推薦システムの評価（ML専用）
- ✅ データ品質管理
- ✅ 設定管理とエラーハンドリング
- ✅ 参考人物検索

## 📂 ファイル分類

### 🎯 主要なクラス（重要）

これらは、推薦システムの基盤となるクラスです。

| ファイル | クラス名 | 役割 |
|---------|---------|------|
| **data_loader.py** | `DataLoader` | **データ読込** - CSVファイルからデータを読み込む |
| **data_transformer.py** | `DataTransformer` | **データ変換** - 読み込んだデータを推薦用に変換する |
| **evaluator.py** | `RecommendationEvaluator` | **評価システム** - ML推薦システムの性能を評価する |
| **reference_persons.py** | `ReferencePersonFinder` | **参考人物検索** - ロールモデルとなる先輩を検索する |

### 📊 データ構造

システムが扱うデータの構造を定義しています。

| ファイル | 主要なクラス | 役割 |
|---------|------------|------|
| **models.py** | `Member`, `Competence`, `MemberCompetence`, `Recommendation` | データ構造の定義（メンバー、力量、推薦結果など） |
| **schemas.py** | Pydantic スキーマ | データのバリデーション（入力チェック） |

### 🔍 計算・分析

推薦に必要な計算や分析を行います。

| ファイル | クラス名 | 役割 |
|---------|---------|------|
| **skill_dependency_analyzer.py** | `SkillDependencyAnalyzer` | スキルの依存関係を分析する |

### 🛡️ 品質管理

データ品質とシステムの信頼性を確保します。

| ファイル | クラス/関数 | 役割 |
|---------|-----------|------|
| **data_quality.py** | 品質チェック関数 | データ品質をチェックする |
| **data_quality_monitor.py** | `DataQualityMonitor` | データ品質を継続的に監視する |

### ⚙️ インフラストラクチャ

設定管理、ロギング、エラー処理など。

| ファイル | 主要なクラス/関数 | 役割 |
|---------|----------------|------|
| **config.py** | `Config` クラス | 推薦パラメータと設定を管理する |
| **config_v2.py** | 新バージョンのConfig | 実験的な設定管理（開発中） |
| **logging_config.py** | ログ設定関数 | 構造化ロギングの設定 |
| **errors.py** | カスタム例外クラス | システム固有のエラー定義 |
| **error_handlers.py** | エラー処理関数 | エラーを適切に処理する |
| **retry.py** | リトライデコレータ | 失敗時の自動リトライ |
| **interfaces.py** | 抽象基底クラス | インターフェース定義 |

### 💾 persistence/ - データ永続化

データベースとの連携、モデルの保存・読み込み。

| ファイル | クラス/関数 | 役割 |
|---------|-----------|------|
| **database.py** | データベース操作 | SQLiteデータベース接続 |
| **models.py** | ORMモデル | データベーステーブルの定義 |
| **repository.py** | `Repository` | データの保存・読み込みパターン |
| **session_manager.py** | `SessionManager` | データベースセッション管理 |
| **model_storage.py** | モデル保存関数 | 機械学習モデルの保存・読み込み |

## 🚀 使い方

### データの読み込みと変換

```python
from skillnote_recommendation.core.data_loader import DataLoader
from skillnote_recommendation.core.data_transformer import DataTransformer

# データ読み込み
loader = DataLoader(data_dir='data/')
data = loader.load_all_data()

# データ変換
transformer = DataTransformer()
competence_master = transformer.create_competence_master(data)
member_competence, valid_members = transformer.create_member_competence(
    data, competence_master
)
```

### ML推薦システムの評価

```python
from skillnote_recommendation.core.evaluator import RecommendationEvaluator
from skillnote_recommendation.ml.ml_recommender import MLRecommender

# MLモデルの学習
ml_recommender = MLRecommender.build(
    member_competence=member_competence,
    competence_master=competence_master,
    member_master=member_master
)

# 評価器の初期化
evaluator = RecommendationEvaluator(recommender=ml_recommender)

# 時系列分割評価
train_data, test_data = evaluator.temporal_train_test_split(
    member_competence, train_ratio=0.8
)
metrics = evaluator.evaluate_recommendations(
    train_data, test_data, competence_master, top_k=10
)
```

### 参考人物検索

```python
from skillnote_recommendation.core.reference_persons import ReferencePersonFinder

# 参考人物検索の初期化
finder = ReferencePersonFinder(
    member_competence=member_competence,
    member_master=member_master,
    competence_master=competence_master
)

# 特定の力量を持つ参考人物を検索
reference_persons = finder.find_reference_persons(
    target_member_code='m48',
    recommended_competence_code='skill_001',
    top_n=3
)
```

## 📖 詳しく知りたい方へ

- **初心者の方**: [初心者向けガイド](../../docs/BEGINNER_GUIDE.md) を読んでください
- **開発者の方**: [コード構造ガイド](../../docs/CODE_STRUCTURE.md) を参照してください
- **アーキテクチャを知りたい方**: [ARCHITECTURE.md](../../ARCHITECTURE.md) を見てください

## 🔗 関連モジュール

- **[ml/](../ml/)** - 機械学習ベースの推薦システム
- **[graph/](../graph/)** - グラフベースの推薦システム
- **[utils/](../utils/)** - ユーティリティ関数

## ⚠️ 注意事項

- **config.py と config_v2.py**: 現在は `config.py` を使用してください。`config_v2.py` は開発中です。
- **persistence/**: データベース機能は Python 側の永続化ロジックです。現行の公開 UI は Cloudflare Functions 側を利用します。
