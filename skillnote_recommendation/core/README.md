# Core モジュール

このディレクトリには、CareerNavigatorの**コアビジネスロジック**が含まれています。

## 📋 このモジュールの役割

- ✅ データの読み込みと変換
- ✅ ルールベースの推薦ロジック
- ✅ 推薦システムの評価
- ✅ データ品質管理
- ✅ 設定管理とエラーハンドリング

## 📂 ファイル分類

### 🎯 主要なクラス（重要）

これらは、推薦システムの中心となるクラスです。

| ファイル | クラス名 | 役割 |
|---------|---------|------|
| **recommendation_system.py** | `RecommendationSystem` | **統合推薦システム** - 推薦システム全体を管理する最上位クラス |
| **recommendation_engine.py** | `RecommendationEngine` | **ルールベース推薦エンジン** - スコア計算と推薦ロジック |
| **data_loader.py** | `DataLoader` | **データ読込** - CSVファイルからデータを読み込む |
| **data_transformer.py** | `DataTransformer` | **データ変換** - 読み込んだデータを推薦用に変換する |

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
| **similarity_calculator.py** | `SimilarityCalculator` | 力量間の類似度を計算する |
| **reference_persons.py** | `ReferencePersonSearch` | 参考となる先輩メンバーを検索する |
| **evaluator.py** | `Evaluator` | 推薦システムの性能を評価する |
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
| **streamlit_integration.py** | Streamlit連携 | StreamlitとDBの統合 |

## 🚀 使い方

### 基本的な使い方

```python
from skillnote_recommendation import RecommendationSystem

# 推薦システムを初期化
system = RecommendationSystem()

# メンバーへの推薦を実行
recommendations = system.recommend_competences(
    member_code='m48',
    top_n=10,
    competence_type='SKILL'
)

# 推薦結果を表示
system.print_recommendations('m48', top_n=10)

# CSV出力
system.export_recommendations('m48', 'output.csv', top_n=20)
```

### データの読み込みと変換

```python
from skillnote_recommendation.core.data_loader import DataLoader
from skillnote_recommendation.core.data_transformer import DataTransformer

# データ読み込み
loader = DataLoader(data_dir='data/')
data = loader.load_all_data()

# データ変換
transformer = DataTransformer()
transformed = transformer.transform(data)
```

### 類似度計算

```python
from skillnote_recommendation.core.similarity_calculator import SimilarityCalculator

calculator = SimilarityCalculator(competence_master, member_competence)
similarity = calculator.calculate_similarity('comp_001', 'comp_002')
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
- **persistence/**: データベース機能は Streamlit アプリで使用されます。CLI使用時は不要です。
