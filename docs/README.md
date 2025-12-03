# CareerNavigator ドキュメント

CareerNavigator WebUIの包括的なドキュメントへようこそ。このガイドは、初心者からエキスパート開発者まで、すべてのユーザーをサポートします。

## 📚 ドキュメント構成

### [01-getting-started/](01-getting-started/) - 最初に読む
初心者向けのガイドとクイックスタート

- **[BEGINNER_GUIDE.md](01-getting-started/BEGINNER_GUIDE.md)** - 初心者向けコード理解ガイド
- **[QUICKSTART.md](01-getting-started/QUICKSTART.md)** - クイックスタートガイド
- **[data_model_specification.md](01-getting-started/data_model_specification.md)** - データモデル仕様

### [02-user-guides/](02-user-guides/) - ユーザーガイド（Streamlit）
Streamlitアプリケーションの使い方

- **[STREAMLIT_GUIDE.md](02-user-guides/STREAMLIT_GUIDE.md)** - Streamlitアプリ完全ガイド
- **[STREAMLIT_APPS.md](02-user-guides/STREAMLIT_APPS.md)** - Streamlitアプリ概要
- **[STREAMLIT_CLOUD_SETUP.md](02-user-guides/STREAMLIT_CLOUD_SETUP.md)** - Streamlit Cloudデプロイガイド

### [03-webui/](03-webui/) - WebUIガイド ⭐
React + FastAPI WebUIの使い方と開発ガイド

- **[WEBUI_GUIDE.md](03-webui/WEBUI_GUIDE.md)** - WebUI完全ガイド（セットアップ・開発・デプロイ）

### [04-technical/](04-technical/) - 技術詳細
アルゴリズムと実装の技術解説

#### [algorithms/](04-technical/algorithms/) - 推薦アルゴリズム
- **[CAUSAL_RECOMMENDATION.md](04-technical/algorithms/CAUSAL_RECOMMENDATION.md)** - 因果推論ベース推薦（LiNGAM）
- **[CAUSAL_RECOMMENDATION_THREE_STAGES.md](04-technical/algorithms/CAUSAL_RECOMMENDATION_THREE_STAGES.md)** - 3段階因果推論
- **[CAUSAL_RECOMMENDATION_AUTO_OPTIMIZATION.md](04-technical/algorithms/CAUSAL_RECOMMENDATION_AUTO_OPTIMIZATION.md)** - 自動最適化
- **[HYBRID_RECOMMENDATION_SYSTEM.md](04-technical/algorithms/HYBRID_RECOMMENDATION_SYSTEM.md)** - ハイブリッド推薦
- **[ML_TECHNICAL_DETAILS.md](04-technical/algorithms/ML_TECHNICAL_DETAILS.md)** - ML技術詳細
- **[ML_MODELS_REFERENCE.md](04-technical/algorithms/ML_MODELS_REFERENCE.md)** - MLモデルリファレンス
- **[MODELS_TECHNICAL_GUIDE.md](04-technical/algorithms/MODELS_TECHNICAL_GUIDE.md)** - モデル技術ガイド

#### その他
- **[EVALUATION.md](04-technical/EVALUATION.md)** - 推薦システム評価ガイド

### [05-architecture/](05-architecture/) - アーキテクチャ・設計
システム設計とコード構造

- **[ARCHITECTURE.md](05-architecture/ARCHITECTURE.md)** - システムアーキテクチャ
- **[CODE_STRUCTURE.md](05-architecture/CODE_STRUCTURE.md)** - コード構造ガイド
- **[REFACTORING_GUIDE.md](05-architecture/REFACTORING_GUIDE.md)** - リファクタリングガイド
- **[API_REFERENCE.md](05-architecture/API_REFERENCE.md)** - FastAPI リファレンス
- **[plantuml/](05-architecture/plantuml/)** - PlantUML図（視覚的理解に最適）

### [06-testing/](06-testing/) - テスト ⭐
テスト設計と実装ガイド（Phase 2完了: 100%カバレッジ達成）

- **[TESTING_QUICKSTART.md](06-testing/TESTING_QUICKSTART.md)** - テスト実装クイックスタート（uv対応）
- **[TEST_DESIGN.md](06-testing/TEST_DESIGN.md)** - テスト設計書（WebUI焦点、2025-12-04更新）

## 🎯 目的別ガイド

### プロジェクトを始める（WebUI開発者向け）
1. [BEGINNER_GUIDE.md](01-getting-started/BEGINNER_GUIDE.md) でプロジェクト構造を理解
2. [QUICKSTART.md](01-getting-started/QUICKSTART.md) でセットアップ（**uvベース**）
3. [WEBUI_GUIDE.md](03-webui/WEBUI_GUIDE.md) でWebUI開発を開始
4. [data_model_specification.md](01-getting-started/data_model_specification.md) でデータ構造を確認

### Streamlitアプリを使う（レガシー）
1. [STREAMLIT_GUIDE.md](02-user-guides/STREAMLIT_GUIDE.md) で使い方を学ぶ
2. [STREAMLIT_APPS.md](02-user-guides/STREAMLIT_APPS.md) で機能を確認
3. [STREAMLIT_CLOUD_SETUP.md](02-user-guides/STREAMLIT_CLOUD_SETUP.md) でデプロイ（pyproject.toml対応）

### WebUIを開発する ⭐
1. [WEBUI_GUIDE.md](03-webui/WEBUI_GUIDE.md) - フロントエンド（React）とバックエンド（FastAPI）の完全ガイド
2. [API_REFERENCE.md](05-architecture/API_REFERENCE.md) - API仕様を確認
3. [TESTING_QUICKSTART.md](06-testing/TESTING_QUICKSTART.md) - テストを書く

### アルゴリズムを理解する
1. [MODELS_TECHNICAL_GUIDE.md](04-technical/algorithms/MODELS_TECHNICAL_GUIDE.md) で基礎を学ぶ
2. [CAUSAL_RECOMMENDATION.md](04-technical/algorithms/CAUSAL_RECOMMENDATION.md) で因果推論（LiNGAM）を理解
3. [ML_TECHNICAL_DETAILS.md](04-technical/algorithms/ML_TECHNICAL_DETAILS.md) でML詳細を確認

### テストを書く（バックエンド）
1. [TESTING_QUICKSTART.md](06-testing/TESTING_QUICKSTART.md) - uv環境でテスト実装を開始
2. [TEST_DESIGN.md](06-testing/TEST_DESIGN.md) - WebUI焦点のテスト設計を確認
3. **現状**: Services/Repositories/Schemas/Middleware層で100%カバレッジ達成（Phase 2完了）

### システムアーキテクチャを理解する
1. [ARCHITECTURE.md](05-architecture/ARCHITECTURE.md) でアーキテクチャを理解
2. [CODE_STRUCTURE.md](05-architecture/CODE_STRUCTURE.md) でコード構造を確認
3. [plantuml/](05-architecture/plantuml/) で視覚的に理解

## 💡 おすすめドキュメント

### 🌟 初めての方に
- **[BEGINNER_GUIDE.md](01-getting-started/BEGINNER_GUIDE.md)** - どのファイルから読むべきかわかりやすく説明
- **[QUICKSTART.md](01-getting-started/QUICKSTART.md)** - uvベースのセットアップガイド

### 🔬 技術者向け（WebUI開発）
- **[WEBUI_GUIDE.md](03-webui/WEBUI_GUIDE.md)** - React + FastAPI開発の完全ガイド
- **[MODELS_TECHNICAL_GUIDE.md](04-technical/algorithms/MODELS_TECHNICAL_GUIDE.md)** - 推薦モデルの選び方と使い方
- **[TEST_DESIGN.md](06-testing/TEST_DESIGN.md)** - Phase 2完了（100%カバレッジ）の最新テスト設計

### 📊 視覚的に理解したい方
- **[plantuml/](05-architecture/plantuml/)** - PlantUML図でアーキテクチャを可視化

## 🔗 関連リンク

- [プロジェクトREADME](../README.md) - プロジェクト概要とuv統一化情報
- [GitHub Repository](https://github.com/ykato27/CareerNavigator)

## 📝 ドキュメントの更新

ドキュメントに誤りや改善点を見つけた場合は、GitHubでIssueまたはPull Requestを作成してください。

## 🚀 最新の変更（2025-12-04）

- ✅ **Python環境管理のuv統一化完了**: requirements.txt削除、pyproject.toml一元管理
- ✅ **Phase 2完了**: バックエンドコア層（Services/Repositories/Schemas/Middleware）100%カバレッジ達成
- ✅ **SEMドキュメント削除**: WebUIで使用しないSEM関連ドキュメントを整理
- ✅ **TEST_DESIGN.md更新**: WebUI焦点、Phase 2完了状況を反映

---

**最終更新:** 2025-12-04
