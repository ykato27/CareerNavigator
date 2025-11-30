# CareerNavigator ドキュメント

このディレクトリには、CareerNavigatorプロジェクトのすべての技術ドキュメントが含まれています。

## 📚 ドキュメント一覧

### アーキテクチャ・設計

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - システムアーキテクチャの概要
- **[API_REFERENCE.md](API_REFERENCE.md)** - APIリファレンス
- **[CODE_STRUCTURE.md](CODE_STRUCTURE.md)** - コード構造ガイド
- **[plantuml/](plantuml/)** - PlantUML図（アーキテクチャ/モジュール/クラス/シーケンス図等）

### 開発ガイド

- **[QUICKSTART.md](QUICKSTART.md)** - クイックスタートガイド
- **[BEGINNER_GUIDE.md](BEGINNER_GUIDE.md)** - 初心者向けガイド
- **[REFACTORING_GUIDE.md](REFACTORING_GUIDE.md)** - リファクタリングガイド
- **[STREAMLIT_GUIDE.md](STREAMLIT_GUIDE.md)** - Streamlit UI 開発ガイド
- **[STREAMLIT_CLOUD_SETUP.md](STREAMLIT_CLOUD_SETUP.md)** - Streamlit Cloud セットアップ

### 技術詳細

- **[ML_TECHNICAL_DETAILS.md](ML_TECHNICAL_DETAILS.md)** - 機械学習の技術詳細
- **[MODELS_TECHNICAL_GUIDE.md](MODELS_TECHNICAL_GUIDE.md)** - モデル技術ガイド
- **[HYBRID_RECOMMENDATION_SYSTEM.md](HYBRID_RECOMMENDATION_SYSTEM.md)** - ハイブリッド推薦システム
- **[EVALUATION.md](EVALUATION.md)** - モデル評価ガイド

### テスト

- **[TESTING_QUICKSTART.md](TESTING_QUICKSTART.md)** - テストクイックスタート
- **[TEST_DESIGN.md](TEST_DESIGN.md)** - テスト設計書

### 機能別ドキュメント

#### メインアプリケーション

- **[STREAMLIT_APPS.md](STREAMLIT_APPS.md)** - Streamlitアプリケーション完全ガイド
  - データ読み込み、因果推論推薦、従業員キャリアダッシュボード、組織スキルマップの詳細
- **[WEBUI_GUIDE.md](WEBUI_GUIDE.md)** - WebUI完全ガイド
  - React + FastAPI構成、セットアップ、APIリファレンス、開発ワークフロー

#### 技術詳細・参考実装

- **[CAUSAL_RECOMMENDATION.md](CAUSAL_RECOMMENDATION.md)** - 因果推論ベース推薦の技術詳細
  - LiNGAMアルゴリズム、3軸スコアリング、Causalフィルタリング、スマートロードマップ
- **[ML_MODELS_REFERENCE.md](ML_MODELS_REFERENCE.md)** - 機械学習/統計モデル参考資料
  - Matrix Factorization、グラフベース推薦、ベイジアンネットワーク、SEM、ハイブリッド推薦

#### SEM（構造方程式モデリング）

- **[SEM_IMPLEMENTATION_SUMMARY.md](SEM_IMPLEMENTATION_SUMMARY.md)** - SEM実装サマリー
- **[CAREER_PATH_SEM_MODEL.md](CAREER_PATH_SEM_MODEL.md)** - キャリアパスSEMモデル
- **[SEM_SCALABILITY_ANALYSIS.md](SEM_SCALABILITY_ANALYSIS.md)** - SEMスケーラビリティ分析
- **[NEW_SEM_FEATURES.md](NEW_SEM_FEATURES.md)** - 新SEM機能紹介

## 🚀 推奨読書順序

### 初めての方

1. [QUICKSTART.md](QUICKSTART.md) - プロジェクトの立ち上げ
2. [BEGINNER_GUIDE.md](BEGINNER_GUIDE.md) - 基本概念の理解
3. [plantuml/](plantuml/) - **視覚的理解**: システム図でアーキテクチャを把握
4. [ARCHITECTURE.md](ARCHITECTURE.md) - システム全体像の詳細
5. [STREAMLIT_GUIDE.md](STREAMLIT_GUIDE.md) - UI操作方法

### 開発者向け

1. [plantuml/](plantuml/) - **推奨開始点**: クラス図・シーケンス図でコード構造を視覚的に理解
2. [CODE_STRUCTURE.md](CODE_STRUCTURE.md) - コード構造の詳細
3. [REFACTORING_GUIDE.md](REFACTORING_GUIDE.md) - リファクタリング方針
4. [TESTING_QUICKSTART.md](TESTING_QUICKSTART.md) - テストの書き方
5. [API_REFERENCE.md](API_REFERENCE.md) - API仕様

### ML/データサイエンティスト向け

1. [plantuml/](plantuml/) - **アルゴリズムフロー図**: 推薦アルゴリズムの処理フローを理解
2. [ML_TECHNICAL_DETAILS.md](ML_TECHNICAL_DETAILS.md) - ML技術詳細
3. [MODELS_TECHNICAL_GUIDE.md](MODELS_TECHNICAL_GUIDE.md) - モデルの詳細
4. [HYBRID_RECOMMENDATION_SYSTEM.md](HYBRID_RECOMMENDATION_SYSTEM.md) - ハイブリッドアプローチ
5. [SEM_IMPLEMENTATION_SUMMARY.md](SEM_IMPLEMENTATION_SUMMARY.md) - SEM実装の詳細
6. [EVALUATION.md](EVALUATION.md) - 評価手法とメトリクス

## 📝 ドキュメント更新

新しいドキュメントを追加した場合は、このREADME.mdも更新してください。

### ドキュメント配置ルール

- **ルート直下** (`/CareerNavigator/`): `README.md` と `CONTRIBUTING.md` のみ
- **docs/** (`/CareerNavigator/docs/`): すべての技術ドキュメント
- **モジュール内** (`skillnote_recommendation/*/`): 各モジュールの `README.md`

詳細は [../.claude/claude.md](../.claude/claude.md) の「フォルダ構成とファイル配置規約」を参照してください。
