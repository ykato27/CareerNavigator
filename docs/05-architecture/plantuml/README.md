
## 🎨 図の詳細説明

### 01. システムアーキテクチャ図

**対象読者**: アーキテクト、テックリード、新規参加者

**概要**:
- CareerNavigator の4層アーキテクチャを可視化
- プレゼンテーション層（WebUI: React + FastAPI）
- アプリケーション層（ビジネスロジック）
- ドメイン層（因果推論・グラフアルゴリズム）
- インフラストラクチャ層（データアクセス、永続化、ログ）

**学べること**:
- システム全体の構造
- レイヤー間の責務分離
- モダンWebアプリケーションアーキテクチャ
- Dependency Inversion の実装

---

### 02. モジュール構成図

**対象読者**: 開発者、アーキテクト

**概要**:
- フロントエンド（React + TypeScript）の全ページコンポーネント
- バックエンド（FastAPI）の全APIエンドポイント
- `skillnote_recommendation` パッケージの因果推論モジュール
- core (基盤)、graph/causal (因果推論)、utils (ユーティリティ) モジュール

**学べること**:
- フロントエンド・バックエンド間の通信フロー
- モジュール間の依存方向
- SOLID 原則の適用
- 再利用可能なコンポーネント設計

---

### 03. クラス図

**対象読者**: 開発者

**概要**:
- データモデル（Member, Competence, MemberCompetence）
- 因果推論エンジン（CausalGraphRecommender, DependencyAnalyzer）
- 因果キャリアパス生成（CausalCareerPath）
- 推薦結果モデル（3軸スコア付きRecommendation）
- FastAPI Pydanticモデル（TrainParams, RecommendRequest等）

**学べること**:
- ドメインモデルの設計
- 因果推論パイプラインのクラス設計
- Pydanticによるデータバリデーション
- Repository パターン（データアクセス）

---

### 04. シーケンス図

**対象読者**: 開発者、QA エンジニア

**概要**:
- 因果推論推薦システムの実行フロー
  1. データアップロードフェーズ（CSV → FastAPI）
  2. モデル学習フェーズ（LiNGAM因果グラフ構築）
  3. 推薦生成フェーズ（3軸スコアリング）
  4. グラフ可視化フェーズ（Pyvisによるインタラクティブグラフ）
  5. 重み調整フェーズ（手動/ベイズ最適化）

**学べること**:
- WebアプリケーションのAPI通信フロー
- 因果推論パイプラインの動的な振る舞い
- React ↔ FastAPI間のデータフロー
- LiNGAMアルゴリズムの実行プロセス

---

### 05. ユースケース図

**対象読者**: プロダクトマネージャー、ビジネスアナリスト、新規参加者

**概要**:
- 4種類のユーザーロール
  - 従業員（一般ユーザー）
  - 人事担当者（HR Manager）
  - システム管理者（Admin）
  - データサイエンティスト
- 5つの機能パッケージ
  - データ管理
  - 推薦機能
  - モデル管理
  - 可視化・分析
  - システム設定

**学べること**:
- ユーザーごとの機能アクセス
- システムの主要機能
- ユースケース間の関係（include, extend）

---

### 06. コンポーネント図

**対象読者**: インフラエンジニア、DevOps、アーキテクト

**概要**:
- 技術スタックの全体像
- フロントエンド（React 18, TypeScript, Vite, Tailwind CSS v4, Recharts）
- バックエンド（FastAPI, Uvicorn, Pydantic）
- 因果推論ライブラリ（lingam, optuna）
- グラフ処理ライブラリ（NetworkX, Pyvis）
- データ永続化（CSV, joblib）

**学べること**:
- モダンWeb技術スタック
- 因果推論に必要な外部ライブラリの依存関係
- データフローとストレージ戦略
- フロントエンド・バックエンド間の技術構成

---

### 07. アルゴリズムフロー図

**対象読者**: データサイエンティスト、開発者

**概要**:
- 因果推論推薦アルゴリズムの詳細フロー
  1. データ準備フェーズ（バイナリ変換、正規化）
  2. 因果グラフ構築フェーズ（DirectLiNGAM）
  3. 3軸スコアリングフェーズ
     - Readiness（準備完了度）: 前提スキル充足率
     - Probability（習得確率）: 条件付き確率
     - Utility（有用性）: キャリア目標との関連性
  4. 統合・フィルタリングフェーズ（因果的制約）
  5. 最終出力（スマートロードマップ、グラフ可視化）

**学べること**:
- LiNGAM因果探索アルゴリズムの詳細
- 3軸スコアリングの計算方法
- 因果的フィルタリングによる実現可能性の保証
- 依存関係に基づくロードマップ生成

---

## 🛠️ PlantUML の使い方

### オンラインで表示

PlantUML ファイル（.puml）を以下のツールで表示できます：

1. **PlantUML Web Server**
   - https://www.plantuml.com/plantuml/uml/
   - ファイルの内容をコピー＆ペーストして表示

2. **VS Code 拡張機能**
   - [PlantUML 拡張機能](https://marketplace.visualstudio.com/items?itemName=jebbs.plantuml)をインストール
   - .puml ファイルを開いて `Alt+D` でプレビュー表示

3. **IntelliJ IDEA / PyCharm プラグイン**
   - PlantUML Integration プラグインをインストール
   - .puml ファイルを開くと自動でプレビュー表示

### ローカルで PNG/SVG に変換

```bash
# PlantUML のインストール（Java が必要）
# macOS
brew install plantuml

# Ubuntu/Debian
sudo apt-get install plantuml

# PNG に変換
plantuml docs/plantuml/*.puml

# SVG に変換（推奨：スケーラブル）
plantuml -tsvg docs/plantuml/*.puml

# 複数ファイルを一括変換
cd docs/plantuml
plantuml -tsvg *.puml
```

### Docker で変換

```bash
# Docker イメージを使用
docker run --rm -v $(pwd)/docs/plantuml:/data \
  plantuml/plantuml:latest -tsvg /data/*.puml
```

---

## 📊 図の用途マトリックス

| 図の種類 | 設計 | 実装 | レビュー | ドキュメント | 教育 |
|---------|------|------|---------|-------------|------|
| システムアーキテクチャ図 | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| モジュール構成図 | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| クラス図 | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| シーケンス図 | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| ユースケース図 | ⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| コンポーネント図 | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| アルゴリズムフロー図 | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |

---

## 🔄 図の更新方法

### 新機能追加時

1. 影響を受ける図を特定
2. .puml ファイルを編集
3. プレビューで確認
4. コミット時にこの README も更新

### 推奨される更新タイミング

- ✅ 新しいモジュールやクラスを追加したとき
- ✅ アーキテクチャを変更したとき
- ✅ 主要な処理フローを変更したとき
- ✅ 外部ライブラリの依存関係を変更したとき
- ✅ 新しいユースケースを追加したとき

### 図の一貫性を保つために

- 色スキームを統一（各図で同じレイヤー/モジュールは同じ色）
- 命名規則を統一（日本語/英語の混在を避ける）
- 定期的にコードと図の同期を確認

---

## 📚 参考資料

### PlantUML 公式ドキュメント

- [PlantUML 公式ドキュメント](https://plantuml.com/ja/)
- [PlantUML クラス図ガイド](https://plantuml.com/ja/class-diagram)
- [PlantUML シーケンス図ガイド](https://plantuml.com/ja/sequence-diagram)
- [PlantUML アクティビティ図ガイド](https://plantuml.com/ja/activity-diagram-beta)
- [PlantUML ユースケース図ガイド](https://plantuml.com/ja/use-case-diagram)
- [PlantUML コンポーネント図ガイド](https://plantuml.com/ja/component-diagram)

### CareerNavigator アーキテクチャドキュメント

**システム設計・構造**:
- [アーキテクチャドキュメント](../ARCHITECTURE.md) - システム全体のアーキテクチャ設計
- [コード構造ドキュメント](../CODE_STRUCTURE.md) - モジュール構成と責務分離
- [リファクタリングガイド](../REFACTORING_GUIDE.md) - エンタープライズパターンとSOLID原則

**推薦システム技術**:
- [ML技術詳細](../ML_TECHNICAL_DETAILS.md) - 機械学習アルゴリズムの詳細解説
- [モデル技術ガイド](../MODELS_TECHNICAL_GUIDE.md) - NMF、多様性戦略の実装
- [ハイブリッド推薦システム](../HYBRID_RECOMMENDATION_SYSTEM.md) - ハイブリッドアプローチの詳細
- [SEM実装サマリー](../SEM_IMPLEMENTATION_SUMMARY.md) - 構造方程式モデリング
- [キャリアパスSEMモデル](../CAREER_PATH_SEM_MODEL.md) - キャリアパス分析
- [SEM新機能](../NEW_SEM_FEATURES.md) - SEM関連の最新機能

**評価・品質**:
- [評価ドキュメント](../EVALUATION.md) - 推薦システムの評価手法とメトリクス
- [テスト設計書](../TEST_DESIGN.md) - テスト戦略とカバレッジ

**ユーザーガイド**:
- [初心者ガイド](../BEGINNER_GUIDE.md) - **推奨**: 初めての方はここから
- [クイックスタート](../QUICKSTART.md) - すぐに始めるための手順
- [Streamlitガイド](../STREAMLIT_GUIDE.md) - Webアプリケーションの使い方
- [テスト実行ガイド](../TESTING_QUICKSTART.md) - テストの実行方法

**運用・デプロイ**:
- [Streamlit Cloudセットアップ](../STREAMLIT_CLOUD_SETUP.md) - クラウドデプロイ手順
- [SEMスケーラビリティ分析](../SEM_SCALABILITY_ANALYSIS.md) - スケーラビリティ考慮事項

**API・開発者向け**:
- [APIリファレンス](../API_REFERENCE.md) - API仕様とエンドポイント
- [貢献ガイド](../../CONTRIBUTING.md) - コントリビューション方法

**モジュール別ドキュメント**:
- [Coreモジュール](../../skillnote_recommendation/core/README.md) - データローダー、変換器、評価器
- [MLモジュール](../../skillnote_recommendation/ml/README.md) - 機械学習推薦エンジン
- [Graphモジュール](../../skillnote_recommendation/graph/README.md) - グラフベース推薦

---

## 💡 Tips

### 図を読む順序（推奨）

新規参加者向けの推奨学習順序：

1. **ユースケース図** (05) → システムの目的と機能を理解
2. **システムアーキテクチャ図** (01) → 全体構造を理解
3. **モジュール構成図** (02) → モジュール間の関係を理解
4. **シーケンス図** (04) → 実行フローを理解
5. **クラス図** (03) → 詳細な設計を理解
6. **コンポーネント図** (06) → 技術スタックを理解
7. **アルゴリズムフロー図** (07) → アルゴリズムの詳細を理解

### 図を活用する場面

- **設計レビュー**: システムアーキテクチャ図、モジュール構成図、クラス図
- **コードレビュー**: クラス図、シーケンス図
- **新規参加者オンボーディング**: ユースケース図、システムアーキテクチャ図、シーケンス図
- **技術選定**: コンポーネント図
- **アルゴリズム改善**: アルゴリズムフロー図

---

## ✨ まとめ

これらの PlantUML 図は、CareerNavigator システムの**理解**、**設計**、**実装**、**保守**を支援するために作成されました。

- **可視化** により複雑なシステムを理解しやすくします
- **ドキュメント化** によりチーム全体で知識を共有します
- **設計の検証** により不整合を早期に発見します
- **コミュニケーション** を促進し、議論の質を向上させます

定期的に図を更新し、コードとドキュメントの同期を保つことをお勧めします。

---

**作成日**: 2025年

**最終更新**: 2025年

**メンテナ**: CareerNavigator Development Team
