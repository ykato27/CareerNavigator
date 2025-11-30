
## 🎨 図の詳細説明

### 01. システムアーキテクチャ図

**対象読者**: アーキテクト、テックリード、新規参加者

**概要**:
- CareerNavigator の4層アーキテクチャを可視化
- プレゼンテーション層（Streamlit UI）
- アプリケーション層（ビジネスロジック）
- ドメイン層（ML/Graph アルゴリズム）
- インフラストラクチャ層（データアクセス、永続化、ログ）

**学べること**:
- システム全体の構造
- レイヤー間の責務分離
- Dependency Inversion の実装

---

### 02. モジュール構成図

**対象読者**: 開発者、アーキテクト

**概要**:
- `skillnote_recommendation` パッケージ内の各モジュール構成
- core (基盤)、ml (機械学習)、graph (グラフ)、utils (ユーティリティ) モジュール
- 各モジュール内のコンポーネントと依存関係

**学べること**:
- モジュール間の依存方向
- SOLID 原則の適用
- 再利用可能なコンポーネント設計

---

### 03. クラス図

**対象読者**: 開発者

**概要**:
- データモデル（Member, Competence, MemberCompetence, Recommendation）
- 推薦エンジンの抽象クラスと実装クラス
- 機械学習コンポーネント（NMF, Diversity Strategy）
- グラフコンポーネント（KnowledgeGraph, RandomWalk）
- 評価・ユーティリティクラス

**学べること**:
- ドメインモデルの設計
- Strategy パターン（多様性戦略）
- Builder パターン（ハイブリッド推薦器）
- Repository パターン（データアクセス）

---

### 04. シーケンス図

**対象読者**: 開発者、QA エンジニア

**概要**:
- 推薦システムの実行フロー
  1. データ読み込みフェーズ
  2. モデル学習フェーズ（並行学習）
  3. 推薦生成フェーズ（並行推薦）
  4. 結果表示フェーズ
  5. 評価フェーズ

**学べること**:
- システムの動的な振る舞い
- コンポーネント間の相互作用
- 並行処理の設計
- エラーハンドリングの流れ

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
- フロントエンド（Streamlit, Plotly）
- 機械学習ライブラリ（scikit-learn, Optuna）
- グラフ処理ライブラリ（NetworkX, node2vec）
- データ永続化（SQLite, CSV, joblib）
- 共通インフラ（structlog, pydantic, tenacity）
- 開発ツール（pytest, black, mypy）

**学べること**:
- 外部ライブラリの依存関係
- データフローとストレージ戦略
- ログと設定管理
- 品質管理ツールチェーン

---

### 07. アルゴリズムフロー図

**対象読者**: データサイエンティスト、開発者

**概要**:
- ハイブリッド推薦アルゴリズムの詳細フロー
  1. グラフベース推薦（Random Walk with Restart）- 40%
  2. 協調フィルタリング（NMF）- 30%
  3. コンテンツベース推薦 - 30%
  4. スコア統合
  5. 多様性再ランキング（4戦略）
  6. ロールモデル検索
  7. 推薦理由生成

**学べること**:
- 各推薦アルゴリズムの詳細
- スコア統合の重み付け
- 多様性戦略（MMR/カテゴリ/タイプ/ハイブリッド）
- 説明可能性の実装

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
