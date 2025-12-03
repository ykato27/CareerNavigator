# CareerNavigator - スキルノート推薦システム

スキルノートのデータを基に、因果推論（Causal Inference）と機械学習を活用して技術者向けの最適なキャリアパスを提案する統合システムです。

## メインアプリケーション

本システムは2つのメインアプリケーションを提供します：

### 🎯 Streamlitアプリ（対話型Webアプリ）

ブラウザベースの対話的なWebアプリケーション。以下の機能を提供：

- **データ読み込み**: CSVファイルのドラッグ&ドロップアップロード、データ変換、品質チェック
- **因果推論ベース推薦**: 因果グラフによるスキル依存関係の可視化、3軸スコアリング（準備完了度・確率・有用性）
- **従業員キャリアダッシュボード**: ロールモデル/役職ベース目標設定、スマートキャリアロードマップ、ギャップ分析
- **組織スキルマップ**: 組織全体のスキル分布可視化、後継者計画、スキルギャップ分析

→ [Streamlitアプリ詳細](docs/STREAMLIT_APPS.md)

### 🌐 WebUI（モダンWebアプリ）

React + FastAPIによるモダンなWebアプリケーション：

- **フロントエンド**: React + TypeScript + Vite
- **バックエンド**: FastAPI + Pydantic
- **機能**: 従業員キャリアダッシュボード、リアルタイムデータ可視化、RESTful API

→ [WebUIガイド](docs/WEBUI_GUIDE.md)

## 主な機能

1. **因果推論ベースのスキル推薦**: 「どのスキルを習得すれば、次のスキル習得が容易になるか」という因果関係を考慮した推薦を行います。
2. **スマートキャリアロードマップ**: 依存関係を考慮したガントチャート形式の学習計画を自動生成します。
3. **キャリアパス比較**: 複数のキャリア目標（ロールモデルや職種）を並列で比較検討できます。
4. **3軸スコアリング**: 準備完了度 (Readiness)、習得確率 (Probability)、将来性 (Utility) の3つの観点からスキルを評価します。
5. **組織スキル分析**: 組織全体のスキル分布を可視化し、スキルギャップや後継者計画を支援します。

## 推薦アルゴリズム（WebUI）

### 因果推論ベース推薦：LiNGAMの仕組み

WebUIでは**LiNGAM (Linear Non-Gaussian Acyclic Model)** という因果推論手法を用いて、スキル間の因果関係を自動的に発見します。

#### なぜLiNGAMなのか？

従来の協調フィルタリング（「似た人が学んだスキル」を推薦）ではなく、**「このスキルを学ぶと、次にどのスキルが学びやすくなるか」という因果関係**を発見できます。

#### LiNGAMの3つのステップ

1. **スキル習得データの収集**  
   組織内の全メンバーの「どのスキルをどのレベルで習得しているか」というデータを分析します。

2. **因果グラフの自動生成**  
   LiNGAMアルゴリズムにより、「スキルA → スキルB」のような因果関係を統計的に推定します。  
   例: 「Python基礎 → Django → REST API」のような依存関係チェーンを自動発見

3. **因果グラフの可視化**  
   発見された因果関係を矢印付きグラフとして可視化し、学習経路を明確化します。

### 3段階スコアリング：スキルを多角的に評価

各推薦スキルを**3つの観点**から評価し、総合スコアを算出します：

#### 1. Readiness Score（準備完了度）: 0.0 ~ 1.0

**「今すぐ学べるか？」**

- **計算方法**: 因果グラフ上で、推薦スキルの前提スキル（親ノード）のうち、どれだけ既に習得済みか
- **例**: 
  - 「Django」の前提スキルが「Python基礎」「Web基礎」の2つ
  - あなたが「Python基礎」のみ習得済み → Readiness = 0.5 (50%)
  - 両方習得済み → Readiness = 1.0 (100%) ✅ 今すぐ学習可能
- **意味**: 高いほど「前提知識が揃っていて学びやすい」

####  2. Probability Score（習得確率）: 0.0 ~ 1.0

**「組織では一般的に学びやすいスキルか？」**

- **計算方法**: 組織内のデータから、類似のバックグラウンドを持つメンバーがこのスキルを習得した確率
- **例**:
  - 組織の80%のPythonエンジニアが「Django」を習得 → Probability = 0.8
  - 希少スキルで10%しか習得していない → Probability = 0.1
- **意味**: 高いほど「一般的に習得されやすい標準的なスキル」

#### 3. Utility Score（有用性）: 0.0 ~ 1.0

**「キャリア目標達成にどれだけ役立つか？」**

- **計算方法**: 目標とするロールモデルや職種が持つスキルとの関連度
- **例**:
  - 目標が「シニアエンジニア」で、そのロールが「Kubernetes」を必須保有 → Utility = 1.0
  - 目標達成に直接は不要なスキル → Utility = 0.3
- **意味**: 高いほど「キャリア目標達成に重要」

### 総合スコアの計算

```
総合スコア = (Readiness^0.4) × (Probability^0.3) × (Utility^0.3)
```

**重み付けの意味**:
- **Readiness優先**: 「学べる状態にある」ことを最重視（指数0.4）
- **Probability/Utility**: 「一般的な学びやすさ」と「目標への有用性」をバランス良く考慮（各0.3）

### 推薦結果の例

| スキル | Readiness | Probability | Utility | 総合スコア | 説明 |
|--------|-----------|-------------|---------|-----------|------|
| Django | 1.0 | 0.8 | 0.9 | **0.90** | ✅ 前提揃い、一般的、有用 → 最優先推薦 |
| Kubernetes | 0.3 | 0.5 | 1.0 | 0.51 | ⚠️ 前提不足 → まだ早い |
| React | 0.8 | 0.9 | 0.4 | 0.67 | 💡 学びやすいが目標への関連度は中程度 |

→ 詳細は [WEBUI_GUIDE.md](docs/WEBUI_GUIDE.md) を参照

## プロジェクト構成

```
CareerNavigator/
├── data/                          # 入力データ（CSVファイル）
│   ├── members/                   # メンバーデータ（複数CSVファイル対応）
│   │   ├── member_1.csv
│   │   ├── member_2.csv
│   │   └── ...                    # ディレクトリ内の全CSVを自動読込・結合
│   ├── acquired/                  # 習得力量データ（複数ファイル対応）
│   ├── skills/                    # スキル力量データ
│   ├── education/                 # 教育力量データ
│   ├── license/                   # 資格力量データ
│   └── categories/                # カテゴリデータ
│
├── output/                        # 出力データ（変換後のCSV）
│   ├── members_clean.csv
│   ├── competence_master.csv
│   ├── member_competence.csv
│   ├── skill_matrix.csv
│   └── competence_similarity.csv
│
├── streamlit_app.py               # Streamlitメインページ
├── pages/                         # Streamlitページ
│   ├── 1_Causal_Recommendation.py           # 因果推論推薦
│   ├── 2_Employee_Career_Dashboard.py       # 従業員キャリアダッシュボード
│   └── 3_Organizational_Skill_Map.py        # 組織スキルマップ
│
├── frontend/                      # WebUI フロントエンド
│   ├── src/                       # Reactアプリケーション
│   │   ├── pages/                 # ページコンポーネント
│   │   ├── components/            # 再利用可能コンポーネント
│   │   └── App.tsx                # メインアプリ
│   └── package.json               # Node.js依存関係
│
├── backend/                       # WebUI バックエンドAPI
│   ├── api/                       # APIエンドポイント
│   │   ├── career_dashboard.py    # キャリアダッシュボードAPI
│   │   └── role_based_dashboard.py # 役職ベース分析API
│   ├── utils/                     # ユーティリティ
│   └── main.py                    # FastAPIアプリ
│
├── pages_disabled/                # アーカイブされたページ（旧バージョン）
│
├── skillnote_recommendation/      # パッケージ
│   ├── __init__.py
│   ├── core/                      # コアモジュール → [詳細](skillnote_recommendation/core/README.md)
│   │   ├── config.py              # 設定管理
│   │   ├── models.py              # データモデル
│   │   ├── data_loader.py         # データ読み込み
│   │   ├── data_transformer.py    # データ変換
│   │   ├── reference_persons.py   # 参考人物検索
│   │   └── evaluator.py           # 評価器（時系列分割・メトリクス・多様性評価）
│   ├── ml/                        # 機械学習モジュール → [詳細](skillnote_recommendation/ml/README.md)
│   │   ├── __init__.py
│   │   ├── matrix_factorization.py  # Matrix Factorizationモデル
│   │   ├── diversity.py           # 多様性再ランキング
│   │   └── ml_recommender.py      # ML推薦システム
│   ├── graph/                     # 因果グラフモジュール → [詳細](skillnote_recommendation/graph/README.md)
│   │   ├── lingam_recommender.py  # LiNGAM因果推論
│   │   └── career_path.py         # キャリアパス生成
│   └── scripts/                   # 実行スクリプト（現在未使用）
│
├── tests/                         # テストコード（Phase 2完了: 81+テスト、コア層100%カバレッジ）
│   ├── backend/                   # WebUI バックエンドテスト
│   │   ├── test_services_*.py     # Servicesレイヤー（96-100%）
│   │   ├── test_repositories_*.py # Repositoriesレイヤー（100%）
│   │   ├── test_schemas_*.py      # Schemasレイヤー（100%）
│   │   ├── test_middleware_*.py   # Middlewareレイヤー（100%）
│   │   └── test_api_*.py          # API統合テスト（進行中）
│   └── その他                      # コアライブラリテスト（レガシー、整理予定）
│   ├── test_data_loader.py
│   ├── test_directory_scan.py
│   ├── test_data_transformer.py
│   ├── test_evaluator.py
│   ├── test_matrix_factorization.py
│   ├── test_diversity.py
│   ├── test_ml_recommender.py
│   ├── test_knowledge_graph.py
│   ├── test_hyperparameter_tuning.py
│   ├── test_feature_engineering.py
│   └── test_visualization.py
│
├── docs/                          # ドキュメント → [一覧](docs/README.md)
│   ├── plantuml/                  # PlantUML図（アーキテクチャ可視化）
│   ├── ARCHITECTURE.md            # システムアーキテクチャ
│   ├── CODE_STRUCTURE.md          # コード構造ガイド
│   ├── BEGINNER_GUIDE.md          # 初心者向けガイド
│   ├── QUICKSTART.md              # クイックスタート
│   ├── STREAMLIT_GUIDE.md         # Streamlit UIガイド
│   ├── ML_TECHNICAL_DETAILS.md    # ML技術詳細
│   ├── EVALUATION.md              # 評価ガイド
│   ├── TESTING_QUICKSTART.md      # テスト実装ガイド
│   └── ... 他多数                 # SEM、ハイブリッド推薦システム等
│
├── pyproject.toml                 # プロジェクト設定
├── .gitignore
└── README.md
```

## 環境構築（uv使用）

### 前提条件

- **Python 3.11以上** （推奨: Python 3.12 または 3.13）
- uv（Pythonパッケージマネージャー）

**注意**: Python 3.10以下はサポートしていません。Python 3.11以降は性能が大幅に向上しており（3.9比で25%以上）、長期サポートが提供されます。

### uvのインストール

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# pipxを使用（すでにPythonがインストールされている場合）
pipx install uv
```

### Pythonバージョンの確認・インストール

uvを使えば、必要なPythonバージョンを自動的にインストールできます：

```bash
# 現在のPythonバージョンを確認
python --version

# Python 3.12以降を推奨（3.11でも動作）
cd CareerNavigator
uv python install 3.12  # または 3.13

# システムにPython 3.11以上がインストール済みの場合はそれを使用
# uv sync が自動的に検出します
```

### プロジェクトのセットアップ

```bash
# 1. プロジェクトをクローンまたはダウンロード
cd CareerNavigator

# 2. 依存関係のインストール（自動的にPython 3.11+の仮想環境も作成されます）
uv sync

# 3. CSVファイルをdataディレクトリに配置
mkdir -p data/members data/acquired data/skills data/education data/license data/categories
cp /path/to/member_*.csv data/members/
cp /path/to/acquired_*.csv data/acquired/
cp /path/to/skill_*.csv data/skills/
cp /path/to/education_*.csv data/education/
cp /path/to/license_*.csv data/license/
cp /path/to/category_*.csv data/categories/
```

### データファイル配置方法

**重要**: v1.2.0から、データファイルは各サブディレクトリに配置する必要があります。

#### 既存プロジェクトからの移行

以前のバージョンで `data/` 直下にCSVファイルを配置していた場合、以下のコマンドで移行してください：

```bash
# 既存のCSVファイルを適切なディレクトリに移動
mv data/member_*.csv data/members/ 2>/dev/null || true
mv data/acquired*.csv data/acquired/ 2>/dev/null || true
mv data/skill_*.csv data/skills/ 2>/dev/null || true
mv data/education_*.csv data/education/ 2>/dev/null || true
mv data/license_*.csv data/license/ 2>/dev/null || true
mv data/*category*.csv data/categories/ 2>/dev/null || true
```

#### ディレクトリ構造

複数のCSVファイルがある場合、各種別ごとにディレクトリに配置すると自動的に読み込み・結合されます：

```
data/
├── members/          # メンバーデータディレクトリ
│   ├── member_dept_a.csv    # 部署Aのメンバー
│   ├── member_dept_b.csv    # 部署Bのメンバー
│   └── member_dept_c.csv    # 部署Cのメンバー
├── acquired/         # 習得力量データディレクトリ
│   ├── acquired_2024.csv    # 2024年のデータ
│   └── acquired_2025.csv    # 2025年のデータ
├── skills/           # スキル力量データディレクトリ
│   └── skills.csv
├── education/        # 教育力量データディレクトリ
│   └── education.csv
├── license/          # 資格力量データディレクトリ
│   └── license.csv
└── categories/       # カテゴリデータディレクトリ
    └── categories.csv
```

各ディレクトリ内の**全ての.csvファイル**が自動的に読み込まれ、1つのDataFrameに結合されます。

#### カラム構造の整合性

同じディレクトリ内の複数CSVファイルは、すべて同じカラム構造を持つ必要があります。
カラムが異なるファイルが含まれている場合、読み込み時にエラーが発生します

#### データ保護

**重要**: CSVデータファイルはgitリポジトリに含まれません（機密情報保護のため）。

- `.gitignore` により `data/**/*.csv` および `output/*.csv` は除外されています
- ディレクトリ構造のみが `.gitkeep` ファイルで保持されます
- 実際のデータファイルは各自で配置してください
- チーム内でデータを共有する場合は、安全な方法（暗号化、アクセス制限）を使用してください

## 使い方

### 🌐 Streamlit Webアプリ（推奨）

ブラウザベースの対話的なWebアプリケーションで推薦システムを利用できます。

```bash
# Streamlitアプリの起動
uv run streamlit run streamlit_app.py
```

ブラウザが自動的に開き、以下の機能が利用できます：

- ✅ CSVファイルのドラッグ&ドロップアップロード
- ✅ 既存メンバー・新規ユーザーへの推薦
- ✅ 機械学習ベースの推薦システム
- ✅ ロールモデル（参考となるメンバー）の表示
- ✅ 多様性メトリクスの可視化
- ✅ 推薦結果のCSVダウンロード

**詳細**: [Streamlitアプリガイド](docs/STREAMLIT_GUIDE.md)

### Pythonコードから利用

#### 機械学習ベース推薦システム

```python
from skillnote_recommendation.ml import MLRecommender
from skillnote_recommendation.core.data_loader import DataLoader

# データ読み込み
loader = DataLoader()
data = loader.load_all_data()

# ML推薦システム初期化
ml_recommender = MLRecommender(data)

# メンバーm48への推薦
recommendations = ml_recommender.recommend(
    member_code='m48',
    top_n=10,
    use_diversity=True,           # 多様性再ランキング有効化
    diversity_strategy='hybrid'   # hybrid/mmr/category/type
)

# 結果表示
for rec in recommendations:
    print(f"{rec['力量名']}: スコア {rec['MLスコア']:.3f} - {rec['推薦理由']}")

# SKILLタイプのみ + 多様性重視
skill_recommendations = ml_recommender.recommend(
    member_code='m48',
    top_n=10,
    competence_type='SKILL',
    diversity_strategy='mmr'
)

# 多様性メトリクス計算
diversity_metrics = ml_recommender.calculate_diversity_metrics(
    recommendations,
    ml_recommender.competence_master
)
print(f"カテゴリ多様性: {diversity_metrics['category_diversity']:.3f}")
print(f"タイプ多様性: {diversity_metrics['type_diversity']:.3f}")
```

### インタラクティブシェルで実行

```bash
# uvのシェルを起動
uv run python

# Pythonシェル内で
>>> from skillnote_recommendation import RecommendationSystem
>>> system = RecommendationSystem()
>>> system.print_recommendations('m48', top_n=5)
```

## 開発環境のセットアップ

### 開発用依存関係のインストール

```bash
# 開発用ツールを含めてインストール
uv sync --all-extras

# または、devグループのみ追加
uv sync --extra dev
```

### コード品質チェック

```bash
# フォーマット（Black）
uv run black skillnote_recommendation/

# Lint（Flake8）
uv run flake8 skillnote_recommendation/

# 型チェック（mypy）
uv run mypy skillnote_recommendation/
```

### テスト実行

```bash
# テスト実行
uv run pytest

# カバレッジ付きテスト
uv run pytest --cov=skillnote_recommendation
```

## プロジェクト管理

### 依存関係の追加

```bash
# 本番依存関係を追加
uv add pandas numpy scikit-learn

# 開発依存関係を追加
uv add --dev pytest black flake8
```

### 依存関係の更新

```bash
# すべての依存関係を更新
uv sync --upgrade

# 特定のパッケージのみ更新
uv add pandas --upgrade
```

### uvによる仮想環境管理

**uvは自動的に仮想環境を管理します。** 手動でアクティベート/ディアクティベートする必要はありません。

```bash
# uvが自動的に.venv内で実行（アクティベート不要）
uv run python script.py
uv run pytest
uv run streamlit run streamlit_app.py

# 必要に応じて仮想環境を再作成
uv venv --clear
`skillnote_recommendation/core/config.py` でMLモデルのパラメータを調整:

```python
MF_PARAMS = {
    'n_components': 20,  # 潜在因子数
    'random_state': 42,
    'max_iter': 200,
    'use_confidence_weighting': False,
    'confidence_alpha': 1.0
}
```

## トラブルシューティング

### エラー: ファイルが見つかりません

```
FileNotFoundError: /path/to/data/member_skillnote.csv が見つかりません
```

**対処法**: 必要なCSVファイルを `data/` ディレクトリに配置してください。

### エラー: 変換済みデータが見つかりません

```
エラー: 変換済みデータが見つかりません
```

**対処法**: 先に `skillnote-convert` を実行してください。

```bash
uv run skillnote-convert
```

### 推薦結果が空

**原因**: 既に全ての力量を習得している可能性があります。

**対処法**: 他のメンバーコードで試すか、フィルタを変更してください。

### エラー: Windows/OneDrive環境での同期エラー

```
error: Failed to install: ... (os error 396)
Caused by: failed to hardlink file ...
```

**原因**: OneDriveフォルダ内では、`uv` がデフォルトで使用するハードリンク作成機能が制限される場合があります。

**対処法**:
1. `pyproject.toml` に以下の設定を追加します（本プロジェクトでは設定済み）:
   ```toml
   [tool.uv]
   link-mode = "copy"
   ```

2. 仮想環境が破損している場合は、再作成します:
   ```bash
   # 既存の環境をクリアして再作成
   uv venv --clear
   uv sync
   ```

## 実行例

### データ変換

```bash
$ uv run skillnote-convert
================================================================================
スキルノート データ変換処理
================================================================================

入力ディレクトリ: /path/to/data
出力ディレクトリ: /path/to/output
================================================================================
データ読み込み
================================================================================
  ✓ member_skillnote.csv: 229行
  ✓ acquiredCompetenceLevel.csv: 6002行
  ...

（中略）

================================================================================
データ変換処理完了
================================================================================

変換データは /path/to/output/ に保存されました
```

### 推薦実行

```bash
$ uv run skillnote-recommend
================================================================================
スキルノート 推薦システム
================================================================================

================================================================================
推薦システム初期化
================================================================================

  メンバー数: 228
  力量数: 423
  習得記録数: 6002
  初期化完了

================================================================================
サンプル実行1: 全タイプの力量を推薦
================================================================================

================================================================================
力量推薦結果
================================================================================
メンバー: 黒崎 国彦 (m48)
役職: 未設定
職能等級: 3等級

保有力量: SKILL 42件 / EDUCATION 20件 / LICENSE 0件
================================================================================

推薦力量 （全タイプ）（上位10件）:

【推薦 1】 図面の読み取り
  タイプ: SKILL
  カテゴリ: 製造部 > 製造部共通力量
  優先度スコア: 7.32
  推薦理由: ...
```

## ビルドと配布

### パッケージのビルド

```bash
# wheelとsdistを作成
uv build
```

### ローカルインストール

```bash
# 編集可能モードでインストール
uv pip install -e .
```

## ドキュメント

### 🎓 初心者向けドキュメント

- **[WebUI社内共有用ガイド (docs/WEBUI_INTRODUCTION_FOR_TEAM.md)](docs/WEBUI_INTRODUCTION_FOR_TEAM.md)** ⭐ **社内共有に最適！**
  - 取り組みの背景と解決したい課題
  - 因果推論ベース推薦システムの説明（初学者向け）
  - WebUIで提供している機能の詳細
  - 実際の使い方（シナリオ別）
  - よくある質問と回答

- **[初心者向けコード理解ガイド (docs/BEGINNER_GUIDE.md)](docs/BEGINNER_GUIDE.md)** ⭐ **開発者向け！**
  - どのファイルが実行ファイルで、どのファイルがライブラリコードか
  - 各ファイルの役割を分かりやすく説明
  - コードの読み方・学習の進め方
  - よくある質問と回答

### 📚 開発者向けドキュメント

- **[PlantUML図 (docs/plantuml/)](docs/plantuml/)** ⭐ **視覚的理解に最適！**
  - システムアーキテクチャ図（4層構造）
  - モジュール構成図（Core/ML/Graph/Utils）
  - クラス図（データモデル、推薦エンジン、アルゴリズム）
  - シーケンス図（推薦プロセスフロー）
  - ユースケース図（ユーザーロールと機能）
  - コンポーネント図（技術スタック）
  - アルゴリズムフロー図（ハイブリッド推薦の処理フロー）

- **[リファクタリングガイド (docs/REFACTORING_GUIDE.md)](docs/REFACTORING_GUIDE.md)** - エンタープライズグレード機能の実装ガイド
  - 構造化ロギング、エラーハンドリング、リトライロジック
  - Config管理、インターフェース定義
  - 入力バリデーション、データ品質チェック
  - マイグレーションガイドと使用例

- **[アーキテクチャドキュメント (docs/ARCHITECTURE.md)](docs/ARCHITECTURE.md)** - システム設計ドキュメント
  - アーキテクチャ概要
  - モジュール構成
  - データフロー

- **[コード構造ガイド (docs/CODE_STRUCTURE.md)](docs/CODE_STRUCTURE.md)** - プロジェクト構造とモジュール説明
  - Streamlitアプリケーション構造
  - コーディング規約とパターン
  - テスト構造と拡張ガイド

- **[APIリファレンス (docs/API_REFERENCE.md)](docs/API_REFERENCE.md)** - API使用方法
  - 構造化ロギング
  - エラーハンドリング
  - Config管理
  - データバリデーション

- **[コントリビューションガイド (CONTRIBUTING.md)](CONTRIBUTING.md)** - 開発者向けガイド
  - 開発環境のセットアップ
  - コーディング規約
  - プルリクエストのガイドライン

### 📖 ユーザー向けドキュメント

- **[Streamlitアプリガイド (docs/STREAMLIT_GUIDE.md)](docs/STREAMLIT_GUIDE.md)** - Webアプリケーション完全ガイド
  - 起動方法と使い方
  - 既存メンバー・新規ユーザーへの推薦
  - ロールモデル機能
  - 多様性メトリクスの見方
  - CSVダウンロード
  - トラブルシューティング

- **[クイックスタート (docs/QUICKSTART.md)](docs/QUICKSTART.md)** - コマンドライン使い方ガイド

- **[データ永続化クイックスタート (docs/PERSISTENCE_QUICKSTART.md)](docs/PERSISTENCE_QUICKSTART.md)** - データ永続化機能の使い方

### 🔬 技術ドキュメント

- **[モデル技術ガイド (docs/MODELS_TECHNICAL_GUIDE.md)](docs/MODELS_TECHNICAL_GUIDE.md)** ⭐ **初級者向け！**
  - Matrix Factorization（行列分解）の仕組みと使い方
  - 多様性再ランキング戦略の選び方
  - グラフベース推薦とハイブリッド推薦
  - モデル選択ガイドとFAQ

- **[機械学習技術詳細 (docs/ML_TECHNICAL_DETAILS.md)](docs/ML_TECHNICAL_DETAILS.md)** - ML推薦システムの技術解説
  - Matrix Factorization (NMF)
  - 多様性再ランキング戦略
  - 評価メトリクス
  - パラメータチューニング

- **[ハイブリッド推薦システム (docs/HYBRID_RECOMMENDATION_SYSTEM.md)](docs/HYBRID_RECOMMENDATION_SYSTEM.md)** - ハイブリッドアプローチの詳細
  - グラフベース推薦（RWR）
  - 協調フィルタリング（NMF）
  - コンテンツベース推薦
  - スコア統合と重み付け

- **[データ永続化 (docs/PERSISTENCE.md)](docs/PERSISTENCE.md)** - データベース設計と永続化アーキテクチャ

- **[評価ガイド (docs/EVALUATION.md)](docs/EVALUATION.md)** - 推薦システムの評価方法
  - 時系列分割による評価
  - 評価メトリクス (Precision@K, Recall@K, NDCG@K, Hit Rate)
  - 多様性評価メトリクス
  - クロスバリデーション

### 🧪 テストドキュメント

- **[テスト設計 (docs/TEST_DESIGN.md)](docs/TEST_DESIGN.md)** - テストコードの設計書（100+テストケース）

- **[テスト実装ガイド (docs/TESTING_QUICKSTART.md)](docs/TESTING_QUICKSTART.md)** - テスト実装手順

### 📊 プレゼンテーション

- **[機械学習推薦システム - 営業向け資料 (docs/ML_PRESENTATION_SALES.md)](docs/ML_PRESENTATION_SALES.md)** - 非技術者向けプレゼンテーション

### 🛠️ その他

- **[新規ユーザーCSVテンプレート (templates/)](templates/)** - 新規ユーザー用CSVテンプレートと使い方

## バージョン履歴

- **v1.2.1 (2025-11-07)** - **テストカバレッジ大幅向上とドキュメント拡充**
  - **テストカバレッジ向上** (30% → 47%)
    - 156個の新規テスト追加（253 → 393テスト）
    - 4つの新規テストファイル追加
      - tests/test_knowledge_graph.py - グラフベース推薦のテスト
      - tests/test_hyperparameter_tuning.py - ハイパーパラメータ調整のテスト
      - tests/test_feature_engineering.py - 特徴量エンジニアリングのテスト
      - tests/test_visualization.py - 可視化機能のテスト
  - **ドキュメント拡充**
    - MODELS_TECHNICAL_GUIDE.md追加（初級者向けモデル技術ガイド）
    - 既存ドキュメント5ファイル更新
      - BEGINNER_GUIDE.md - テスト情報と新規ドキュメントへの参照追加
      - EVALUATION.md - 評価手法の詳細説明追加
      - ML_TECHNICAL_DETAILS.md - 機械学習技術の詳細解説更新
      - CODE_STRUCTURE.md - コード構造ガイド更新
      - TESTING_QUICKSTART.md - テスト実装ガイド更新

- **v1.2.0 (2025-10-24)** - **機械学習推薦システム + エンタープライズ機能強化**
  - **機械学習ベース推薦システム追加** (skillnote_recommendation/ml/)
    - Matrix Factorization (NMF) による協調フィルタリング
    - 多様性再ランキング (MMR, Category, Type, Hybrid)
    - MLRecommender: ML推薦システムの統合インターフェース
  - **エンタープライズグレードの機能追加**
    - 構造化ロギング（structlog + JSON出力）
    - エラーハンドリング体系化（エラーコード体系、リトライロジック）
    - Config管理リファクタリング（不変設計、環境分離）
    - 入力バリデーション強化（Pydantic v2）
    - データ品質チェック導入
    - モデル保存形式改善（joblib + JSON、バージョン管理）
    - インターフェース定義（Protocol、ABC）
    - データ正規化・バリデーションユーティリティ
  - **多様性評価メトリクス追加**
    - カテゴリ多様性、タイプ多様性、カバレッジ、リスト内多様性
    - evaluate_with_diversity() による統合評価
  - **データ入力方式の一本化**
    - ディレクトリ構造のみをサポート（単一ファイル対応を削除）
    - data/members/, data/acquired/ など各ディレクトリに複数CSVを配置
  - **データ保護強化**
    - .gitignoreでCSV/Excelファイルを除外（機密情報保護）
    - .gitkeepでディレクトリ構造のみ保持
  - **Pythonバージョン要件更新**
    - Python 3.11以上を要求（3.9はEOL間近のため）
    - 性能向上（3.9比で25%以上高速）
  - **テストスイート拡充** (253テスト)
    - ML関連テスト追加 (matrix_factorization, diversity, ml_recommender)
    - ディレクトリスキャン機能の包括的テスト
  - **ドキュメント整理**
    - docs/ディレクトリに全ドキュメントを集約
    - README更新（ML使用例、多様性戦略の説明）
    - データ保護・移行ガイドの追加

- v1.1.0 (2025-10-23)
  - 推薦システム評価機能追加
  - 時系列分割による評価 (Temporal Split)
  - 評価メトリクス実装 (Precision@K, Recall@K, NDCG@K, Hit Rate)
  - クロスバリデーション機能
  - ディレクトリスキャンによる複数CSV対応
  - カラム構造検証機能
  - 包括的テストスイート (194テスト)

- **v1.3.0 (2025-11-21)** - **MVP完成: Causal Recommendation & キャリア支援機能**
  - **因果推論ベース推薦システムの実装**
    - LiNGAMによる因果グラフ構築
    - 3軸スコアリング (Readiness, Probability, Utility)
    - 依存関係を考慮したスキル推薦
  - **従業員向けキャリアダッシュボード**
    - スマートロードマップ（ガントチャート）
    - キャリアパス比較機能（Decision Support）
    - インタラクティブな因果グラフ可視化
  - **リファクタリングと整備**
    - 設定の外部化 (settings.py)
    - 共通UIコンポーネントの整備
    - プロジェクト構成のクリーンアップ

- v1.2.1 (2025-11-07)
