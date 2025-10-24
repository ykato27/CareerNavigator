# スキルノート推薦システム

スキルノートのデータを基に、技術者向けの力量推薦を行うシステムです。

## プロジェクト構成

```
CareerNavigator/
├── data/                          # 入力データ（CSVファイル）
│   ├── members/                   # 会員データ（複数CSVファイル対応）
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
├── skillnote_recommendation/      # パッケージ
│   ├── __init__.py
│   ├── core/                      # コアモジュール（ルールベース推薦）
│   │   ├── config.py              # 設定管理
│   │   ├── models.py              # データモデル
│   │   ├── data_loader.py         # データ読み込み
│   │   ├── data_transformer.py    # データ変換
│   │   ├── similarity_calculator.py  # 類似度計算
│   │   ├── recommendation_engine.py  # ルールベース推薦エンジン
│   │   ├── recommendation_system.py  # 推薦システム
│   │   └── evaluator.py           # 評価器（時系列分割・メトリクス・多様性評価）
│   ├── ml/                        # 機械学習モジュール
│   │   ├── __init__.py
│   │   ├── matrix_factorization.py  # Matrix Factorizationモデル
│   │   ├── diversity.py           # 多様性再ランキング
│   │   └── ml_recommender.py      # ML推薦システム
│   └── scripts/                   # 実行スクリプト
│       ├── convert_data.py        # データ変換
│       └── run_recommendation.py  # 推薦実行
│
├── tests/                         # テストコード（253テスト）
│   ├── test_basic.py
│   ├── test_data_loader.py
│   ├── test_directory_scan.py
│   ├── test_data_transformer.py
│   ├── test_similarity_calculator.py
│   ├── test_recommendation_engine.py
│   ├── test_recommendation_system.py
│   ├── test_evaluator.py
│   ├── test_matrix_factorization.py
│   ├── test_diversity.py
│   └── test_ml_recommender.py
│
├── docs/                          # ドキュメント
│   ├── EVALUATION.md              # 評価ガイド
│   ├── QUICKSTART.md              # クイックスタート
│   ├── TESTING_QUICKSTART.md      # テスト実装ガイド
│   └── TEST_DESIGN.md             # テスト設計書
│
├── pyproject.toml                 # プロジェクト設定
├── .gitignore
└── README.md
```

## 環境構築（uv使用）

### 前提条件

- Python 3.9以上
- uv（Pythonパッケージマネージャー）

### uvのインストール

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# pipxを使用（すでにPythonがインストールされている場合）
pipx install uv
```

### プロジェクトのセットアップ

```bash
# 1. プロジェクトをクローンまたはダウンロード
cd CareerNavigator

# 2. 依存関係のインストール（自動的に仮想環境も作成されます）
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

複数のCSVファイルがある場合、各種別ごとにディレクトリに配置すると自動的に読み込み・結合されます：

```
data/
├── members/          # 会員データディレクトリ
│   ├── member_dept_a.csv    # 部署Aの会員
│   ├── member_dept_b.csv    # 部署Bの会員
│   └── member_dept_c.csv    # 部署Cの会員
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

## 使い方

### コマンドラインから実行

```bash
# データ変換
uv run skillnote-convert

# 推薦実行
uv run skillnote-recommend
```

または、直接Pythonモジュールとして実行:

```bash
# データ変換
uv run python -m skillnote_recommendation.scripts.convert_data

# 推薦実行
uv run python -m skillnote_recommendation.scripts.run_recommendation
```

### Pythonコードから利用

#### ルールベース推薦システム

```python
from skillnote_recommendation import RecommendationSystem

# 推薦システム初期化
system = RecommendationSystem()

# 特定の会員に推薦
system.print_recommendations('m48', top_n=10)

# SKILLタイプのみ推薦
recommendations = system.recommend_competences(
    'm48',
    competence_type='SKILL',
    top_n=5
)

for rec in recommendations:
    print(f"{rec.competence_name}: {rec.priority_score:.2f}")

# CSV出力
system.export_recommendations('m48', 'recommendations_m48.csv', top_n=20)
```

#### 機械学習ベース推薦システム

```python
from skillnote_recommendation.ml import MLRecommender
from skillnote_recommendation.core.data_loader import DataLoader

# データ読み込み
loader = DataLoader()
data = loader.load_all_data()

# ML推薦システム初期化
ml_recommender = MLRecommender(data)

# 会員m48への推薦
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

### 仮想環境の操作

```bash
# 仮想環境をアクティベート
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# 仮想環境をディアクティベート
deactivate
```

## 推薦アルゴリズム

本システムは**ルールベース**と**機械学習ベース**の2つの推薦手法を提供します。

### ルールベース推薦

```
優先度スコア = (カテゴリ重要度 × 0.4) + (習得容易性 × 0.3) + (人気度 × 0.3)
```

#### 評価要素

1. **カテゴリ重要度**: カテゴリ内での習得者数に基づく重要度
2. **習得容易性**: 類似力量を保有している場合の習得しやすさ
3. **人気度**: 全体での習得率

### 機械学習ベース推薦

#### Matrix Factorization (協調フィルタリング)

- **手法**: NMF (Non-negative Matrix Factorization)
- **入力**: 会員×力量マトリクス（習得=1, 未習得=0）
- **出力**: 未習得力量に対する予測スコア
- **潜在因子数**: 20次元（デフォルト）
- **特徴**: 会員の習得パターンから類似会員の習得傾向を学習

#### 多様性再ランキング

ML予測スコアに基づく推薦結果を多様性の観点から再ランキング：

1. **MMR (Maximal Marginal Relevance)**
   - 関連度と多様性のバランスを取る
   - λパラメータで調整可能（デフォルト=0.7）

2. **カテゴリ多様性**
   - 異なるカテゴリから推薦
   - カテゴリごとの上限設定可能

3. **タイプ多様性**
   - SKILL/EDUCATION/LICENSE をバランス良く推薦
   - タイプ比率の指定可能

4. **ハイブリッド戦略**
   - 上記3手法を組み合わせた総合的な多様性確保

## カスタマイズ

### パラメータ調整

`skillnote_recommendation/core/config.py` の `RECOMMENDATION_PARAMS` で調整:

```python
RECOMMENDATION_PARAMS = {
    'category_importance_weight': 0.4,
    'acquisition_ease_weight': 0.3,
    'popularity_weight': 0.3,
    'similarity_threshold': 0.3,
    'similarity_sample_size': 100
}
```

### 推薦エンジンの拡張

```python
from skillnote_recommendation.core.recommendation_engine import RecommendationEngine

class CustomEngine(RecommendationEngine):
    def calculate_custom_score(self, member_code, competence_code):
        # カスタムロジック
        pass
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

**対処法**: 他の会員コードで試すか、フィルタを変更してください。

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

  会員数: 228
  力量数: 423
  習得記録数: 6002
  初期化完了

================================================================================
サンプル実行1: 全タイプの力量を推薦
================================================================================

================================================================================
力量推薦結果
================================================================================
会員: 黒崎 国彦 (m48)
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

- [評価ガイド (docs/EVALUATION.md)](docs/EVALUATION.md) - 推薦システムの評価方法
  - 時系列分割による評価
  - 評価メトリクス (Precision@K, Recall@K, NDCG@K, Hit Rate)
  - 多様性評価メトリクス
  - クロスバリデーション
  - ベストプラクティス

- [クイックスタート (docs/QUICKSTART.md)](docs/QUICKSTART.md) - 使い方ガイド

- [テスト設計 (docs/TEST_DESIGN.md)](docs/TEST_DESIGN.md) - テストコードの設計書（100+テストケース）

- [テスト実装ガイド (docs/TESTING_QUICKSTART.md)](docs/TESTING_QUICKSTART.md) - テスト実装手順

## バージョン履歴

- v1.2.0 (2025-10-24)
  - **機械学習ベース推薦システム追加** (skillnote_recommendation/ml/)
    - Matrix Factorization (NMF) による協調フィルタリング
    - 多様性再ランキング (MMR, Category, Type, Hybrid)
    - MLRecommender: ML推薦システムの統合インターフェース
  - **多様性評価メトリクス追加**
    - カテゴリ多様性、タイプ多様性、カバレッジ、リスト内多様性
    - evaluate_with_diversity() による統合評価
  - **データ入力方式の一本化**
    - ディレクトリ構造のみをサポート（単一ファイル対応を削除）
    - data/members/, data/acquired/ など各ディレクトリに複数CSVを配置
  - **テストスイート拡充** (253テスト)
    - ML関連テスト追加 (matrix_factorization, diversity, ml_recommender)
    - ディレクトリスキャン機能の包括的テスト
  - **ドキュメント整理**
    - docs/ディレクトリに全ドキュメントを集約
    - README更新（ML使用例、多様性戦略の説明）

- v1.1.0 (2025-10-23)
  - 推薦システム評価機能追加
  - 時系列分割による評価 (Temporal Split)
  - 評価メトリクス実装 (Precision@K, Recall@K, NDCG@K, Hit Rate)
  - クロスバリデーション機能
  - ディレクトリスキャンによる複数CSV対応
  - カラム構造検証機能
  - 包括的テストスイート (194テスト)

- v1.0.0 (2025-10-23)
  - 初回リリース
  - uv対応
  - オブジェクト指向設計による実装
  - モジュール分割による保守性向上
