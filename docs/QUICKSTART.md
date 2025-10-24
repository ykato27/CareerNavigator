# クイックスタートガイド

## 1分で始めるスキルノート推薦システム

### 前提条件

- **Python 3.11以上**（推奨: 3.12 または 3.13）

### ステップ1: uvのインストール（初回のみ）

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### ステップ2: プロジェクトのセットアップ

```bash
# プロジェクトディレクトリに移動
cd CareerNavigator

# 依存関係のインストール（自動的に仮想環境も作成）
uv sync
```

### ステップ3: データを配置

```bash
# ディレクトリ構造を作成
mkdir -p data/members data/acquired data/skills data/education data/license data/categories

# CSVファイルを各ディレクトリにコピー
cp /path/to/member*.csv data/members/
cp /path/to/acquired*.csv data/acquired/
cp /path/to/skill*.csv data/skills/
cp /path/to/education*.csv data/education/
cp /path/to/license*.csv data/license/
cp /path/to/category*.csv data/categories/
```

データ構造（各ディレクトリに複数CSVファイルを配置可能）:
```
data/
├── members/          # メンバーデータ（複数ファイル可）
├── acquired/         # 習得力量データ（複数ファイル可）
├── skills/           # スキル力量データ
├── education/        # 教育力量データ
├── license/          # 資格力量データ
└── categories/       # カテゴリデータ
```

### ステップ4: 実行

```bash
# データ変換
uv run skillnote-convert

# 推薦実行
uv run skillnote-recommend
```

## よく使うコマンド

```bash
# データ変換
uv run skillnote-convert

# 推薦実行
uv run skillnote-recommend

# Pythonシェルで対話的に使用
uv run python
>>> from skillnote_recommendation import RecommendationSystem
>>> system = RecommendationSystem()
>>> system.print_recommendations('m48', top_n=10)

# 依存関係の追加
uv add パッケージ名

# 依存関係の更新
uv sync --upgrade

# テスト実行
uv run pytest

# コードフォーマット
uv run black skillnote_recommendation/
```

## Pythonコードで使う

### ルールベース推薦（基本）

```python
from skillnote_recommendation import RecommendationSystem

# システム初期化
system = RecommendationSystem()

# 推薦実行
system.print_recommendations('m48', top_n=10)
```

### ルールベース推薦（詳細）

```python
from skillnote_recommendation import RecommendationSystem

system = RecommendationSystem()

# メンバー情報を取得
info = system.get_member_info('m48')
print(f"メンバー: {info['name']}")
print(f"SKILL: {info['skill_count']}件")

# SKILLのみ推薦
recommendations = system.recommend_competences(
    'm48',
    competence_type='SKILL',
    top_n=5
)

for rec in recommendations:
    print(f"{rec.competence_name}: {rec.priority_score:.2f}")

# 特定カテゴリのみ推薦
recommendations = system.recommend_competences(
    'm48',
    category_filter='製造部',
    top_n=10
)

# CSV出力
system.export_recommendations(
    'm48',
    'recommendations_m48.csv',
    top_n=20
)
```

### 機械学習ベース推薦

```python
from skillnote_recommendation.ml import MLRecommender
from skillnote_recommendation.core.data_loader import DataLoader

# データ読み込み
loader = DataLoader()
data = loader.load_all_data()

# ML推薦システム初期化
ml_recommender = MLRecommender(data)

# 基本的な推薦
recommendations = ml_recommender.recommend(
    member_code='m48',
    top_n=10
)

# 推薦結果表示
for rec in recommendations:
    print(f"{rec['力量名']}: {rec['MLスコア']:.3f}")
    print(f"  理由: {rec['推薦理由']}")
```

### 多様性を重視した推薦

```python
# MMR戦略で多様性重視
recommendations = ml_recommender.recommend(
    member_code='m48',
    top_n=10,
    use_diversity=True,
    diversity_strategy='mmr'  # mmr/category/type/hybrid
)

# 多様性メトリクス確認
diversity = ml_recommender.calculate_diversity_metrics(
    recommendations,
    ml_recommender.competence_master
)
print(f"カテゴリ多様性: {diversity['category_diversity']:.3f}")
print(f"タイプ多様性: {diversity['type_diversity']:.3f}")
```

## トラブルシューティング

### Q: uvが見つからない

```bash
# パスを確認
echo $PATH

# uvを再インストール
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # または source ~/.zshrc
```

### Q: CSVファイルが見つからない

```bash
# dataディレクトリを確認
ls -la data/

# ファイルをコピー
cp /path/to/*.csv data/
```

### Q: 変換済みデータが見つからない

```bash
# データ変換を実行
uv run skillnote-convert

# outputディレクトリを確認
ls -la output/
```

## 次のステップ

- [README](../README.md) - すべての機能と使い方
- [評価ガイド](EVALUATION.md) - 推薦システムの評価方法（時系列分割、多様性メトリクス）
- [テスト設計](TEST_DESIGN.md) - テストコードの設計書（100+テストケース）
- [開発ガイド](../README.md#開発環境のセットアップ) - 開発環境の構築とテスト方法
- [カスタマイズ](../README.md#カスタマイズ) - パラメータ調整と拡張方法
