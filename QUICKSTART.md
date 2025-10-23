# クイックスタートガイド

## 1分で始めるスキルノート推薦システム

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
# CSVファイルをdataディレクトリにコピー
cp /path/to/csvfiles/*.csv data/
```

必要なファイル:
- member_skillnote.csv
- acquiredCompetenceLevel.csv
- skill_skillnote.csv
- education_skillnote.csv
- license_skillnote.csv
- competence_category_skillnote.csv

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

### 基本的な使い方

```python
from skillnote_recommendation import RecommendationSystem

# システム初期化
system = RecommendationSystem()

# 推薦実行
system.print_recommendations('m48', top_n=10)
```

### 詳細な使い方

```python
from skillnote_recommendation import RecommendationSystem

system = RecommendationSystem()

# 会員情報を取得
info = system.get_member_info('m48')
print(f"会員: {info['name']}")
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

- [詳細ドキュメント](README.md) - すべての機能と使い方
- [開発ガイド](README.md#開発環境のセットアップ) - 開発環境の構築とテスト方法
- [カスタマイズ](README.md#カスタマイズ) - パラメータ調整と拡張方法
