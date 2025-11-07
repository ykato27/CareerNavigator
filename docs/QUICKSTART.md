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
# Streamlitアプリを起動
uv run streamlit run streamlit_app.py
```

ブラウザが自動的に開き、WebUIから推薦システムを使用できます。

## よく使うコマンド

```bash
# Streamlitアプリを起動
uv run streamlit run streamlit_app.py

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

### 機械学習ベース推薦

```python
from skillnote_recommendation.ml.ml_recommender import MLRecommender
from skillnote_recommendation.core.data_loader import DataLoader
from skillnote_recommendation.core.data_transformer import DataTransformer

# データ読み込み
loader = DataLoader()
data = loader.load_all_data()

# データ変換
transformer = DataTransformer()
competence_master = transformer.create_competence_master(data)
member_competence, _ = transformer.create_member_competence(data, competence_master)

# ML推薦システム初期化
ml_recommender = MLRecommender.build(
    member_competence=member_competence,
    competence_master=competence_master,
    member_master=data['members'],
    use_preprocessing=False,
    use_tuning=False,
    n_components=20  # 潜在因子数を指定（省略可能、デフォルトは20）
)

# 基本的な推薦
recommendations = ml_recommender.recommend(
    member_code='m48',
    top_n=10,
    use_diversity=False
)

# 推薦結果表示
for rec in recommendations:
    print(f"{rec.competence_name}: {rec.priority_score:.3f}")
    print(f"  理由: {rec.reason}")
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
diversity = ml_recommender.calculate_diversity_metrics(recommendations)
print(f"カテゴリ多様性: {diversity['category_diversity']:.3f}")
print(f"タイプ多様性: {diversity['type_diversity']:.3f}")
print(f"カバレッジ: {diversity['coverage']:.3f}")
print(f"リスト内多様性: {diversity['intra_list_diversity']:.3f}")
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
- [MODELS_TECHNICAL_GUIDE.md](MODELS_TECHNICAL_GUIDE.md) - モデル実装の詳細、データ前処理、ハイパーパラメータチューニング
- [ML_TECHNICAL_DETAILS.md](ML_TECHNICAL_DETAILS.md) - 機械学習推薦システムの技術詳細
- [EVALUATION.md](EVALUATION.md) - 推薦システムの評価方法（時系列分割、多様性メトリクス）
- [CODE_STRUCTURE.md](CODE_STRUCTURE.md) - コード構造とモジュール設計
- [STREAMLIT_GUIDE.md](STREAMLIT_GUIDE.md) - StreamlitアプリケーションガイD
- [TEST_DESIGN.md](TEST_DESIGN.md) - テストコードの設計書（238テストケース）
