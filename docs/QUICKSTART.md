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

### 因果推論ベース推薦

```python
from skillnote_recommendation.ml.causal_graph_recommender import CausalGraphRecommender
from skillnote_recommendation.core.data_loader import DataLoader
from skillnote_recommendation.core.data_transformer import DataTransformer

# データ読み込み
loader = DataLoader()
data = loader.load_all_data()

# データ変換
transformer = DataTransformer()
competence_master = transformer.create_competence_master(data)
member_competence, _ = transformer.create_member_competence(data, competence_master)

# 因果推論推薦システム初期化
causal_recommender = CausalGraphRecommender(
    member_competence=member_competence,
    competence_master=competence_master,
    skill_matrix=transformer.create_skill_matrix(member_competence)
)

# 因果グラフ学習
causal_recommender.fit()

# メンバーへの推薦（3軸スコアリング）
recommendations = causal_recommender.recommend(
    member_code='m48',
    top_n=10
)

# 推薦結果表示
for rec in recommendations:
    print(f"{rec['力量名']}: 総合スコア {rec['causal_score']:.2f}")
    print(f"  準備度: {rec['readiness_score']:.2f}")
    print(f"  習得確率: {rec['bayesian_prob']:.2f}")
    print(f"  将来性: {rec['utility_score']:.2f}")
```

### キャリアパス分析

```python
from skillnote_recommendation.graph.career_path import CareerGapAnalyzer, LearningPathGenerator

# ギャップ分析
gap_analyzer = CareerGapAnalyzer(member_competence, competence_master)
gap_analysis = gap_analyzer.analyze(
    current_member_code='m48',
    target_member_code='m100'  # ロールモデル
)

print(f"不足スキル数: {len(gap_analysis['missing_competencies'])}")

# 学習パス生成
path_generator = LearningPathGenerator(member_competence, competence_master)
learning_path = path_generator.generate(
    member_code='m48',
    target_competencies=gap_analysis['missing_competencies']
)

# 学習順序を表示
for step in learning_path:
    print(f"{step['competence_name']} - 推定学習期間: {step['estimated_months']}ヶ月")
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
