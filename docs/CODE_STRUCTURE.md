# コード構造ガイド

このドキュメントでは、CareerNavigatorプロジェクトのコード構造と各モジュールの役割について説明します。

## テスト品質

- **テスト数**: 238テスト
- **カバレッジ**: 30%
- **テスト実行**: `uv run pytest`

> **関連ドキュメント**: モデル実装の詳細については、[MODELS_TECHNICAL_GUIDE.md](MODELS_TECHNICAL_GUIDE.md)を参照してください。

## プロジェクト構造

```
CareerNavigator/
├── skillnote_recommendation/       # メインパッケージ
│   ├── core/                       # コアビジネスロジック
│   ├── ml/                         # 機械学習モジュール
│   ├── graph/                      # グラフベース推薦モジュール
│   ├── utils/                      # ユーティリティモジュール
│   └── scripts/                    # 実行スクリプト
├── pages/                          # Streamlitページ
│   ├── 1_Model_Training.py         # モデル学習ページ
│   ├── 4_Inference.py              # 推論ページ
│   ├── 5_Data_Quality.py           # データ品質ページ
│   └── 6_Model_Comparison.py       # モデル比較ページ
├── streamlit_app.py                # メインアプリ（データ読み込み）
├── tests/                          # テストコード
└── docs/                           # ドキュメント
```

## モジュール詳細

### skillnote_recommendation/core/

基盤コンポーネントとデータ処理の実装。

#### 主要モジュール:

- **config.py**: 設定管理
  - 推薦パラメータ
  - ファイルパス設定

- **models.py**: データモデル定義
  - Recommendation
  - ReferencePerson
  - その他のデータクラス

- **data_loader.py**: データ読み込み
  - CSVファイルの読み込み
  - ディレクトリスキャン
  - データ検証

- **data_transformer.py**: データ変換
  - 力量マスタ作成
  - メンバー力量データ作成
  - スキルマトリクス作成

- **reference_persons.py**: 参考人物検索
  - 類似キャリアの人物検索
  - ロールモデル検索
  - 異なるキャリアパスの人物検索
  - **上位者フィルタリング機能**

- **evaluator.py**: 推薦システム評価
  - 時系列分割
  - 評価メトリクス (Precision, Recall, NDCG, Hit Rate)
  - 多様性評価

### skillnote_recommendation/ml/

機械学習ベースの推薦システム。

#### 主要モジュール:

- **matrix_factorization.py**: 行列分解モデル
  - NMF (Non-negative Matrix Factorization)
  - 潜在因子学習
  - スコア予測

- **diversity.py**: 多様性再ランキング
  - MMR (Maximal Marginal Relevance)
  - カテゴリ多様性
  - タイプ多様性
  - ハイブリッド戦略

- **ml_recommender.py**: ML推薦システム統合
  - MatrixFactorizationModelのラッパー
  - 推薦生成
  - 参考人物の統合
  - 多様性メトリクス計算

### skillnote_recommendation/graph/

グラフベース推薦システム（Random Walk with Restart）。

#### 主要モジュール:

- **knowledge_graph.py**: 知識グラフの構築
  - CompetenceKnowledgeGraphクラス
  - メンバー、力量、カテゴリのノード管理
  - エッジ関係の管理（acquired, belongs_to, similar, parent_of）
  - グラフクエリ機能

- **random_walk.py**: Random Walk with Restart推薦
  - RandomWalkRecommenderクラス
  - PageRankアルゴリズムによる推薦スコア計算
  - フォールバック機能（カテゴリベース、類似メンバーベース）
  - 推薦パス抽出
  - PageRankキャッシュ機能

- **hybrid_recommender.py**: ハイブリッド推薦
  - HybridGraphRecommenderクラス
  - RWR + NMF + コンテンツベースの統合
  - スコア融合とランキング
  - 推薦理由生成

- **hybrid_builder.py**: ハイブリッド推薦ビルダー
  - build_hybrid_recommender関数
  - quick_recommend関数
  - システム全体の構築と初期化

- **category_hierarchy.py**: カテゴリ階層管理
  - カテゴリの階層構造
  - 親子関係の管理

### skillnote_recommendation/utils/

再利用可能なユーティリティ関数。

#### モジュール:

- **streamlit_helpers.py**: Streamlit UI ユーティリティ
  - `init_session_state()`: セッション状態の初期化
  - `check_data_loaded()`: データ読み込み確認
  - `check_model_trained()`: モデル学習確認
  - `display_error_details()`: エラー詳細表示
  - `show_metric_cards()`: メトリクスカード表示

- **visualization.py**: 可視化ユーティリティ
  - `create_member_positioning_data()`: メンバー位置データ作成
  - `create_positioning_plot()`: 散布図作成
  - `prepare_positioning_display_dataframe()`: 表示用データ整形
  - 色スキーム定数 (COLOR_TARGET_MEMBER, etc.)

## Streamlitアプリケーション構造

### 3ページ構成

#### streamlit_app.py - データ読み込み
**ステップ1**: CSVファイルのアップロードとデータ準備

- 6種類のCSVファイルのアップロード
  1. メンバーマスタ
  2. 力量（スキル）マスタ
  3. 力量（教育）マスタ
  4. 力量（資格）マスタ
  5. 力量カテゴリーマスタ
  6. 保有力量データ

- DataLoaderによる読み込み
- DataTransformerによる変換
- セッション状態への保存

#### pages/1_Model_Training.py - モデル学習
**ステップ2**: MLモデルの学習と分析

- NMFモデルの学習
- 学習結果の分析
  - 潜在因子の分析
  - メンバーの潜在因子分布
  - 力量の潜在因子分布
  - モデル評価指標

#### pages/4_Inference.py - 推論
**ステップ3**: 推薦の実行と可視化

- 推論対象メンバーの選択
- 推論設定（推薦数、フィルタ、多様性戦略）
- 推薦実行
- 推薦結果の詳細表示
  - 推薦理由
  - 参考人物（上位者のみ）
- **メンバーポジショニングマップ**
  - 総合スキルレベル vs 保有力量数
  - 平均レベル vs 保有力量数
  - 潜在因子マップ
  - データテーブル
- 推薦結果のCSVダウンロード

## コーディング規約

### 関数設計

#### 単一責任の原則
各関数は1つの責任のみを持つべきです。

```python
# Good: 単一の責任
def load_csv_file(file_path: str) -> pd.DataFrame:
    """CSVファイルを読み込む"""
    return pd.read_csv(file_path, encoding="utf-8-sig")

# Bad: 複数の責任
def load_and_process_csv(file_path: str) -> pd.DataFrame:
    """CSVファイルを読み込んで処理する"""
    df = pd.read_csv(file_path, encoding="utf-8-sig")
    df = df.dropna()
    df = df[df["status"] == "active"]
    return df
```

#### ドキュメント文字列

すべての公開関数にはdocstringを付けます。

```python
def create_member_positioning_data(
    member_competence: pd.DataFrame,
    member_master: pd.DataFrame,
    mf_model: MatrixFactorizationModel
) -> pd.DataFrame:
    """
    全メンバーの位置データを作成します。

    Args:
        member_competence: メンバー力量データ
        member_master: メンバーマスタ
        mf_model: 学習済みMatrixFactorizationModel

    Returns:
        メンバー位置データのDataFrame

    Example:
        >>> position_df = create_member_positioning_data(
        ...     member_comp_df, member_df, trained_model
        ... )
    """
    # 実装...
```

### Streamlit コーディングパターン

#### セッション状態管理

```python
from skillnote_recommendation.utils.streamlit_helpers import init_session_state

# アプリ起動時に初期化
init_session_state()

# セッション状態へのアクセス
if st.session_state.data_loaded:
    # 処理...
```

#### 前提条件チェック

```python
from skillnote_recommendation.utils.streamlit_helpers import (
    check_data_loaded,
    check_model_trained
)

# データ読み込みチェック
check_data_loaded()  # 未読み込みの場合は警告を表示してstop

# モデル学習チェック
check_model_trained()  # 未学習の場合は警告を表示してstop
```

#### エラー処理

```python
from skillnote_recommendation.utils.streamlit_helpers import display_error_details

try:
    # 処理...
    pass
except Exception as e:
    display_error_details(e, "処理中")
```

## テスト構造

### テストディレクトリ

```
tests/
├── test_data_loader.py             # データ読み込みテスト
├── test_data_transformer.py        # データ変換テスト
├── test_evaluator.py               # 評価システムテスト
├── test_matrix_factorization.py    # 行列分解テスト
├── test_diversity.py               # 多様性テスト
├── test_ml_recommender.py          # ML推薦システムテスト
├── test_hyperparameter_tuning.py   # ハイパーパラメータチューニングテスト
└── conftest.py                     # テストフィクスチャ
```

### テスト統計

- **総テスト数**: 238テスト
- **テストカバレッジ**: 30%
- **テストフレームワーク**: pytest

### テストの実行

```bash
# すべてのテストを実行（238テスト）
uv run pytest

# 特定のモジュールのテストのみ
uv run pytest tests/test_ml_recommender.py

# カバレッジ付き（現在30%）
uv run pytest --cov=skillnote_recommendation

# カバレッジレポートをHTMLで出力
uv run pytest --cov=skillnote_recommendation --cov-report=html
```

## 拡張ガイド

### 新しいページの追加

1. **pages/** ディレクトリに新しいファイルを作成
   ```python
   # pages/3_New_Page.py
   import streamlit as st
   from skillnote_recommendation.utils.streamlit_helpers import (
       check_data_loaded
   )

   st.set_page_config(
       page_title="新しいページ",
       page_icon="📊",
       layout="wide"
   )

   check_data_loaded()

   # ページの実装...
   ```

2. Streamlitが自動的に検出してサイドバーに追加

### 新しい可視化の追加

1. **skillnote_recommendation/utils/visualization.py** に関数を追加
2. 適切なdocstringとtype hintsを付ける
3. 色スキーム定数を使用
4. テストを書く

### 新しい推薦アルゴリズムの追加

1. **skillnote_recommendation/ml/** に新しいモジュールを作成
2. 既存のインターフェースに準拠
3. **ml_recommender.py** に統合
4. テストを書く

## パフォーマンス最適化

### データキャッシング

Streamlitの`@st.cache_data`を使用してデータをキャッシュ:

```python
@st.cache_data
def load_and_transform_data(data_dir: str):
    loader = DataLoader(data_dir)
    data = loader.load_all_data()
    transformer = DataTransformer()
    return transformer.transform(data)
```

### セッション状態の活用

重い処理の結果はセッション状態に保存:

```python
if "ml_recommender" not in st.session_state:
    with st.spinner("モデル学習中..."):
        st.session_state.ml_recommender = build_ml_recommender(data)
```

## デバッグ

### Streamlitデバッグモード

```bash
# デバッグモードで起動
streamlit run streamlit_app.py --logger.level=debug
```

### エラートレースバック

`display_error_details()`関数は自動的に詳細なトレースバックを表示します。

### ログ出力

```python
import logging

logger = logging.getLogger(__name__)
logger.info("処理開始")
logger.error("エラーが発生しました")
```

## まとめ

このコード構造により、以下のメリットがあります:

1. **モジュール性**: 各モジュールは独立して機能し、再利用可能
2. **可読性**: 明確な命名規則とドキュメント
3. **保守性**: 単一責任の原則により、変更が容易
4. **拡張性**: 新しい機能の追加が簡単
5. **テスト性**: 小さな関数単位でテスト可能（238テスト、カバレッジ30%）

新しいコードを書く際は、この構造とパターンに従ってください。

---

## 関連ドキュメント

- [MODELS_TECHNICAL_GUIDE.md](MODELS_TECHNICAL_GUIDE.md) - モデル実装の詳細、データ前処理、ハイパーパラメータチューニング
- [ML_TECHNICAL_DETAILS.md](ML_TECHNICAL_DETAILS.md) - 機械学習推薦システムの技術詳細
- [EVALUATION.md](EVALUATION.md) - 推薦システムの評価方法
- [TEST_DESIGN.md](TEST_DESIGN.md) - テスト設計ドキュメント（238テストケースの詳細）
- [QUICKSTART.md](QUICKSTART.md) - クイックスタートガイド
- [STREAMLIT_GUIDE.md](STREAMLIT_GUIDE.md) - Streamlitアプリケーションガイド
