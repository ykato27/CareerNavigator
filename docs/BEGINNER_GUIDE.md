# 初心者向けコード理解ガイド

このガイドでは、CareerNavigatorプロジェクトのコード構造を初心者向けに説明します。

## 📋 目次

1. [実行ファイル vs ライブラリコード](#実行ファイル-vs-ライブラリコード)
2. [実行ファイル一覧](#実行ファイル一覧)
3. [ライブラリコード一覧](#ライブラリコード一覧)
4. [コードの読み方](#コードの読み方)
5. [よくある質問](#よくある質問)

---

## 実行ファイル vs ライブラリコード

### 🚀 実行ファイル（スクリプト）とは？

**実行ファイル**は、直接実行できるPythonファイルです。これらのファイルには `if __name__ == "__main__":` というコードが含まれており、コマンドラインやStreamlitから起動します。

**特徴:**
- ✅ `python ファイル名.py` で直接実行できる
- ✅ ユーザーインターフェース（UI）を持つ
- ✅ プログラムのエントリーポイント（入口）

### 📚 ライブラリコード（関数・クラス定義）とは？

**ライブラリコード**は、関数やクラスの定義が書かれたファイルで、他のファイルから `import` して使います。直接実行することはありません。

**特徴:**
- ✅ `from skillnote_recommendation import ～` で読み込んで使う
- ✅ 再利用可能な関数やクラスを提供
- ✅ ビジネスロジックや計算処理を実装

---

## 実行ファイル一覧

### 🌐 Webアプリ（Streamlit）

これらのファイルは、ブラウザで動くWebアプリケーションです。

| ファイル | 実行方法 | 役割 |
|---------|---------|------|
| **streamlit_app.py** | `uv run streamlit run streamlit_app.py` | **メインページ**: CSVファイルをアップロードしてデータを読み込む |
| **pages/1_Model_Training.py** | サイドバーから選択 | **モデル訓練ページ**: 機械学習モデルを訓練する |
| **pages/4_Inference.py** | サイドバーから選択 | **推論ページ**: メンバーへの力量推薦を実行する |
| **pages/5_Data_Quality.py** | サイドバーから選択 | **データ品質ページ**: データ品質をモニタリングする |
| **pages/6_Model_Comparison.py** | サイドバーから選択 | **モデル比較ページ**: 複数のモデルを比較する |

**使い方:**
```bash
# Streamlitアプリを起動
uv run streamlit run streamlit_app.py

# ブラウザが自動的に開き、上記のページにアクセスできます
```

**推奨:** 基本的にStreamlit Webアプリからすべての機能を利用できます。

---

## ライブラリコード一覧

以下のファイルは、上記の実行ファイルから `import` されて使われます。直接実行することはありません。

### 📁 skillnote_recommendation/core/ - コアロジック

**データ関連:**

| ファイル | 主要なクラス・関数 | 役割 |
|---------|------------------|------|
| **models.py** | `Member`, `Competence`, `Recommendation` など | データ構造を定義（メンバー情報、力量情報など） |
| **data_loader.py** | `DataLoader` クラス | CSVファイルを読み込む |
| **data_transformer.py** | `DataTransformer` クラス | データを推薦システム用に変換する |

**推薦補助機能:**

| ファイル | 主要なクラス・関数 | 役割 |
|---------|------------------|------|
| **reference_persons.py** | `ReferencePersonFinder` クラス | 参考となる先輩メンバーを検索する |
| **evaluator.py** | `RecommendationEvaluator` クラス | ML推薦システムの性能を評価する |

**品質管理:**

| ファイル | 主要なクラス・関数 | 役割 |
|---------|------------------|------|
| **data_quality.py** | 品質チェック関数 | データ品質をチェックする |
| **data_quality_monitor.py** | `DataQualityMonitor` クラス | データ品質を継続的に監視する |

**その他:**

| ファイル | 主要なクラス・関数 | 役割 |
|---------|------------------|------|
| **config.py** | `Config` クラス、各種設定 | 推薦パラメータや設定を管理する |
| **logging_config.py** | ログ設定関数 | ログ出力の設定を行う |
| **errors.py** | カスタム例外クラス | エラーの種類を定義する |
| **error_handlers.py** | エラー処理関数 | エラーを適切に処理する |

### 📁 skillnote_recommendation/ml/ - 機械学習

| ファイル | 主要なクラス・関数 | 役割 |
|---------|------------------|------|
| **ml_recommender.py** | `MLRecommender` クラス | 機械学習ベースの推薦システム（メイン） |
| **matrix_factorization.py** | `MatrixFactorizationModel` クラス | 行列分解による協調フィルタリング |
| **diversity.py** | `DiversityReranker` クラス | 推薦結果を多様性の観点で再ランキング |
| **hyperparameter_tuning.py** | `HyperparameterTuner` クラス | ハイパーパラメータを自動調整 |
| **data_preprocessing.py** | 前処理関数 | 機械学習用にデータを前処理する |
| **ml_evaluation.py** | 評価指標関数 | 機械学習モデルの性能を評価する |

### 📁 skillnote_recommendation/graph/ - グラフベース推薦

| ファイル | 主要なクラス・関数 | 役割 |
|---------|------------------|------|
| **knowledge_graph.py** | `CompetenceKnowledgeGraph` クラス | 力量の知識グラフを構築する |
| **random_walk.py** | `RandomWalkRecommender` クラス | Random Walk with Restartによる推薦 |
| **hybrid_recommender.py** | `HybridGraphRecommender` クラス | RWR + NMF + コンテンツベースの統合推薦 |
| **hybrid_builder.py** | ビルダー関数 | ハイブリッド推薦システムを簡単に構築 |
| **category_hierarchy.py** | カテゴリ階層管理 | カテゴリの階層構造を管理 |

### 📁 skillnote_recommendation/utils/ - ユーティリティ

| ファイル | 主要な関数 | 役割 |
|---------|-----------|------|
| **streamlit_helpers.py** | `init_session_state()`, `check_data_loaded()` など | Streamlitアプリの共通処理 |
| **visualization.py** | `create_positioning_plot()` など | グラフ・チャートの作成 |
| **data_validators.py** | バリデーション関数 | データの妥当性をチェック |
| **data_normalizers.py** | 正規化関数 | データを正規化する |

### 📁 skillnote_recommendation/core/persistence/ - データ永続化

| ファイル | 主要なクラス・関数 | 役割 |
|---------|------------------|------|
| **database.py** | データベース操作関数 | SQLiteデータベースとの接続 |
| **models.py** | ORMモデル | データベーステーブルの定義 |
| **repository.py** | `Repository` クラス | データの保存・読み込み |
| **model_storage.py** | モデル保存・読込関数 | 機械学習モデルの保存管理 |

---

## コードの読み方

### 🎯 学習の進め方

初心者がコードを理解するための推奨順序：

#### **ステップ1: データ構造を理解する**

まず、システムが扱うデータの構造を理解しましょう。

1. **skillnote_recommendation/core/models.py** を読む
   - `Member`: メンバー（社員）の情報
   - `Competence`: 力量（スキル、教育、資格）の情報
   - `MemberCompetence`: メンバーが保有する力量
   - `Recommendation`: 推薦結果

#### **ステップ2: データの流れを追う**

データがどのように読み込まれ、変換されるかを見ます。

2. **skillnote_recommendation/core/data_loader.py** を読む
   - CSVファイルからデータを読み込む処理

3. **skillnote_recommendation/core/data_transformer.py** を読む
   - 読み込んだデータを推薦システム用に変換する処理

#### **ステップ3: 推薦ロジックを理解する**

どのように推薦が行われるかを見ます。

4. **skillnote_recommendation/ml/ml_recommender.py** を読む
   - 機械学習ベースの推薦アルゴリズム
   - NMFによる協調フィルタリング
   - 多様性再ランキング

5. **skillnote_recommendation/core/reference_persons.py** を読む
   - 参考人物（ロールモデル）の検索ロジック

#### **ステップ4: 実行ファイルを読む**

最後に、これらがどう組み合わされて動くかを見ます。

6. **streamlit_app.py** を読む
   - Webアプリのメインページ
   - データ読み込みの流れ

7. **pages/4_Inference.py** を読む
   - 推薦実行のUI
   - ライブラリコードの使い方

### 📖 コードを読むときのヒント

1. **クラス定義から読む**
   ```python
   class MLRecommender:
       """機械学習ベース推薦エンジン"""

       def __init__(self, mf_model, competence_master, ...):
           # 初期化処理
           pass

       def recommend(self, member_code, top_n=10):
           # 推薦を実行
           pass
   ```
   - クラス名とdocstring（説明）を見る
   - メソッド名を見て何をするクラスか理解する

2. **関数のシグネチャを見る**
   ```python
   def find_reference_persons(target_member_code: str,
                               recommended_competence_code: str,
                               top_n: int = 3) -> List[dict]:
       """参考人物を検索する"""
       pass
   ```
   - 引数（何を受け取るか）
   - 戻り値（何を返すか）
   - docstring（何をする関数か）

3. **importを追う**
   ```python
   from skillnote_recommendation.core.data_loader import DataLoader
   from skillnote_recommendation.ml.ml_recommender import MLRecommender
   ```
   - どのライブラリコードを使っているかが分かる

---

## よくある質問

### Q1: どのファイルから読み始めればいいですか？

**A:** まずは **skillnote_recommendation/core/models.py** から読んでください。システムが扱うデータ構造が理解できます。

その後、実行ファイルの **streamlit_app.py** を見て、全体の流れを掴むと良いでしょう。

### Q2: 実行ファイルとライブラリコードの違いが分かりません

**A:** 簡単に言うと：

- **実行ファイル**: `uv run streamlit run streamlit_app.py` のように、コマンドで起動するファイル
- **ライブラリコード**: `from ～ import ～` で読み込んで使うファイル

実行ファイルには `if __name__ == "__main__":` というコードがあります。

### Q3: どのファイルがどのファイルを使っているか分かりません

**A:** 各ファイルの冒頭にある `import` 文を見てください。

例えば、**streamlit_app.py** の冒頭：
```python
from skillnote_recommendation.core.data_loader import DataLoader
from skillnote_recommendation.core.data_transformer import DataTransformer
```

これは、`streamlit_app.py` が `DataLoader` と `DataTransformer` というライブラリコードを使っていることを示します。

### Q4: Streamlitのページはどう動いていますか？

**A:** Streamlitは以下のような構造です：

```
streamlit_app.py        ← メインページ（ホーム）
└── pages/              ← サブページ（自動的にサイドバーに表示される）
    ├── 1_Model_Training.py
    ├── 2_Inference.py
    └── ...
```

`streamlit_app.py` を起動すると、`pages/` フォルダ内のファイルが自動的にサイドバーに表示されます。

### Q5: コマンドラインから実行できますか？

**A:** 基本的に **Streamlit Webアプリ** から全ての機能を利用できます。
コマンドラインツールは現在提供していません。

Streamlitアプリの起動：
```bash
uv run streamlit run streamlit_app.py
```

### Q6: テストコードは読むべきですか？

**A:** **はい、おすすめです！** テストコード（`tests/` フォルダ）を読むと、各ライブラリコードの使い方が分かります。

**現在のテスト状況（v1.2.1）:**
- **テスト数**: 393テスト
- **カバレッジ**: 47%
- **テストファイル数**: 12ファイル

**主なテストファイル:**
- `tests/test_data_loader.py` - データ読み込みのテスト
- `tests/test_data_transformer.py` - データ変換のテスト
- `tests/test_evaluator.py` - 評価機能のテスト
- `tests/test_matrix_factorization.py` - 行列分解モデルのテスト
- `tests/test_diversity.py` - 多様性再ランキングのテスト
- `tests/test_ml_recommender.py` - ML推薦システムのテスト
- `tests/test_knowledge_graph.py` - グラフベース推薦のテスト（v1.2.1新規）
- `tests/test_hyperparameter_tuning.py` - ハイパーパラメータ調整のテスト（v1.2.1新規）
- `tests/test_feature_engineering.py` - 特徴量エンジニアリングのテスト（v1.2.1新規）
- `tests/test_visualization.py` - 可視化機能のテスト（v1.2.1新規）

例えば、`tests/test_data_loader.py` を見れば、`DataLoader` クラスの使い方が分かります。

**テストの実行方法:**
```bash
# すべてのテストを実行
uv run pytest

# カバレッジ付きでテスト実行
uv run pytest --cov=skillnote_recommendation

# 特定のテストファイルのみ実行
uv run pytest tests/test_data_loader.py
```

### Q7: どのファイルを修正すれば機能を追加できますか？

**A:** 追加したい機能によって異なります：

- **推薦ロジックを変更**: `ml/ml_recommender.py` または `ml/matrix_factorization.py`
- **新しいページを追加**: `pages/` に新しいファイルを作成
- **データ処理を変更**: `core/data_transformer.py`
- **新しいモデルを追加**: `ml/` に新しいファイルを作成
- **グラフベース推薦**: `graph/` のファイルを修正

ただし、**コードを修正する前に、まず既存のコードをよく読んで理解してください。**

---

## 📚 次のステップ

このガイドでコードの全体像が理解できたら、以下のドキュメントも読んでみてください：

### 🎓 初級者向け（推奨）

1. **[モデル技術ガイド (MODELS_TECHNICAL_GUIDE.md)](MODELS_TECHNICAL_GUIDE.md)** ⭐ **初級者向け！**
   - Matrix Factorization（行列分解）の仕組みと使い方
   - 多様性再ランキング戦略の選び方
   - グラフベース推薦とハイブリッド推薦
   - モデル選択ガイドとFAQ

2. **[Streamlitアプリガイド (STREAMLIT_GUIDE.md)](STREAMLIT_GUIDE.md)** - Webアプリの使い方

### 📖 中級者向け

3. **[コード構造ガイド (CODE_STRUCTURE.md)](CODE_STRUCTURE.md)** - より詳細な技術ドキュメント
4. **[クイックスタート (QUICKSTART.md)](QUICKSTART.md)** - コマンドライン使い方
5. **[機械学習技術詳細 (ML_TECHNICAL_DETAILS.md)](ML_TECHNICAL_DETAILS.md)** - ML推薦システムの仕組み
6. **[テスト実装ガイド (TESTING_QUICKSTART.md)](TESTING_QUICKSTART.md)** - テストの書き方

---

## 🤔 困ったときは

- **エラーが出た**: [README.md](../README.md) の「トラブルシューティング」セクションを見る
- **使い方が分からない**: [STREAMLIT_GUIDE.md](STREAMLIT_GUIDE.md) を読む
- **コードが理解できない**: このドキュメントの「コードの読み方」を参照
- **テストを書きたい**: [TESTING_QUICKSTART.md](TESTING_QUICKSTART.md) を読む

コードを読むのは最初は難しいですが、順を追って読めば必ず理解できます。頑張ってください！
