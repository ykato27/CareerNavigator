# テスト実装クイックスタートガイド

CareerNavigatorプロジェクトのテスト実装をすぐに始めるためのガイドです。

---

## 📋 目次

1. [セットアップ](#セットアップ)
2. [テスト実行](#テスト実行)
3. [作成済みファイル](#作成済みファイル)
4. [テスト作成の例](#テスト作成の例)
5. [次のステップ](#次のステップ)

---

## ⚙️ セットアップ

### 1. 開発依存関係のインストール

```bash
# uv を使用する場合（推奨）
uv sync --dev

# pip を使用する場合
pip install -e ".[dev]"
```

### 2. 必要なパッケージの確認

以下がインストールされていることを確認：
- pytest (テストフレームワーク)
- pandas (データ処理)
- numpy (数値計算)
- scikit-learn (機械学習ライブラリ)
- networkx (グラフライブラリ)

---

## 🧪 テスト実行

### 基本的なテスト実行

```bash
# 全テストを実行
uv run pytest tests/

# 詳細モードで実行
uv run pytest tests/ -v

# 特定のファイルのみ実行
uv run pytest tests/test_data_loader.py -v

# 特定のテストクラスのみ実行
uv run pytest tests/test_data_loader.py::TestCleanColumnName -v

# 特定の1つのテストのみ実行
uv run pytest tests/test_data_loader.py::TestCleanColumnName::test_clean_column_name_with_marker -v
```

### カバレッジ付き実行

```bash
# カバレッジを測定
uv run pytest --cov=skillnote_recommendation --cov-report=term tests/

# HTMLレポート生成
uv run pytest --cov=skillnote_recommendation --cov-report=html tests/

# HTMLレポートを開く
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### 失敗したテストの詳細表示

```bash
# 詳細なトレースバック表示
uv run pytest tests/ -v --tb=long

# 短いトレースバック
uv run pytest tests/ -v --tb=short
```

---

## 📁 作成済みファイル

### 1. TEST_DESIGN.md

**内容**: 包括的なテスト設計ドキュメント
- テスト戦略
- コンポーネント別テストケース設計
- テストデータ戦略
- 実装優先度

**使い方**: テスト実装前に必ず確認し、設計に従って実装してください。

### 2. tests/conftest.py

**内容**: 共通テストフィクスチャ
- サンプルデータ生成
- テスト用ヘルパー関数

### 3. 既存テスト

以下のテストが既に実装されています：
- `tests/test_data_loader.py` - データローダーテスト
- `tests/test_data_transformer.py` - データ変換テスト
- `tests/test_evaluator.py` - 評価器テスト
- `tests/test_matrix_factorization.py` - NMFモデルテスト
- `tests/test_diversity.py` - 多様性再ランキングテスト
- `tests/test_ml_recommender.py` - ML推薦エンジンテスト

---

## 🎯 テスト作成の例

### 例1: 知識グラフのテスト（NEW）

**ファイル**: `tests/test_knowledge_graph.py`

```python
import pytest
import pandas as pd
from skillnote_recommendation.graph.knowledge_graph import CompetenceKnowledgeGraph


def test_graph_construction(sample_members, sample_competences, sample_member_competence):
    """グラフ構築のテスト"""
    kg = CompetenceKnowledgeGraph(
        member_competence=sample_member_competence,
        member_master=sample_members,
        competence_master=sample_competences,
        use_category_hierarchy=True
    )

    # ノード数の確認
    assert kg.G.number_of_nodes() > 0

    # エッジ数の確認
    assert kg.G.number_of_edges() > 0


def test_get_neighbors(sample_members, sample_competences, sample_member_competence):
    """隣接ノード取得のテスト"""
    kg = CompetenceKnowledgeGraph(
        member_competence=sample_member_competence,
        member_master=sample_members,
        competence_master=sample_competences
    )

    # メンバーの習得力量を取得
    neighbors = kg.get_neighbors("member_M001", edge_type="acquired")

    # 習得力量が存在することを確認
    assert len(neighbors) > 0

    # 力量ノードであることを確認
    for neighbor in neighbors:
        assert neighbor.startswith("competence_")


def test_get_member_acquired_competences(sample_members, sample_competences, sample_member_competence):
    """習得力量取得のテスト"""
    kg = CompetenceKnowledgeGraph(
        member_competence=sample_member_competence,
        member_master=sample_members,
        competence_master=sample_competences
    )

    # 習得力量を取得
    acquired = kg.get_member_acquired_competences("M001")

    # 習得力量が存在することを確認
    assert len(acquired) > 0

    # 力量コードの形式を確認
    for comp_code in acquired:
        assert isinstance(comp_code, str)
```

### 例2: Random Walkのテスト（NEW）

**ファイル**: `tests/test_random_walk.py`

```python
import pytest
from skillnote_recommendation.graph.knowledge_graph import CompetenceKnowledgeGraph
from skillnote_recommendation.graph.random_walk import RandomWalkRecommender


def test_rwr_recommend(sample_members, sample_competences, sample_member_competence):
    """RWR推薦のテスト"""
    # グラフ構築
    kg = CompetenceKnowledgeGraph(
        member_competence=sample_member_competence,
        member_master=sample_members,
        competence_master=sample_competences
    )

    # RWR推薦エンジン
    rwr = RandomWalkRecommender(
        knowledge_graph=kg,
        restart_prob=0.15
    )

    # 推薦生成
    recommendations = rwr.recommend(
        member_code="M001",
        top_n=5,
        return_paths=True
    )

    # 推薦結果が存在することを確認
    assert len(recommendations) > 0

    # 推薦形式の確認
    for comp_code, score, paths in recommendations:
        assert isinstance(comp_code, str)
        assert isinstance(score, float)
        assert 0 <= score <= 1
        assert isinstance(paths, list)


def test_rwr_cache(sample_members, sample_competences, sample_member_competence):
    """RWRキャッシュのテスト"""
    kg = CompetenceKnowledgeGraph(
        member_competence=sample_member_competence,
        member_master=sample_members,
        competence_master=sample_competences
    )

    rwr = RandomWalkRecommender(
        knowledge_graph=kg,
        enable_cache=True
    )

    # 初回実行
    _ = rwr.recommend("M001", top_n=5)

    # キャッシュ統計確認
    cache_stats = rwr.get_cache_stats()
    assert cache_stats['cached_members'] == 1

    # キャッシュクリア
    rwr.clear_cache()
    cache_stats = rwr.get_cache_stats()
    assert cache_stats['cached_members'] == 0
```

---

## 📝 次のステップ

### Phase 3: グラフ推薦テスト（未実装）

1. **知識グラフテスト** (`test_knowledge_graph.py`)
   - グラフ構築
   - クエリ機能
   - メンバー間類似度

2. **Random Walkテスト** (`test_random_walk.py`)
   - RWR推薦
   - フォールバック処理
   - 推薦パス抽出
   - キャッシング

3. **ハイブリッド推薦テスト** (`test_hybrid_recommender.py`)
   - スコア融合
   - 推薦生成
   - 推薦理由生成

### Phase 4: 統合・E2Eテスト（未実装）

4. **統合テスト** (`test_integration.py`)
   - データパイプライン全体の検証

5. **E2Eテスト** (`test_e2e.py`)
   - エンドツーエンドシナリオの検証

---

## 🔗 関連ドキュメント

- [TEST_DESIGN.md](TEST_DESIGN.md) - 詳細なテスト設計
- [TESTING_QUICKSTART.md](TESTING_QUICKSTART.md) - このドキュメント
- [pytest ドキュメント](https://docs.pytest.org/) - pytestの公式ドキュメント

---

## 💡 ヒント

### テスト実行が遅い場合

```bash
# 並列実行（pytest-xdist使用）
uv run pytest tests/ -n auto

# 失敗したテストのみ再実行
uv run pytest tests/ --lf

# 最初の失敗で停止
uv run pytest tests/ -x
```

### テストデバッグ

```bash
# pdbでデバッグ
uv run pytest tests/ --pdb

# 標準出力を表示
uv run pytest tests/ -s
```

### テストマーカー

```python
# 遅いテストをスキップ
@pytest.mark.slow
def test_slow_operation():
    ...

# 実行時
pytest tests/ -m "not slow"
```

---

## 📊 現在の状況

**カバレッジ**: 約40-50%（推定）
**目標カバレッジ**: 80%以上

**完了済み**:
- ✅ Core Module テスト
- ✅ ML Module テスト

**未実装**:
- ⏳ Graph Module テスト
- ⏳ 統合テスト
- ⏳ E2Eテスト

---

テスト実装を進めて、品質の高いシステムを構築しましょう！
