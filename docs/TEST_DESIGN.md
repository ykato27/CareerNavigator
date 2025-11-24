# テスト設計ドキュメント - CareerNavigator

**バージョン**: 2.0
**最終更新**: 2025-11-06
**対象プロジェクト**: skillnote-recommendation（スキルノート推薦システム）

---

## 目次

1. [概要](#概要)
2. [テスト戦略](#テスト戦略)
3. [テスト構成](#テスト構成)
4. [コンポーネント別テスト設計](#コンポーネント別テスト設計)
5. [テストデータ戦略](#テストデータ戦略)
6. [テスト実行環境](#テスト実行環境)
7. [カバレッジ目標](#カバレッジ目標)
8. [実装優先度](#実装優先度)

---

## 1. 概要

### 1.1 目的

このドキュメントは、CareerNavigator（スキルノート推薦システム）の包括的なテスト設計を定義します。品質保証とリグレッション防止のため、各コンポーネントに対する詳細なテストケースを提供します。

### 1.2 対象システム

**システム名**: スキルノート推薦システム

**主要機能**:
- 技術者のスキルプロファイル分析
- 力量（コンピテンシー）の推薦
- 機械学習ベース推薦（NMF）
- グラフベース推薦（RWR）
- ハイブリッド推薦（RWR + NMF + Content-Based）
- 推薦理由の生成

**主要コンポーネント**:
1. **Core Module**
   - データモデル（`models.py`）
   - データローダー（`data_loader.py`）
   - データ変換（`data_transformer.py`）
   - 評価器（`evaluator.py`）

2. **ML Module**
   - Matrix Factorization（`matrix_factorization.py`）
   - ML推薦エンジン（`ml_recommender.py`）
   - 多様性再ランキング（`diversity.py`）
   - コンテンツベース推薦（`content_based_recommender.py`）

3. **Graph Module**
   - 知識グラフ（`knowledge_graph.py`）
   - Random Walk with Restart（`random_walk.py`）
   - ハイブリッド推薦（`hybrid_recommender.py`）
   - カテゴリー階層（`category_hierarchy.py`）

### 1.3 現在のテスト状況

**既存テスト**:
- `tests/test_data_loader.py`
- `tests/test_data_transformer.py`
- `tests/test_evaluator.py`
- `tests/test_matrix_factorization.py`
- `tests/test_diversity.py`
- `tests/test_ml_recommender.py`

**カバレッジ**: 約40-50%（推定）

**課題**:
- グラフモジュールのテストが不足
- 統合テストが不足
- エッジケースのテストが不足

---

## 2. テスト戦略

### 2.1 テストレベル

| レベル | 目的 | スコープ |
|--------|------|----------|
| **単体テスト** | 個別関数/メソッドの正確性検証 | 全コンポーネント（80%以上のカバレッジ目標）|
| **統合テスト** | コンポーネント間の連携検証 | データパイプライン全体 |
| **E2Eテスト** | エンドツーエンドシナリオ検証 | データ読込→変換→推薦→出力 |

### 2.2 テストアプローチ

- **TDD（テスト駆動開発）**: 新規機能実装時に推奨
- **リグレッションテスト**: CI/CD パイプラインで自動実行
- **境界値分析**: 数値計算、レベル正規化等で実施
- **等価分割**: 力量タイプ（SKILL/EDUCATION/LICENSE）ごとに検証

### 2.3 テストツール

- **テストフレームワーク**: pytest 7.0+
- **モック/スタブ**: pytest-mock, unittest.mock
- **フィクスチャ**: pytest fixtures（conftest.py）
- **カバレッジ**: pytest-cov
- **データ生成**: pandas, numpy（テストデータ作成用）

---

## 3. テスト構成

### 3.1 テストディレクトリ構造

```
tests/
├── __init__.py
├── conftest.py                      # 共通フィクスチャ
├── test_models.py                   # データモデルテスト
├── test_data_loader.py              # データローダーテスト
├── test_data_transformer.py         # データ変換テスト
├── test_evaluator.py                # 評価器テスト
├── test_matrix_factorization.py     # NMFモデルテスト
├── test_ml_recommender.py           # ML推薦エンジンテスト
├── test_diversity.py                # 多様性再ランキングテスト
├── test_knowledge_graph.py          # 知識グラフテスト（NEW）
├── test_random_walk.py              # RWRテスト（NEW）
├── test_hybrid_recommender.py       # ハイブリッド推薦テスト（NEW）
├── test_integration.py              # 統合テスト
├── test_e2e.py                      # E2Eテスト
└── fixtures/                        # テストデータフィクスチャ
    ├── sample_members.csv
    ├── sample_competences.csv
    └── sample_acquired.csv
```

### 3.2 共通フィクスチャ（conftest.py）

```python
import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_members():
    """サンプルメンバーデータ"""
    return pd.DataFrame({
        'メンバーコード': ['M001', 'M002', 'M003'],
        'メンバー名': ['山田太郎', '佐藤花子', '鈴木一郎'],
        '職能等級': ['E3', 'E4', 'E3'],
        '役職': ['エンジニア', 'シニアエンジニア', 'エンジニア']
    })


@pytest.fixture
def sample_competences():
    """サンプル力量マスタデータ"""
    return pd.DataFrame({
        '力量コード': ['C001', 'C002', 'C003'],
        '力量名': ['Python基礎', 'Django', 'データ分析'],
        '力量タイプ': ['SKILL', 'SKILL', 'SKILL'],
        '力量カテゴリー名': ['プログラミング', 'Webフレームワーク', 'データサイエンス']
    })


@pytest.fixture
def sample_member_competence():
    """サンプル習得力量データ"""
    return pd.DataFrame({
        'メンバーコード': ['M001', 'M001', 'M002', 'M002', 'M003'],
        '力量コード': ['C001', 'C002', 'C001', 'C003', 'C001'],
        '正規化レベル': [0.8, 0.6, 1.0, 0.7, 0.5]
    })
```

---

## 4. コンポーネント別テスト設計

### 4.1 データローダー（`data_loader.py`）

#### テストファイル: `test_data_loader.py`

**テストケース**:
1. **ディレクトリスキャン**
   - 複数CSVファイルの自動検出
   - サブディレクトリの処理
   - 空ディレクトリの処理

2. **CSV読み込み**
   - エンコーディング自動判定
   - カラム構造の検証
   - 欠損値の処理

3. **データ結合**
   - 複数ファイルの結合
   - カラム不一致の検出

### 4.2 データ変換（`data_transformer.py`）

#### テストファイル: `test_data_transformer.py`

**テストケース**:
1. **力量マスタ変換**
   - 力量タイプの正規化
   - カテゴリー抽出
   - 重複除去

2. **メンバー習得力量変換**
   - レベル正規化
   - 日付パース

3. **スキルマトリックス生成**
   - ピボットテーブル作成
   - 欠損値の0埋め

### 4.3 ML推薦エンジン（`ml_recommender.py`）

#### テストファイル: `test_ml_recommender.py`

**テストケース**:
1. **推薦生成**
   - top_nパラメータの動作
   - 既習得力量の除外
   - 力量タイプフィルタ

2. **多様性再ランキング**
   - MMR戦略
   - カテゴリー戦略
   - ハイブリッド戦略

3. **参考人物検索**
   - 類似メンバーの発見
   - 異なるキャリアパスの発見

### 4.4 知識グラフ（`knowledge_graph.py`）

#### テストファイル: `test_knowledge_graph.py`（NEW）

**テストケース**:
1. **グラフ構築**
   - ノード追加（メンバー、力量、カテゴリー）
   - エッジ追加（acquired, belongs_to, similar, parent_of）
   - カテゴリー階層の構築

2. **クエリ機能**
   - `get_neighbors()`: 隣接ノードの取得
   - `get_node_info()`: ノード情報の取得
   - `get_member_acquired_competences()`: 習得力量の取得
   - `get_competence_category()`: カテゴリー取得

3. **メンバー間類似度**
   - コサイン類似度の計算
   - top_kフィルタリング
   - 閾値フィルタリング

### 4.5 Random Walk with Restart（`random_walk.py`）

#### テストファイル: `test_random_walk.py`（NEW）

**テストケース**:
1. **RWR推薦**
   - PageRankスコアの計算
   - 力量候補の抽出
   - フィルタリング

2. **フォールバック処理**
   - カテゴリーベース推薦
   - 類似メンバーベース推薦
   - パス生成

3. **推薦パス抽出**
   - k-shortest pathsアルゴリズム
   - パス長フィルタリング
   - 代替パス生成

4. **キャッシング**
   - PageRankキャッシュの動作
   - キャッシュクリア
   - キャッシュ統計

### 4.6 ハイブリッド推薦（`hybrid_recommender.py`）

#### テストファイル: `test_hybrid_recommender.py`（NEW）

**テストケース**:
1. **スコア融合**
   - RWR + NMF + Content-Basedの統合
   - 重み付けの動作
   - スコア正規化

2. **推薦生成**
   - Top-N選択
   - 力量タイプフィルタ
   - カテゴリーフィルタ

3. **推薦理由生成**
   - 複数手法からの理由統合
   - パス情報の含有

### 4.7 評価器（`evaluator.py`）

#### テストファイル: `test_evaluator.py`

**テストケース**:
1. **時系列分割**
   - 訓練データと評価データの分割
   - 日付ベース分割
   - ユーザーベース分割

2. **評価メトリクス**
   - Precision@K
   - Recall@K
   - NDCG@K
   - MRR
   - MAP

3. **多様性メトリクス**
   - Gini Index
   - Novelty
   - Coverage

---

## 5. テストデータ戦略

### 5.1 テストデータの種類

1. **最小データセット**: 単体テスト用（3-5行）
2. **小規模データセット**: 統合テスト用（10-20行）
3. **中規模データセット**: E2Eテスト用（100行程度）

### 5.2 テストデータの配置

```
tests/fixtures/
├── minimal/          # 最小データセット
│   ├── members.csv
│   ├── competences.csv
│   └── acquired.csv
├── small/            # 小規模データセット
└── medium/           # 中規模データセット
```

### 5.3 データ生成戦略

- **実データの匿名化**: 本番データから個人情報を除去
- **合成データ生成**: Faker, numpy.randomを使用
- **境界値データ**: エッジケース検証用

---

## 6. テスト実行環境

### 6.1 ローカル環境

```bash
# 全テスト実行
uv run pytest tests/

# カバレッジ付き実行
uv run pytest --cov=skillnote_recommendation --cov-report=html tests/

# 特定のモジュールのみ
uv run pytest tests/test_random_walk.py -v
```

### 6.2 CI/CD環境

- **GitHub Actions**: プルリクエスト時に自動実行
- **カバレッジ閾値**: 80%以上
- **失敗時の動作**: マージをブロック

---

## 7. カバレッジ目標

| コンポーネント | 優先度 | カバレッジ目標 |
|---------------|--------|--------------|
| `models.py` | 中 | 70% |
| `data_loader.py` | 高 | 85% |
| `data_transformer.py` | 高 | 85% |
| `matrix_factorization.py` | 最高 | 90% |
| `ml_recommender.py` | 最高 | 90% |
| `diversity.py` | 高 | 85% |
| `knowledge_graph.py` | 最高 | 90% |
| `random_walk.py` | 最高 | 90% |
| `hybrid_recommender.py` | 最高 | 85% |
| `evaluator.py` | 高 | 85% |

**全体目標**: 80%以上

---

## 8. 実装優先度

### Phase 1: 基盤テスト（完了）

1. ✅ **データモデルテスト** (`test_models.py`)
2. ✅ **データローダーテスト** (`test_data_loader.py`)
3. ✅ **データ変換テスト** (`test_data_transformer.py`)

### Phase 2: ML推薦テスト（完了）

4. ✅ **Matrix Factorizationテスト** (`test_matrix_factorization.py`)
5. ✅ **多様性再ランキングテスト** (`test_diversity.py`)
6. ✅ **ML推薦エンジンテスト** (`test_ml_recommender.py`)

### Phase 3: グラフ推薦テスト（未実装）

7. ⏳ **知識グラフテスト** (`test_knowledge_graph.py`)
8. ⏳ **Random Walkテスト** (`test_random_walk.py`)
9. ⏳ **ハイブリッド推薦テスト** (`test_hybrid_recommender.py`)

### Phase 4: 統合・E2Eテスト（未実装）

10. ⏳ **統合テスト** (`test_integration.py`)
11. ⏳ **E2Eテスト** (`test_e2e.py`)

---

## まとめ

このテスト設計に従って段階的にテストを実装することで、CareerNavigatorシステムの品質を保証し、リグレッションを防止します。

**次のステップ**:
1. Phase 3のグラフ推薦テストを実装
2. Phase 4の統合テストを実装
3. カバレッジ80%達成
4. CI/CDパイプラインへの統合
