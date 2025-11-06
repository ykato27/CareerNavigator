# テスト実装クイックスタートガイド

CareerNavigatorプロジェクトのテスト実装をすぐに始めるためのガイドです。

---

## 📋 目次

1. [セットアップ](#セットアップ)
2. [テスト実行](#テスト実行)
3. [作成済みファイル](#作成済みファイル)
4. [次のステップ](#次のステップ)
5. [テスト作成の例](#テスト作成の例)

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
- コンポーネント別テストケース設計（合計100+ケース）
- テストデータ戦略
- 実装優先度

**使い方**: テスト実装前に必ず確認し、設計に従って実装してください。

### 2. tests/conftest.py

**内容**: 共通テストフィクスチャ
- サンプルデータフィクスチャ（メンバー、力量、習得データ等）
- 一時ディレクトリフィクスチャ
- CSVファイル生成ヘルパー
- カスタムpytestマーカー定義

**使い方**: 他のテストファイルから自動的にインポートされます。新しいフィクスチャを追加する場合はこのファイルに記述してください。

### 3. tests/test_data_loader.py

**内容**: DataLoaderクラスの完全なテスト実装例（23テストケース）
- カラム名クリーニング（6テスト）
- CSV読み込み（5テスト）
- 全データ読み込み（3テスト）
- データ検証（4テスト）
- 初期化（2テスト）
- エッジケース（3テスト）

**使い方**: テスト実装の参考例として活用してください。

**実行結果**:
```
✅ 23 passed (全テスト成功)
```

### 4. tests/test_basic.py

**内容**: 既存の基本テスト（6テストケース）
- Config設定
- データモデル（Member, Competence, Recommendation）

**状態**: 既存テストも正常動作中

---

## 🚀 次のステップ

### フェーズ1: データ変換テストの実装（優先度: 高）

**ファイル**: `tests/test_data_transformer.py`

**実装すべきテスト**:
1. レベル正規化テスト
   - SKILLタイプ（1-5の整数）
   - EDUCATION/LICENSEタイプ（●→1, 空→0）
2. 統合力量マスタ作成
3. メンバー習得力量データ作成
4. スキルマトリクス作成

**サンプルコード**:

```python
"""
tests/test_data_transformer.py
"""
import pytest
import pandas as pd
from skillnote_recommendation.core.data_transformer import DataTransformer


class TestNormalizeLevel:
    """レベル正規化のテスト"""

    @pytest.mark.parametrize("level,expected", [
        ('1', 1), ('3', 3), ('5', 5),
        ('invalid', 0), ('', 0), (None, 0)
    ])
    def test_normalize_level_skill(self, level, expected):
        """SKILLタイプのレベル正規化"""
        result = DataTransformer.normalize_level(level, 'SKILL')
        assert result == expected

    @pytest.mark.parametrize("comp_type,level,expected", [
        ('EDUCATION', '●', 1),
        ('EDUCATION', '', 0),
        ('LICENSE', '●', 1),
        ('LICENSE', None, 0),
    ])
    def test_normalize_level_non_skill(self, comp_type, level, expected):
        """EDUCATION/LICENSEタイプのレベル正規化"""
        result = DataTransformer.normalize_level(level, comp_type)
        assert result == expected


class TestCreateCompetenceMaster:
    """統合力量マスタ作成のテスト"""

    def test_create_competence_master(self, sample_skills, sample_education,
                                       sample_license, sample_categories):
        """統合力量マスタが正しく作成される"""
        data = {
            'skills': sample_skills,
            'education': sample_education,
            'license': sample_license,
            'categories': sample_categories
        }

        transformer = DataTransformer()
        master = transformer.create_competence_master(data)

        # 3タイプすべてが含まれること
        assert 'SKILL' in master['力量タイプ'].values
        assert 'EDUCATION' in master['力量タイプ'].values
        assert 'LICENSE' in master['力量タイプ'].values

        # 件数確認
        skill_count = len(sample_skills)
        edu_count = len(sample_education)
        lic_count = len(sample_license)
        assert len(master) == skill_count + edu_count + lic_count
```

**実行**:
```bash
uv run pytest tests/test_data_transformer.py -v
```

---

### フェーズ2: 類似度計算テストの実装（優先度: 高）

**ファイル**: `tests/test_similarity_calculator.py`

**実装すべきテスト**:
1. Jaccard係数の正確性検証
2. 類似度閾値フィルタリング
3. サンプリング機能
4. 対称性（(A,B)のみで(B,A)は含まれない）

**サンプルコード**:

```python
"""
tests/test_similarity_calculator.py
"""
import pytest
import pandas as pd
import numpy as np
from skillnote_recommendation.core.similarity_calculator import SimilarityCalculator


class TestJaccardCoefficient:
    """Jaccard係数計算のテスト"""

    def test_jaccard_coefficient_accuracy(self):
        """Jaccard係数の正確性を手計算で検証"""
        # テストデータ: s1とs2の習得者が一部重複
        # s1: {m1, m2} (2人)
        # s2: {m1, m3} (2人)
        # intersection: {m1} (1人)
        # union: {m1, m2, m3} (3人)
        # Jaccard = 1/3 = 0.333...

        data = pd.DataFrame({
            'メンバーコード': ['m1', 'm1', 'm2', 'm3'],
            '力量コード': ['s1', 's2', 's1', 's2'],
            '正規化レベル': [3, 4, 2, 3]
        })

        calculator = SimilarityCalculator(sample_size=100, threshold=0.1)
        result = calculator.calculate_similarity(data)

        # s1-s2の類似度を取得
        similarity = result[
            ((result['力量1'] == 's1') & (result['力量2'] == 's2')) |
            ((result['力量1'] == 's2') & (result['力量2'] == 's1'))
        ]['類似度'].values[0]

        # 小数第2位まで一致することを確認
        assert abs(similarity - 1/3) < 0.01

    def test_similarity_threshold(self):
        """閾値以下の類似度が除外される"""
        data = pd.DataFrame({
            'メンバーコード': ['m1', 'm2', 'm3', 'm4'],
            '力量コード': ['s1', 's1', 's2', 's2'],
            '正規化レベル': [3, 2, 4, 5]
        })

        # 閾値を高く設定
        calculator = SimilarityCalculator(sample_size=100, threshold=0.8)
        result = calculator.calculate_similarity(data)

        # 高い閾値で類似ペアが少なくなる（または0件）
        assert len(result) >= 0
        if len(result) > 0:
            assert result['類似度'].min() > 0.8
```

---

### フェーズ3: 推薦エンジンテストの実装（優先度: 最高）

**ファイル**: `tests/test_recommendation_engine.py`

**実装すべきテスト**:
1. カテゴリ重要度計算
2. 習得容易性計算
3. 人気度計算
4. 優先度スコア計算
5. 推薦理由生成
6. 推薦実行とソート

---

## 💡 テスト作成の例

### パターン1: 単純な関数テスト

```python
def test_simple_function():
    """説明文"""
    # Arrange (準備)
    input_value = "test"

    # Act (実行)
    result = my_function(input_value)

    # Assert (検証)
    assert result == "expected_output"
```

### パターン2: フィクスチャを使用したテスト

```python
def test_with_fixture(sample_members):
    """フィクスチャを使用したテスト"""
    # sample_membersはconftest.pyで定義されたフィクスチャ
    assert len(sample_members) == 5
    assert 'メンバーコード' in sample_members.columns
```

### パターン3: パラメータ化テスト

```python
@pytest.mark.parametrize("input_val,expected", [
    (1, 1),
    (3, 3),
    (5, 5),
    ('invalid', 0),
])
def test_with_parameters(input_val, expected):
    """複数のパターンを一度にテスト"""
    result = normalize_value(input_val)
    assert result == expected
```

### パターン4: 例外テスト

```python
def test_exception_raised():
    """例外が発生することを確認"""
    with pytest.raises(ValueError) as exc_info:
        dangerous_function("invalid")

    assert "エラーメッセージ" in str(exc_info.value)
```

### パターン5: 一時ファイルを使用したテスト

```python
def test_file_operation(tmp_path):
    """一時ファイルを使ったテスト"""
    # tmp_pathはpytestが提供する一時ディレクトリ
    test_file = tmp_path / "test.csv"
    df = pd.DataFrame({'col': [1, 2, 3]})
    df.to_csv(test_file, index=False)

    # ファイル操作をテスト
    result = load_csv_file(test_file)
    assert len(result) == 3
```

---

## 📊 現在のテスト状況

### 実装済み

| テストファイル | テスト数 | 状態 | カバレッジ |
|--------------|---------|------|-----------|
| `test_basic.py` | 6 | ✅ 全成功 | Config, Models |
| `test_data_loader.py` | 23 | ✅ 全成功 | DataLoader完全カバー |
| **合計** | **29** | ✅ | **約30%** |

### 未実装（優先順）

1. `test_data_transformer.py` - 18テスト（高優先度）
2. `test_similarity_calculator.py` - 12テスト（高優先度）
3. `test_recommendation_engine.py` - 24テスト（最高優先度）
4. `test_recommendation_system.py` - 12テスト（高優先度）
5. `test_integration.py` - 5テスト（中優先度）
6. `test_e2e.py` - 5テスト（中優先度）

**目標**: 合計100+テストケース、80%以上のカバレッジ

---

## 🎯 ベストプラクティス

### 1. テストの命名規則

```python
# ✅ 良い例
def test_normalize_level_skill_with_valid_value():
    """有効な値でのSKILLレベル正規化"""
    pass

# ❌ 悪い例
def test1():
    pass
```

### 2. Arrange-Act-Assert パターン

```python
def test_calculate_score():
    # Arrange: テストデータを準備
    input_data = create_test_data()

    # Act: テスト対象を実行
    result = calculate_score(input_data)

    # Assert: 結果を検証
    assert result > 0
    assert result <= 10
```

### 3. 1テスト1検証

```python
# ✅ 良い例: 1つの機能を明確にテスト
def test_category_importance_in_range():
    """カテゴリ重要度が0-10の範囲内"""
    score = calculate_category_importance('cat01')
    assert 0 <= score <= 10

# ❌ 悪い例: 複数の異なる機能を1つのテストで検証
def test_everything():
    assert func1() == 1
    assert func2() == 2
    assert func3() == 3
```

### 4. フィクスチャの活用

```python
# conftest.pyで定義
@pytest.fixture
def sample_ml_recommender(sample_members, sample_competence_master,
                          sample_member_competence):
    """ML推薦エンジンのフィクスチャ"""
    from skillnote_recommendation.ml.ml_recommender import MLRecommender

    return MLRecommender.build(
        member_competence=sample_member_competence,
        competence_master=sample_competence_master,
        member_master=sample_members,
        use_preprocessing=False,
        use_tuning=False
    )

# テストで使用
def test_recommend(sample_ml_recommender):
    """sample_ml_recommenderフィクスチャを使用"""
    recommendations = sample_ml_recommender.recommend('m001', top_n=5, use_diversity=False)
    assert len(recommendations) <= 5
```

---

## 🔍 デバッグのヒント

### テストが失敗した場合

```bash
# 詳細なエラー情報を表示
uv run pytest tests/test_data_loader.py::test_name -v --tb=long

# pdbデバッガを起動
uv run pytest tests/test_data_loader.py::test_name --pdb

# 最初の失敗で停止
uv run pytest tests/ -x

# 失敗したテストのみ再実行
uv run pytest tests/ --lf
```

### printデバッグ

```python
def test_with_debug(sample_members):
    """デバッグ情報を出力"""
    print(f"\nDataFrame shape: {sample_members.shape}")
    print(f"Columns: {sample_members.columns.tolist()}")

    result = process_data(sample_members)

    print(f"Result: {result}")
    assert result is not None
```

実行時に `-s` オプションを追加：
```bash
uv run pytest tests/test_file.py::test_with_debug -v -s
```

---

## 📚 参考資料

- **TEST_DESIGN.md**: 詳細なテスト設計ドキュメント
- **pytest公式ドキュメント**: https://docs.pytest.org/
- **pandas testing**: https://pandas.pydata.org/docs/reference/api/pandas.testing.assert_frame_equal.html

---

## ✅ チェックリスト

テスト実装前に確認：

- [ ] TEST_DESIGN.mdを読んで全体像を把握
- [ ] 開発環境がセットアップ済み（`uv sync --dev`）
- [ ] 既存テストが動作することを確認（`uv run pytest tests/`）
- [ ] conftest.pyのフィクスチャを理解
- [ ] test_data_loader.pyを参考例として確認

テスト実装中：

- [ ] 明確なテスト名とdocstring
- [ ] Arrange-Act-Assertパターンに従う
- [ ] エッジケースも考慮
- [ ] 必要に応じてフィクスチャを追加

テスト完了後：

- [ ] 全テストが成功（`uv run pytest tests/ -v`）
- [ ] カバレッジを確認（`uv run pytest --cov`）
- [ ] コードレビュー依頼

---

**作成日**: 2025-10-23
**バージョン**: 1.0
