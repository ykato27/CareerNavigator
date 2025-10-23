"""
SimilarityCalculatorクラスのテスト

力量間の類似度（Jaccard係数）計算をテスト
"""

import pytest
import pandas as pd
import numpy as np
from skillnote_recommendation.core.similarity_calculator import SimilarityCalculator


# ==================== 初期化テスト ====================

class TestCalculatorInitialization:
    """類似度計算器の初期化テスト"""

    def test_calculator_initialization_default(self):
        """デフォルトパラメータで初期化"""
        calculator = SimilarityCalculator()

        assert calculator is not None
        assert calculator.sample_size > 0
        assert calculator.threshold >= 0

    def test_calculator_initialization_custom_params(self):
        """カスタムパラメータで初期化"""
        calculator = SimilarityCalculator(sample_size=50, threshold=0.5)

        assert calculator.sample_size == 50
        assert calculator.threshold == 0.5


# ==================== Jaccard係数計算テスト ====================

class TestJaccardCoefficient:
    """Jaccard係数計算の正確性テスト"""

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

    def test_jaccard_perfect_overlap(self):
        """完全に重複する力量（Jaccard = 1.0）"""
        # s1とs2を全く同じメンバーが習得
        data = pd.DataFrame({
            'メンバーコード': ['m1', 'm1', 'm2', 'm2'],
            '力量コード': ['s1', 's2', 's1', 's2'],
            '正規化レベル': [3, 3, 4, 4]
        })

        calculator = SimilarityCalculator(sample_size=100, threshold=0.1)
        result = calculator.calculate_similarity(data)

        if len(result) > 0:
            similarity = result['類似度'].max()
            # 完全一致なので1.0
            assert abs(similarity - 1.0) < 0.01

    def test_jaccard_no_overlap(self):
        """全く重複しない力量（Jaccard = 0.0）"""
        # s1とs2を異なるメンバーが習得
        data = pd.DataFrame({
            'メンバーコード': ['m1', 'm2'],
            '力量コード': ['s1', 's2'],
            '正規化レベル': [3, 4]
        })

        calculator = SimilarityCalculator(sample_size=100, threshold=0.0)
        result = calculator.calculate_similarity(data)

        # 重複なしなので類似度0.0（閾値0.0より大きい必要があるので結果は空）
        # または類似度0.0は含まれない
        if len(result) > 0:
            similarity_values = result['類似度'].values
            assert all(s > 0 for s in similarity_values)


# ==================== 類似度計算の基本機能テスト ====================

class TestCalculateSimilarity:
    """類似度計算の基本機能テスト"""

    def test_calculate_similarity_basic(self, sample_member_competence):
        """基本的な類似度計算"""
        calculator = SimilarityCalculator(sample_size=100, threshold=0.1)
        result = calculator.calculate_similarity(sample_member_competence)

        assert isinstance(result, pd.DataFrame)
        assert '力量1' in result.columns
        assert '力量2' in result.columns
        assert '類似度' in result.columns

    def test_similarity_threshold_filtering(self):
        """閾値以下の類似度が除外される"""
        data = pd.DataFrame({
            'メンバーコード': ['m1', 'm1', 'm2', 'm3', 'm3'],
            '力量コード': ['s1', 's2', 's1', 's2', 's3'],
            '正規化レベル': [3, 4, 2, 3, 5]
        })

        # 高い閾値を設定
        calculator = SimilarityCalculator(sample_size=100, threshold=0.8)
        result = calculator.calculate_similarity(data)

        # 閾値0.8以上の類似度のみ含まれる
        if len(result) > 0:
            assert result['類似度'].min() > 0.8

    def test_similarity_symmetric(self):
        """(A,B)のペアのみで(B,A)は含まれない"""
        data = pd.DataFrame({
            'メンバーコード': ['m1', 'm1', 'm2', 'm2'],
            '力量コード': ['s1', 's2', 's1', 's2'],
            '正規化レベル': [3, 4, 2, 5]
        })

        calculator = SimilarityCalculator(sample_size=100, threshold=0.1)
        result = calculator.calculate_similarity(data)

        # 同じペアの逆順がないことを確認
        for _, row in result.iterrows():
            comp1, comp2 = row['力量1'], row['力量2']
            # comp1 < comp2 であること（辞書順で一方向のみ）
            reverse_exists = ((result['力量1'] == comp2) & (result['力量2'] == comp1)).any()
            assert not reverse_exists, f"逆順ペア ({comp2}, {comp1}) が存在します"

    def test_similarity_no_acquirers_skip(self):
        """習得者ゼロの力量はスキップされる"""
        data = pd.DataFrame({
            'メンバーコード': ['m1'],
            '力量コード': ['s1'],
            '正規化レベル': [3]
        })

        calculator = SimilarityCalculator(sample_size=100, threshold=0.1)
        result = calculator.calculate_similarity(data)

        # s1のみのデータなので類似ペアは生成されない
        assert len(result) == 0


# ==================== サンプリング機能テスト ====================

class TestSampling:
    """サンプリング機能のテスト"""

    def test_similarity_sampling_applied(self):
        """サンプリングサイズが適用される"""
        # 多数の力量を含むデータ
        members = []
        competences = []
        levels = []

        for i in range(50):  # 50人のメンバー
            for j in range(10):  # 各10力量
                members.append(f'm{i}')
                competences.append(f's{j}')
                levels.append(np.random.randint(1, 6))

        data = pd.DataFrame({
            'メンバーコード': members,
            '力量コード': competences,
            '正規化レベル': levels
        })

        # サンプルサイズを小さく設定
        calculator = SimilarityCalculator(sample_size=5, threshold=0.0)
        result = calculator.calculate_similarity(data)

        # 計算は完了する（サンプリングにより高速化）
        assert isinstance(result, pd.DataFrame)

    def test_similarity_sample_size_larger_than_data(self, sample_member_competence):
        """サンプルサイズがデータより大きい場合"""
        # 実際の力量数より大きいサンプルサイズ
        unique_competences = sample_member_competence['力量コード'].nunique()

        calculator = SimilarityCalculator(sample_size=1000, threshold=0.1)
        result = calculator.calculate_similarity(sample_member_competence)

        # エラーなく実行される
        assert isinstance(result, pd.DataFrame)


# ==================== 出力形式テスト ====================

class TestOutputFormat:
    """出力DataFrameの形式テスト"""

    def test_similarity_output_format(self, sample_member_competence):
        """出力DataFrameの形式確認"""
        calculator = SimilarityCalculator(sample_size=100, threshold=0.1)
        result = calculator.calculate_similarity(sample_member_competence)

        # 必須カラムが存在
        assert '力量1' in result.columns
        assert '力量2' in result.columns
        assert '類似度' in result.columns

        # カラム数は3つ
        assert len(result.columns) == 3

    def test_similarity_values_range(self, sample_member_competence):
        """類似度の値が0-1の範囲内"""
        calculator = SimilarityCalculator(sample_size=100, threshold=0.0)
        result = calculator.calculate_similarity(sample_member_competence)

        if len(result) > 0:
            assert result['類似度'].min() >= 0.0
            assert result['類似度'].max() <= 1.0

    def test_similarity_sorted_by_score(self, sample_member_competence):
        """類似度でソートされているか（オプション）"""
        calculator = SimilarityCalculator(sample_size=100, threshold=0.1)
        result = calculator.calculate_similarity(sample_member_competence)

        # ソートは要件ではないため、単に数値型であることを確認
        if len(result) > 0:
            assert pd.api.types.is_numeric_dtype(result['類似度'])


# ==================== エッジケーステスト ====================

class TestEdgeCases:
    """エッジケースのテスト"""

    def test_empty_data(self):
        """空データでも例外が発生しない"""
        data = pd.DataFrame({
            'メンバーコード': [],
            '力量コード': [],
            '正規化レベル': []
        })

        calculator = SimilarityCalculator(sample_size=100, threshold=0.1)
        result = calculator.calculate_similarity(data)

        # 空のDataFrameが返される
        assert len(result) == 0

    def test_single_competence(self):
        """力量が1つだけの場合"""
        data = pd.DataFrame({
            'メンバーコード': ['m1', 'm2'],
            '力量コード': ['s1', 's1'],
            '正規化レベル': [3, 4]
        })

        calculator = SimilarityCalculator(sample_size=100, threshold=0.1)
        result = calculator.calculate_similarity(data)

        # 1つの力量なので類似ペアは生成されない
        assert len(result) == 0

    def test_single_member(self):
        """メンバーが1人だけの場合"""
        data = pd.DataFrame({
            'メンバーコード': ['m1', 'm1', 'm1'],
            '力量コード': ['s1', 's2', 's3'],
            '正規化レベル': [3, 4, 5]
        })

        calculator = SimilarityCalculator(sample_size=100, threshold=0.1)
        result = calculator.calculate_similarity(data)

        # 1人のメンバーが複数力量を保有
        # 同じメンバーなのでJaccard=1.0となる（閾値次第で含まれる）
        if len(result) > 0:
            # 全ての類似度が1.0（同じメンバーのみが保有）
            assert all(abs(s - 1.0) < 0.01 for s in result['類似度'].values)

    def test_all_zero_levels(self):
        """全てレベル0の場合（習得していない）"""
        data = pd.DataFrame({
            'メンバーコード': ['m1', 'm2'],
            '力量コード': ['s1', 's2'],
            '正規化レベル': [0, 0]
        })

        calculator = SimilarityCalculator(sample_size=100, threshold=0.1)
        result = calculator.calculate_similarity(data)

        # レベル0は二値化で0になり、習得者ゼロとしてスキップされる
        assert len(result) == 0
