"""
UnifiedSEMEstimatorのテスト
"""

import sys
import numpy as np
import pandas as pd

# pytest があればimport、なければダミークラス
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    class pytest:
        @staticmethod
        def fixture(func):
            return func
        @staticmethod
        def raises(*args, **kwargs):
            class RaisesContext:
                def __enter__(self):
                    return self
                def __exit__(self, exc_type, exc_val, exc_tb):
                    if exc_type is None:
                        raise AssertionError("Expected exception was not raised")
                    return True
            return RaisesContext()

from skillnote_recommendation.ml.unified_sem_estimator import (
    UnifiedSEMEstimator,
    MeasurementModelSpec,
    StructuralModelSpec,
    SEMFitIndices,
)


class TestUnifiedSEMEstimator:
    """UnifiedSEMEstimatorのテストクラス"""

    @pytest.fixture
    def sample_data(self):
        """
        テスト用のサンプルデータを生成

        構造:
        - 初級力量 → Python基礎, SQL基礎
        - 中級力量 → Web開発, データ分析
        - 初級力量 → 中級力量
        """
        np.random.seed(42)
        n = 200

        # 潜在変数を生成
        beginner = np.random.normal(0, 1, n)
        intermediate = 0.7 * beginner + np.random.normal(0, 0.5, n)

        # 観測変数を生成
        data = pd.DataFrame({
            'Python基礎': 0.8 * beginner + np.random.normal(0, 0.3, n),
            'SQL基礎': 0.75 * beginner + np.random.normal(0, 0.35, n),
            'Web開発': 0.85 * intermediate + np.random.normal(0, 0.25, n),
            'データ分析': 0.80 * intermediate + np.random.normal(0, 0.30, n),
        })

        return data

    @pytest.fixture
    def simple_model_specs(self):
        """シンプルなモデル仕様"""
        measurement = [
            MeasurementModelSpec(
                '初級力量',
                ['Python基礎', 'SQL基礎'],
                reference_indicator='Python基礎'
            ),
            MeasurementModelSpec(
                '中級力量',
                ['Web開発', 'データ分析'],
                reference_indicator='Web開発'
            ),
        ]

        structural = [
            StructuralModelSpec('初級力量', '中級力量'),
        ]

        return measurement, structural

    def test_initialization(self, simple_model_specs):
        """初期化のテスト"""
        measurement, structural = simple_model_specs
        sem = UnifiedSEMEstimator(measurement, structural)

        assert len(sem.latent_vars) == 2
        assert len(sem.observed_vars) == 4
        assert sem.is_fitted is False

    def test_fit(self, sample_data, simple_model_specs):
        """推定のテスト"""
        measurement, structural = simple_model_specs
        sem = UnifiedSEMEstimator(measurement, structural)

        # 推定
        sem.fit(sample_data)

        assert sem.is_fitted is True
        assert sem.Lambda is not None
        assert sem.B is not None
        assert sem.Psi is not None
        assert sem.Theta is not None
        assert sem.fit_indices is not None

    def test_fit_indices(self, sample_data, simple_model_specs):
        """適合度指標のテスト"""
        measurement, structural = simple_model_specs
        sem = UnifiedSEMEstimator(measurement, structural)
        sem.fit(sample_data)

        fit = sem.fit_indices

        # 適合度指標が計算されているか
        assert fit.chi_square > 0
        assert fit.df > 0
        assert 0 <= fit.rmsea <= 1
        assert 0 <= fit.cfi <= 1
        assert 0 <= fit.tli <= 1.5  # TLIは1を超えることがある
        assert fit.aic > 0
        assert fit.bic > 0

        # 良好な適合か（シミュレーションデータなので期待）
        print(f"\n適合度指標:")
        print(f"  RMSEA: {fit.rmsea:.3f} (< 0.08: 良好)")
        print(f"  CFI: {fit.cfi:.3f} (> 0.90: 良好)")
        print(f"  TLI: {fit.tli:.3f} (> 0.90: 良好)")
        print(f"  SRMR: {fit.srmr:.3f} (< 0.08: 良好)")

    def test_get_skill_relationships(self, sample_data, simple_model_specs):
        """力量関係性の取得テスト"""
        measurement, structural = simple_model_specs
        sem = UnifiedSEMEstimator(measurement, structural)
        sem.fit(sample_data)

        relationships = sem.get_skill_relationships()

        assert len(relationships) == 1
        assert relationships.iloc[0]['from_skill'] == '初級力量'
        assert relationships.iloc[0]['to_skill'] == '中級力量'

        # 係数が0.7付近であることを確認（真の値）
        coef = relationships.iloc[0]['coefficient']
        print(f"\n構造係数: {coef:.3f} (真の値: 0.70)")
        assert 0.5 < coef < 0.9  # ある程度の範囲で正しい

    def test_get_indirect_effects(self, sample_data):
        """間接効果の計算テスト"""
        # 3層モデルを作成
        np.random.seed(42)
        n = 200

        # 潜在変数
        beginner = np.random.normal(0, 1, n)
        intermediate = 0.6 * beginner + np.random.normal(0, 0.4, n)
        advanced = 0.5 * intermediate + 0.2 * beginner + np.random.normal(0, 0.3, n)

        data = pd.DataFrame({
            'Python基礎': 0.8 * beginner + np.random.normal(0, 0.3, n),
            'SQL基礎': 0.75 * beginner + np.random.normal(0, 0.35, n),
            'Web開発': 0.85 * intermediate + np.random.normal(0, 0.25, n),
            'データ分析': 0.80 * intermediate + np.random.normal(0, 0.30, n),
            'システム設計': 0.9 * advanced + np.random.normal(0, 0.2, n),
            '機械学習': 0.85 * advanced + np.random.normal(0, 0.25, n),
        })

        measurement = [
            MeasurementModelSpec('初級力量', ['Python基礎', 'SQL基礎'], 'Python基礎'),
            MeasurementModelSpec('中級力量', ['Web開発', 'データ分析'], 'Web開発'),
            MeasurementModelSpec('上級力量', ['システム設計', '機械学習'], 'システム設計'),
        ]

        structural = [
            StructuralModelSpec('初級力量', '中級力量'),
            StructuralModelSpec('中級力量', '上級力量'),
            StructuralModelSpec('初級力量', '上級力量'),
        ]

        sem = UnifiedSEMEstimator(measurement, structural)
        sem.fit(data)

        indirect = sem.get_indirect_effects()

        print(f"\n間接効果:")
        print(indirect)

        # 初級→上級の間接効果が存在するはず
        indirect_beginner_to_advanced = indirect[
            (indirect['from_skill'] == '初級力量') &
            (indirect['to_skill'] == '上級力量')
        ]

        if len(indirect_beginner_to_advanced) > 0:
            ind_effect = indirect_beginner_to_advanced.iloc[0]['indirect_effect']
            print(f"初級→上級の間接効果: {ind_effect:.3f}")
            assert ind_effect > 0

    def test_predict_latent_scores(self, sample_data, simple_model_specs):
        """潜在変数スコアの予測テスト"""
        measurement, structural = simple_model_specs
        sem = UnifiedSEMEstimator(measurement, structural)
        sem.fit(sample_data)

        scores = sem.predict_latent_scores(sample_data)

        assert len(scores) == len(sample_data)
        assert list(scores.columns) == ['初級力量', '中級力量']

        print(f"\n潜在変数スコア（最初の5人）:")
        print(scores.head())

    def test_invalid_model_spec(self):
        """不正なモデル仕様のテスト"""
        # 重複した潜在変数
        measurement = [
            MeasurementModelSpec('力量A', ['x1', 'x2']),
            MeasurementModelSpec('力量A', ['x3', 'x4']),  # 重複
        ]

        with pytest.raises(ValueError, match="重複"):
            UnifiedSEMEstimator(measurement, [])

        # 存在しない潜在変数への構造パス
        measurement = [
            MeasurementModelSpec('力量A', ['x1', 'x2']),
        ]
        structural = [
            StructuralModelSpec('力量A', '力量B'),  # 力量Bは存在しない
        ]

        with pytest.raises(ValueError, match="存在しません"):
            UnifiedSEMEstimator(measurement, structural)

    def test_unfitted_error(self, simple_model_specs):
        """未推定状態でのエラーテスト"""
        measurement, structural = simple_model_specs
        sem = UnifiedSEMEstimator(measurement, structural)

        with pytest.raises(ValueError, match="推定されていません"):
            sem.get_skill_relationships()

        with pytest.raises(ValueError, match="推定されていません"):
            sem.get_indirect_effects()


if __name__ == '__main__':
    # 簡易テスト実行
    test = TestUnifiedSEMEstimator()

    print("=" * 60)
    print("UnifiedSEMEstimator テスト")
    print("=" * 60)

    # サンプルデータ生成
    data = test.sample_data()
    print(f"\nサンプルデータ（n={len(data)}）:")
    print(data.head())

    # モデル推定
    measurement, structural = test.simple_model_specs()
    sem = UnifiedSEMEstimator(measurement, structural)

    print("\nモデル推定中...")
    sem.fit(data)

    # 適合度指標
    print("\n【適合度指標】")
    fit = sem.fit_indices
    print(f"  χ² = {fit.chi_square:.2f}, df = {fit.df}, p = {fit.p_value:.3f}")
    print(f"  RMSEA = {fit.rmsea:.3f} (90% CI: [{fit.rmsea_ci_lower:.3f}, {fit.rmsea_ci_upper:.3f}])")
    print(f"  CFI = {fit.cfi:.3f}")
    print(f"  TLI = {fit.tli:.3f}")
    print(f"  SRMR = {fit.srmr:.3f}")
    print(f"  AIC = {fit.aic:.2f}")
    print(f"  BIC = {fit.bic:.2f}")

    if fit.is_excellent_fit():
        print("\n  ✅ 優れた適合度です！")
    elif fit.is_good_fit():
        print("\n  ✅ 良好な適合度です")
    else:
        print("\n  ⚠️  適合度が低いです")

    # 力量関係性
    print("\n【力量同士の関係性】")
    relationships = sem.get_skill_relationships()
    print(relationships.to_string(index=False))

    # パラメータ
    print("\n【推定パラメータ】")
    print("\nファクターローディング（Λ）:")
    print(sem.Lambda)
    print("\n構造係数（B）:")
    print(sem.B)

    print("\n" + "=" * 60)
    print("テスト完了")
    print("=" * 60)
