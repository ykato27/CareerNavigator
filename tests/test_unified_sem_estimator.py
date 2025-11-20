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

        # measurement_specsとstructural_specsの確認
        assert len(sem.measurement_specs) == 2
        assert len(sem.structural_specs) == 1
        assert sem.is_fitted is False

    def test_fit(self, sample_data, simple_model_specs):
        """推定のテスト"""
        measurement, structural = simple_model_specs
        sem = UnifiedSEMEstimator(measurement, structural)

        # 推定
        sem.fit(sample_data)

        assert sem.is_fitted is True
        assert sem.model is not None  # semopyモデル
        assert sem.parameters_ is not None  # パラメータ辞書
        assert sem.fit_indices_ is not None  # 適合度指標

    def test_fit_indices(self, sample_data, simple_model_specs):
        """適合度指標のテスト"""
        measurement, structural = simple_model_specs
        sem = UnifiedSEMEstimator(measurement, structural)
        sem.fit(sample_data)

        fit = sem.fit_indices_

        # 適合度指標が計算されているか
        assert fit.chi_square >= 0
        assert fit.df >= 0
        assert 0 <= fit.rmsea <= 1
        assert 0 <= fit.cfi <= 1.05  # CFIは1を超えることがある
        assert 0 <= fit.tli <= 1.5  # TLIは1を超えることがある
        assert fit.aic > 0
        assert fit.bic > 0

        # 良好な適合か（シミュレーションデータなので期待）
        print(f"\n適合度指標:")
        print(f"  RMSEA: {fit.rmsea:.3f} (< 0.08: 良好)")
        print(f"  CFI: {fit.cfi:.3f} (> 0.90: 良好)")
        print(f"  TLI: {fit.tli:.3f} (> 0.90: 良好)")

    def test_get_parameters(self, sample_data, simple_model_specs):
        """パラメータ取得のテスト"""
        measurement, structural = simple_model_specs
        sem = UnifiedSEMEstimator(measurement, structural)
        sem.fit(sample_data)

        # paramsプロパティでパラメータを取得
        params = sem.params

        assert len(params) > 0

        # 構造パラメータの確認
        param_name = "中級力量 ~ 初級力量"
        assert param_name in params

        # 係数が0.7付近であることを確認（真の値）
        coef = params[param_name].value
        print(f"\n構造係数: {coef:.3f} (真の値: 0.70)")
        assert 0.5 < coef < 0.9  # ある程度の範囲で正しい

    # TODO: 間接効果の計算機能は将来実装予定
    # def test_get_indirect_effects(self, sample_data):
    #     """間接効果の計算テスト（未実装）"""
    #     pass

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

    def test_unfitted_error(self, sample_data, simple_model_specs):
        """未推定状態でのエラーテスト"""
        measurement, structural = simple_model_specs
        sem = UnifiedSEMEstimator(measurement, structural)

        # predict_latent_scores()を未推定状態で呼ぶとエラー
        with pytest.raises(RuntimeError, match="学習されていません"):
            sem.predict_latent_scores(sample_data)


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
    fit = sem.fit_indices_
    print(f"  χ² = {fit.chi_square:.2f}, df = {fit.df}, p = {fit.p_value:.3f}")
    print(f"  RMSEA = {fit.rmsea:.3f}")
    print(f"  CFI = {fit.cfi:.3f}")
    print(f"  TLI = {fit.tli:.3f}")
    print(f"  AIC = {fit.aic:.2f}")
    print(f"  BIC = {fit.bic:.2f}")

    if fit.is_excellent_fit():
        print("\n  ✅ 優れた適合度です！")
    elif fit.is_good_fit():
        print("\n  ✅ 良好な適合度です")
    else:
        print("\n  ⚠️  適合度が低いです")

    # パラメータ
    print("\n【推定パラメータ】")
    print("\n構造パラメータ:")
    param_name = "中級力量 ~ 初級力量"
    if param_name in sem.params:
        param = sem.params[param_name]
        print(f"  {param_name}: {param.value:.3f} (SE: {param.std_error:.3f})")

    # 潜在変数スコア
    print("\n【潜在変数スコア（最初の5件）】")
    scores = sem.predict_latent_scores(data)
    print(scores.head())

    print("\n" + "=" * 60)
    print("テスト完了")
    print("=" * 60)
