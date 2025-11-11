"""
SEM標準誤差計算のテスト

ヘッセ行列の数値微分による標準誤差計算が正しく機能するかをテストします。
"""

import pytest
import numpy as np
import pandas as pd
from skillnote_recommendation.ml.unified_sem_estimator import (
    UnifiedSEMEstimator,
    MeasurementModelSpec,
    StructuralModelSpec,
)


@pytest.fixture
def simple_sem_data():
    """シンプルなSEMテストデータ"""
    np.random.seed(42)
    n = 200

    # 潜在変数: 技術力
    latent_tech = np.random.normal(0, 1, n)

    # 観測変数: Python, SQL (技術力に依存)
    python_skill = 0.8 * latent_tech + np.random.normal(0, 0.3, n)
    sql_skill = 0.7 * latent_tech + np.random.normal(0, 0.3, n)

    data = pd.DataFrame({
        'Python': python_skill,
        'SQL': sql_skill,
    })

    return data


class TestSEMStandardErrors:
    """SEM標準誤差のテスト"""

    def test_hessian_computation_shape(self, simple_sem_data):
        """ヘッセ行列の形状が正しいかテスト"""
        # モデル仕様
        measurement = [
            MeasurementModelSpec('技術力', ['Python', 'SQL'], reference_indicator='Python')
        ]
        structural = []

        sem = UnifiedSEMEstimator(measurement, structural)

        # 推定を実行せずに、ヘッセ行列計算のみテスト
        S = simple_sem_data.cov().values
        n_params = 1 + 1 + 2  # Lambda(1個) + Psi(1個) + Theta(2個) = 4個

        # 初期パラメータ
        theta = np.array([0.7, 1.0, 0.5, 0.5])  # [λ_SQL, ψ_技術力, θ_Python, θ_SQL]

        hessian = sem._compute_hessian_numerical(S, theta)

        # 形状チェック
        assert hessian.shape == (n_params, n_params)

        # 対称行列チェック
        assert np.allclose(hessian, hessian.T, atol=1e-4)

    def test_hessian_positive_definite(self, simple_sem_data):
        """ヘッセ行列が正定値かテスト（最適化後）"""
        measurement = [
            MeasurementModelSpec('技術力', ['Python', 'SQL'], reference_indicator='Python')
        ]
        structural = []

        sem = UnifiedSEMEstimator(measurement, structural)
        sem.fit(simple_sem_data)

        # 推定後のパラメータでヘッセ行列を計算
        S = simple_sem_data.cov().values
        theta = sem._pack_params()

        hessian = sem._compute_hessian_numerical(S, theta)

        # 固有値がすべて正（正定値）
        eigenvalues = np.linalg.eigvals(hessian)

        # 最小固有値が正または微小な負（数値誤差許容）
        assert np.min(eigenvalues) > -1e-3, f"最小固有値: {np.min(eigenvalues)}"

    def test_standard_errors_computed(self, simple_sem_data):
        """標準誤差が正しく計算されるかテスト"""
        measurement = [
            MeasurementModelSpec('技術力', ['Python', 'SQL'], reference_indicator='Python')
        ]
        structural = []

        sem = UnifiedSEMEstimator(measurement, structural)
        sem.fit(simple_sem_data)

        # 標準誤差が計算されているか
        assert len(sem.params) > 0

        # すべてのパラメータに標準誤差が設定されているか
        for param_name, param in sem.params.items():
            assert param.std_error is not None
            assert param.std_error > 0, f"{param_name}の標準誤差が0以下"

    def test_p_values_computed(self, simple_sem_data):
        """p値が正しく計算されるかテスト"""
        measurement = [
            MeasurementModelSpec('技術力', ['Python', 'SQL'], reference_indicator='Python')
        ]
        structural = []

        sem = UnifiedSEMEstimator(measurement, structural)
        sem.fit(simple_sem_data)

        # すべてのパラメータにp値が設定されているか
        for param_name, param in sem.params.items():
            assert param.p_value is not None
            assert 0 <= param.p_value <= 1, f"{param_name}のp値が範囲外: {param.p_value}"

    def test_significant_parameters_detected(self, simple_sem_data):
        """統計的に有意なパラメータが検出されるかテスト"""
        measurement = [
            MeasurementModelSpec('技術力', ['Python', 'SQL'], reference_indicator='Python')
        ]
        structural = []

        sem = UnifiedSEMEstimator(measurement, structural)
        sem.fit(simple_sem_data)

        # λ_SQL→技術力 は有意なはず（真の値0.7）
        lambda_param = None
        for param_name, param in sem.params.items():
            if 'SQL' in param_name and '技術力' in param_name:
                lambda_param = param
                break

        assert lambda_param is not None, "λ_SQL→技術力 が見つかりません"
        assert lambda_param.is_significant, f"λ_SQL→技術力 が有意でない (p={lambda_param.p_value})"

    def test_confidence_intervals(self, simple_sem_data):
        """信頼区間が妥当かテスト（将来実装用）"""
        # 現在は実装していないため、スキップ
        pytest.skip("信頼区間の計算は将来実装予定")

    def test_bootstrap_fallback(self, simple_sem_data):
        """ブートストラップフォールバックが機能するかテスト"""
        measurement = [
            MeasurementModelSpec('技術力', ['Python', 'SQL'], reference_indicator='Python')
        ]
        structural = []

        sem = UnifiedSEMEstimator(measurement, structural)

        # ブートストラップメソッドを直接テスト
        S = simple_sem_data.cov().values
        theta = np.array([0.7, 1.0, 0.5, 0.5])

        se_bootstrap = sem._compute_standard_errors_bootstrap(S, theta, n_samples=10)

        # 標準誤差が計算される
        assert len(se_bootstrap) == len(theta)
        assert all(se > 0 for se in se_bootstrap)

    def test_standard_errors_with_structural_model(self):
        """構造モデルありの標準誤差計算テスト"""
        np.random.seed(42)
        n = 200

        # 潜在変数: 初級技術力 → 中級技術力
        latent_basic = np.random.normal(0, 1, n)
        latent_advanced = 0.6 * latent_basic + np.random.normal(0, 0.5, n)

        # 観測変数
        python_basic = 0.8 * latent_basic + np.random.normal(0, 0.3, n)
        sql_basic = 0.7 * latent_basic + np.random.normal(0, 0.3, n)

        web_advanced = 0.75 * latent_advanced + np.random.normal(0, 0.3, n)
        data_advanced = 0.65 * latent_advanced + np.random.normal(0, 0.3, n)

        data = pd.DataFrame({
            'Python基礎': python_basic,
            'SQL基礎': sql_basic,
            'Web開発': web_advanced,
            'データ分析': data_advanced,
        })

        # モデル仕様
        measurement = [
            MeasurementModelSpec('初級技術力', ['Python基礎', 'SQL基礎'], reference_indicator='Python基礎'),
            MeasurementModelSpec('中級技術力', ['Web開発', 'データ分析'], reference_indicator='Web開発'),
        ]
        structural = [
            StructuralModelSpec('初級技術力', '中級技術力'),
        ]

        sem = UnifiedSEMEstimator(measurement, structural)
        sem.fit(data)

        # 構造係数の標準誤差が計算されているか
        beta_param = None
        for param_name, param in sem.params.items():
            if 'β_初級技術力→中級技術力' == param_name:
                beta_param = param
                break

        assert beta_param is not None, "構造係数が見つかりません"
        assert beta_param.std_error is not None
        assert beta_param.std_error > 0

        # 構造係数が有意（真の値0.6）
        # サンプルサイズ200で0.6の効果は十分検出されるはず
        assert beta_param.is_significant or beta_param.p_value < 0.1, (
            f"構造係数が有意でない (β={beta_param.value:.3f}, se={beta_param.std_error:.3f}, p={beta_param.p_value:.3f})"
        )


class TestSEMStandardErrorsEdgeCases:
    """エッジケースのテスト"""

    def test_small_sample_size(self):
        """サンプルサイズが小さい場合"""
        np.random.seed(42)
        n = 50  # 小サンプル

        latent = np.random.normal(0, 1, n)
        x1 = 0.8 * latent + np.random.normal(0, 0.3, n)
        x2 = 0.7 * latent + np.random.normal(0, 0.3, n)

        data = pd.DataFrame({'X1': x1, 'X2': x2})

        measurement = [
            MeasurementModelSpec('F', ['X1', 'X2'], reference_indicator='X1')
        ]

        sem = UnifiedSEMEstimator(measurement, [])
        sem.fit(data)

        # 警告が出るが、計算は完了する
        assert sem.is_fitted

    def test_near_singular_hessian(self):
        """ヘッセ行列が特異に近い場合（多重共線性）"""
        np.random.seed(42)
        n = 100

        latent = np.random.normal(0, 1, n)
        # X1とX2がほぼ完全相関
        x1 = 0.9 * latent + np.random.normal(0, 0.1, n)
        x2 = 0.9 * latent + np.random.normal(0, 0.1, n)

        data = pd.DataFrame({'X1': x1, 'X2': x2})

        measurement = [
            MeasurementModelSpec('F', ['X1', 'X2'], reference_indicator='X1')
        ]

        sem = UnifiedSEMEstimator(measurement, [])

        # エラーなく実行される（正定値化により対処）
        try:
            sem.fit(data)
            assert sem.is_fitted
        except Exception as e:
            pytest.fail(f"特異に近いヘッセ行列でエラー: {e}")
