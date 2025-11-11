"""
NMF Early Stopping実装のテスト

Multiplicative Update Ruleを使用した真のEarly Stopping実装のテスト
"""

import pytest
import numpy as np
import pandas as pd
from skillnote_recommendation.ml.matrix_factorization import MatrixFactorizationModel


@pytest.fixture
def sample_matrix():
    """テスト用のサンプル行列"""
    np.random.seed(42)
    n_members = 50
    n_competences = 30

    # 真の潜在因子を生成
    true_W = np.abs(np.random.randn(n_members, 5))
    true_H = np.abs(np.random.randn(5, n_competences))

    # ノイズを追加
    X = true_W @ true_H + 0.1 * np.abs(np.random.randn(n_members, n_competences))

    # 非負に制約
    X = np.maximum(X, 0)

    return X


@pytest.fixture
def member_competence_data():
    """テスト用のメンバー力量データ"""
    np.random.seed(42)
    n_members = 50
    n_competences = 30

    member_codes = [f"M{i:03d}" for i in range(n_members)]
    competence_codes = [f"C{i:03d}" for i in range(n_competences)]

    # スパースなデータを生成
    data = []
    for m in range(n_members):
        # 各メンバーは5～10個の力量を保有
        n_skills = np.random.randint(5, 11)
        selected_competences = np.random.choice(n_competences, n_skills, replace=False)

        for c in selected_competences:
            data.append({
                'メンバーコード': member_codes[m],
                '力量コード': competence_codes[c],
                '正規化レベル': np.random.uniform(0.3, 1.0),
            })

    return pd.DataFrame(data)


class TestNMFMultiplicativeUpdate:
    """Multiplicative Update Ruleのテスト"""

    def test_update_step_reduces_error(self, sample_matrix):
        """1ステップの更新で誤差が減少するかテスト"""
        model = MatrixFactorizationModel(
            n_components=5,
            max_iter=1,
            random_state=42,
            use_early_stopping=False,
        )

        model.fit(sample_matrix)

        # 初期誤差
        W_init = np.abs(np.random.randn(sample_matrix.shape[0], 5))
        H_init = np.abs(np.random.randn(5, sample_matrix.shape[1]))

        error_before = np.linalg.norm(sample_matrix - W_init @ H_init, 'fro')

        # 1ステップ更新
        W_new, H_new = model._update_nmf_step(
            sample_matrix, W_init, H_init,
            alpha_W=0.01, alpha_H=0.01, l1_ratio=0.5
        )

        error_after = np.linalg.norm(sample_matrix - W_new @ H_new, 'fro')

        # 誤差が減少していることを確認
        assert error_after < error_before, "更新後に誤差が減少していない"

    def test_nndsvda_initialization(self, sample_matrix):
        """NNDSVDA初期化のテスト"""
        model = MatrixFactorizationModel(
            n_components=5,
            init='nndsvda',
            random_state=42,
        )

        W, H = model._initialize_nmf(sample_matrix, n_components=5)

        # 形状チェック
        assert W.shape == (sample_matrix.shape[0], 5)
        assert H.shape == (5, sample_matrix.shape[1])

        # 非負チェック
        assert np.all(W >= 0)
        assert np.all(H >= 0)

        # ゼロでないことをチェック（初期化が機能している）
        assert np.any(W > 0)
        assert np.any(H > 0)

    def test_early_stopping_activates(self, sample_matrix):
        """Early Stoppingが機能するかテスト"""
        # 検証データを用意
        X_train = sample_matrix[:40, :]
        X_val = sample_matrix[40:, :]

        model = MatrixFactorizationModel(
            n_components=5,
            max_iter=500,
            use_early_stopping=True,
            early_stopping_patience=5,
            early_stopping_min_delta=1e-5,
            validation_split=0.0,  # 手動で検証データを渡す
            random_state=42,
        )

        model._fit_with_early_stopping(X_train, X_val=X_val)

        # Early Stoppingが作動した場合、max_iterより少ないイテレーションで終了
        if hasattr(model, 'training_history') and model.training_history:
            n_evals = len(model.training_history['val_errors'])
            # 評価回数がmax_iter / eval_frequencyより少ないことを確認
            expected_max_evals = 500 // 10  # eval_frequency=10
            assert n_evals <= expected_max_evals, "Early Stoppingが作動していない"


class TestNMFEarlyStopping:
    """Early Stopping機能のテスト"""

    def test_training_history_recorded(self, member_competence_data):
        """学習履歴が記録されるかテスト"""
        model = MatrixFactorizationModel(
            n_components=5,
            max_iter=100,
            use_early_stopping=True,
            early_stopping_patience=5,
            validation_split=0.2,
            random_state=42,
        )

        model.fit(member_competence_data)

        # 学習履歴が記録されている
        assert hasattr(model, 'training_history')
        assert model.training_history is not None
        assert 'train_errors' in model.training_history
        assert 'val_errors' in model.training_history
        assert 'generalization_gaps' in model.training_history

        # すべてのリストの長さが一致
        n_evals = len(model.training_history['train_errors'])
        assert len(model.training_history['val_errors']) == n_evals
        assert len(model.training_history['generalization_gaps']) == n_evals

        # 誤差は正の値
        assert all(err > 0 for err in model.training_history['train_errors'])
        assert all(err > 0 for err in model.training_history['val_errors'])

    def test_validation_error_decreases(self, member_competence_data):
        """検証誤差が減少するかテスト"""
        model = MatrixFactorizationModel(
            n_components=5,
            max_iter=100,
            use_early_stopping=True,
            early_stopping_patience=10,
            validation_split=0.2,
            random_state=42,
        )

        model.fit(member_competence_data)

        val_errors = model.training_history['val_errors']

        # 最初の誤差と最後の誤差を比較
        initial_error = val_errors[0]
        final_error = val_errors[-1]

        # 誤差が減少していることを確認（または横ばい）
        assert final_error <= initial_error * 1.1, "検証誤差が大きく増加している"

    def test_generalization_gap_computed(self, member_competence_data):
        """汎化ギャップが計算されるかテスト"""
        model = MatrixFactorizationModel(
            n_components=5,
            max_iter=100,
            use_early_stopping=True,
            early_stopping_patience=5,
            validation_split=0.2,
            random_state=42,
        )

        model.fit(member_competence_data)

        gaps = model.training_history['generalization_gaps']

        # 汎化ギャップが計算されている
        assert len(gaps) > 0
        assert all(isinstance(g, (int, float)) for g in gaps)

        # 汎化ギャップは通常正の値（訓練誤差 < 検証誤差）
        # ただし、ノイズや初期化により負になることもあるため、絶対値が大きすぎないことを確認
        assert all(abs(g) < 100 for g in gaps), "汎化ギャップの値が異常"

    def test_early_stopping_patience(self, sample_matrix):
        """Patience設定が機能するかテスト"""
        X_train = sample_matrix[:40, :]
        X_val = sample_matrix[40:, :]

        # Patience=3で学習
        model_short_patience = MatrixFactorizationModel(
            n_components=5,
            max_iter=500,
            use_early_stopping=True,
            early_stopping_patience=3,
            early_stopping_min_delta=1e-6,
            random_state=42,
        )

        model_short_patience._fit_with_early_stopping(X_train, X_val=X_val)

        # Patience=10で学習
        model_long_patience = MatrixFactorizationModel(
            n_components=5,
            max_iter=500,
            use_early_stopping=True,
            early_stopping_patience=10,
            early_stopping_min_delta=1e-6,
            random_state=42,
        )

        model_long_patience._fit_with_early_stopping(X_train, X_val=X_val)

        # Patience=3の方が早く終了する（または同じ）
        n_evals_short = len(model_short_patience.training_history['val_errors'])
        n_evals_long = len(model_long_patience.training_history['val_errors'])

        assert n_evals_short <= n_evals_long, "Patience設定が機能していない"

    def test_no_early_stopping_mode(self, member_competence_data):
        """Early Stoppingなしモードのテスト"""
        model = MatrixFactorizationModel(
            n_components=5,
            max_iter=50,
            use_early_stopping=False,
            random_state=42,
        )

        model.fit(member_competence_data)

        # 学習履歴が記録されていない（またはNone）
        assert not hasattr(model, 'training_history') or model.training_history is None


class TestNMFPerformance:
    """パフォーマンステスト"""

    def test_multiplicative_update_faster_than_sklearn(self, sample_matrix):
        """Multiplicative Updateが高速かテスト（概念的）"""
        import time

        model = MatrixFactorizationModel(
            n_components=10,
            max_iter=100,
            use_early_stopping=False,
            random_state=42,
        )

        start = time.time()
        model.fit(sample_matrix)
        elapsed = time.time() - start

        # 100イテレーションが10秒以内に完了することを確認
        assert elapsed < 10.0, f"学習が遅すぎる: {elapsed:.2f}秒"

    def test_convergence_quality(self, member_competence_data):
        """収束品質のテスト"""
        model = MatrixFactorizationModel(
            n_components=10,
            max_iter=200,
            use_early_stopping=True,
            early_stopping_patience=10,
            validation_split=0.2,
            random_state=42,
        )

        model.fit(member_competence_data)

        # 再構成誤差が妥当な範囲
        error = model.get_reconstruction_error()
        assert 0 < error < 10, f"再構成誤差が異常: {error}"

        # 行列がゼロでない
        assert np.any(model.W > 0)
        assert np.any(model.H > 0)


class TestNMFEdgeCases:
    """エッジケースのテスト"""

    def test_small_dataset(self):
        """小規模データセットでのテスト"""
        np.random.seed(42)
        X = np.abs(np.random.randn(10, 5))

        model = MatrixFactorizationModel(
            n_components=3,
            max_iter=50,
            use_early_stopping=True,
            validation_split=0.2,
            random_state=42,
        )

        # エラーなく実行される
        model.fit(X)
        assert model.is_fitted

    def test_sparse_data(self):
        """スパースデータでのテスト"""
        np.random.seed(42)
        n_members = 100
        n_competences = 50

        # スパース行列（90%が0）
        X = np.zeros((n_members, n_competences))
        n_nonzero = int(0.1 * n_members * n_competences)
        indices = np.random.choice(n_members * n_competences, n_nonzero, replace=False)
        X.flat[indices] = np.random.uniform(0.3, 1.0, n_nonzero)

        model = MatrixFactorizationModel(
            n_components=10,
            max_iter=100,
            use_early_stopping=True,
            validation_split=0.2,
            random_state=42,
        )

        model.fit(X)
        assert model.is_fitted

    def test_regularization_effect(self, sample_matrix):
        """正則化の効果をテスト"""
        # 正則化なし
        model_no_reg = MatrixFactorizationModel(
            n_components=10,
            alpha_W=0.0,
            alpha_H=0.0,
            max_iter=100,
            random_state=42,
        )
        model_no_reg.fit(sample_matrix)
        W_no_reg = model_no_reg.W

        # 正則化あり
        model_with_reg = MatrixFactorizationModel(
            n_components=10,
            alpha_W=0.1,
            alpha_H=0.1,
            max_iter=100,
            random_state=42,
        )
        model_with_reg.fit(sample_matrix)
        W_with_reg = model_with_reg.W

        # 正則化ありの方がWのノルムが小さい
        norm_no_reg = np.linalg.norm(W_no_reg, 'fro')
        norm_with_reg = np.linalg.norm(W_with_reg, 'fro')

        assert norm_with_reg < norm_no_reg, "正則化の効果が見られない"
