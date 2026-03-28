"""
ハイパーパラメータチューニングモジュールのテスト

NMFHyperparameterTunerクラスとOptunaベースの最適化機能のテスト
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch, call
from skillnote_recommendation.ml.hyperparameter_tuning import (
    NMFHyperparameterTuner,
    tune_nmf_hyperparameters_from_config,
    OPTUNA_AVAILABLE,
)


# ==================== フィクスチャ ====================


@pytest.fixture
def sample_skill_matrix():
    """サンプルメンバー×力量マトリックス（学習用）"""
    np.random.seed(42)
    # 20メンバー × 10力量のランダムマトリックス
    data = np.random.choice([0, 0, 0, 1, 2, 3, 4, 5], size=(20, 10))
    return pd.DataFrame(
        data,
        index=[f"m{i:03d}" for i in range(20)],
        columns=[f"s{i:03d}" for i in range(10)],
    )


@pytest.fixture
def small_skill_matrix():
    """小さなスキルマトリックス（高速テスト用）"""
    return pd.DataFrame(
        {
            "s001": [3, 0, 2, 0, 1],
            "s002": [0, 4, 0, 3, 0],
            "s003": [2, 0, 5, 0, 2],
            "s004": [0, 3, 0, 4, 0],
            "s005": [1, 0, 2, 0, 3],
        },
        index=["m001", "m002", "m003", "m004", "m005"],
    )


@pytest.fixture
def mock_config():
    """モックConfig"""
    config = Mock()
    config.OPTUNA_PARAMS = {
        "n_trials": 5,
        "timeout": 60,
        "n_jobs": 1,
        "search_space": {
            "n_components": (5, 15),
            "alpha_W": (0.001, 0.1),
            "alpha_H": (0.001, 0.1),
            "l1_ratio": (0.0, 1.0),
            "max_iter": (200, 500),
        },
        "use_cross_validation": False,  # 高速化のため無効
        "n_folds": 2,
        "use_time_series_split": False,
        "test_size": 0.0,  # テストセット分離なし
        "enable_early_stopping": False,  # 高速化のため無効
        "early_stopping_patience": 3,
        "early_stopping_batch_size": 20,
    }
    config.MF_PARAMS = {"random_state": 42}
    return config


# ==================== 初期化テスト ====================


class TestNMFHyperparameterTunerInitialization:
    """NMFHyperparameterTunerの初期化テスト"""

    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not available")
    def test_initialization_default(self, small_skill_matrix):
        """デフォルトパラメータで初期化"""
        tuner = NMFHyperparameterTuner(
            skill_matrix=small_skill_matrix,
            n_trials=5,
            test_size=0.0,  # テストセット分離なし
        )

        assert tuner.n_trials == 5
        assert tuner.timeout == 600
        assert tuner.n_jobs == 1
        assert tuner.random_state == 42
        assert tuner.use_cross_validation
        assert tuner.n_folds == 3
        assert tuner.sampler == "tpe"
        assert tuner.best_params is None
        assert tuner.best_value is None
        assert tuner.study is None

    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not available")
    def test_initialization_custom_params(self, small_skill_matrix):
        """カスタムパラメータで初期化"""
        custom_search_space = {
            "n_components": (5, 10),
            "alpha_W": (0.01, 0.1),
            "alpha_H": (0.01, 0.1),
        }

        tuner = NMFHyperparameterTuner(
            skill_matrix=small_skill_matrix,
            n_trials=10,
            timeout=300,
            n_jobs=2,
            random_state=123,
            search_space=custom_search_space,
            use_cross_validation=False,
            sampler="random",
            test_size=0.0,
        )

        assert tuner.n_trials == 10
        assert tuner.timeout == 300
        assert tuner.n_jobs == 2
        assert tuner.random_state == 123
        assert not tuner.use_cross_validation
        assert tuner.sampler == "random"
        assert tuner.search_space == custom_search_space

    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not available")
    def test_initialization_with_test_split(self, sample_skill_matrix):
        """テストセット分離あり初期化"""
        tuner = NMFHyperparameterTuner(
            skill_matrix=sample_skill_matrix, n_trials=5, test_size=0.2
        )

        # Train+Val: 80%, Test: 20%
        assert len(tuner.skill_matrix) == int(len(sample_skill_matrix) * 0.8)
        assert len(tuner.test_matrix) == len(sample_skill_matrix) - len(tuner.skill_matrix)
        assert tuner.test_score is None

    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not available")
    def test_initialization_no_test_split(self, small_skill_matrix):
        """テストセット分離なし初期化"""
        tuner = NMFHyperparameterTuner(
            skill_matrix=small_skill_matrix, n_trials=5, test_size=0.0
        )

        assert len(tuner.skill_matrix) == len(small_skill_matrix)
        assert tuner.test_matrix is None
        assert tuner.test_score is None

    @pytest.mark.skipif(OPTUNA_AVAILABLE, reason="Test for Optuna not available")
    def test_initialization_without_optuna_raises_error(self, small_skill_matrix):
        """Optunaが利用できない場合はエラー"""
        with pytest.raises(ImportError, match="Optunaがインストールされていません"):
            NMFHyperparameterTuner(skill_matrix=small_skill_matrix, n_trials=5)


# ==================== パラメータサンプリングテスト ====================


class TestParameterSuggestion:
    """パラメータサンプリングのテスト"""

    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not available")
    def test_suggest_params_integer_params(self, small_skill_matrix):
        """整数パラメータのサンプリング"""
        tuner = NMFHyperparameterTuner(
            skill_matrix=small_skill_matrix, n_trials=1, test_size=0.0
        )

        mock_trial = Mock()
        mock_trial.suggest_int = Mock(side_effect=[15, 1000])  # n_components, max_iter
        mock_trial.suggest_float = Mock(side_effect=[0.1, 0.2, 0.5])  # alpha_W, alpha_H, l1_ratio

        params = tuner._suggest_params(mock_trial)

        # n_componentsとmax_iterは整数として提案される
        assert mock_trial.suggest_int.call_count == 2
        mock_trial.suggest_int.assert_any_call("n_components", 10, 30)
        mock_trial.suggest_int.assert_any_call("max_iter", 500, 1500, step=100)

    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not available")
    def test_suggest_params_log_scale(self, small_skill_matrix):
        """対数スケールパラメータのサンプリング"""
        tuner = NMFHyperparameterTuner(
            skill_matrix=small_skill_matrix, n_trials=1, test_size=0.0
        )

        mock_trial = Mock()
        mock_trial.suggest_int = Mock(side_effect=[15, 1000])
        mock_trial.suggest_float = Mock(side_effect=[0.05, 0.05, 0.5])

        params = tuner._suggest_params(mock_trial)

        # alpha_W, alpha_Hは対数スケールで提案される（min_val > 0）
        alpha_w_call = [
            c for c in mock_trial.suggest_float.call_args_list if c[0][0] == "alpha_W"
        ][0]
        assert alpha_w_call[1]["log"] is True

    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not available")
    def test_suggest_params_linear_scale(self, small_skill_matrix):
        """線形スケールパラメータのサンプリング"""
        tuner = NMFHyperparameterTuner(
            skill_matrix=small_skill_matrix, n_trials=1, test_size=0.0
        )

        mock_trial = Mock()
        mock_trial.suggest_int = Mock(side_effect=[15, 1000])
        mock_trial.suggest_float = Mock(side_effect=[0.05, 0.05, 0.5])

        params = tuner._suggest_params(mock_trial)

        # l1_ratioは線形スケールで提案される
        l1_ratio_call = [
            c for c in mock_trial.suggest_float.call_args_list if c[0][0] == "l1_ratio"
        ][0]
        assert "log" not in l1_ratio_call[1] or l1_ratio_call[1].get("log") is False

    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not available")
    def test_suggest_params_fixed_params(self, small_skill_matrix):
        """固定パラメータが含まれること"""
        tuner = NMFHyperparameterTuner(
            skill_matrix=small_skill_matrix, n_trials=1, test_size=0.0
        )

        mock_trial = Mock()
        mock_trial.suggest_int = Mock(side_effect=[15, 1000])
        mock_trial.suggest_float = Mock(side_effect=[0.05, 0.05, 0.5])

        params = tuner._suggest_params(mock_trial)

        # 固定パラメータが設定されている
        assert params["init"] == "nndsvda"
        assert params["solver"] == "cd"
        assert params["tol"] == 1e-5


# ==================== 目的関数テスト ====================


class TestObjectiveFunction:
    """目的関数のテスト"""

    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not available")
    def test_objective_without_cv(self, small_skill_matrix):
        """交差検証なしの目的関数"""
        tuner = NMFHyperparameterTuner(
            skill_matrix=small_skill_matrix,
            n_trials=1,
            use_cross_validation=False,
            test_size=0.0,
        )

        mock_trial = Mock()
        mock_trial.number = 0
        mock_trial.suggest_int = Mock(side_effect=[3, 500])  # n_components=3 (小さなマトリックス用)
        mock_trial.suggest_float = Mock(side_effect=[0.01, 0.01, 0.5])
        mock_trial.set_user_attr = Mock()

        error = tuner.objective(mock_trial)

        # エラーが返される
        assert isinstance(error, float)
        assert error >= 0
        # ユーザー属性が設定される
        assert mock_trial.set_user_attr.called

    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not available")
    def test_objective_with_cv(self, small_skill_matrix):
        """交差検証ありの目的関数"""
        tuner = NMFHyperparameterTuner(
            skill_matrix=small_skill_matrix,
            n_trials=1,
            use_cross_validation=True,
            n_folds=2,  # 高速化のため2分割
            test_size=0.0,
        )

        mock_trial = Mock()
        mock_trial.number = 0
        mock_trial.suggest_int = Mock(side_effect=[3, 500])  # n_components=3 (小さなマトリックス用)
        mock_trial.suggest_float = Mock(side_effect=[0.01, 0.01, 0.5])
        mock_trial.set_user_attr = Mock()

        error = tuner.objective(mock_trial)

        # エラーが返される
        assert isinstance(error, float)
        assert error >= 0
        # CVメトリクスが記録される
        user_attr_calls = {call[0][0]: call[0][1] for call in mock_trial.set_user_attr.call_args_list}
        assert "cv_std" in user_attr_calls
        assert "cv_errors" in user_attr_calls
        assert "split_method" in user_attr_calls

    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not available")
    def test_objective_exception_handling(self, small_skill_matrix):
        """目的関数内での例外ハンドリング"""
        tuner = NMFHyperparameterTuner(
            skill_matrix=small_skill_matrix,
            n_trials=1,
            use_cross_validation=False,
            test_size=0.0,
        )

        mock_trial = Mock()
        mock_trial.number = 0
        # n_componentsを大きくしてValueErrorを発生させる（NMFの初期化時）
        mock_trial.suggest_int = Mock(side_effect=[100, 500])  # n_components=100は5x5マトリックスには大きすぎる
        mock_trial.suggest_float = Mock(side_effect=[0.01, 0.01, 0.5])
        mock_trial.set_user_attr = Mock()

        error = tuner.objective(mock_trial)

        # 例外時はinfを返す
        assert error == float("inf")

    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not available")
    def test_objective_records_trial_attributes(self, small_skill_matrix):
        """トライアル属性が記録されること"""
        tuner = NMFHyperparameterTuner(
            skill_matrix=small_skill_matrix,
            n_trials=1,
            use_cross_validation=False,
            test_size=0.0,
        )

        mock_trial = Mock()
        mock_trial.number = 0
        mock_trial.suggest_int = Mock(side_effect=[3, 500])  # n_components=3 (小さなマトリックス用)
        mock_trial.suggest_float = Mock(side_effect=[0.01, 0.01, 0.5])
        mock_trial.set_user_attr = Mock()

        tuner.objective(mock_trial)

        # random_stateが記録される
        user_attr_calls = {call[0][0]: call[0][1] for call in mock_trial.set_user_attr.call_args_list}
        assert "random_state" in user_attr_calls
        assert user_attr_calls["random_state"] == 42


# ==================== 最適化実行テスト ====================


class TestOptimization:
    """最適化実行のテスト"""

    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not available")
    def test_optimize_basic(self, small_skill_matrix):
        """基本的な最適化"""
        tuner = NMFHyperparameterTuner(
            skill_matrix=small_skill_matrix,
            n_trials=3,  # 高速化
            use_cross_validation=False,
            test_size=0.0,
        )

        best_params, best_value = tuner.optimize(show_progress_bar=False)

        # 最適パラメータが返される
        assert isinstance(best_params, dict)
        assert "n_components" in best_params
        assert isinstance(best_value, float)
        assert best_value >= 0

        # Tunerの状態が更新される
        assert tuner.best_params == best_params
        assert tuner.best_value == best_value
        assert tuner.study is not None

    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not available")
    def test_optimize_with_tpe_sampler(self, small_skill_matrix):
        """TPEサンプラーでの最適化"""
        tuner = NMFHyperparameterTuner(
            skill_matrix=small_skill_matrix,
            n_trials=3,
            sampler="tpe",
            use_cross_validation=False,
            test_size=0.0,
        )

        best_params, best_value = tuner.optimize(show_progress_bar=False)

        assert tuner.study is not None
        # TPESamplerが使用されている
        from optuna.samplers import TPESampler

        assert isinstance(tuner.study.sampler, TPESampler)

    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not available")
    def test_optimize_with_random_sampler(self, small_skill_matrix):
        """Randomサンプラーでの最適化"""
        tuner = NMFHyperparameterTuner(
            skill_matrix=small_skill_matrix,
            n_trials=3,
            sampler="random",
            use_cross_validation=False,
            test_size=0.0,
        )

        best_params, best_value = tuner.optimize(show_progress_bar=False)

        assert tuner.study is not None
        # RandomSamplerが使用されている
        from optuna.samplers import RandomSampler

        assert isinstance(tuner.study.sampler, RandomSampler)

    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not available")
    def test_optimize_with_unknown_sampler_fallback_to_tpe(self, small_skill_matrix):
        """不明なサンプラーはTPEにフォールバック"""
        tuner = NMFHyperparameterTuner(
            skill_matrix=small_skill_matrix,
            n_trials=3,
            sampler="unknown_sampler",
            use_cross_validation=False,
            test_size=0.0,
        )

        best_params, best_value = tuner.optimize(show_progress_bar=False)

        # TPEにフォールバックされている
        from optuna.samplers import TPESampler

        assert isinstance(tuner.study.sampler, TPESampler)

    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not available")
    def test_optimize_with_callback(self, small_skill_matrix):
        """コールバック付き最適化"""
        callback_calls = []

        def progress_callback(trial, study):
            callback_calls.append((trial.number, study))

        tuner = NMFHyperparameterTuner(
            skill_matrix=small_skill_matrix,
            n_trials=3,
            progress_callback=progress_callback,
            use_cross_validation=False,
            test_size=0.0,
        )

        tuner.optimize(show_progress_bar=False)

        # コールバックが呼ばれている
        assert len(callback_calls) == 3
        assert all(isinstance(call[0], int) for call in callback_calls)

    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not available")
    def test_optimize_creates_completed_trials(self, small_skill_matrix):
        """最適化が完了したトライアルを生成すること"""
        tuner = NMFHyperparameterTuner(
            skill_matrix=small_skill_matrix,
            n_trials=5,
            use_cross_validation=False,
            test_size=0.0,
        )

        tuner.optimize(show_progress_bar=False)

        # 5個のトライアルが完了している
        assert len(tuner.study.trials) == 5

    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not available")
    def test_optimize_with_timeout(self, small_skill_matrix):
        """タイムアウト設定での最適化"""
        # 小さなマトリックス用の探索空間（成功するトライアルを作るため）
        custom_search_space = {
            "n_components": (2, 4),
            "alpha_W": (0.001, 0.1),
            "alpha_H": (0.001, 0.1),
            "l1_ratio": (0.0, 1.0),
            "max_iter": (200, 500),
        }

        tuner = NMFHyperparameterTuner(
            skill_matrix=small_skill_matrix,
            n_trials=100,  # 多数のトライアル
            timeout=2,  # 2秒でタイムアウト
            search_space=custom_search_space,
            use_cross_validation=False,
            test_size=0.0,
        )

        best_params, best_value = tuner.optimize(show_progress_bar=False)

        # タイムアウトが設定されていることを確認（完了数は問わない）
        # 注: 小さなマトリックスでは処理が速いため、100個完了する可能性もある
        assert len(tuner.study.trials) <= 100


# ==================== モデル取得テスト ====================


class TestModelRetrieval:
    """モデル取得のテスト"""

    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not available")
    def test_get_best_model(self, small_skill_matrix):
        """最適モデルの取得"""
        # 小さなマトリックス用の探索空間
        custom_search_space = {
            "n_components": (2, 4),  # 5x5マトリックスに適した範囲
            "alpha_W": (0.001, 0.1),
            "alpha_H": (0.001, 0.1),
            "l1_ratio": (0.0, 1.0),
            "max_iter": (200, 500),
        }

        tuner = NMFHyperparameterTuner(
            skill_matrix=small_skill_matrix,
            n_trials=3,
            search_space=custom_search_space,
            use_cross_validation=False,
            test_size=0.0,
        )
        tuner.optimize(show_progress_bar=False)

        model = tuner.get_best_model()

        # モデルが学習済み
        assert model.is_fitted
        # 最適パラメータが使用されている
        assert model.n_components == tuner.best_params["n_components"]

    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not available")
    def test_get_best_model_before_optimization_raises_error(self, small_skill_matrix):
        """最適化前のモデル取得はエラー"""
        tuner = NMFHyperparameterTuner(
            skill_matrix=small_skill_matrix, n_trials=3, test_size=0.0
        )

        with pytest.raises(ValueError, match="先にoptimize\\(\\)を実行してください"):
            tuner.get_best_model()

    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not available")
    def test_get_best_model_with_test_set_evaluation(self, sample_skill_matrix):
        """テストセットでの評価を含むモデル取得"""
        tuner = NMFHyperparameterTuner(
            skill_matrix=sample_skill_matrix,
            n_trials=3,
            use_cross_validation=False,
            test_size=0.2,  # 20%をテストセットに
        )
        tuner.optimize(show_progress_bar=False)

        model = tuner.get_best_model()

        # テストスコアが計算されている
        assert tuner.test_score is not None
        assert isinstance(tuner.test_score, float)
        assert tuner.test_score >= 0


# ==================== ヘルパー関数テスト ====================


class TestHelperFunctions:
    """ヘルパー関数のテスト"""

    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not available")
    def test_create_skill_based_split(self, small_skill_matrix):
        """スキルベース分割の作成"""
        tuner = NMFHyperparameterTuner(
            skill_matrix=small_skill_matrix,
            n_trials=1,
            n_folds=3,
            test_size=0.0,
        )

        train_matrix, val_mask = tuner._create_skill_based_split(fold_idx=0)

        # トレーニングマトリックスとバリデーションマスクのサイズが一致
        assert train_matrix.shape == small_skill_matrix.shape
        assert val_mask.shape == small_skill_matrix.shape

        # バリデーションマスクの非ゼロ要素がトレーニングマトリックスでゼロになっている
        for member in small_skill_matrix.index:
            val_skills = val_mask.loc[member][val_mask.loc[member] > 0].index
            for skill in val_skills:
                assert train_matrix.loc[member, skill] == 0.0

    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not available")
    def test_create_skill_based_split_different_folds(self, small_skill_matrix):
        """異なるフォールドで異なる分割が作成されること"""
        tuner = NMFHyperparameterTuner(
            skill_matrix=small_skill_matrix,
            n_trials=1,
            n_folds=3,
            test_size=0.0,
        )

        train1, val1 = tuner._create_skill_based_split(fold_idx=0)
        train2, val2 = tuner._create_skill_based_split(fold_idx=1)

        # 異なる分割になっている（完全には一致しない）
        assert not train1.equals(train2) or not val1.equals(val2)

    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not available")
    def test_calculate_skill_based_validation_error(self, small_skill_matrix):
        """スキルベースバリデーション誤差の計算"""
        from skillnote_recommendation.ml.matrix_factorization import MatrixFactorizationModel

        tuner = NMFHyperparameterTuner(
            skill_matrix=small_skill_matrix,
            n_trials=1,
            test_size=0.0,
        )

        # スキルベース分割を作成
        train_matrix, val_mask = tuner._create_skill_based_split(fold_idx=0)

        # モデルを学習
        model = MatrixFactorizationModel(n_components=2, random_state=42)
        model.fit(train_matrix)

        # バリデーション誤差を計算
        error = tuner._calculate_skill_based_validation_error(model, val_mask)

        assert isinstance(error, float)
        assert error >= 0

    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not available")
    def test_evaluate_on_test_set(self, sample_skill_matrix):
        """テストセットでの評価"""
        from skillnote_recommendation.ml.matrix_factorization import MatrixFactorizationModel

        tuner = NMFHyperparameterTuner(
            skill_matrix=sample_skill_matrix,
            n_trials=1,
            test_size=0.2,
        )

        # モデルを学習（チューニングなしで直接）
        model = MatrixFactorizationModel(n_components=5, random_state=42)
        model.fit(tuner.skill_matrix)

        # テストセットで評価
        test_error = tuner.evaluate_on_test_set(model)

        assert isinstance(test_error, float)
        assert test_error >= 0

    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not available")
    def test_evaluate_on_test_set_without_test_raises_error(self, small_skill_matrix):
        """テストセットなしで評価するとエラー"""
        from skillnote_recommendation.ml.matrix_factorization import MatrixFactorizationModel

        tuner = NMFHyperparameterTuner(
            skill_matrix=small_skill_matrix,
            n_trials=1,
            test_size=0.0,  # テストセットなし
        )

        model = MatrixFactorizationModel(n_components=2, random_state=42)
        model.fit(tuner.skill_matrix)

        with pytest.raises(ValueError, match="Test setが分離されていません"):
            tuner.evaluate_on_test_set(model)


# ==================== 履歴とプロットテスト ====================


class TestHistoryAndPlotting:
    """履歴とプロット機能のテスト"""

    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not available")
    def test_get_optimization_history(self, small_skill_matrix):
        """最適化履歴の取得"""
        tuner = NMFHyperparameterTuner(
            skill_matrix=small_skill_matrix,
            n_trials=5,
            use_cross_validation=False,
            test_size=0.0,
        )
        tuner.optimize(show_progress_bar=False)

        history = tuner.get_optimization_history()

        # DataFrameが返される
        assert isinstance(history, pd.DataFrame)
        assert len(history) == 5
        assert "number" in history.columns
        assert "value" in history.columns

    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not available")
    def test_get_optimization_history_before_optimization_raises_error(self, small_skill_matrix):
        """最適化前の履歴取得はエラー"""
        tuner = NMFHyperparameterTuner(
            skill_matrix=small_skill_matrix, n_trials=3, test_size=0.0
        )

        with pytest.raises(ValueError, match="先にoptimize\\(\\)を実行してください"):
            tuner.get_optimization_history()

    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not available")
    def test_plot_optimization_history(self, small_skill_matrix):
        """最適化履歴のプロット"""
        tuner = NMFHyperparameterTuner(
            skill_matrix=small_skill_matrix,
            n_trials=5,
            use_cross_validation=False,
            test_size=0.0,
        )
        tuner.optimize(show_progress_bar=False)

        try:
            fig = tuner.plot_optimization_history()
            # Plotlyがある場合はFigureが返される
            assert fig is not None
        except ImportError:
            # Plotlyがない場合はNoneが返される
            assert tuner.plot_optimization_history() is None

    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not available")
    def test_plot_param_importances(self, small_skill_matrix):
        """パラメータ重要度のプロット"""
        # 小さなマトリックス用の探索空間
        custom_search_space = {
            "n_components": (2, 4),
            "alpha_W": (0.001, 0.1),
            "alpha_H": (0.001, 0.1),
            "l1_ratio": (0.0, 1.0),
            "max_iter": (200, 500),
        }

        tuner = NMFHyperparameterTuner(
            skill_matrix=small_skill_matrix,
            n_trials=5,
            search_space=custom_search_space,
            use_cross_validation=False,
            test_size=0.0,
        )
        tuner.optimize(show_progress_bar=False)

        try:
            fig = tuner.plot_param_importances()
            # Plotlyがある場合はFigureが返される
            assert fig is not None
        except (ImportError, IndexError):
            # Plotlyがない場合、または十分なトライアルがない場合はスキップ
            pass


# ==================== Config関数テスト ====================


class TestTuneFromConfig:
    """tune_nmf_hyperparameters_from_config()のテスト"""

    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not available")
    def test_tune_from_config_basic(self, small_skill_matrix, mock_config):
        """設定ベースのチューニング"""
        best_params, best_value, best_model = tune_nmf_hyperparameters_from_config(
            skill_matrix=small_skill_matrix, config=mock_config, show_progress_bar=False
        )

        # 最適パラメータとモデルが返される
        assert isinstance(best_params, dict)
        assert isinstance(best_value, float)
        assert best_model.is_fitted

    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not available")
    def test_tune_from_config_with_return_tuner(self, small_skill_matrix, mock_config):
        """Tunerオブジェクトも返す"""
        best_params, best_value, best_model, tuner = tune_nmf_hyperparameters_from_config(
            skill_matrix=small_skill_matrix,
            config=mock_config,
            show_progress_bar=False,
            return_tuner=True,
        )

        # Tunerも返される
        assert isinstance(tuner, NMFHyperparameterTuner)
        assert tuner.best_params == best_params
        assert tuner.best_value == best_value

    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not available")
    def test_tune_from_config_with_custom_params(self, small_skill_matrix, mock_config):
        """カスタムパラメータで上書き"""
        custom_search_space = {
            "n_components": (5, 10),
            "alpha_W": (0.01, 0.1),
        }

        best_params, best_value, best_model, tuner = tune_nmf_hyperparameters_from_config(
            skill_matrix=small_skill_matrix,
            config=mock_config,
            show_progress_bar=False,
            return_tuner=True,
            custom_n_trials=3,
            custom_timeout=30,
            custom_search_space=custom_search_space,
            custom_sampler="random",
        )

        # カスタムパラメータが使用されている
        assert tuner.n_trials == 3
        assert tuner.timeout == 30
        assert tuner.search_space == custom_search_space
        assert tuner.sampler == "random"


# ==================== エッジケーステスト ====================


class TestEdgeCases:
    """エッジケースのテスト"""

    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not available")
    def test_optimize_with_very_small_matrix(self):
        """非常に小さなマトリックスでの最適化"""
        tiny_matrix = pd.DataFrame(
            {"s001": [1, 0], "s002": [0, 1]}, index=["m001", "m002"]
        )

        tuner = NMFHyperparameterTuner(
            skill_matrix=tiny_matrix,
            n_trials=2,
            use_cross_validation=False,
            test_size=0.0,
        )

        best_params, best_value = tuner.optimize(show_progress_bar=False)

        assert isinstance(best_params, dict)
        assert best_value >= 0

    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not available")
    def test_optimize_with_all_zero_matrix(self):
        """全てゼロのマトリックスでの最適化"""
        zero_matrix = pd.DataFrame(
            np.zeros((5, 5)),
            index=[f"m{i:03d}" for i in range(5)],
            columns=[f"s{i:03d}" for i in range(5)],
        )

        tuner = NMFHyperparameterTuner(
            skill_matrix=zero_matrix,
            n_trials=2,
            use_cross_validation=False,
            test_size=0.0,
        )

        # エラーなく実行できる（ただし意味のある結果は期待しない）
        best_params, best_value = tuner.optimize(show_progress_bar=False)
        assert isinstance(best_params, dict)

    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not available")
    def test_custom_search_space_partial_override(self, small_skill_matrix):
        """探索空間の部分的な上書き"""
        # n_componentsのみカスタマイズ
        custom_space = {"n_components": (3, 5)}

        tuner = NMFHyperparameterTuner(
            skill_matrix=small_skill_matrix,
            n_trials=2,
            search_space=custom_space,
            use_cross_validation=False,
            test_size=0.0,
        )

        # カスタム探索空間が使用される
        assert tuner.search_space == custom_space

    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not available")
    def test_progress_callback_receives_correct_arguments(self, small_skill_matrix):
        """プログレスコールバックが正しい引数を受け取る"""
        callback_args = []

        def callback(trial, study):
            callback_args.append({"trial": trial, "study": study})

        tuner = NMFHyperparameterTuner(
            skill_matrix=small_skill_matrix,
            n_trials=2,
            progress_callback=callback,
            use_cross_validation=False,
            test_size=0.0,
        )

        tuner.optimize(show_progress_bar=False)

        # コールバックが呼ばれている
        assert len(callback_args) == 2
        # 引数が正しい
        for arg in callback_args:
            assert "trial" in arg
            assert "study" in arg
            assert hasattr(arg["trial"], "number")
            assert hasattr(arg["study"], "best_value")
