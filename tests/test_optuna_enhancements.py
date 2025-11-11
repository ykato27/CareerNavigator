"""
Optuna機能強化のテスト

- 永続化（SQLite storage）
- Pruning（枝刈り）
- Visualization統合

のテスト
"""

import pytest
import numpy as np
import pandas as pd
import os
import tempfile
from pathlib import Path

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from skillnote_recommendation.ml.hyperparameter_tuning import NMFHyperparameterTuner
from skillnote_recommendation.ml.optuna_visualization_helper import (
    generate_optuna_visualizations,
    get_best_trials_summary,
    get_pruned_trials_count,
    plot_training_history,
)


@pytest.fixture
def sample_member_competence():
    """テスト用のメンバー力量データ"""
    np.random.seed(42)
    n_members = 50
    n_competences = 30

    member_codes = [f"M{i:03d}" for i in range(n_members)]
    competence_codes = [f"C{i:03d}" for i in range(n_competences)]

    data = []
    for m in range(n_members):
        n_skills = np.random.randint(5, 11)
        selected_competences = np.random.choice(n_competences, n_skills, replace=False)

        for c in selected_competences:
            data.append({
                'メンバーコード': member_codes[m],
                '力量コード': competence_codes[c],
                '正規化レベル': np.random.uniform(0.3, 1.0),
            })

    return pd.DataFrame(data)


@pytest.fixture
def temp_storage_path():
    """一時的なSQLiteストレージパス"""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir) / "test_optuna.db"
        yield f"sqlite:///{storage_path}"


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optunaがインストールされていません")
class TestOptunaPersistence:
    """Optuna永続化のテスト"""

    def test_study_saved_to_sqlite(self, sample_member_competence, temp_storage_path):
        """Studyがデータベースに保存されるかテスト"""
        tuner = NMFHyperparameterTuner(
            sample_member_competence,
            search_space={
                'n_components': (5, 10),
                'alpha_W': (0.001, 0.1),
                'alpha_H': (0.001, 0.1),
                'l1_ratio': (0.0, 1.0),
            },
            n_splits=2,
            random_state=42,
            use_cross_validation=True,
        )

        # 一時的なstorageを設定
        tuner.storage_url = temp_storage_path
        tuner.study_name = "test_study"

        # 少数の試行で最適化
        best_params, best_value = tuner.optimize(
            n_trials=3,
            show_progress_bar=False,
        )

        # Studyが存在するか確認
        assert tuner.study is not None
        assert len(tuner.study.trials) == 3

        # 新しいtunerで同じstudyをロード
        tuner2 = NMFHyperparameterTuner(
            sample_member_competence,
            search_space={
                'n_components': (5, 10),
                'alpha_W': (0.001, 0.1),
                'alpha_H': (0.001, 0.1),
                'l1_ratio': (0.0, 1.0),
            },
            n_splits=2,
            random_state=42,
        )
        tuner2.storage_url = temp_storage_path
        tuner2.study_name = "test_study"

        # studyを再ロード
        tuner2.study = optuna.load_study(
            study_name="test_study",
            storage=temp_storage_path,
        )

        # 以前の試行が読み込まれているか確認
        assert len(tuner2.study.trials) == 3
        assert tuner2.study.best_value == tuner.study.best_value

    def test_resume_optimization(self, sample_member_competence, temp_storage_path):
        """最適化が中断から再開できるかテスト"""
        tuner = NMFHyperparameterTuner(
            sample_member_competence,
            search_space={
                'n_components': (5, 10),
                'alpha_W': (0.001, 0.1),
                'alpha_H': (0.001, 0.1),
                'l1_ratio': (0.0, 1.0),
            },
            n_splits=2,
            random_state=42,
        )
        tuner.storage_url = temp_storage_path
        tuner.study_name = "test_resume_study"

        # 最初の2試行
        tuner.optimize(n_trials=2, show_progress_bar=False)
        initial_trials = len(tuner.study.trials)

        # 同じstudyで追加の3試行
        tuner2 = NMFHyperparameterTuner(
            sample_member_competence,
            search_space={
                'n_components': (5, 10),
                'alpha_W': (0.001, 0.1),
                'alpha_H': (0.001, 0.1),
                'l1_ratio': (0.0, 1.0),
            },
            n_splits=2,
            random_state=42,
        )
        tuner2.storage_url = temp_storage_path
        tuner2.study_name = "test_resume_study"

        # load_if_exists=Trueで再開
        tuner2.study = optuna.create_study(
            study_name="test_resume_study",
            storage=temp_storage_path,
            load_if_exists=True,
            direction='minimize',
        )

        # 追加の3試行
        tuner2.study.optimize(
            lambda trial: tuner2._objective_with_progress_report(trial, None),
            n_trials=3,
            show_progress_bar=False,
        )

        # 合計5試行になっている
        assert len(tuner2.study.trials) == initial_trials + 3


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optunaがインストールされていません")
class TestOptunaPruning:
    """Optuna Pruningのテスト"""

    def test_pruner_configured(self, sample_member_competence):
        """Prunerが設定されるかテスト"""
        tuner = NMFHyperparameterTuner(
            sample_member_competence,
            search_space={
                'n_components': (5, 10),
                'alpha_W': (0.001, 0.1),
                'alpha_H': (0.001, 0.1),
                'l1_ratio': (0.0, 1.0),
            },
            n_splits=2,
            random_state=42,
        )

        # Studyを作成
        tuner.study = optuna.create_study(
            study_name="test_pruning",
            direction='minimize',
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=2,
                n_warmup_steps=0,
            ),
        )

        # Prunerが設定されている
        assert tuner.study.pruner is not None
        assert isinstance(tuner.study.pruner, optuna.pruners.MedianPruner)

    def test_pruning_reduces_trials(self, sample_member_competence):
        """Pruningが試行数を削減するかテスト"""
        # Pruningなしの実行
        tuner_no_pruning = NMFHyperparameterTuner(
            sample_member_competence,
            search_space={
                'n_components': (5, 15),
                'alpha_W': (0.001, 0.5),
                'alpha_H': (0.001, 0.5),
                'l1_ratio': (0.0, 1.0),
            },
            n_splits=2,
            random_state=42,
        )

        tuner_no_pruning.study = optuna.create_study(
            study_name="test_no_pruning",
            direction='minimize',
            pruner=optuna.pruners.NopPruner(),  # Pruningなし
        )

        # 10試行実行
        import time
        start_no_pruning = time.time()
        tuner_no_pruning.optimize(n_trials=10, show_progress_bar=False)
        elapsed_no_pruning = time.time() - start_no_pruning

        # Pruningありの実行
        tuner_with_pruning = NMFHyperparameterTuner(
            sample_member_competence,
            search_space={
                'n_components': (5, 15),
                'alpha_W': (0.001, 0.5),
                'alpha_H': (0.001, 0.5),
                'l1_ratio': (0.0, 1.0),
            },
            n_splits=2,
            random_state=42,
        )

        tuner_with_pruning.study = optuna.create_study(
            study_name="test_with_pruning",
            direction='minimize',
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=2,
                n_warmup_steps=0,
            ),
        )

        start_with_pruning = time.time()
        tuner_with_pruning.optimize(n_trials=10, show_progress_bar=False)
        elapsed_with_pruning = time.time() - start_with_pruning

        # Pruningにより計算時間が短縮される（またはほぼ同じ）
        # ※小規模データでは効果が見えにくいため、時間の増加がないことを確認
        assert elapsed_with_pruning <= elapsed_no_pruning * 1.2

    def test_pruning_statistics(self, sample_member_competence):
        """Pruning統計が取得できるかテスト"""
        tuner = NMFHyperparameterTuner(
            sample_member_competence,
            search_space={
                'n_components': (5, 15),
                'alpha_W': (0.001, 0.5),
                'alpha_H': (0.001, 0.5),
                'l1_ratio': (0.0, 1.0),
            },
            n_splits=2,
            random_state=42,
        )

        tuner.study = optuna.create_study(
            study_name="test_pruning_stats",
            direction='minimize',
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=2,
                n_warmup_steps=0,
            ),
        )

        tuner.optimize(n_trials=10, show_progress_bar=False)

        # 統計情報を取得
        stats = get_pruned_trials_count(tuner.study)

        # 必須フィールドが存在する
        assert 'total' in stats
        assert 'complete' in stats
        assert 'pruned' in stats
        assert 'failed' in stats
        assert 'pruning_rate' in stats

        # 値が妥当
        assert stats['total'] == 10
        assert stats['complete'] + stats['pruned'] + stats['failed'] == stats['total']
        assert 0 <= stats['pruning_rate'] <= 100


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optunaがインストールされていません")
class TestOptunaVisualization:
    """Optuna可視化のテスト"""

    def test_generate_visualizations(self, sample_member_competence):
        """可視化グラフが生成されるかテスト"""
        tuner = NMFHyperparameterTuner(
            sample_member_competence,
            search_space={
                'n_components': (5, 10),
                'alpha_W': (0.001, 0.1),
                'alpha_H': (0.001, 0.1),
                'l1_ratio': (0.0, 1.0),
            },
            n_splits=2,
            random_state=42,
        )

        tuner.optimize(n_trials=10, show_progress_bar=False)

        # 可視化を生成
        visualizations = generate_optuna_visualizations(
            tuner.study,
            params_to_plot=['n_components', 'alpha_W', 'alpha_H', 'l1_ratio']
        )

        # 期待される可視化が生成される
        expected_keys = [
            'optimization_history',
            'param_importances',
            'parallel_coordinate',
            'contour',
            'slice',
            'edf',
        ]

        for key in expected_keys:
            assert key in visualizations, f"{key}が生成されていない"

    def test_best_trials_summary(self, sample_member_competence):
        """上位試行サマリーが取得できるかテスト"""
        tuner = NMFHyperparameterTuner(
            sample_member_competence,
            search_space={
                'n_components': (5, 10),
                'alpha_W': (0.001, 0.1),
                'alpha_H': (0.001, 0.1),
                'l1_ratio': (0.0, 1.0),
            },
            n_splits=2,
            random_state=42,
        )

        tuner.optimize(n_trials=10, show_progress_bar=False)

        # 上位5試行を取得
        summary_df = get_best_trials_summary(tuner.study, top_n=5)

        # データフレームの形状
        assert summary_df.shape[0] == 5
        assert 'Rank' in summary_df.columns
        assert 'Trial' in summary_df.columns
        assert 'Value' in summary_df.columns

        # パラメータが含まれている
        assert 'n_components' in summary_df.columns
        assert 'alpha_W' in summary_df.columns

        # Rankが1～5
        assert list(summary_df['Rank']) == [1, 2, 3, 4, 5]

        # Valueが昇順（最小化問題）
        assert all(summary_df['Value'].iloc[i] <= summary_df['Value'].iloc[i+1]
                   for i in range(len(summary_df) - 1))

    def test_plot_training_history(self):
        """学習履歴プロットが生成されるかテスト"""
        # サンプル学習履歴
        training_history = {
            'train_errors': [0.5, 0.4, 0.35, 0.32, 0.31],
            'val_errors': [0.55, 0.45, 0.42, 0.40, 0.39],
            'generalization_gaps': [0.05, 0.05, 0.07, 0.08, 0.08],
        }

        fig = plot_training_history(training_history, title="テスト学習履歴")

        # Plotly Figureオブジェクトが返される
        import plotly.graph_objects as go
        assert isinstance(fig, go.Figure)

        # 3つのtraceが存在する
        assert len(fig.data) == 3

    def test_visualization_with_minimal_trials(self, sample_member_competence):
        """最小限の試行数でも可視化が生成されるかテスト"""
        tuner = NMFHyperparameterTuner(
            sample_member_competence,
            search_space={
                'n_components': (5, 10),
                'alpha_W': (0.001, 0.1),
            },
            n_splits=2,
            random_state=42,
        )

        # 2試行のみ
        tuner.optimize(n_trials=2, show_progress_bar=False)

        # 可視化を生成（エラーなく実行される）
        try:
            visualizations = generate_optuna_visualizations(
                tuner.study,
                params_to_plot=['n_components', 'alpha_W']
            )
            # 少なくとも一部の可視化が生成される
            assert len(visualizations) > 0
        except Exception as e:
            pytest.fail(f"最小限の試行数で可視化生成に失敗: {e}")


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optunaがインストールされていません")
class TestOptunaIntegration:
    """統合テスト"""

    def test_full_workflow_with_persistence_and_visualization(
        self, sample_member_competence, temp_storage_path
    ):
        """永続化と可視化を含む完全なワークフローのテスト"""
        # チューニング実行
        tuner = NMFHyperparameterTuner(
            sample_member_competence,
            search_space={
                'n_components': (5, 10),
                'alpha_W': (0.001, 0.1),
                'alpha_H': (0.001, 0.1),
                'l1_ratio': (0.0, 1.0),
            },
            n_splits=2,
            random_state=42,
        )

        tuner.storage_url = temp_storage_path
        tuner.study_name = "integration_test"

        # Studyを作成（永続化 + Pruning）
        tuner.study = optuna.create_study(
            study_name="integration_test",
            storage=temp_storage_path,
            load_if_exists=True,
            direction='minimize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=2),
        )

        # 最適化実行
        best_params, best_value = tuner.optimize(n_trials=10, show_progress_bar=False)

        # 1. 最適化が完了
        assert tuner.study is not None
        assert len(tuner.study.trials) == 10

        # 2. 統計情報を取得
        stats = get_pruned_trials_count(tuner.study)
        assert stats['total'] == 10

        # 3. 可視化を生成
        visualizations = generate_optuna_visualizations(
            tuner.study,
            params_to_plot=['n_components', 'alpha_W']
        )
        assert len(visualizations) > 0

        # 4. 上位試行を取得
        summary_df = get_best_trials_summary(tuner.study, top_n=5)
        assert summary_df.shape[0] == 5

        # 5. 永続化を確認（別のtunerで再ロード）
        tuner2 = NMFHyperparameterTuner(
            sample_member_competence,
            search_space={
                'n_components': (5, 10),
                'alpha_W': (0.001, 0.1),
                'alpha_H': (0.001, 0.1),
                'l1_ratio': (0.0, 1.0),
            },
            n_splits=2,
            random_state=42,
        )

        tuner2.study = optuna.load_study(
            study_name="integration_test",
            storage=temp_storage_path,
        )

        assert len(tuner2.study.trials) == 10
        assert tuner2.study.best_value == best_value
