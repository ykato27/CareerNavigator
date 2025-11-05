"""
Optunaベースのハイパーパラメータチューニングモジュール

NMFモデルのハイパーパラメータをベイズ最適化で探索
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Callable
import logging

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning("Optunaがインストールされていません。ハイパーパラメータチューニング機能が使用できません。")

from skillnote_recommendation.ml.matrix_factorization import MatrixFactorizationModel

logger = logging.getLogger(__name__)


class NMFHyperparameterTuner:
    """NMFモデルのハイパーパラメータチューナー（Optuna版）"""

    def __init__(
        self,
        skill_matrix: pd.DataFrame,
        n_trials: int = 50,
        timeout: Optional[int] = 600,
        n_jobs: int = 1,
        random_state: int = 42,
        search_space: Optional[Dict] = None
    ):
        """
        初期化

        Args:
            skill_matrix: メンバー×力量マトリクス
            n_trials: 試行回数
            timeout: タイムアウト（秒）
            n_jobs: 並列実行数
            random_state: 乱数シード
            search_space: 探索空間の辞書
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optunaがインストールされていません。\n"
                "pip install optuna でインストールしてください。"
            )

        self.skill_matrix = skill_matrix
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.random_state = random_state

        # デフォルトの探索空間
        self.search_space = search_space or {
            'n_components': (10, 40),
            'alpha_W': (0.0, 0.2),
            'alpha_H': (0.0, 0.2),
            'l1_ratio': (0.0, 1.0),
            'max_iter': (500, 2000),
        }

        self.best_params = None
        self.best_value = None
        self.study = None

    def objective(self, trial: 'optuna.Trial') -> float:
        """
        Optunaの目的関数（再構成誤差を最小化）

        Args:
            trial: Optunaのtrialオブジェクト

        Returns:
            再構成誤差
        """
        # ハイパーパラメータをサンプリング
        params = self._suggest_params(trial)

        try:
            # モデルを学習
            model = MatrixFactorizationModel(**params, random_state=self.random_state)
            model.fit(self.skill_matrix)

            # 再構成誤差を取得
            reconstruction_error = model.get_reconstruction_error()

            # 追加メトリクスをログ
            trial.set_user_attr('n_iter', model.model.n_iter_)
            trial.set_user_attr('sparsity_W', np.sum(model.W == 0) / model.W.size)
            trial.set_user_attr('sparsity_H', np.sum(model.H == 0) / model.H.size)

            return reconstruction_error

        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {e}")
            # 失敗した場合は大きな値を返す
            return float('inf')

    def _suggest_params(self, trial: 'optuna.Trial') -> Dict:
        """
        探索空間からハイパーパラメータをサンプリング

        Args:
            trial: Optunaのtrialオブジェクト

        Returns:
            パラメータの辞書
        """
        params = {}

        # n_components: 整数型
        if 'n_components' in self.search_space:
            min_val, max_val = self.search_space['n_components']
            params['n_components'] = trial.suggest_int('n_components', min_val, max_val)

        # max_iter: 整数型
        if 'max_iter' in self.search_space:
            min_val, max_val = self.search_space['max_iter']
            params['max_iter'] = trial.suggest_int('max_iter', min_val, max_val, step=100)

        # alpha_W: 対数スケールで探索（0に近い値も探索しやすくする）
        if 'alpha_W' in self.search_space:
            min_val, max_val = self.search_space['alpha_W']
            params['alpha_W'] = trial.suggest_float('alpha_W', min_val, max_val, log=True) if min_val > 0 else trial.suggest_float('alpha_W', min_val, max_val)

        # alpha_H: 対数スケールで探索
        if 'alpha_H' in self.search_space:
            min_val, max_val = self.search_space['alpha_H']
            params['alpha_H'] = trial.suggest_float('alpha_H', min_val, max_val, log=True) if min_val > 0 else trial.suggest_float('alpha_H', min_val, max_val)

        # l1_ratio: 線形スケールで探索
        if 'l1_ratio' in self.search_space:
            min_val, max_val = self.search_space['l1_ratio']
            params['l1_ratio'] = trial.suggest_float('l1_ratio', min_val, max_val)

        # 固定パラメータ
        params['init'] = 'nndsvda'
        params['solver'] = 'cd'
        params['tol'] = 1e-5

        return params

    def optimize(
        self,
        show_progress_bar: bool = True,
        callbacks: Optional[list] = None
    ) -> Tuple[Dict, float]:
        """
        ハイパーパラメータ最適化を実行

        Args:
            show_progress_bar: プログレスバーを表示するか
            callbacks: Optunaのコールバック関数リスト

        Returns:
            (最適パラメータ, 最小再構成誤差)
        """
        logger.info("=" * 60)
        logger.info("Optunaハイパーパラメータチューニング開始")
        logger.info("=" * 60)
        logger.info(f"試行回数: {self.n_trials}")
        logger.info(f"タイムアウト: {self.timeout}秒")
        logger.info(f"探索空間:")
        for key, value in self.search_space.items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 60)

        # Optunaのログレベルを調整（大量のログを抑制）
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Studyを作成（TPESamplerでベイズ最適化）
        sampler = TPESampler(seed=self.random_state)
        self.study = optuna.create_study(
            direction='minimize',  # 再構成誤差を最小化
            sampler=sampler
        )

        # 最適化実行
        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs,
            show_progress_bar=show_progress_bar,
            callbacks=callbacks
        )

        # 最良の結果を取得
        self.best_params = self.study.best_params
        self.best_value = self.study.best_value

        logger.info("\n" + "=" * 60)
        logger.info("チューニング完了")
        logger.info("=" * 60)
        logger.info(f"最適パラメータ: {self.best_params}")
        logger.info(f"最小再構成誤差: {self.best_value:.6f}")
        logger.info(f"完了した試行数: {len(self.study.trials)}")
        logger.info("=" * 60 + "\n")

        return self.best_params, self.best_value

    def get_best_model(self) -> MatrixFactorizationModel:
        """
        最適なパラメータで学習したモデルを取得

        Returns:
            学習済みMatrixFactorizationModel
        """
        if self.best_params is None:
            raise ValueError("先にoptimize()を実行してください。")

        # 固定パラメータを追加
        params = self.best_params.copy()
        params['init'] = 'nndsvda'
        params['solver'] = 'cd'
        params['tol'] = 1e-5

        # モデルを学習
        model = MatrixFactorizationModel(**params, random_state=self.random_state)
        model.fit(self.skill_matrix)

        return model

    def get_optimization_history(self) -> pd.DataFrame:
        """
        最適化の履歴を取得

        Returns:
            最適化履歴のDataFrame
        """
        if self.study is None:
            raise ValueError("先にoptimize()を実行してください。")

        trials_df = self.study.trials_dataframe()
        return trials_df

    def plot_optimization_history(self) -> 'plotly.graph_objects.Figure':
        """
        最適化履歴をプロット（Plotly版）

        Returns:
            Plotly figure
        """
        if self.study is None:
            raise ValueError("先にoptimize()を実行してください。")

        try:
            import plotly.graph_objects as go

            trials_df = self.get_optimization_history()

            fig = go.Figure()

            # 各試行の再構成誤差
            fig.add_trace(go.Scatter(
                x=trials_df['number'],
                y=trials_df['value'],
                mode='markers',
                name='各試行',
                marker=dict(size=8, opacity=0.6)
            ))

            # ベストバリューの推移
            best_values = trials_df['value'].cummin()
            fig.add_trace(go.Scatter(
                x=trials_df['number'],
                y=best_values,
                mode='lines',
                name='最良値の推移',
                line=dict(color='red', width=2)
            ))

            fig.update_layout(
                title='ハイパーパラメータ最適化の履歴',
                xaxis_title='Trial',
                yaxis_title='再構成誤差',
                height=500
            )

            return fig

        except ImportError:
            logger.warning("Plotlyがインストールされていません。")
            return None

    def plot_param_importances(self) -> 'plotly.graph_objects.Figure':
        """
        パラメータの重要度をプロット

        Returns:
            Plotly figure
        """
        if self.study is None:
            raise ValueError("先にoptimize()を実行してください。")

        try:
            import plotly.graph_objects as go

            # パラメータの重要度を計算
            importances = optuna.importance.get_param_importances(self.study)

            fig = go.Figure(go.Bar(
                x=list(importances.values()),
                y=list(importances.keys()),
                orientation='h'
            ))

            fig.update_layout(
                title='パラメータの重要度',
                xaxis_title='重要度',
                yaxis_title='パラメータ',
                height=400
            )

            return fig

        except ImportError:
            logger.warning("Plotlyがインストールされていません。")
            return None


def tune_nmf_hyperparameters_from_config(
    skill_matrix: pd.DataFrame,
    config,
    show_progress_bar: bool = True,
    return_tuner: bool = False
) -> Tuple:
    """
    Configを使ってハイパーパラメータチューニングを実行

    Args:
        skill_matrix: メンバー×力量マトリクス
        config: Configクラスインスタンス
        show_progress_bar: プログレスバーを表示するか
        return_tuner: Tunerオブジェクトも返すか

    Returns:
        return_tuner=False: (最適パラメータ, 最小再構成誤差, 最良モデル)
        return_tuner=True: (最適パラメータ, 最小再構成誤差, 最良モデル, Tuner)
    """
    optuna_params = config.OPTUNA_PARAMS

    tuner = NMFHyperparameterTuner(
        skill_matrix=skill_matrix,
        n_trials=optuna_params['n_trials'],
        timeout=optuna_params['timeout'],
        n_jobs=optuna_params['n_jobs'],
        random_state=config.MF_PARAMS['random_state'],
        search_space=optuna_params['search_space']
    )

    best_params, best_value = tuner.optimize(show_progress_bar=show_progress_bar)
    best_model = tuner.get_best_model()

    if return_tuner:
        return best_params, best_value, best_model, tuner
    else:
        return best_params, best_value, best_model
