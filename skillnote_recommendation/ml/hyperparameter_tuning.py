"""
Optunaベースのハイパーパラメータチューニングモジュール

NMFモデルのハイパーパラメータをベイズ最適化で探索

改善内容:
1. random_state固定化（パラメータの効果と初期値の効果を分離）
2. K-Fold交差検証の導入（過学習の検出）
3. 探索空間の最適化（効率的な探索）
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Callable
import logging
from sklearn.model_selection import KFold

try:
    import optuna
    from optuna.samplers import TPESampler, RandomSampler, GridSampler, CmaEsSampler
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
        search_space: Optional[Dict] = None,
        use_cross_validation: bool = True,
        n_folds: int = 3,
        sampler: str = "tpe",
        progress_callback: Optional[Callable] = None
    ):
        """
        初期化

        Args:
            skill_matrix: メンバー×力量マトリクス
            n_trials: 試行回数
            timeout: タイムアウト（秒）
            n_jobs: 並列実行数
            random_state: 乱数シード（固定化により再現性を確保）
            search_space: 探索空間の辞書
            use_cross_validation: 交差検証を使用するか（推奨: True）
            n_folds: 交差検証の分割数（デフォルト: 3, 計算時間とのトレードオフ）
            sampler: サンプラー種別 ("tpe", "random", "cmaes")
            progress_callback: 進捗コールバック関数（trial, study を受け取る）
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
        self.use_cross_validation = use_cross_validation
        self.n_folds = n_folds
        self.sampler = sampler
        self.progress_callback = progress_callback

        # 最適化された探索空間（より狭い範囲で効率的に探索）
        self.search_space = search_space or {
            'n_components': (10, 30),  # 40 -> 30に縮小（計算効率改善）
            'alpha_W': (0.001, 0.5),   # 1.0 -> 0.5に縮小（過度な正則化を避ける）
            'alpha_H': (0.001, 0.5),   # 1.0 -> 0.5に縮小
            'l1_ratio': (0.0, 1.0),
            'max_iter': (500, 1500),   # 2000 -> 1500に縮小（計算効率改善）
        }

        self.best_params = None
        self.best_value = None
        self.study = None

    def objective(self, trial: 'optuna.Trial') -> float:
        """
        Optunaの目的関数（交差検証による再構成誤差を最小化）

        改善点:
        1. random_stateを固定（パラメータの効果のみを評価）
        2. 交差検証による汎化性能の評価
        3. より詳細なメトリクスの記録

        Args:
            trial: Optunaのtrialオブジェクト

        Returns:
            交差検証による平均再構成誤差、またはfull-data再構成誤差
        """
        # ハイパーパラメータをサンプリング
        params = self._suggest_params(trial)

        # デバッグ：パラメータをログ出力
        logger.info(f"Trial {trial.number}: {params}")

        try:
            if self.use_cross_validation:
                # 交差検証を使用
                cv_errors = []
                kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

                for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(self.skill_matrix)):
                    # 訓練データと検証データに分割
                    train_matrix = self.skill_matrix.iloc[train_idx]
                    val_matrix = self.skill_matrix.iloc[val_idx]

                    # モデルを学習（random_stateは固定）
                    model = MatrixFactorizationModel(**params, random_state=self.random_state)
                    model.fit(train_matrix)

                    # 検証データで評価
                    val_error = self._calculate_validation_error(model, train_matrix, val_matrix)
                    cv_errors.append(val_error)

                    logger.debug(f"  Fold {fold_idx+1}/{self.n_folds}: error={val_error:.6f}")

                # 平均誤差を計算
                mean_cv_error = np.mean(cv_errors)
                std_cv_error = np.std(cv_errors)

                logger.info(f"Trial {trial.number} CV error: {mean_cv_error:.6f} (±{std_cv_error:.6f})")

                # 追加メトリクスをログ
                trial.set_user_attr('cv_std', std_cv_error)
                trial.set_user_attr('cv_errors', cv_errors)

                return mean_cv_error
            else:
                # 交差検証なし（従来の方法、ただしrandom_stateは固定）
                model = MatrixFactorizationModel(**params, random_state=self.random_state)
                model.fit(self.skill_matrix)

                # 再構成誤差を取得
                reconstruction_error = model.get_reconstruction_error()

                logger.info(f"Trial {trial.number} reconstruction error: {reconstruction_error:.6f}")

                # 追加メトリクスをログ
                trial.set_user_attr('n_iter', model.model.n_iter_)
                trial.set_user_attr('sparsity_W', np.sum(model.W == 0) / model.W.size)
                trial.set_user_attr('sparsity_H', np.sum(model.H == 0) / model.H.size)

                return reconstruction_error

        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {e}")
            # 失敗した場合は大きな値を返す
            return float('inf')

    def _calculate_validation_error(
        self,
        model: MatrixFactorizationModel,
        train_matrix: pd.DataFrame,
        val_matrix: pd.DataFrame
    ) -> float:
        """
        検証データでの再構成誤差を計算

        検証データのメンバーのうち、訓練データに存在するメンバーのみを対象に、
        力量の予測精度を評価します。

        Args:
            model: 学習済みモデル
            train_matrix: 訓練データ
            val_matrix: 検証データ

        Returns:
            検証データでの再構成誤差（Frobenius norm）
        """
        total_squared_error = 0.0
        total_elements = 0

        for member_code in val_matrix.index:
            # 訓練データに存在するメンバーのみ評価可能
            if member_code not in model.member_index:
                continue

            # 予測スコアを取得
            try:
                pred_scores = model.predict(member_code)
            except ValueError:
                continue

            # 実際の値
            actual_scores = val_matrix.loc[member_code]

            # 共通の力量コードのみを比較
            common_codes = list(set(pred_scores.index) & set(actual_scores.index))

            if not common_codes:
                continue

            # 二乗誤差を計算
            pred_values = pred_scores[common_codes].values
            actual_values = actual_scores[common_codes].values

            squared_error = np.sum((pred_values - actual_values) ** 2)
            total_squared_error += squared_error
            total_elements += len(common_codes)

        # Frobenius norm（平方根を取る）
        if total_elements == 0:
            return float('inf')

        reconstruction_error = np.sqrt(total_squared_error)

        return reconstruction_error

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

        # Optunaのログレベルを調整（INFOレベルで詳細を表示）
        optuna.logging.set_verbosity(optuna.logging.INFO)

        # サンプラーを選択
        if self.sampler == "tpe":
            sampler_instance = TPESampler(seed=self.random_state)
            logger.info("サンプラー: TPE (Tree-structured Parzen Estimator)")
        elif self.sampler == "random":
            sampler_instance = RandomSampler(seed=self.random_state)
            logger.info("サンプラー: Random Sampler")
        elif self.sampler == "cmaes":
            sampler_instance = CmaEsSampler(seed=self.random_state)
            logger.info("サンプラー: CMA-ES (Covariance Matrix Adaptation Evolution Strategy)")
        else:
            sampler_instance = TPESampler(seed=self.random_state)
            logger.warning(f"不明なサンプラー '{self.sampler}'。TPEを使用します。")

        # Studyを作成
        self.study = optuna.create_study(
            direction='minimize',  # 再構成誤差を最小化
            sampler=sampler_instance
        )

        # プログレスコールバックを設定
        all_callbacks = callbacks or []
        if self.progress_callback:
            def progress_wrapper(study, trial):
                self.progress_callback(trial, study)
            all_callbacks.append(progress_wrapper)

        # 最適化実行
        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs,
            show_progress_bar=show_progress_bar,
            callbacks=all_callbacks if all_callbacks else None
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

        改善点: random_stateを固定（self.random_state）し、全データで学習

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

        logger.info(f"最良モデルの学習: params={params}, random_state={self.random_state}")

        # モデルを学習（全データで、固定されたrandom_stateを使用）
        model = MatrixFactorizationModel(**params, random_state=self.random_state)
        model.fit(self.skill_matrix)

        logger.info(f"最良モデル学習完了: reconstruction_error={model.get_reconstruction_error():.6f}")

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
    return_tuner: bool = False,
    custom_n_trials: Optional[int] = None,
    custom_timeout: Optional[int] = None,
    custom_search_space: Optional[Dict] = None,
    custom_sampler: Optional[str] = None,
    progress_callback: Optional[Callable] = None
) -> Tuple:
    """
    Configを使ってハイパーパラメータチューニングを実行

    Args:
        skill_matrix: メンバー×力量マトリクス
        config: Configクラスインスタンス
        show_progress_bar: プログレスバーを表示するか
        return_tuner: Tunerオブジェクトも返すか
        custom_n_trials: カスタム試行回数（Noneの場合はconfigから取得）
        custom_timeout: カスタムタイムアウト（Noneの場合はconfigから取得）
        custom_search_space: カスタム探索空間（Noneの場合はconfigから取得）
        custom_sampler: カスタムサンプラー（Noneの場合は"tpe"）
        progress_callback: 進捗コールバック関数

    Returns:
        return_tuner=False: (最適パラメータ, 最小再構成誤差, 最良モデル)
        return_tuner=True: (最適パラメータ, 最小再構成誤差, 最良モデル, Tuner)
    """
    optuna_params = config.OPTUNA_PARAMS

    tuner = NMFHyperparameterTuner(
        skill_matrix=skill_matrix,
        n_trials=custom_n_trials or optuna_params['n_trials'],
        timeout=custom_timeout or optuna_params['timeout'],
        n_jobs=optuna_params['n_jobs'],
        random_state=config.MF_PARAMS['random_state'],
        search_space=custom_search_space or optuna_params['search_space'],
        use_cross_validation=optuna_params.get('use_cross_validation', True),
        n_folds=optuna_params.get('n_folds', 3),
        sampler=custom_sampler or "tpe",
        progress_callback=progress_callback
    )

    best_params, best_value = tuner.optimize(show_progress_bar=show_progress_bar)
    best_model = tuner.get_best_model()

    if return_tuner:
        return best_params, best_value, best_model, tuner
    else:
        return best_params, best_value, best_model
