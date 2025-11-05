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
from sklearn.model_selection import KFold, TimeSeriesSplit

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
        progress_callback: Optional[Callable] = None,
        use_time_series_split: bool = True,
        test_size: float = 0.15,
        enable_early_stopping: bool = True,
        early_stopping_patience: int = 5,
        early_stopping_batch_size: int = 50
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
            use_time_series_split: TimeSeriesSplitを使用するか（推奨: True, 時系列データの場合）
            test_size: テストセットのサイズ（0.0-1.0）デフォルト15%
            enable_early_stopping: Early stoppingを有効にするか
            early_stopping_patience: Early stopping用の待機回数（デフォルト: 5）
            early_stopping_batch_size: Early stopping用のバッチサイズ（デフォルト: 50）
                                       大きくすると高速化（推奨: 100-200）
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optunaがインストールされていません。\n"
                "pip install optuna でインストールしてください。"
            )

        # Test setを分離（チューニング時は触らない）
        if test_size > 0:
            # 時系列順にソート（indexがメンバーコードの場合、取得日でソートできない）
            # そのため、単純に後ろから切り取る（最近のデータがテスト）
            split_idx = int(len(skill_matrix) * (1 - test_size))
            self.skill_matrix = skill_matrix.iloc[:split_idx]  # Train + Validation
            self.test_matrix = skill_matrix.iloc[split_idx:]   # Test (隔離)
            print(f"[Data Split] Train+Val: {len(self.skill_matrix)}, Test: {len(self.test_matrix)} (Test ratio: {test_size:.1%})")
        else:
            self.skill_matrix = skill_matrix
            self.test_matrix = None
            print("[Data Split] No test set separation (test_size=0)")

        self.n_trials = n_trials
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.use_cross_validation = use_cross_validation
        self.n_folds = n_folds
        self.sampler = sampler
        self.progress_callback = progress_callback
        self.use_time_series_split = use_time_series_split
        self.test_size = test_size
        self.enable_early_stopping = enable_early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_batch_size = early_stopping_batch_size

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
        self.test_score = None  # Test setでの最終評価スコア

    def objective(self, trial: 'optuna.Trial') -> float:
        """
        Optunaの目的関数（交差検証による再構成誤差を最小化）

        改善点:
        1. random_stateを固定（パラメータの効果のみを評価）
        2. スキルベースの交差検証による汎化性能の評価
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
        print(f"[DEBUG] Trial {trial.number} started with params: {params}")

        try:
            if self.use_cross_validation:
                # Matrix Factorization用のスキルベース交差検証
                print(f"[DEBUG] Trial {trial.number}: Using skill-based cross-validation with {self.n_folds} folds")
                split_method = "SkillBasedCV"

                # 交差検証を実行
                cv_errors = []
                for fold_idx in range(self.n_folds):
                    print(f"[DEBUG] Trial {trial.number}, Fold {fold_idx+1}/{self.n_folds}: Training...")

                    # スキルベースの分割：各メンバーのスキルをランダムにマスク
                    train_matrix, val_mask = self._create_skill_based_split(fold_idx)

                    # モデルを学習（random_stateは固定、Early stopping有効）
                    model = MatrixFactorizationModel(
                        **params,
                        random_state=self.random_state,
                        early_stopping=self.enable_early_stopping,
                        early_stopping_patience=self.early_stopping_patience,
                        early_stopping_min_delta=1e-5,
                        early_stopping_batch_size=self.early_stopping_batch_size
                    )
                    model.fit(train_matrix)

                    # 検証データで評価（マスクされたスキルのみ）
                    val_error = self._calculate_skill_based_validation_error(model, val_mask)
                    cv_errors.append(val_error)

                    logger.debug(f"  Fold {fold_idx+1}/{self.n_folds}: error={val_error:.6f}")
                    print(f"[DEBUG] Trial {trial.number}, Fold {fold_idx+1}/{self.n_folds}: error={val_error:.6f}")

                # 平均誤差を計算
                mean_cv_error = np.mean(cv_errors)
                std_cv_error = np.std(cv_errors)

                logger.info(f"Trial {trial.number} CV error: {mean_cv_error:.6f} (±{std_cv_error:.6f})")
                print(f"[DEBUG] Trial {trial.number} completed with CV error: {mean_cv_error:.6f} ({split_method})")

                # 追加メトリクスをログ
                trial.set_user_attr('cv_std', std_cv_error)
                trial.set_user_attr('cv_errors', cv_errors)
                trial.set_user_attr('random_state', self.random_state)
                trial.set_user_attr('split_method', split_method)

                return mean_cv_error
            else:
                print(f"[DEBUG] Trial {trial.number}: Using full data (no cross-validation)")
                # 交差検証なし（従来の方法、ただしrandom_stateは固定）
                model = MatrixFactorizationModel(**params, random_state=self.random_state)

                print(f"[DEBUG] Trial {trial.number}: Fitting model...")
                model.fit(self.skill_matrix)

                # 再構成誤差を取得
                reconstruction_error = model.get_reconstruction_error()

                logger.info(f"Trial {trial.number} reconstruction error: {reconstruction_error:.6f}")
                print(f"[DEBUG] Trial {trial.number} completed with error: {reconstruction_error:.6f}")

                # 追加メトリクスをログ
                trial.set_user_attr('n_iter', model.model.n_iter_)
                trial.set_user_attr('random_state', self.random_state)
                trial.set_user_attr('sparsity_W', np.sum(model.W == 0) / model.W.size)
                trial.set_user_attr('sparsity_H', np.sum(model.H == 0) / model.H.size)

                return reconstruction_error

        except Exception as e:
            import traceback
            error_msg = f"Trial {trial.number} failed: {type(e).__name__}: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            print(f"[ERROR] {error_msg}")
            print(f"[ERROR] Traceback:\n{traceback.format_exc()}")
            # 失敗した場合は大きな値を返す
            return float('inf')

    def _create_skill_based_split(self, fold_idx: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        スキルベースの分割を作成

        各メンバーのスキル（力量）をランダムにマスクし、トレーニング用とバリデーション用に分割します。
        これにより、Matrix Factorizationモデルの評価が可能になります。

        Args:
            fold_idx: フォールドのインデックス（0から始まる）

        Returns:
            (train_matrix, val_mask): トレーニングマトリックス（マスク適用後）と
                                      バリデーションマスク（元の値を保持）
        """
        # 元のマトリックスをコピー
        train_matrix = self.skill_matrix.copy()
        val_mask = pd.DataFrame(0.0, index=self.skill_matrix.index, columns=self.skill_matrix.columns)

        # 各メンバーについて、スキルをランダムにマスク
        np.random.seed(self.random_state + fold_idx)  # fold毎に異なるseed

        val_ratio = 1.0 / self.n_folds  # 各foldで同じ割合をバリデーションに

        for member_code in self.skill_matrix.index:
            # このメンバーが持つスキル（非ゼロの要素）を取得
            member_skills = self.skill_matrix.loc[member_code]
            non_zero_skills = member_skills[member_skills > 0].index.tolist()

            if len(non_zero_skills) == 0:
                continue

            # バリデーション用にマスクするスキル数を計算
            n_val = max(1, int(len(non_zero_skills) * val_ratio))

            # ランダムにバリデーションスキルを選択
            val_skills = np.random.choice(non_zero_skills, size=n_val, replace=False)

            # バリデーションスキルをマスク（トレーニングマトリックスから削除）
            train_matrix.loc[member_code, val_skills] = 0.0

            # バリデーションマスクに元の値を保存
            val_mask.loc[member_code, val_skills] = member_skills[val_skills]

        return train_matrix, val_mask

    def _calculate_skill_based_validation_error(
        self,
        model: MatrixFactorizationModel,
        val_mask: pd.DataFrame
    ) -> float:
        """
        スキルベースのバリデーション誤差を計算

        マスクされたスキル（バリデーション用）の予測精度を評価します。

        Args:
            model: 学習済みモデル
            val_mask: バリデーションマスク（マスクされたスキルの元の値）

        Returns:
            バリデーション誤差（Frobenius norm）
        """
        total_squared_error = 0.0
        total_elements = 0

        for member_code in val_mask.index:
            # このメンバーのバリデーションスキルを取得
            member_val_skills = val_mask.loc[member_code]
            val_skills = member_val_skills[member_val_skills > 0]

            if len(val_skills) == 0:
                continue

            # 予測スコアを取得
            try:
                pred_scores = model.predict(member_code, competence_codes=val_skills.index.tolist())
            except ValueError:
                continue

            # 実際の値
            actual_values = val_skills.values
            pred_values = pred_scores.values

            # 二乗誤差を計算
            squared_error = np.sum((pred_values - actual_values) ** 2)
            total_squared_error += squared_error
            total_elements += len(val_skills)

        # Frobenius norm（平方根を取る）
        if total_elements == 0:
            return float('inf')

        reconstruction_error = np.sqrt(total_squared_error)

        return reconstruction_error

    def _calculate_validation_error(
        self,
        model: MatrixFactorizationModel,
        train_matrix: pd.DataFrame,
        val_matrix: pd.DataFrame
    ) -> float:
        """
        検証データでの再構成誤差を計算（レガシー - メンバーベース分割用）

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
        logger.info(f"交差検証: {self.use_cross_validation}")
        logger.info(f"探索空間:")
        for key, value in self.search_space.items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 60)

        # デバッグ: n_trialsが0または不正な値でないか確認
        if self.n_trials <= 0:
            raise ValueError(f"n_trials must be positive, got: {self.n_trials}")

        print(f"[DEBUG] Optimize started with n_trials={self.n_trials}")
        print(f"[DEBUG] skill_matrix shape: {self.skill_matrix.shape}")
        print(f"[DEBUG] timeout: {self.timeout}")
        print(f"[DEBUG] n_jobs: {self.n_jobs}")
        print(f"[DEBUG] use_cross_validation: {self.use_cross_validation}")
        print(f"[DEBUG] progress_callback is set: {self.progress_callback is not None}")

        # Optunaのログレベルを調整（INFOレベルで詳細を表示）
        optuna.logging.set_verbosity(optuna.logging.INFO)

        # サンプラーを選択
        if self.sampler == "tpe":
            sampler_instance = TPESampler(seed=self.random_state)
            logger.info("サンプラー: TPE (Tree-structured Parzen Estimator)")
            print("[DEBUG] Sampler: TPE")
        elif self.sampler == "random":
            sampler_instance = RandomSampler(seed=self.random_state)
            logger.info("サンプラー: Random Sampler")
            print("[DEBUG] Sampler: Random")
        elif self.sampler == "cmaes":
            sampler_instance = CmaEsSampler(seed=self.random_state)
            logger.info("サンプラー: CMA-ES (Covariance Matrix Adaptation Evolution Strategy)")
            print("[DEBUG] Sampler: CMA-ES")
        else:
            sampler_instance = TPESampler(seed=self.random_state)
            logger.warning(f"不明なサンプラー '{self.sampler}'。TPEを使用します。")
            print(f"[DEBUG] Unknown sampler '{self.sampler}', using TPE")

        # Studyを作成
        print("[DEBUG] Creating Optuna study...")
        self.study = optuna.create_study(
            direction='minimize',  # 再構成誤差を最小化
            sampler=sampler_instance
        )
        print("[DEBUG] Study created successfully")

        # プログレスコールバックを設定
        all_callbacks = callbacks or []
        if self.progress_callback:
            print("[DEBUG] Adding progress callback")
            def progress_wrapper(study, trial):
                self.progress_callback(trial, study)
            all_callbacks.append(progress_wrapper)

        # 最適化実行
        print(f"[DEBUG] Starting study.optimize with n_trials={self.n_trials}")
        try:
            self.study.optimize(
                self.objective,
                n_trials=self.n_trials,
                timeout=self.timeout,
                n_jobs=self.n_jobs,
                show_progress_bar=show_progress_bar,
                callbacks=all_callbacks if all_callbacks else None
            )
            print(f"[DEBUG] Study.optimize completed. Total trials: {len(self.study.trials)}")
        except Exception as e:
            import traceback
            print(f"[ERROR] Study.optimize failed: {type(e).__name__}: {e}")
            print(f"[ERROR] Traceback:\n{traceback.format_exc()}")
            logger.error(f"Study.optimize failed: {e}")
            logger.error(traceback.format_exc())
            raise

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
        model = MatrixFactorizationModel(
            **params,
            random_state=self.random_state,
            early_stopping=self.enable_early_stopping,
            early_stopping_patience=self.early_stopping_patience,
            early_stopping_min_delta=1e-5,
            early_stopping_batch_size=self.early_stopping_batch_size
        )
        model.fit(self.skill_matrix)

        logger.info(f"最良モデル学習完了: reconstruction_error={model.get_reconstruction_error():.6f}")

        # Test setが存在する場合、最終評価を実行
        if self.test_matrix is not None:
            self.test_score = self.evaluate_on_test_set(model)
            logger.info(f"Test set evaluation: reconstruction_error={self.test_score:.6f}")
            print(f"[Test Evaluation] Final test error: {self.test_score:.6f}")

        return model

    def evaluate_on_test_set(self, model: MatrixFactorizationModel) -> float:
        """
        チューニング完了後、Test setで最終評価

        重要: この評価は、ハイパーパラメータチューニング完了後に
        1度だけ実行すべきです。Test setは絶対にチューニングに使用してはいけません。

        Args:
            model: 学習済みモデル

        Returns:
            Test setでの再構成誤差
        """
        if self.test_matrix is None:
            raise ValueError("Test setが分離されていません。test_size > 0で初期化してください。")

        print(f"[Test Evaluation] Evaluating on test set (size: {len(self.test_matrix)})")

        # Test setで評価
        test_error = self._calculate_validation_error(
            model,
            self.skill_matrix,  # Train+Valで学習したモデル
            self.test_matrix    # Test setで評価
        )

        logger.info(f"Test set reconstruction error: {test_error:.6f}")

        return test_error

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
        progress_callback=progress_callback,
        use_time_series_split=optuna_params.get('use_time_series_split', True),
        test_size=optuna_params.get('test_size', 0.15),
        enable_early_stopping=optuna_params.get('enable_early_stopping', True),
        early_stopping_patience=optuna_params.get('early_stopping_patience', 5),
        early_stopping_batch_size=optuna_params.get('early_stopping_batch_size', 50)
    )

    best_params, best_value = tuner.optimize(show_progress_bar=show_progress_bar)
    best_model = tuner.get_best_model()

    if return_tuner:
        return best_params, best_value, best_model, tuner
    else:
        return best_params, best_value, best_model
