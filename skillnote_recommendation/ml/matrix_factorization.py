"""
Matrix Factorizationベースの推薦モデル

NMF (Non-negative Matrix Factorization)またはALS (Alternating Least Squares)を
使用してメンバー×力量マトリクスを潜在因子に分解し、未習得力量のスコアを予測する

改善内容:
1. Implicit feedback対応（ALS with confidence weighting）
2. 正規化レベルに基づくconfidence weighting
3. より適切なスコアリング

注意:
- confidence weighting使用時はALS（implicitライブラリ）を使用（推奨）
- NMFはconfidence weightingに非対応（理論的に不正確）
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from sklearn.decomposition import NMF
import pickle
import warnings

try:
    from implicit.als import AlternatingLeastSquares
    from scipy.sparse import csr_matrix
    IMPLICIT_AVAILABLE = True
except ImportError:
    IMPLICIT_AVAILABLE = False
    AlternatingLeastSquares = None
    csr_matrix = None


class MatrixFactorizationModel:
    """Matrix Factorizationベースの推薦モデル"""

    def __init__(
        self,
        n_components: int = 20,
        random_state: int = 42,
        use_confidence_weighting: bool = False,
        confidence_alpha: float = 1.0,
        early_stopping: bool = False,
        early_stopping_patience: int = 5,
        early_stopping_min_delta: float = 1e-5,
        early_stopping_batch_size: int = 50,
        **nmf_params,
    ):
        """
        初期化

        Args:
            n_components: 潜在因子の数（次元数）
            random_state: 乱数シード
            use_confidence_weighting: confidence weightingを使用するか
                                     Trueの場合、ALS (implicit library)を使用（推奨）
                                     NMFはconfidence weightingに理論的に非対応
            confidence_alpha: confidence weight計算の係数 (confidence = 1 + alpha * rating)
            early_stopping: Early stoppingを使用するか（NMFのみ）
            early_stopping_patience: Early stopping用の待機回数（改善が見られないエポック数）
            early_stopping_min_delta: 改善とみなす最小の誤差減少量
            early_stopping_batch_size: Early stopping用のイテレーションバッチサイズ（デフォルト: 50）
                                       大きくすると高速化するが、Early stoppingの精度が下がる
                                       推奨値: 50（標準）、100（高速）、200（最速）
            **nmf_params: NMFまたはALSへの追加パラメータ
        """
        self.n_components = n_components
        self.random_state = random_state
        self.use_confidence_weighting = use_confidence_weighting
        self.confidence_alpha = confidence_alpha
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.early_stopping_batch_size = early_stopping_batch_size
        self.nmf_params = nmf_params

        # モデルの種類を決定
        self.model_type = None  # 'nmf' or 'als'

        if use_confidence_weighting:
            # Confidence weighting使用時はALSを使用
            if not IMPLICIT_AVAILABLE:
                raise ImportError(
                    "confidence weighting使用時は implicit ライブラリが必要です。\n"
                    "以下のコマンドでインストールしてください:\n"
                    "  pip install implicit"
                )

            self.model_type = 'als'
            # ALSモデル（implicit library）
            # iterations, regularizationなどのパラメータを取得
            als_iterations = nmf_params.get('iterations', nmf_params.get('max_iter', 15))
            als_regularization = nmf_params.get('regularization', 0.01)

            self.model = AlternatingLeastSquares(
                factors=n_components,
                iterations=als_iterations,
                regularization=als_regularization,
                random_state=random_state,
                use_gpu=False,  # CPU使用（GPUは環境依存）
            )
        else:
            # Confidence weightingなしの場合はNMFを使用
            self.model_type = 'nmf'
            # NMFモデル
            # max_iterがnmf_paramsに含まれていない場合はデフォルト値500を使用
            default_params = {"init": "nndsvda", "max_iter": 500}
            final_params = {**default_params, **nmf_params}
            self.model = NMF(n_components=n_components, random_state=random_state, **final_params)

        # 学習後のデータ
        self.X = None  # 元のデータマトリクス（再構成誤差計算用）
        self.W = None  # メンバー因子行列（item factors for ALS）
        self.H = None  # 力量因子行列（user factors for ALS）
        self.member_codes = None  # メンバーコードのリスト
        self.competence_codes = None  # 力量コードのリスト
        self.member_index = None  # メンバーコード → インデックス
        self.competence_index = None  # 力量コード → インデックス
        self.is_fitted = False
        self.actual_n_iter_ = None  # 実際のイテレーション数（Early stopping時）

    def fit(self, skill_matrix: pd.DataFrame, validation_matrix: Optional[pd.DataFrame] = None) -> "MatrixFactorizationModel":
        """
        モデルを学習

        改善点:
        - ALS使用時: confidence weightingにより、高レベルのスキルをより重視（Hu et al. 2008）
        - NMF使用時: 標準的な非負値行列分解
        - 暗黙的フィードバック（有無のみ）と明示的フィードバック（レベル）の両方に対応
        - Early stoppingによる効率的な学習（NMFのみ）
        - 検証セット監視による過学習防止

        Args:
            skill_matrix: メンバー×力量マトリクス (index=メンバーコード, columns=力量コード)
            validation_matrix: 検証用マトリクス（Early stopping時に使用、NMFのみ）

        Returns:
            self
        """
        # メンバー・力量のコードを保存
        self.member_codes = skill_matrix.index.tolist()
        self.competence_codes = skill_matrix.columns.tolist()

        # インデックスマッピングを作成
        self.member_index = {code: idx for idx, code in enumerate(self.member_codes)}
        self.competence_index = {code: idx for idx, code in enumerate(self.competence_codes)}

        if self.model_type == 'als':
            # ALS（implicit library）を使用した学習
            self._fit_als(skill_matrix)
        else:
            # NMFを使用した学習
            training_matrix = skill_matrix.values

            # 検証用マトリクスを準備
            if validation_matrix is not None:
                validation_matrix_values = validation_matrix.values
            else:
                validation_matrix_values = None

            # Early stoppingの有無で分岐
            if self.early_stopping:
                self._fit_with_early_stopping(training_matrix, validation_matrix_values)
            else:
                self._fit_normal(training_matrix)

        self.is_fitted = True
        return self

    def _fit_als(self, skill_matrix: pd.DataFrame) -> None:
        """
        ALS (Alternating Least Squares)を使用した学習

        implicit libraryのALSを使用し、confidence weightingを適切に適用。
        Hu et al. (2008)のimplicit feedbackアプローチに基づく。

        Args:
            skill_matrix: メンバー×力量マトリクス (index=メンバーコード, columns=力量コード)
        """
        # Preference matrix (binary: 1 if acquired, 0 otherwise)
        preference_matrix = (skill_matrix > 0).astype(np.float32)

        # Confidence matrix: c = 1 + alpha * rating
        # ratingは正規化されたスキルレベル（0-1）
        confidence_matrix = 1 + self.confidence_alpha * skill_matrix.values

        # Weighted matrix for ALS: preference * confidence
        # implicit libraryではスパース行列を使用
        # 転置が必要: implicit libraryはitem×userを期待（我々はmember×competence）
        user_item_data = csr_matrix((preference_matrix * confidence_matrix).T)

        # ALSで学習
        print(f"[ALS] 学習開始（factors={self.n_components}, iterations={self.model.iterations}）")
        self.model.fit(user_item_data, show_progress=True)

        # 因子行列を取得
        # implicit library: item_factors (competence×factors), user_factors (member×factors)
        # 我々の定義: W (member×factors), H (factors×competence)
        self.W = self.model.user_factors  # member × factors
        self.H = self.model.item_factors.T  # factors × competence

        # 元のデータマトリクスを保存（再構成誤差計算用）
        self.X = skill_matrix.values

        # イテレーション数を保存
        self.actual_n_iter_ = self.model.iterations

        print(f"[ALS] 学習完了")

    def _fit_normal(self, X: np.ndarray) -> None:
        """通常の学習（Early stoppingなし）"""
        self.X = X  # 元のデータマトリクスを保存（再構成誤差計算用）
        self.W = self.model.fit_transform(X)
        self.H = self.model.components_
        self.actual_n_iter_ = self.model.n_iter_

    def _fit_with_early_stopping(self, X: np.ndarray, X_val: Optional[np.ndarray] = None) -> None:
        """
        Early stoppingを使用した学習（Multiplicative Update Rule）

        1ステップずつ更新しながら、検証セットの誤差が増加し始めたら早期終了する。
        Lee & Seung (2001)のMultiplicative Update Ruleを使用。

        Args:
            X: 訓練用マトリクス (m × n)
            X_val: 検証用マトリクス（Noneの場合は訓練誤差で判定）
        """
        m, n = X.shape
        max_iter = self.model.max_iter

        # 初期化（NNDSVDA）
        print("[Early Stopping] 初期化中（NNDSVDA）...")
        W, H = self._initialize_nmf(X, self.n_components)

        best_error = float("inf")
        patience_counter = 0
        best_W = W.copy()
        best_H = H.copy()
        best_iter = 0

        # 学習履歴
        train_errors = []
        val_errors = []
        generalization_gaps = []

        # アルファ値（正則化係数）を取得
        alpha_W = self.nmf_params.get('alpha_W', 0.0)
        alpha_H = self.nmf_params.get('alpha_H', 0.0)
        l1_ratio = self.nmf_params.get('l1_ratio', 0.0)

        print(f"[Early Stopping] 学習開始（max_iter={max_iter}, patience={self.early_stopping_patience}）")

        for iter_num in range(1, max_iter + 1):
            # Multiplicative Update Rule（1ステップ）
            W, H = self._update_nmf_step(X, W, H, alpha_W, alpha_H, l1_ratio)

            # 評価頻度ごとに誤差を計算
            if iter_num % 10 == 0 or iter_num == max_iter:
                # 訓練誤差を計算
                reconstructed = W @ H
                train_error = np.linalg.norm(X - reconstructed, "fro")
                train_errors.append(train_error)

                # 検証誤差を計算
                if X_val is not None:
                    # 検証セット用のW（訓練データで学習済みのHを使用）
                    W_val = self._transform_new_data(X_val, H, alpha_W, l1_ratio)
                    reconstructed_val = W_val @ H
                    val_error = np.linalg.norm(X_val - reconstructed_val, "fro")
                    val_errors.append(val_error)
                    generalization_gap = val_error - train_error
                    generalization_gaps.append(generalization_gap)
                else:
                    val_error = train_error
                    val_errors.append(val_error)
                    generalization_gap = 0.0

                # Early stopping判定
                current_error = val_error if X_val is not None else train_error
                improvement = best_error - current_error

                if improvement > self.early_stopping_min_delta:
                    best_error = current_error
                    best_W = W.copy()
                    best_H = H.copy()
                    best_iter = iter_num
                    patience_counter = 0

                    if iter_num % 50 == 0:  # 50イテレーションごとに表示
                        if X_val is not None:
                            print(
                                f"[Early Stopping] Iter {iter_num:4d}: "
                                f"train={train_error:.6f}, val={val_error:.6f}, "
                                f"gap={generalization_gap:.6f} ✓"
                            )
                        else:
                            print(f"[Early Stopping] Iter {iter_num:4d}: error={train_error:.6f} ✓")
                else:
                    patience_counter += 1

                    if iter_num % 50 == 0:  # 50イテレーションごとに表示
                        if X_val is not None:
                            print(
                                f"[Early Stopping] Iter {iter_num:4d}: "
                                f"train={train_error:.6f}, val={val_error:.6f}, "
                                f"gap={generalization_gap:.6f} (patience={patience_counter}/{self.early_stopping_patience})"
                            )
                        else:
                            print(
                                f"[Early Stopping] Iter {iter_num:4d}: error={train_error:.6f} "
                                f"(patience={patience_counter}/{self.early_stopping_patience})"
                            )

                    if patience_counter >= self.early_stopping_patience:
                        print(
                            f"\n[Early Stopping] ✅ 停止（Iter {best_iter}, best_error={best_error:.6f}）"
                        )
                        if X_val is not None:
                            avg_gap = np.mean(generalization_gaps[-self.early_stopping_patience:])
                            print(f"[Early Stopping] 平均汎化ギャップ: {avg_gap:.6f}")
                        break

        # ベストモデルを設定
        self.X = X
        self.W = best_W
        self.H = best_H
        self.actual_n_iter_ = best_iter

        # 学習履歴を保存
        self.training_history = {
            'train_errors': train_errors,
            'val_errors': val_errors,
            'generalization_gaps': generalization_gaps,
        }

        print(f"[Early Stopping] 学習完了（最終Iter: {best_iter}/{max_iter}）")

    def _initialize_nmf(self, X: np.ndarray, n_components: int) -> tuple:
        """
        NMFの初期化（NNDSVDA法）

        SVD分解に基づく初期化により、収束を高速化する。

        Args:
            X: データマトリクス (m × n)
            n_components: 潜在因子数

        Returns:
            W: (m × n_components), H: (n_components × n)
        """
        from sklearn.utils.extmath import randomized_svd

        m, n = X.shape

        # SVD分解
        U, S, Vt = randomized_svd(X, n_components=n_components, random_state=self.random_state)

        # W, Hの初期化
        W = np.zeros((m, n_components))
        H = np.zeros((n_components, n))

        # 正の成分のみ抽出
        W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
        H[0, :] = np.sqrt(S[0]) * np.abs(Vt[0, :])

        for j in range(1, n_components):
            x = U[:, j]
            y = Vt[j, :]

            x_pos = np.maximum(x, 0)
            y_pos = np.maximum(y, 0)
            x_neg = np.abs(np.minimum(x, 0))
            y_neg = np.abs(np.minimum(y, 0))

            x_pos_norm = np.linalg.norm(x_pos)
            y_pos_norm = np.linalg.norm(y_pos)
            x_neg_norm = np.linalg.norm(x_neg)
            y_neg_norm = np.linalg.norm(y_neg)

            m_pos = x_pos_norm * y_pos_norm
            m_neg = x_neg_norm * y_neg_norm

            if m_pos >= m_neg:
                u = x_pos / (x_pos_norm + 1e-10)
                v = y_pos / (y_pos_norm + 1e-10)
                sigma = m_pos
            else:
                u = x_neg / (x_neg_norm + 1e-10)
                v = y_neg / (y_neg_norm + 1e-10)
                sigma = m_neg

            W[:, j] = np.sqrt(S[j] * sigma) * u
            H[j, :] = np.sqrt(S[j] * sigma) * v

        # 小さな正の値を追加（ゼロ除算防止）
        W = np.maximum(W, 1e-10)
        H = np.maximum(H, 1e-10)

        return W, H

    def _update_nmf_step(
        self,
        X: np.ndarray,
        W: np.ndarray,
        H: np.ndarray,
        alpha_W: float = 0.0,
        alpha_H: float = 0.0,
        l1_ratio: float = 0.0
    ) -> tuple:
        """
        NMFの1ステップ更新（Multiplicative Update Rule with regularization）

        Lee & Seung (2001)のアルゴリズムに正則化を追加。

        Args:
            X: データマトリクス (m × n)
            W: メンバー因子行列 (m × k)
            H: 力量因子行列 (k × n)
            alpha_W: Wの正則化係数
            alpha_H: Hの正則化係数
            l1_ratio: L1正則化の比率（0=L2のみ, 1=L1のみ）

        Returns:
            更新されたW, H
        """
        eps = 1e-10

        # Hの更新
        # H = H * (W^T @ X) / (W^T @ W @ H + alpha_H * (l1 + (1-l1)*H))
        numerator_H = W.T @ X
        denominator_H = W.T @ W @ H + eps

        if alpha_H > 0:
            l1_H = l1_ratio
            l2_H = 1 - l1_ratio
            denominator_H += alpha_H * (l1_H + l2_H * H)

        H *= numerator_H / denominator_H

        # Wの更新
        # W = W * (X @ H^T) / (W @ H @ H^T + alpha_W * (l1 + (1-l1)*W))
        numerator_W = X @ H.T
        denominator_W = W @ H @ H.T + eps

        if alpha_W > 0:
            l1_W = l1_ratio
            l2_W = 1 - l1_ratio
            denominator_W += alpha_W * (l1_W + l2_W * W)

        W *= numerator_W / denominator_W

        return W, H

    def _transform_new_data(
        self,
        X_new: np.ndarray,
        H: np.ndarray,
        alpha: float = 0.0,
        l1_ratio: float = 0.0,
        max_iter: int = 100
    ) -> np.ndarray:
        """
        新しいデータに対してWを計算（Hは固定）

        Args:
            X_new: 新しいデータマトリクス (m_new × n)
            H: 学習済みの力量因子行列 (k × n)
            alpha: 正則化係数
            l1_ratio: L1正則化の比率
            max_iter: 最大イテレーション数

        Returns:
            W_new: (m_new × k)
        """
        m_new, n = X_new.shape
        k = H.shape[0]

        # Wの初期化（ランダム）
        W_new = np.random.rand(m_new, k) * 0.01 + 0.01

        eps = 1e-10

        for _ in range(max_iter):
            # Wの更新（Hは固定）
            numerator = X_new @ H.T
            denominator = W_new @ H @ H.T + eps

            if alpha > 0:
                l1 = l1_ratio
                l2 = 1 - l1_ratio
                denominator += alpha * (l1 + l2 * W_new)

            W_new *= numerator / denominator

        return W_new

    def predict(self, member_code: str, competence_codes: Optional[List[str]] = None) -> pd.Series:
        """
        特定メンバーに対する力量のスコアを予測

        Args:
            member_code: メンバーコード
            competence_codes: 予測対象の力量コードリスト（Noneの場合は全力量）

        Returns:
            力量コードをインデックスとするスコアのSeries

        Raises:
            ValueError: モデルが未学習、またはメンバーコードが不明な場合
        """
        if not self.is_fitted:
            raise ValueError("モデルが学習されていません。先にfit()を呼んでください。")

        if member_code not in self.member_index:
            raise ValueError(f"メンバーコード '{member_code}' は学習データに存在しません。")

        # メンバーのインデックスを取得
        member_idx = self.member_index[member_code]

        # 全力量のスコアを予測: W[member_idx] × H
        # W[member_idx]: (n_components,)
        # H: (n_components, n_competences)
        # scores: (n_competences,)
        scores = self.W[member_idx] @ self.H

        # 力量コードでインデックス化
        scores_series = pd.Series(scores, index=self.competence_codes)

        # 特定の力量のみ返す場合
        if competence_codes is not None:
            # 存在する力量のみフィルタ
            valid_codes = [c for c in competence_codes if c in scores_series.index]
            scores_series = scores_series[valid_codes]

        return scores_series

    def predict_top_k(
        self,
        member_code: str,
        k: int = 10,
        exclude_acquired: bool = True,
        acquired_competences: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """
        特定メンバーに対するTop-K推薦を生成

        Args:
            member_code: メンバーコード
            k: 推薦数
            exclude_acquired: 既習得力量を除外するか
            acquired_competences: 既習得力量のリスト（exclude_acquiredがTrueの場合必須）

        Returns:
            (力量コード, スコア)のタプルリスト（スコア降順）
        """
        # 全力量のスコアを予測
        scores = self.predict(member_code)

        # 既習得力量を除外
        if exclude_acquired:
            if acquired_competences is None:
                raise ValueError(
                    "exclude_acquired=Trueの場合、acquired_competencesを指定してください。"
                )
            scores = scores.drop(labels=acquired_competences, errors="ignore")

        # Top-Kを取得
        top_k_scores = scores.nlargest(k)

        return list(zip(top_k_scores.index, top_k_scores.values))

    def get_member_factors(self, member_code: str) -> np.ndarray:
        """
        メンバーの潜在因子ベクトルを取得

        Args:
            member_code: メンバーコード

        Returns:
            潜在因子ベクトル (n_components,)
        """
        if not self.is_fitted:
            raise ValueError("モデルが学習されていません。")

        if member_code not in self.member_index:
            raise ValueError(f"メンバーコード '{member_code}' は学習データに存在しません。")

        member_idx = self.member_index[member_code]
        return self.W[member_idx]

    def get_competence_factors(self, competence_code: str) -> np.ndarray:
        """
        力量の潜在因子ベクトルを取得

        Args:
            competence_code: 力量コード

        Returns:
            潜在因子ベクトル (n_components,)
        """
        if not self.is_fitted:
            raise ValueError("モデルが学習されていません。")

        if competence_code not in self.competence_index:
            raise ValueError(f"力量コード '{competence_code}' は学習データに存在しません。")

        competence_idx = self.competence_index[competence_code]
        return self.H[:, competence_idx]

    def get_reconstruction_error(self) -> float:
        """
        再構成誤差を取得

        Returns:
            再構成誤差（Frobenius norm）
        """
        if not self.is_fitted:
            raise ValueError("モデルが学習されていません。")

        # scikit-learn 1.0以降ではreconstruction_err_属性が削除されたため、手動で計算
        if hasattr(self.model, "reconstruction_err_"):
            return self.model.reconstruction_err_
        else:
            # 手動で再構成誤差を計算: ||X - WH||_F
            X_reconstructed = self.W @ self.H
            reconstruction_error = np.linalg.norm(self.X - X_reconstructed, "fro")
            return reconstruction_error

    def get_normalized_reconstruction_error(self) -> float:
        """
        正規化再構成誤差を取得（相対誤差）

        元のデータのスケールに依存しない相対的な誤差を計算。
        異なるスケールのデータセット間での比較が可能。

        計算式: error / ||X||_F

        Returns:
            正規化再構成誤差 [0, 1]に正規化されることが多い
            値が小さいほどモデルの品質が高い
        """
        if not self.is_fitted:
            raise ValueError("モデルが学習されていません。")

        reconstruction_error = self.get_reconstruction_error()
        X_frobenius = np.linalg.norm(self.X, "fro")

        if X_frobenius == 0:
            return 0.0  # 元のデータがすべて0の場合

        return reconstruction_error / X_frobenius

    def get_model_sparsity(self) -> dict:
        """
        W, H行列のスパース性（疎行列性）を計算

        スパース性が高い（値が大きい）ほど、モデルが解釈しやすく効率的。
        ただし、スパース性が高すぎる場合は潜在因子数（n_components）が
        多すぎる可能性を示唆する。

        Returns:
            dict: スパース性と診断結果
                - W_sparsity: メンバー因子行列のスパース性（%）
                - H_sparsity: 力量因子行列のスパース性（%）
                - unused_factors: 完全に0の潜在因子のインデックスリスト
                - recommendation: 診断結果の文字列
        """
        if not self.is_fitted:
            raise ValueError("モデルが学習されていません。")

        # スパース性を計算（0の要素の割合）
        sparsity_W = np.sum(self.W == 0) / self.W.size * 100
        sparsity_H = np.sum(self.H == 0) / self.H.size * 100

        # 完全に0の潜在因子を検出（すべての力量に対して重みが0）
        unused_factors = []
        for factor_idx in range(self.H.shape[0]):
            if np.allclose(self.H[factor_idx, :], 0, atol=1e-10):
                unused_factors.append(factor_idx)

        # 診断メッセージを生成
        if unused_factors:
            recommendation = (
                f"⚠️ {len(unused_factors)}個の不使用潜在因子が検出されました（インデックス: {unused_factors}）。\n"
                f"n_components を {self.n_components - len(unused_factors)} に削減して再学習することをお勧めします。"
            )
        else:
            recommendation = "✅ すべての潜在因子が使用されています。"

        return {
            "W_sparsity": sparsity_W,
            "H_sparsity": sparsity_H,
            "unused_factors": unused_factors,
            "recommendation": recommendation,
        }

    def save(self, filepath: str):
        """
        モデルを保存

        Args:
            filepath: 保存先ファイルパス
        """
        if not self.is_fitted:
            raise ValueError("モデルが学習されていません。")

        model_data = {
            "n_components": self.n_components,
            "random_state": self.random_state,
            "use_confidence_weighting": self.use_confidence_weighting,
            "confidence_alpha": self.confidence_alpha,
            "nmf_params": self.nmf_params,
            "W": self.W,
            "H": self.H,
            "member_codes": self.member_codes,
            "competence_codes": self.competence_codes,
            "member_index": self.member_index,
            "competence_index": self.competence_index,
            "reconstruction_err": self.model.reconstruction_err_,
            "n_iter": self.model.n_iter_,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

    @classmethod
    def load(cls, filepath: str) -> "MatrixFactorizationModel":
        """
        モデルを読み込み

        Args:
            filepath: モデルファイルパス

        Returns:
            読み込まれたモデル
        """
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        # モデルインスタンスを作成（新しいパラメータに対応、後方互換性も維持）
        model = cls(
            n_components=model_data["n_components"],
            random_state=model_data["random_state"],
            use_confidence_weighting=model_data.get("use_confidence_weighting", False),
            confidence_alpha=model_data.get("confidence_alpha", 1.0),
            **model_data["nmf_params"],
        )

        # 学習済みデータを復元
        model.W = model_data["W"]
        model.H = model_data["H"]
        model.member_codes = model_data["member_codes"]
        model.competence_codes = model_data["competence_codes"]
        model.member_index = model_data["member_index"]
        model.competence_index = model_data["competence_index"]
        model.is_fitted = True

        # NMFモデルの属性を復元（参考情報）
        model.model.reconstruction_err_ = model_data["reconstruction_err"]
        model.model.n_iter_ = model_data["n_iter"]

        return model

    def __repr__(self):
        status = "fitted" if self.is_fitted else "not fitted"
        model_info = f"model_type={self.model_type}" if self.model_type else "model_type=unknown"
        return (
            f"MatrixFactorizationModel(n_components={self.n_components}, "
            f"{model_info}, status={status})"
        )
