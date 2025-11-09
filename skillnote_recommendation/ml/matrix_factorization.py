"""
Matrix Factorizationベースの推薦モデル

NMF (Non-negative Matrix Factorization)を使用してメンバー×力量マトリクスを
潜在因子に分解し、未習得力量のスコアを予測する

改善内容:
1. Weighted NMF対応（暗黙的フィードバック考慮）
2. 正規化レベルに基づくconfidence weighting
3. より適切なスコアリング
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from sklearn.decomposition import NMF
import pickle


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
            confidence_alpha: confidence weight計算の係数 (confidence = 1 + alpha * rating)
            early_stopping: Early stoppingを使用するか
            early_stopping_patience: Early stopping用の待機回数（改善が見られないエポック数）
            early_stopping_min_delta: 改善とみなす最小の誤差減少量
            early_stopping_batch_size: Early stopping用のイテレーションバッチサイズ（デフォルト: 50）
                                       大きくすると高速化するが、Early stoppingの精度が下がる
                                       推奨値: 50（標準）、100（高速）、200（最速）
            **nmf_params: NMFへの追加パラメータ
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

        # NMFモデル
        # max_iterがnmf_paramsに含まれていない場合はデフォルト値500を使用
        default_params = {"init": "nndsvda", "max_iter": 500}
        final_params = {**default_params, **nmf_params}

        self.model = NMF(n_components=n_components, random_state=random_state, **final_params)

        # 学習後のデータ
        self.X = None  # 元のデータマトリクス（再構成誤差計算用）
        self.W = None  # メンバー因子行列
        self.H = None  # 力量因子行列
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
        - confidence weightingにより、高レベルのスキルをより重視
        - 暗黙的フィードバック（有無のみ）と明示的フィードバック（レベル）の両方に対応
        - Early stoppingによる効率的な学習
        - 検証セット監視による過学習防止

        Args:
            skill_matrix: メンバー×力量マトリクス (index=メンバーコード, columns=力量コード)
            validation_matrix: 検証用マトリクス（Early stopping時に使用）

        Returns:
            self
        """
        # メンバー・力量のコードを保存
        self.member_codes = skill_matrix.index.tolist()
        self.competence_codes = skill_matrix.columns.tolist()

        # インデックスマッピングを作成
        self.member_index = {code: idx for idx, code in enumerate(self.member_codes)}
        self.competence_index = {code: idx for idx, code in enumerate(self.competence_codes)}

        # 学習用マトリクスを準備
        if self.use_confidence_weighting:
            # 正規化レベルにconfidence weightを適用
            weighted_matrix = skill_matrix.copy()
            non_zero_mask = weighted_matrix > 0
            weighted_matrix[non_zero_mask] = (
                1 + self.confidence_alpha * weighted_matrix[non_zero_mask]
            )
            training_matrix = weighted_matrix.values
        else:
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

    def _fit_normal(self, X: np.ndarray) -> None:
        """通常の学習（Early stoppingなし）"""
        self.X = X  # 元のデータマトリクスを保存（再構成誤差計算用）
        self.W = self.model.fit_transform(X)
        self.H = self.model.components_
        self.actual_n_iter_ = self.model.n_iter_

    def _fit_with_early_stopping(self, X: np.ndarray, X_val: Optional[np.ndarray] = None) -> None:
        """
        Early stoppingを使用した学習

        段階的にmax_iterを増やしながら学習し、
        検証セットの誤差が増加し始めたら早期終了する

        Args:
            X: 訓練用マトリクス
            X_val: 検証用マトリクス（Noneの場合は訓練誤差で判定）
        """
        max_iter_total = self.model.max_iter
        batch_size = self.early_stopping_batch_size

        best_error = float("inf")
        best_val_error = float("inf")
        patience_counter = 0
        best_W = None
        best_H = None
        best_iter = 0
        generalization_gap_history = []  # 汎化ギャップの履歴

        for current_max_iter in range(batch_size, max_iter_total + 1, batch_size):
            # NMFモデルを再作成（max_iterを更新）
            nmf_params = self.nmf_params.copy()
            nmf_params["max_iter"] = current_max_iter
            # 常にnndsvdaを使用（customは初期値W,Hが必要で複雑になるため）
            nmf_params["init"] = "nndsvda"

            temp_model = NMF(
                n_components=self.n_components, random_state=self.random_state, **nmf_params
            )

            # 学習
            W = temp_model.fit_transform(X)
            H = temp_model.components_

            # 訓練誤差を計算
            reconstructed = W @ H
            train_error = np.linalg.norm(X - reconstructed, "fro")

            # 検証誤差を計算（検証セットが提供されている場合）
            if X_val is not None:
                reconstructed_val = (W @ H)[:X_val.shape[0], :] if W.shape[0] >= X_val.shape[0] else (W @ H)
                # マトリクスサイズが異なる場合は、共通部分のみで計算
                min_rows = min(X_val.shape[0], reconstructed_val.shape[0])
                val_error = np.linalg.norm(X_val[:min_rows] - reconstructed_val[:min_rows], "fro")
                generalization_gap = val_error - train_error
            else:
                val_error = train_error
                generalization_gap = 0.0

            # Early stopping判定
            # 検証セットがある場合は検証誤差で判定、ない場合は訓練誤差で判定
            current_error = val_error if X_val is not None else train_error
            improvement = best_error - current_error

            if improvement > self.early_stopping_min_delta:
                best_error = current_error
                best_val_error = val_error if X_val is not None else train_error
                best_W = W.copy()
                best_H = H.copy()
                best_iter = current_max_iter
                patience_counter = 0
                generalization_gap_history.append(generalization_gap)

                if X_val is not None:
                    print(
                        f"[Early Stopping] Iter {current_max_iter}: "
                        f"train_error={train_error:.6f}, val_error={val_error:.6f}, "
                        f"gap={generalization_gap:.6f} (improved by {improvement:.6f})"
                    )
                else:
                    print(
                        f"[Early Stopping] Iter {current_max_iter}: error={train_error:.6f} (improved by {improvement:.6f})"
                    )
            else:
                patience_counter += 1
                generalization_gap_history.append(generalization_gap)

                if X_val is not None:
                    print(
                        f"[Early Stopping] Iter {current_max_iter}: "
                        f"train_error={train_error:.6f}, val_error={val_error:.6f}, "
                        f"gap={generalization_gap:.6f} (no improvement, patience={patience_counter}/{self.early_stopping_patience})"
                    )
                else:
                    print(
                        f"[Early Stopping] Iter {current_max_iter}: error={train_error:.6f} (no improvement, patience={patience_counter}/{self.early_stopping_patience})"
                    )

                if patience_counter >= self.early_stopping_patience:
                    print(
                        f"[Early Stopping] Stopped at iteration {best_iter} (best error: {best_error:.6f})"
                    )
                    if X_val is not None:
                        avg_gap = np.mean(generalization_gap_history[-self.early_stopping_patience:])
                        print(f"[Early Stopping] Average generalization gap: {avg_gap:.6f}")
                    break

        # ベストモデルを設定
        self.X = X  # 元のデータマトリクスを保存（再構成誤差計算用）
        self.W = best_W if best_W is not None else W
        self.H = best_H if best_H is not None else H
        self.actual_n_iter_ = best_iter if best_W is not None else current_max_iter

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
        return f"MatrixFactorizationModel(n_components={self.n_components}, " f"status={status})"
