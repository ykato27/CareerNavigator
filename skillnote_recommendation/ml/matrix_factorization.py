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
        **nmf_params
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
        default_params = {'init': 'nndsvda', 'max_iter': 500}
        final_params = {**default_params, **nmf_params}

        self.model = NMF(
            n_components=n_components,
            random_state=random_state,
            **final_params
        )

        # 学習後のデータ
        self.W = None  # メンバー因子行列
        self.H = None  # 力量因子行列
        self.member_codes = None  # メンバーコードのリスト
        self.competence_codes = None  # 力量コードのリスト
        self.member_index = None  # メンバーコード → インデックス
        self.competence_index = None  # 力量コード → インデックス
        self.is_fitted = False
        self.actual_n_iter_ = None  # 実際のイテレーション数（Early stopping時）

    def fit(self, skill_matrix: pd.DataFrame) -> 'MatrixFactorizationModel':
        """
        モデルを学習

        改善点:
        - confidence weightingにより、高レベルのスキルをより重視
        - 暗黙的フィードバック（有無のみ）と明示的フィードバック（レベル）の両方に対応
        - Early stoppingによる効率的な学習

        Args:
            skill_matrix: メンバー×力量マトリクス (index=メンバーコード, columns=力量コード)

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
            weighted_matrix[non_zero_mask] = 1 + self.confidence_alpha * weighted_matrix[non_zero_mask]
            training_matrix = weighted_matrix.values
        else:
            training_matrix = skill_matrix.values

        # Early stoppingの有無で分岐
        if self.early_stopping:
            self._fit_with_early_stopping(training_matrix)
        else:
            self._fit_normal(training_matrix)

        self.is_fitted = True
        return self

    def _fit_normal(self, X: np.ndarray) -> None:
        """通常の学習（Early stoppingなし）"""
        self.W = self.model.fit_transform(X)
        self.H = self.model.components_
        self.actual_n_iter_ = self.model.n_iter_

    def _fit_with_early_stopping(self, X: np.ndarray) -> None:
        """
        Early stoppingを使用した学習

        段階的にmax_iterを増やしながら学習し、
        改善が止まったら早期終了する
        """
        max_iter_total = self.model.max_iter
        batch_size = self.early_stopping_batch_size

        best_error = float('inf')
        patience_counter = 0
        best_W = None
        best_H = None
        best_iter = 0

        for current_max_iter in range(batch_size, max_iter_total + 1, batch_size):
            # NMFモデルを再作成（max_iterを更新）
            nmf_params = self.nmf_params.copy()
            nmf_params['max_iter'] = current_max_iter
            # 常にnndsvdaを使用（customは初期値W,Hが必要で複雑になるため）
            nmf_params['init'] = 'nndsvda'

            temp_model = NMF(
                n_components=self.n_components,
                random_state=self.random_state,
                **nmf_params
            )

            # 学習
            W = temp_model.fit_transform(X)
            H = temp_model.components_

            # 再構成誤差を計算
            reconstructed = W @ H
            error = np.linalg.norm(X - reconstructed, 'fro')

            # Early stopping判定
            improvement = best_error - error
            if improvement > self.early_stopping_min_delta:
                best_error = error
                best_W = W.copy()
                best_H = H.copy()
                best_iter = current_max_iter
                patience_counter = 0
                print(f"[Early Stopping] Iter {current_max_iter}: error={error:.6f} (improved by {improvement:.6f})")
            else:
                patience_counter += 1
                print(f"[Early Stopping] Iter {current_max_iter}: error={error:.6f} (no improvement, patience={patience_counter}/{self.early_stopping_patience})")

                if patience_counter >= self.early_stopping_patience:
                    print(f"[Early Stopping] Stopped at iteration {best_iter} (best error: {best_error:.6f})")
                    break

        # ベストモデルを設定
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

    def predict_top_k(self, member_code: str, k: int = 10,
                      exclude_acquired: bool = True,
                      acquired_competences: Optional[List[str]] = None) -> List[Tuple[str, float]]:
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
                raise ValueError("exclude_acquired=Trueの場合、acquired_competencesを指定してください。")
            scores = scores.drop(labels=acquired_competences, errors='ignore')

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
        if hasattr(self.model, 'reconstruction_err_'):
            return self.model.reconstruction_err_
        else:
            # 手動で再構成誤差を計算: ||X - WH||_F
            X_reconstructed = self.W @ self.H
            reconstruction_error = np.linalg.norm(self.X - X_reconstructed, 'fro')
            return reconstruction_error

    def save(self, filepath: str):
        """
        モデルを保存

        Args:
            filepath: 保存先ファイルパス
        """
        if not self.is_fitted:
            raise ValueError("モデルが学習されていません。")

        model_data = {
            'n_components': self.n_components,
            'random_state': self.random_state,
            'use_confidence_weighting': self.use_confidence_weighting,
            'confidence_alpha': self.confidence_alpha,
            'nmf_params': self.nmf_params,
            'W': self.W,
            'H': self.H,
            'member_codes': self.member_codes,
            'competence_codes': self.competence_codes,
            'member_index': self.member_index,
            'competence_index': self.competence_index,
            'reconstruction_err': self.model.reconstruction_err_,
            'n_iter': self.model.n_iter_
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    @classmethod
    def load(cls, filepath: str) -> 'MatrixFactorizationModel':
        """
        モデルを読み込み

        Args:
            filepath: モデルファイルパス

        Returns:
            読み込まれたモデル
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        # モデルインスタンスを作成（新しいパラメータに対応、後方互換性も維持）
        model = cls(
            n_components=model_data['n_components'],
            random_state=model_data['random_state'],
            use_confidence_weighting=model_data.get('use_confidence_weighting', False),
            confidence_alpha=model_data.get('confidence_alpha', 1.0),
            **model_data['nmf_params']
        )

        # 学習済みデータを復元
        model.W = model_data['W']
        model.H = model_data['H']
        model.member_codes = model_data['member_codes']
        model.competence_codes = model_data['competence_codes']
        model.member_index = model_data['member_index']
        model.competence_index = model_data['competence_index']
        model.is_fitted = True

        # NMFモデルの属性を復元（参考情報）
        model.model.reconstruction_err_ = model_data['reconstruction_err']
        model.model.n_iter_ = model_data['n_iter']

        return model

    def __repr__(self):
        status = "fitted" if self.is_fitted else "not fitted"
        return (f"MatrixFactorizationModel(n_components={self.n_components}, "
                f"status={status})")
