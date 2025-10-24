"""
Matrix Factorizationベースの推薦モデル

NMF (Non-negative Matrix Factorization)を使用して会員×力量マトリクスを
潜在因子に分解し、未習得力量のスコアを予測する
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from sklearn.decomposition import NMF
import pickle


class MatrixFactorizationModel:
    """Matrix Factorizationベースの推薦モデル"""

    def __init__(self, n_components: int = 20, random_state: int = 42, **nmf_params):
        """
        初期化

        Args:
            n_components: 潜在因子の数（次元数）
            random_state: 乱数シード
            **nmf_params: NMFへの追加パラメータ
        """
        self.n_components = n_components
        self.random_state = random_state
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
        self.W = None  # 会員因子行列
        self.H = None  # 力量因子行列
        self.member_codes = None  # 会員コードのリスト
        self.competence_codes = None  # 力量コードのリスト
        self.member_index = None  # 会員コード → インデックス
        self.competence_index = None  # 力量コード → インデックス
        self.is_fitted = False

    def fit(self, skill_matrix: pd.DataFrame) -> 'MatrixFactorizationModel':
        """
        モデルを学習

        Args:
            skill_matrix: 会員×力量マトリクス (index=会員コード, columns=力量コード)

        Returns:
            self
        """
        # 会員・力量のコードを保存
        self.member_codes = skill_matrix.index.tolist()
        self.competence_codes = skill_matrix.columns.tolist()

        # インデックスマッピングを作成
        self.member_index = {code: idx for idx, code in enumerate(self.member_codes)}
        self.competence_index = {code: idx for idx, code in enumerate(self.competence_codes)}

        # NMFで分解
        # W: 会員 × 潜在因子 (n_members × n_components)
        # H: 潜在因子 × 力量 (n_components × n_competences)
        self.W = self.model.fit_transform(skill_matrix.values)
        self.H = self.model.components_

        self.is_fitted = True

        return self

    def predict(self, member_code: str, competence_codes: Optional[List[str]] = None) -> pd.Series:
        """
        特定会員に対する力量のスコアを予測

        Args:
            member_code: 会員コード
            competence_codes: 予測対象の力量コードリスト（Noneの場合は全力量）

        Returns:
            力量コードをインデックスとするスコアのSeries

        Raises:
            ValueError: モデルが未学習、または会員コードが不明な場合
        """
        if not self.is_fitted:
            raise ValueError("モデルが学習されていません。先にfit()を呼んでください。")

        if member_code not in self.member_index:
            raise ValueError(f"会員コード '{member_code}' は学習データに存在しません。")

        # 会員のインデックスを取得
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
        特定会員に対するTop-K推薦を生成

        Args:
            member_code: 会員コード
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
        会員の潜在因子ベクトルを取得

        Args:
            member_code: 会員コード

        Returns:
            潜在因子ベクトル (n_components,)
        """
        if not self.is_fitted:
            raise ValueError("モデルが学習されていません。")

        if member_code not in self.member_index:
            raise ValueError(f"会員コード '{member_code}' は学習データに存在しません。")

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

        return self.model.reconstruction_err_

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

        # モデルインスタンスを作成
        model = cls(
            n_components=model_data['n_components'],
            random_state=model_data['random_state'],
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
