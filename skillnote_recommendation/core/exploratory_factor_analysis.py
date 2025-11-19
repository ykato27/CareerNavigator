"""
探索的因子分析（Exploratory Factor Analysis: EFA）

データ駆動で潜在因子を発見し、カテゴリー定義に依存しない分析を実現。
大規模データセット（200+スキル）での高速化にも貢献。
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore

logger = logging.getLogger(__name__)


class ExploratoryFactorAnalyzer:
    """
    探索的因子分析を実行するクラス

    特徴:
    - 因子数の自動決定（Kaiser基準、累積寄与率）
    - 大規模データセット対応（200+スキル）
    - 因子ローディングの抽出と解釈
    """

    def __init__(
        self,
        member_competence: pd.DataFrame,
        competence_master: pd.DataFrame,
        n_factors: Optional[int] = None,
        variance_threshold: float = 0.80,
        min_factors: int = 3,
        max_factors: int = 15,
    ):
        """
        初期化

        Args:
            member_competence: メンバー力量データ
            competence_master: 力量マスター
            n_factors: 因子数（Noneの場合は自動決定）
            variance_threshold: 累積寄与率の閾値（デフォルト: 80%）
            min_factors: 最小因子数
            max_factors: 最大因子数
        """
        self.member_competence = member_competence
        self.competence_master = competence_master
        self.n_factors = n_factors
        self.variance_threshold = variance_threshold
        self.min_factors = min_factors
        self.max_factors = max_factors

        # 分析結果の保存
        self.fa_model = None
        self.factor_loadings = None
        self.skill_codes = None
        self.skill_names = None
        self.explained_variance = None

    def prepare_data(self) -> Tuple[np.ndarray, List[str], Dict[str, str]]:
        """
        データ準備：スキルスコア行列の構築

        Returns:
            (スキルスコア行列, スキルコードリスト, スキル名マッピング)
        """
        # メンバーコード取得
        member_codes = self.member_competence['メンバーコード'].unique()

        # スキルコード取得（力量タイプ='スキル'のみ）
        skill_master = self.competence_master[
            self.competence_master['力量タイプ'] == 'スキル'
        ]
        skill_codes = skill_master['力量コード'].tolist()

        # スキル名マッピング
        skill_name_mapping = dict(
            zip(skill_master['力量コード'], skill_master['力量名'])
        )

        # スキルスコア行列を構築（メンバー × スキル）
        skill_scores = []
        valid_members = []

        for member_code in member_codes:
            member_data = self.member_competence[
                self.member_competence['メンバーコード'] == member_code
            ]

            # スキルスコアを取得
            scores = []
            for skill_code in skill_codes:
                skill_row = member_data[member_data['力量コード'] == skill_code]
                if not skill_row.empty:
                    score = skill_row.iloc[0]['スコア']
                    scores.append(score if pd.notna(score) else 0)
                else:
                    scores.append(0)

            # 全てゼロのメンバーは除外
            if sum(scores) > 0:
                skill_scores.append(scores)
                valid_members.append(member_code)

        skill_score_matrix = np.array(skill_scores)

        # 標準化（各スキルを平均0、分散1に）
        scaler = StandardScaler()
        skill_score_matrix_scaled = scaler.fit_transform(skill_score_matrix)

        logger.info(f"Data prepared: {len(valid_members)} members × {len(skill_codes)} skills")

        return skill_score_matrix_scaled, skill_codes, skill_name_mapping

    def determine_n_factors(self, skill_score_matrix: np.ndarray) -> int:
        """
        因子数を自動決定

        Kaiser基準（固有値>1）と累積寄与率を組み合わせて決定

        Args:
            skill_score_matrix: スキルスコア行列（標準化済み）

        Returns:
            最適な因子数
        """
        # 相関行列を計算
        corr_matrix = np.corrcoef(skill_score_matrix.T)

        # 固有値分解
        eigenvalues, _ = np.linalg.eig(corr_matrix)
        eigenvalues = np.real(eigenvalues)
        eigenvalues = np.sort(eigenvalues)[::-1]  # 降順ソート

        # Kaiser基準: 固有値 > 1
        kaiser_n = np.sum(eigenvalues > 1)

        # 累積寄与率
        total_variance = np.sum(eigenvalues)
        cumulative_variance = np.cumsum(eigenvalues) / total_variance

        # 累積寄与率が閾値を超える因子数
        variance_n = np.argmax(cumulative_variance >= self.variance_threshold) + 1

        # Kaiser基準と累積寄与率の平均を取る
        auto_n = int(np.mean([kaiser_n, variance_n]))

        # 範囲制約を適用
        auto_n = max(self.min_factors, min(auto_n, self.max_factors))

        logger.info(f"Auto-determined factors: Kaiser={kaiser_n}, Variance={variance_n}, Final={auto_n}")
        logger.info(f"Explained variance at {auto_n} factors: {cumulative_variance[auto_n-1]:.2%}")

        return auto_n

    def fit(self) -> Dict:
        """
        因子分析を実行

        Returns:
            分析結果の辞書
        """
        # データ準備
        skill_score_matrix, skill_codes, skill_name_mapping = self.prepare_data()

        # 因子数決定
        if self.n_factors is None:
            n_factors = self.determine_n_factors(skill_score_matrix)
        else:
            n_factors = self.n_factors

        logger.info(f"Running EFA with {n_factors} factors on {skill_score_matrix.shape}")

        # 因子分析実行
        fa = FactorAnalysis(n_components=n_factors, random_state=42)
        fa.fit(skill_score_matrix)

        # 因子ローディング取得（スキル × 因子）
        factor_loadings = fa.components_.T

        # 結果を保存
        self.fa_model = fa
        self.factor_loadings = factor_loadings
        self.skill_codes = skill_codes
        self.skill_names = skill_name_mapping

        # 因子の寄与率計算
        eigenvalues = np.var(fa.transform(skill_score_matrix), axis=0)
        total_variance = np.sum(eigenvalues)
        explained_variance = eigenvalues / total_variance
        self.explained_variance = explained_variance

        # 因子名を自動生成
        factor_names = [f"因子{i+1}" for i in range(n_factors)]

        logger.info(f"EFA completed: {len(skill_codes)} skills → {n_factors} factors")
        logger.info(f"Total explained variance: {np.sum(explained_variance):.2%}")

        return {
            'n_factors': n_factors,
            'factor_names': factor_names,
            'factor_loadings': factor_loadings,
            'explained_variance': explained_variance,
            'skill_codes': skill_codes,
            'skill_name_mapping': skill_name_mapping,
        }

    def get_factor_interpretation(self, top_n: int = 5) -> Dict[str, List[Tuple[str, float]]]:
        """
        各因子の解釈を支援（ローディング上位スキルを表示）

        Args:
            top_n: 各因子につき表示する上位スキル数

        Returns:
            因子名 → [(スキル名, ローディング), ...] のマッピング
        """
        if self.factor_loadings is None:
            raise ValueError("fit() を先に実行してください")

        interpretation = {}

        for factor_idx in range(self.factor_loadings.shape[1]):
            loadings = self.factor_loadings[:, factor_idx]

            # 絶対値でソート
            top_indices = np.argsort(np.abs(loadings))[::-1][:top_n]

            top_skills = []
            for idx in top_indices:
                skill_code = self.skill_codes[idx]
                skill_name = self.skill_names.get(skill_code, skill_code)
                loading = loadings[idx]
                top_skills.append((skill_name, loading))

            interpretation[f"因子{factor_idx + 1}"] = top_skills

        return interpretation

    def get_lambda_matrix_compatible(self) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        SEM互換のLambda行列を取得

        Returns:
            (Lambda行列, 潜在変数名リスト, 観測変数名リスト)
        """
        if self.factor_loadings is None:
            raise ValueError("fit() を先に実行してください")

        # Lambda行列（スキル × 因子）
        lambda_matrix = self.factor_loadings

        # 潜在変数名（因子名）
        latent_vars = [f"因子{i+1}" for i in range(lambda_matrix.shape[1])]

        # 観測変数名（スキルコード）
        observed_vars = self.skill_codes

        return lambda_matrix, latent_vars, observed_vars
