"""
ロールモデル検索モジュール

推薦された力量を既に習得している会員（ロールモデル）を検索する機能を提供します。
"""

from typing import List, Dict, Any
import pandas as pd


class RoleModelFinder:
    """ロールモデル検索クラス

    特定の力量を持つ会員を検索し、参考となる会員情報を提供します。
    """

    def __init__(
        self,
        members: pd.DataFrame,
        member_competence: pd.DataFrame,
        competence_master: pd.DataFrame
    ):
        """初期化

        Args:
            members: 会員マスターDataFrame
            member_competence: 会員×力量習得データDataFrame
            competence_master: 力量マスターDataFrame
        """
        self.members = members
        self.member_competence = member_competence
        self.competence_master = competence_master

    def find_role_models(
        self,
        competence_code: str,
        target_member_code: str = None,
        top_n: int = 3,
        min_level: int = 1
    ) -> List[Dict[str, Any]]:
        """指定された力量を持つロールモデルを検索

        Args:
            competence_code: 力量コード
            target_member_code: 対象会員コード（除外する場合）
            top_n: 返す会員数
            min_level: 最小レベル（このレベル以上の会員のみ）

        Returns:
            ロールモデル情報のリスト
            [
                {
                    'member_code': 会員コード,
                    'member_name': 会員名,
                    'competence_level': 力量レベル,
                    'total_competences': 総習得力量数,
                    'skill_count': SKILL数,
                    'education_count': EDUCATION数,
                    'license_count': LICENSE数,
                    'position': 役職,
                    'grade': 職能等級
                },
                ...
            ]
        """
        # 指定された力量を持つ会員を検索
        has_competence = self.member_competence[
            (self.member_competence['力量コード'] == competence_code) &
            (self.member_competence['レベル'] >= min_level)
        ].copy()

        # 対象会員を除外
        if target_member_code:
            has_competence = has_competence[
                has_competence['会員コード'] != target_member_code
            ]

        if has_competence.empty:
            return []

        # 各会員の習得力量統計を計算
        role_models = []

        for _, row in has_competence.iterrows():
            member_code = row['会員コード']
            competence_level = row['レベル']

            # 会員情報を取得
            member_info = self.members[
                self.members['会員コード'] == member_code
            ]

            if member_info.empty:
                continue

            member_info = member_info.iloc[0]

            # 会員の全習得力量を取得
            member_comps = self.member_competence[
                self.member_competence['会員コード'] == member_code
            ]

            # 力量タイプ別にカウント
            comp_with_type = member_comps.merge(
                self.competence_master[['力量コード', '力量種別']],
                on='力量コード',
                how='left'
            )

            skill_count = len(comp_with_type[comp_with_type['力量種別'] == 'SKILL'])
            education_count = len(comp_with_type[comp_with_type['力量種別'] == 'EDUCATION'])
            license_count = len(comp_with_type[comp_with_type['力量種別'] == 'LICENSE'])

            role_models.append({
                'member_code': member_code,
                'member_name': member_info.get('会員名', '不明'),
                'competence_level': int(competence_level),
                'total_competences': len(member_comps),
                'skill_count': skill_count,
                'education_count': education_count,
                'license_count': license_count,
                'position': member_info.get('役職', '未設定'),
                'grade': member_info.get('職能等級', '未設定')
            })

        # 総習得力量数で降順ソート（力量レベルも考慮）
        role_models.sort(
            key=lambda x: (x['competence_level'], x['total_competences']),
            reverse=True
        )

        return role_models[:top_n]

    def find_similar_members(
        self,
        member_code: str,
        top_n: int = 5,
        min_similarity: float = 0.3
    ) -> List[Dict[str, Any]]:
        """類似する会員を検索（習得力量パターンが似ている会員）

        Args:
            member_code: 基準となる会員コード
            top_n: 返す会員数
            min_similarity: 最小類似度（Jaccard係数）

        Returns:
            類似会員情報のリスト
        """
        # 対象会員の習得力量セット
        target_comps = set(
            self.member_competence[
                self.member_competence['会員コード'] == member_code
            ]['力量コード'].values
        )

        if not target_comps:
            return []

        # 全会員の類似度を計算
        similarities = []

        for other_member_code in self.member_competence['会員コード'].unique():
            if other_member_code == member_code:
                continue

            # 他会員の習得力量セット
            other_comps = set(
                self.member_competence[
                    self.member_competence['会員コード'] == other_member_code
                ]['力量コード'].values
            )

            # Jaccard係数を計算
            intersection = len(target_comps & other_comps)
            union = len(target_comps | other_comps)

            if union == 0:
                continue

            similarity = intersection / union

            if similarity < min_similarity:
                continue

            # 会員情報を取得
            member_info = self.members[
                self.members['会員コード'] == other_member_code
            ]

            if member_info.empty:
                continue

            member_info = member_info.iloc[0]

            similarities.append({
                'member_code': other_member_code,
                'member_name': member_info.get('会員名', '不明'),
                'similarity': similarity,
                'common_competences': intersection,
                'total_competences': len(other_comps),
                'position': member_info.get('役職', '未設定'),
                'grade': member_info.get('職能等級', '未設定')
            })

        # 類似度で降順ソート
        similarities.sort(key=lambda x: x['similarity'], reverse=True)

        return similarities[:top_n]
