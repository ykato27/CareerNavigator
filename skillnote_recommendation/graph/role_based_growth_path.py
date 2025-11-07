"""
役職ベースの成長パス分析

同じ役職のメンバーがどのような順序でスキルを習得してきたかを分析し、
実データに基づいた成長ルートを抽出する。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class SkillAcquisitionPattern:
    """スキル取得パターン"""
    competence_code: str
    competence_name: str
    average_order: float  # 平均取得順序（小さいほど早期に取得）
    std_order: float  # 取得順序の標準偏差
    acquisition_count: int  # このスキルを取得した人数
    total_members: int  # この役職の総人数
    acquisition_rate: float  # 取得率（acquisition_count / total_members）
    competence_type: str
    category: str


@dataclass
class RoleGrowthPath:
    """役職ごとの成長パス"""
    role_name: str
    total_members: int
    skills_in_order: List[SkillAcquisitionPattern]  # 取得順序でソートされたスキルリスト

    def get_early_stage_skills(self, threshold: float = 0.3) -> List[SkillAcquisitionPattern]:
        """
        初期段階のスキルを取得

        Args:
            threshold: 早期と判定する閾値（0.0-1.0）

        Returns:
            初期段階のスキルリスト
        """
        if not self.skills_in_order:
            return []

        total_skills = len(self.skills_in_order)
        cutoff_index = int(total_skills * threshold)
        return self.skills_in_order[:cutoff_index]

    def get_mid_stage_skills(self, early_threshold: float = 0.3,
                            late_threshold: float = 0.7) -> List[SkillAcquisitionPattern]:
        """
        中期段階のスキルを取得

        Args:
            early_threshold: 初期の閾値
            late_threshold: 後期の閾値

        Returns:
            中期段階のスキルリスト
        """
        if not self.skills_in_order:
            return []

        total_skills = len(self.skills_in_order)
        early_cutoff = int(total_skills * early_threshold)
        late_cutoff = int(total_skills * late_threshold)
        return self.skills_in_order[early_cutoff:late_cutoff]

    def get_late_stage_skills(self, threshold: float = 0.7) -> List[SkillAcquisitionPattern]:
        """
        後期段階のスキルを取得

        Args:
            threshold: 後期と判定する閾値（0.0-1.0）

        Returns:
            後期段階のスキルリスト
        """
        if not self.skills_in_order:
            return []

        total_skills = len(self.skills_in_order)
        cutoff_index = int(total_skills * threshold)
        return self.skills_in_order[cutoff_index:]


class RoleBasedGrowthPathAnalyzer:
    """
    役職ベースの成長パス分析クラス

    同じ役職のメンバーのスキル取得履歴を分析し、
    典型的な成長ルートを抽出する。
    """

    def __init__(self,
                 members_df: pd.DataFrame,
                 member_competence_df: pd.DataFrame,
                 competence_master_df: pd.DataFrame):
        """
        初期化

        Args:
            members_df: メンバーマスタ（役職情報を含む）
            member_competence_df: メンバー保有力量データ（取得日を含む）
            competence_master_df: 力量マスタ
        """
        self.members_df = members_df
        self.member_competence_df = member_competence_df
        self.competence_master_df = competence_master_df

        # カラム名のマッピング（###[...]### を除去した形式）
        self.role_column = '役職'
        self.member_code_column = 'メンバーコード'
        self.competence_code_column = '力量コード'
        self.acquired_date_column = '取得日'

        # 役職ごとの成長パスキャッシュ
        self._growth_paths_cache: Dict[str, RoleGrowthPath] = {}

    def analyze_all_roles(self, min_members: int = 3) -> Dict[str, RoleGrowthPath]:
        """
        全ての役職について成長パスを分析

        Args:
            min_members: 分析対象とする最小メンバー数

        Returns:
            役職名をキーとした成長パス辞書
        """
        logger.info("=" * 80)
        logger.info("役職ベースの成長パス分析開始")
        logger.info("=" * 80)

        growth_paths = {}

        # 役職ごとにグループ化
        roles = self.members_df[self.role_column].unique()
        logger.info(f"\n分析対象の役職数: {len(roles)}")

        for role in roles:
            if pd.isna(role) or str(role).strip() == '':
                continue

            # この役職のメンバーを取得
            role_members = self.members_df[
                self.members_df[self.role_column] == role
            ][self.member_code_column].unique()

            if len(role_members) < min_members:
                logger.debug(f"役職 '{role}': メンバー数が少ない（{len(role_members)}名）ためスキップ")
                continue

            # 成長パスを分析
            growth_path = self._analyze_role_growth_path(role, role_members)

            if growth_path and len(growth_path.skills_in_order) > 0:
                growth_paths[role] = growth_path
                logger.info(f"\n役職 '{role}':")
                logger.info(f"  メンバー数: {growth_path.total_members}名")
                logger.info(f"  分析されたスキル数: {len(growth_path.skills_in_order)}個")

        self._growth_paths_cache = growth_paths
        logger.info(f"\n分析完了: {len(growth_paths)}個の役職で成長パスを生成")

        return growth_paths

    def _analyze_role_growth_path(self,
                                   role: str,
                                   role_members: np.ndarray) -> Optional[RoleGrowthPath]:
        """
        特定の役職について成長パスを分析

        Args:
            role: 役職名
            role_members: この役職のメンバーコード一覧

        Returns:
            成長パスオブジェクト
        """
        # この役職のメンバーの保有力量データを取得
        role_competence_df = self.member_competence_df[
            self.member_competence_df[self.member_code_column].isin(role_members)
        ].copy()

        if role_competence_df.empty:
            return None

        # 各メンバーのスキル取得順序を分析
        skill_orders = self._calculate_skill_acquisition_orders(role_competence_df, role_members)

        if not skill_orders:
            return None

        # スキルパターンを生成
        skill_patterns = self._generate_skill_patterns(skill_orders, len(role_members))

        # 平均取得順序でソート
        skill_patterns_sorted = sorted(skill_patterns, key=lambda x: x.average_order)

        return RoleGrowthPath(
            role_name=role,
            total_members=len(role_members),
            skills_in_order=skill_patterns_sorted
        )

    def _calculate_skill_acquisition_orders(self,
                                             role_competence_df: pd.DataFrame,
                                             role_members: np.ndarray) -> Dict[str, List[int]]:
        """
        各スキルの取得順序を計算

        Args:
            role_competence_df: この役職のメンバーの保有力量データ
            role_members: メンバーコード一覧

        Returns:
            {スキルコード: [取得順序のリスト]} の辞書
        """
        skill_orders: Dict[str, List[int]] = {}

        for member_code in role_members:
            # このメンバーの保有力量を取得
            member_skills = role_competence_df[
                role_competence_df[self.member_code_column] == member_code
            ].copy()

            if member_skills.empty:
                continue

            # 取得日が存在するデータのみを使用
            member_skills = member_skills[
                member_skills[self.acquired_date_column].notna()
            ].copy()

            if member_skills.empty:
                continue

            # 取得日を日付型に変換
            member_skills[self.acquired_date_column] = pd.to_datetime(
                member_skills[self.acquired_date_column],
                errors='coerce'
            )

            # 取得日でソート
            member_skills = member_skills.sort_values(self.acquired_date_column)

            # 各スキルに順序番号を付与（0始まり）
            for order, (_, row) in enumerate(member_skills.iterrows()):
                competence_code = row[self.competence_code_column]

                if competence_code not in skill_orders:
                    skill_orders[competence_code] = []

                skill_orders[competence_code].append(order)

        return skill_orders

    def _generate_skill_patterns(self,
                                  skill_orders: Dict[str, List[int]],
                                  total_members: int) -> List[SkillAcquisitionPattern]:
        """
        スキル取得パターンを生成

        Args:
            skill_orders: {スキルコード: [取得順序のリスト]}
            total_members: 役職の総メンバー数

        Returns:
            スキル取得パターンのリスト
        """
        patterns = []

        for competence_code, orders in skill_orders.items():
            # スキル情報を取得
            skill_info = self.competence_master_df[
                self.competence_master_df['力量コード'] == competence_code
            ]

            if skill_info.empty:
                continue

            skill_info = skill_info.iloc[0]

            # 統計値を計算
            average_order = np.mean(orders)
            std_order = np.std(orders) if len(orders) > 1 else 0.0
            acquisition_count = len(orders)
            acquisition_rate = acquisition_count / total_members

            pattern = SkillAcquisitionPattern(
                competence_code=competence_code,
                competence_name=skill_info.get('力量名', competence_code),
                average_order=average_order,
                std_order=std_order,
                acquisition_count=acquisition_count,
                total_members=total_members,
                acquisition_rate=acquisition_rate,
                competence_type=skill_info.get('力量タイプ', 'UNKNOWN'),
                category=skill_info.get('力量カテゴリー名', '')
            )

            patterns.append(pattern)

        return patterns

    def get_growth_path_for_member(self, member_code: str) -> Optional[RoleGrowthPath]:
        """
        メンバーの役職に基づいて成長パスを取得

        Args:
            member_code: メンバーコード

        Returns:
            成長パスオブジェクト（該当なしの場合はNone）
        """
        # メンバーの役職を取得
        member_info = self.members_df[
            self.members_df[self.member_code_column] == member_code
        ]

        if member_info.empty:
            logger.warning(f"メンバーコード '{member_code}' が見つかりません")
            return None

        role = member_info.iloc[0][self.role_column]

        if pd.isna(role) or str(role).strip() == '':
            logger.warning(f"メンバーコード '{member_code}' の役職情報がありません")
            return None

        # キャッシュから取得
        if not self._growth_paths_cache:
            self.analyze_all_roles()

        return self._growth_paths_cache.get(role)

    def recommend_next_skills(self,
                              member_code: str,
                              top_n: int = 10,
                              min_acquisition_rate: float = 0.3) -> List[Dict]:
        """
        メンバーに対して次に習得すべきスキルを推薦

        Args:
            member_code: メンバーコード
            top_n: 推薦するスキル数
            min_acquisition_rate: 推薦対象とする最小取得率

        Returns:
            推薦スキルのリスト（優先度順）
        """
        # 成長パスを取得
        growth_path = self.get_growth_path_for_member(member_code)

        if not growth_path:
            logger.warning(f"メンバー {member_code} の成長パスが見つかりません。全スキルから推薦します。")
            # フォールバック：全スキルから推薦
            return self._fallback_recommend_from_all_skills(
                member_code, top_n, min_acquisition_rate
            )

        # メンバーの現在の保有スキルを取得
        member_skills = self.member_competence_df[
            self.member_competence_df[self.member_code_column] == member_code
        ][self.competence_code_column].unique()

        member_skills_set = set(member_skills)

        # 未習得のスキルを抽出
        recommendations = []

        for skill_pattern in growth_path.skills_in_order:
            # 既に習得済みのスキルはスキップ
            if skill_pattern.competence_code in member_skills_set:
                continue

            # 取得率が低すぎるスキルはスキップ
            if skill_pattern.acquisition_rate < min_acquisition_rate:
                continue

            # 推薦理由を生成
            reason = self._generate_recommendation_reason(skill_pattern, growth_path)

            recommendations.append({
                'competence_code': skill_pattern.competence_code,
                'competence_name': skill_pattern.competence_name,
                'competence_type': skill_pattern.competence_type,
                'category': skill_pattern.category,
                'priority_score': 1.0 / (skill_pattern.average_order + 1),  # 早いほど高スコア
                'average_order': skill_pattern.average_order,
                'acquisition_rate': skill_pattern.acquisition_rate,
                'reason': reason
            })

        # 優先度スコアでソート
        recommendations.sort(key=lambda x: x['priority_score'], reverse=True)

        # フォールバック：推薦が0件の場合、フィルタを緩める
        if len(recommendations) == 0 and min_acquisition_rate > 0:
            logger.info(f"推薦が0件のため、取得率フィルタを緩和: {min_acquisition_rate} → 0.0")
            # 取得率フィルタなしで再試行
            for skill_pattern in growth_path.skills_in_order:
                # 既に習得済みのスキルはスキップ
                if skill_pattern.competence_code in member_skills_set:
                    continue

                # 推薦理由を生成
                reason = self._generate_recommendation_reason(skill_pattern, growth_path)
                reason += "\n\n※ 取得率は低いですが、この役職での成長パス上にあるスキルです。"

                recommendations.append({
                    'competence_code': skill_pattern.competence_code,
                    'competence_name': skill_pattern.competence_name,
                    'competence_type': skill_pattern.competence_type,
                    'category': skill_pattern.category,
                    'priority_score': 1.0 / (skill_pattern.average_order + 1),
                    'average_order': skill_pattern.average_order,
                    'acquisition_rate': skill_pattern.acquisition_rate,
                    'reason': reason
                })

            recommendations.sort(key=lambda x: x['priority_score'], reverse=True)

        return recommendations[:top_n]

    def _generate_recommendation_reason(self,
                                        skill_pattern: SkillAcquisitionPattern,
                                        growth_path: RoleGrowthPath) -> str:
        """
        推薦理由を生成

        Args:
            skill_pattern: スキル取得パターン
            growth_path: 成長パス

        Returns:
            推薦理由の文字列
        """
        # 取得率をパーセント表示
        acquisition_pct = skill_pattern.acquisition_rate * 100

        # 取得順序から段階を判定
        total_skills = len(growth_path.skills_in_order)
        skill_index = growth_path.skills_in_order.index(skill_pattern)
        progress_pct = (skill_index / total_skills) * 100

        if progress_pct < 30:
            stage = "基礎段階"
        elif progress_pct < 70:
            stage = "中級段階"
        else:
            stage = "上級段階"

        reason = (
            f"【役職ベースの成長パス推薦】\n"
            f"役職「{growth_path.role_name}」の{stage}で習得されるスキルです。\n"
            f"{growth_path.total_members}名中{skill_pattern.acquisition_count}名（{acquisition_pct:.1f}%）が習得しており、"
            f"平均して{skill_pattern.average_order:.1f}番目に取得されています。"
        )

        return reason

    def get_member_progress(self, member_code: str) -> Optional[Dict]:
        """
        メンバーの成長パス上での進捗状況を取得

        Args:
            member_code: メンバーコード

        Returns:
            進捗情報の辞書
        """
        growth_path = self.get_growth_path_for_member(member_code)

        if not growth_path:
            return None

        # メンバーの保有スキル
        member_skills = self.member_competence_df[
            self.member_competence_df[self.member_code_column] == member_code
        ][self.competence_code_column].unique()

        member_skills_set = set(member_skills)

        # 成長パス上のスキルと照合
        acquired_skills = []
        not_acquired_skills = []

        for skill_pattern in growth_path.skills_in_order:
            if skill_pattern.competence_code in member_skills_set:
                acquired_skills.append(skill_pattern)
            else:
                not_acquired_skills.append(skill_pattern)

        # 進捗率を計算
        total_path_skills = len(growth_path.skills_in_order)
        acquired_count = len(acquired_skills)
        progress_rate = acquired_count / total_path_skills if total_path_skills > 0 else 0.0

        return {
            'role_name': growth_path.role_name,
            'total_path_skills': total_path_skills,
            'acquired_count': acquired_count,
            'not_acquired_count': len(not_acquired_skills),
            'progress_rate': progress_rate,
            'acquired_skills': acquired_skills,
            'not_acquired_skills': not_acquired_skills
        }

    def recommend_next_skills_with_paths(self,
                                          member_code: str,
                                          top_n: int = 10,
                                          min_acquisition_rate: float = 0.3,
                                          max_paths: int = 5) -> List[Dict]:
        """
        メンバーに対して次に習得すべきスキルをパス情報付きで推薦

        Args:
            member_code: メンバーコード
            top_n: 推薦するスキル数
            min_acquisition_rate: 推薦対象とする最小取得率
            max_paths: 各スキルに対する最大パス数

        Returns:
            パス情報を含む推薦スキルのリスト
        """
        # 基本の推薦を取得
        recommendations = self.recommend_next_skills(member_code, top_n, min_acquisition_rate)

        # 各推薦にパス情報を追加
        recommendations_with_paths = []
        for rec in recommendations:
            paths = self._generate_paths_for_skill(
                member_code=member_code,
                competence_code=rec['competence_code'],
                max_paths=max_paths
            )

            rec_with_paths = rec.copy()
            rec_with_paths['paths'] = paths
            recommendations_with_paths.append(rec_with_paths)

        return recommendations_with_paths

    def _generate_paths_for_skill(self,
                                   member_code: str,
                                   competence_code: str,
                                   max_paths: int = 5) -> List[List[Dict]]:
        """
        特定のスキルに対するパス（メンバー → 類似メンバー → スキル）を生成

        Args:
            member_code: メンバーコード
            competence_code: スキルコード
            max_paths: 最大パス数

        Returns:
            パスのリスト
        """
        # メンバー情報を取得
        member_info = self.members_df[
            self.members_df[self.member_code_column] == member_code
        ]

        if member_info.empty:
            return []

        member_name = member_info.iloc[0].get('メンバー名', member_code)
        role = member_info.iloc[0][self.role_column]

        # 同じ役職のメンバーでこのスキルを習得している人を取得
        role_members = self.members_df[
            self.members_df[self.role_column] == role
        ][self.member_code_column].unique()

        # このスキルを習得しているメンバーを取得
        skill_holders = self.member_competence_df[
            (self.member_competence_df[self.competence_code_column] == competence_code) &
            (self.member_competence_df[self.member_code_column].isin(role_members))
        ].copy()

        if skill_holders.empty:
            return []

        # 取得日でソート（早期に習得した人を優先）
        skill_holders = skill_holders[
            skill_holders[self.acquired_date_column].notna()
        ].copy()

        if skill_holders.empty:
            # 取得日がない場合は、そのまま使用
            skill_holders = self.member_competence_df[
                (self.member_competence_df[self.competence_code_column] == competence_code) &
                (self.member_competence_df[self.member_code_column].isin(role_members))
            ].copy()

        skill_holders[self.acquired_date_column] = pd.to_datetime(
            skill_holders[self.acquired_date_column],
            errors='coerce'
        )
        skill_holders = skill_holders.sort_values(self.acquired_date_column)

        # スキル情報を取得
        skill_info = self.competence_master_df[
            self.competence_master_df['力量コード'] == competence_code
        ]

        skill_name = skill_info.iloc[0]['力量名'] if not skill_info.empty else competence_code

        # パスを生成
        paths = []
        for _, holder_row in skill_holders.head(max_paths).iterrows():
            similar_member_code = holder_row[self.member_code_column]

            # 自分自身は除外
            if similar_member_code == member_code:
                continue

            # 類似メンバーの名前を取得
            similar_member_info = self.members_df[
                self.members_df[self.member_code_column] == similar_member_code
            ]

            similar_member_name = similar_member_info.iloc[0].get('メンバー名', similar_member_code) if not similar_member_info.empty else similar_member_code

            # パスを構築
            path = [
                {
                    'id': f'member_{member_code}',
                    'type': 'member',
                    'name': member_name,
                    'code': member_code
                },
                {
                    'id': f'similar_member_{similar_member_code}',
                    'type': 'similar_member',
                    'name': similar_member_name,
                    'code': similar_member_code,
                    'role': role
                },
                {
                    'id': f'competence_{competence_code}',
                    'type': 'competence',
                    'name': skill_name
                }
            ]

            paths.append(path)

        return paths

    def recommend_all_roles(self,
                            top_n_per_role: int = 10,
                            min_acquisition_rate: float = 0.3,
                            max_paths: int = 5) -> Dict[str, List[Dict]]:
        """
        全役職について推薦を生成

        Args:
            top_n_per_role: 各役職での推薦数
            min_acquisition_rate: 推薦対象とする最小取得率
            max_paths: 各スキルに対する最大パス数

        Returns:
            役職名をキーとした推薦辞書
        """
        if not self._growth_paths_cache:
            self.analyze_all_roles()

        all_recommendations = {}

        for role_name, growth_path in self._growth_paths_cache.items():
            # この役職のメンバーを取得（代表として最初のメンバーを使用）
            role_members = self.members_df[
                self.members_df[self.role_column] == role_name
            ][self.member_code_column].unique()

            if len(role_members) == 0:
                continue

            # 各メンバーについて推薦を生成（重複を避けるため、スキルのリストとして）
            role_recommendations = []
            processed_skills = set()

            for member_code in role_members[:5]:  # 最大5名まで
                member_recs = self.recommend_next_skills_with_paths(
                    member_code=member_code,
                    top_n=top_n_per_role * 2,  # 多めに取得
                    min_acquisition_rate=min_acquisition_rate,
                    max_paths=max_paths
                )

                for rec in member_recs:
                    if rec['competence_code'] not in processed_skills:
                        role_recommendations.append(rec)
                        processed_skills.add(rec['competence_code'])

                if len(role_recommendations) >= top_n_per_role:
                    break

            # 優先度スコアでソート
            role_recommendations.sort(key=lambda x: x['priority_score'], reverse=True)
            all_recommendations[role_name] = role_recommendations[:top_n_per_role]

        return all_recommendations

    def _fallback_recommend_from_all_skills(self,
                                           member_code: str,
                                           top_n: int,
                                           min_acquisition_rate: float) -> List[Dict]:
        """
        フォールバック：全スキルから推薦

        成長パスが存在しない場合の最終フォールバック。
        人気スキル（多くのメンバーが保有）を推薦する。

        Args:
            member_code: メンバーコード
            top_n: 推薦するスキル数
            min_acquisition_rate: 最小取得率（未使用、常に0）

        Returns:
            推薦スキルのリスト
        """
        logger.info("フォールバック：全スキルから人気スキルを推薦")

        # メンバーの現在の保有スキルを取得
        member_skills = self.member_competence_df[
            self.member_competence_df[self.member_code_column] == member_code
        ][self.competence_code_column].unique()

        member_skills_set = set(member_skills)

        # 全スキルについて保有メンバー数をカウント
        skill_popularity = {}

        for competence_code in self.competence_master_df['力量コード'].unique():
            # 既に保有しているスキルはスキップ
            if competence_code in member_skills_set:
                continue

            # このスキルを保有しているメンバー数
            holder_count = len(self.member_competence_df[
                self.member_competence_df[self.competence_code_column] == competence_code
            ][self.member_code_column].unique())

            if holder_count > 0:
                skill_popularity[competence_code] = holder_count

        # 人気順にソート
        sorted_skills = sorted(
            skill_popularity.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # 推薦リストを生成
        recommendations = []
        for competence_code, holder_count in sorted_skills[:top_n]:
            # スキル情報を取得
            skill_info = self.competence_master_df[
                self.competence_master_df['力量コード'] == competence_code
            ]

            if skill_info.empty:
                continue

            skill_info = skill_info.iloc[0]

            # 全メンバーに対する取得率
            total_members = len(self.members_df)
            acquisition_rate = holder_count / total_members if total_members > 0 else 0

            reason = (
                f"【人気スキル推薦】\n"
                f"このスキルは{total_members}名中{holder_count}名（{acquisition_rate*100:.1f}%）が保有しています。\n"
                f"役職ベースの成長パスが見つからなかったため、人気のあるスキルから推薦しています。"
            )

            recommendations.append({
                'competence_code': competence_code,
                'competence_name': skill_info.get('力量名', competence_code),
                'competence_type': skill_info.get('力量タイプ', 'UNKNOWN'),
                'category': skill_info.get('力量カテゴリー名', ''),
                'priority_score': acquisition_rate,  # 取得率をスコアとして使用
                'average_order': 0,  # 順序情報なし
                'acquisition_rate': acquisition_rate,
                'reason': reason
            })

        logger.info(f"フォールバック推薦: {len(recommendations)}件")

        return recommendations
