"""
キャリアパス階層構造の定義

役職ごとの標準的なスキル習得パスを定義し、
キャリアの進行段階を階層化します。
"""

from typing import Dict, List, Optional, Set, Tuple
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class CareerPathHierarchy:
    """
    キャリアパス階層構造を定義するクラス

    役職ごとに3～5段階のキャリアステージを定義:
    - Stage 0: 入門期（基礎スキル習得）
    - Stage 1: 成長期（応用スキル習得）
    - Stage 2: 熟達期（専門スキル習得）
    - Stage 3: エキスパート期（高度な専門スキル）
    """

    # デフォルトのキャリアパス階層定義
    DEFAULT_CAREER_PATHS = {
        '一般社員': {
            'stages': [
                {
                    'stage': 0,
                    'name': '入門期',
                    'description': '基礎スキルの習得',
                    'typical_duration_months': 6,
                    'skill_keywords': ['基礎', '入門', '初級', 'マナー', '報連相'],
                },
                {
                    'stage': 1,
                    'name': '成長期',
                    'description': '実務スキルの習得',
                    'typical_duration_months': 12,
                    'skill_keywords': ['実務', '業務', '実践', 'プロジェクト参加'],
                },
                {
                    'stage': 2,
                    'name': '熟達期',
                    'description': '独立した業務遂行',
                    'typical_duration_months': 18,
                    'skill_keywords': ['独立', '自律', '品質管理', '効率化'],
                },
            ],
        },
        '主任': {
            'stages': [
                {
                    'stage': 0,
                    'name': '入門期',
                    'description': 'チームリーダーとしての基礎',
                    'typical_duration_months': 6,
                    'skill_keywords': ['リーダーシップ', 'チーム', '進捗管理', 'タスク管理'],
                },
                {
                    'stage': 1,
                    'name': '成長期',
                    'description': 'プロジェクトマネジメント',
                    'typical_duration_months': 12,
                    'skill_keywords': ['プロジェクト管理', 'リスク管理', '目標設定', 'スケジュール'],
                },
                {
                    'stage': 2,
                    'name': '熟達期',
                    'description': '複数プロジェクトの統括',
                    'typical_duration_months': 18,
                    'skill_keywords': ['統括', '複数', 'リソース配分', '優先順位'],
                },
            ],
        },
        '課長': {
            'stages': [
                {
                    'stage': 0,
                    'name': '入門期',
                    'description': '組織マネジメントの基礎',
                    'typical_duration_months': 6,
                    'skill_keywords': ['組織', 'マネジメント', '人事評価', 'メンバー育成'],
                },
                {
                    'stage': 1,
                    'name': '成長期',
                    'description': '戦略立案と実行',
                    'typical_duration_months': 12,
                    'skill_keywords': ['戦略', '計画', '予算', '目標達成'],
                },
                {
                    'stage': 2,
                    'name': '熟達期',
                    'description': '部門の変革推進',
                    'typical_duration_months': 18,
                    'skill_keywords': ['変革', '改革', '組織開発', 'イノベーション'],
                },
                {
                    'stage': 3,
                    'name': 'エキスパート期',
                    'description': '経営層との連携',
                    'typical_duration_months': 24,
                    'skill_keywords': ['経営', '戦略策定', '事業計画', '意思決定'],
                },
            ],
        },
        '部長': {
            'stages': [
                {
                    'stage': 0,
                    'name': '入門期',
                    'description': '部門統括の基礎',
                    'typical_duration_months': 6,
                    'skill_keywords': ['部門', '統括', '組織運営', '予算管理'],
                },
                {
                    'stage': 1,
                    'name': '成長期',
                    'description': '事業戦略の立案',
                    'typical_duration_months': 12,
                    'skill_keywords': ['事業戦略', '中長期計画', '経営指標', 'KPI'],
                },
                {
                    'stage': 2,
                    'name': '熟達期',
                    'description': '経営への参画',
                    'typical_duration_months': 18,
                    'skill_keywords': ['経営参画', '全社戦略', 'ステークホルダー', '経営判断'],
                },
            ],
        },
    }

    def __init__(
        self,
        member_master: pd.DataFrame,
        member_competence: pd.DataFrame,
        competence_master: pd.DataFrame,
        custom_career_paths: Optional[Dict] = None,
    ):
        """
        Args:
            member_master: メンバーマスタ（役職情報を含む）
            member_competence: メンバー力量データ
            competence_master: 力量マスタ
            custom_career_paths: カスタムキャリアパス定義（Noneの場合はデフォルト）
        """
        self.member_master = member_master
        self.member_competence = member_competence
        self.competence_master = competence_master
        self.career_paths = custom_career_paths or self.DEFAULT_CAREER_PATHS

        # 役職ごとのメンバー数を集計
        self.role_member_counts = self._count_members_by_role()

        logger.info("\nCareer Path Hierarchy 構築完了")
        logger.info("  定義された役職数: %d", len(self.career_paths))
        logger.info("  役職別メンバー数: %s", dict(self.role_member_counts))

    def _count_members_by_role(self) -> Dict[str, int]:
        """役職ごとのメンバー数を集計"""
        if '役職' in self.member_master.columns:
            return self.member_master['役職'].value_counts().to_dict()
        return {}

    def get_role_stages(self, role: str) -> List[Dict]:
        """
        特定の役職のステージリストを取得

        Args:
            role: 役職名

        Returns:
            ステージ情報のリスト
        """
        if role not in self.career_paths:
            return []

        return self.career_paths[role]['stages']

    def get_stage_info(self, role: str, stage: int) -> Optional[Dict]:
        """
        特定の役職・ステージの情報を取得

        Args:
            role: 役職名
            stage: ステージ番号

        Returns:
            ステージ情報（存在しない場合はNone）
        """
        stages = self.get_role_stages(role)

        for stage_info in stages:
            if stage_info['stage'] == stage:
                return stage_info

        return None

    def classify_skill_to_stage(
        self,
        role: str,
        competence_name: str
    ) -> Optional[int]:
        """
        スキルをキャリアステージに分類

        Args:
            role: 役職名
            competence_name: 力量名

        Returns:
            ステージ番号（分類できない場合はNone）
        """
        stages = self.get_role_stages(role)

        if not stages:
            return None

        competence_name_lower = competence_name.lower()

        # 各ステージのキーワードとマッチング
        best_match = None
        best_match_score = 0

        for stage_info in stages:
            keywords = stage_info['skill_keywords']

            match_count = sum(
                1 for keyword in keywords
                if keyword.lower() in competence_name_lower
            )

            if match_count > best_match_score:
                best_match_score = match_count
                best_match = stage_info['stage']

        return best_match

    def get_skills_by_stage(
        self,
        role: str,
        stage: int,
        acquired_skills: Optional[Set[str]] = None
    ) -> List[str]:
        """
        特定の役職・ステージに該当する力量を取得

        Args:
            role: 役職名
            stage: ステージ番号
            acquired_skills: 既習得スキルのセット（除外用）

        Returns:
            力量コードのリスト
        """
        if acquired_skills is None:
            acquired_skills = set()

        # 同じ役職のメンバーが習得しているスキルを取得
        role_members = self.member_master[
            self.member_master['役職'] == role
        ]['メンバーコード'].tolist()

        if not role_members:
            return []

        role_skills = self.member_competence[
            self.member_competence['メンバーコード'].isin(role_members)
        ]

        # 力量名を取得してステージに分類
        stage_skills = []

        for _, row in self.competence_master.iterrows():
            comp_code = row['力量コード']
            comp_name = row['力量名']

            # 既習得は除外
            if comp_code in acquired_skills:
                continue

            # ステージに分類
            classified_stage = self.classify_skill_to_stage(role, comp_name)

            if classified_stage == stage:
                stage_skills.append(comp_code)

        return stage_skills

    def estimate_member_stage(
        self,
        member_code: str,
        role: str
    ) -> Tuple[int, float]:
        """
        メンバーの現在のキャリアステージを推定

        Args:
            member_code: メンバーコード
            role: 役職名

        Returns:
            (ステージ番号, 進捗率)
        """
        stages = self.get_role_stages(role)

        if not stages:
            return (0, 0.0)

        # メンバーの習得スキルを取得
        member_skills = self.member_competence[
            self.member_competence['メンバーコード'] == member_code
        ]['力量コード'].tolist()

        if not member_skills:
            return (0, 0.0)

        # 各ステージのスキル習得数をカウント
        stage_acquisition_counts = {}

        for stage_info in stages:
            stage_num = stage_info['stage']

            # このステージの推奨スキルを取得
            stage_skills = self.get_skills_by_stage(role, stage_num)

            if not stage_skills:
                stage_acquisition_counts[stage_num] = 0.0
                continue

            # 習得している割合を計算
            acquired_count = sum(1 for skill in member_skills if skill in stage_skills)
            stage_acquisition_counts[stage_num] = acquired_count / len(stage_skills)

        # 最も進捗率が高いステージを現在のステージとする
        if stage_acquisition_counts:
            current_stage = max(stage_acquisition_counts, key=stage_acquisition_counts.get)
            progress = stage_acquisition_counts[current_stage]

            return (current_stage, progress)

        return (0, 0.0)

    def get_next_stage_skills(
        self,
        member_code: str,
        role: str,
        top_n: int = 10
    ) -> List[Dict]:
        """
        メンバーの次のステージで習得すべきスキルを取得

        Args:
            member_code: メンバーコード
            role: 役職名
            top_n: 取得数

        Returns:
            推奨スキルのリスト
        """
        current_stage, progress = self.estimate_member_stage(member_code, role)

        stages = self.get_role_stages(role)

        if not stages:
            return []

        # 進捗率が80%以上なら次のステージを推薦
        if progress >= 0.8 and current_stage < len(stages) - 1:
            next_stage = current_stage + 1
        else:
            next_stage = current_stage

        # 既習得スキルを取得
        acquired_skills = set(
            self.member_competence[
                self.member_competence['メンバーコード'] == member_code
            ]['力量コード'].tolist()
        )

        # 次のステージのスキルを取得
        next_stage_skills = self.get_skills_by_stage(
            role, next_stage, acquired_skills
        )

        # 力量情報を付加
        recommendations = []

        for comp_code in next_stage_skills[:top_n]:
            comp_info = self.competence_master[
                self.competence_master['力量コード'] == comp_code
            ]

            if len(comp_info) == 0:
                continue

            stage_info = self.get_stage_info(role, next_stage)

            recommendations.append({
                'competence_code': comp_code,
                'competence_name': comp_info.iloc[0]['力量名'],
                'stage': next_stage,
                'stage_name': stage_info['name'] if stage_info else f'Stage {next_stage}',
                'reason': self._generate_reason(role, current_stage, next_stage, progress),
            })

        return recommendations

    def _generate_reason(
        self,
        role: str,
        current_stage: int,
        next_stage: int,
        progress: float
    ) -> str:
        """推薦理由を生成"""
        current_stage_info = self.get_stage_info(role, current_stage)
        next_stage_info = self.get_stage_info(role, next_stage)

        if current_stage == next_stage:
            return (
                f"{role}の{current_stage_info['name'] if current_stage_info else 'Stage ' + str(current_stage)}"
                f"（進捗{progress*100:.0f}%）を強化するため、このスキルをおすすめします。"
            )
        else:
            return (
                f"{role}の{current_stage_info['name'] if current_stage_info else 'Stage ' + str(current_stage)}"
                f"（進捗{progress*100:.0f}%）を完了し、"
                f"{next_stage_info['name'] if next_stage_info else 'Stage ' + str(next_stage)}"
                f"へのステップアップとして、このスキルをおすすめします。"
            )

    def get_career_path_statistics(self) -> pd.DataFrame:
        """
        キャリアパスの統計情報を取得

        Returns:
            DataFrame with columns: [Role, Stage, Stage_Name, Member_Count, Avg_Progress]
        """
        stats_data = []

        for role in self.career_paths.keys():
            # この役職のメンバー
            role_members = self.member_master[
                self.member_master['役職'] == role
            ]['メンバーコード'].tolist()

            if not role_members:
                continue

            # 各ステージの統計
            stages = self.get_role_stages(role)

            for stage_info in stages:
                stage_num = stage_info['stage']

                # 各メンバーのこのステージの進捗を計算
                stage_progresses = []

                for member_code in role_members:
                    member_stage, progress = self.estimate_member_stage(member_code, role)

                    if member_stage == stage_num:
                        stage_progresses.append(progress)

                avg_progress = np.mean(stage_progresses) if stage_progresses else 0.0

                stats_data.append({
                    'Role': role,
                    'Stage': stage_num,
                    'Stage_Name': stage_info['name'],
                    'Member_Count': len(stage_progresses),
                    'Avg_Progress': avg_progress,
                })

        return pd.DataFrame(stats_data)
