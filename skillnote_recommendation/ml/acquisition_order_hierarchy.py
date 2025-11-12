"""
スキル取得順序ベースの階層構造

メンバーのスキル取得順序から、初級→中級→上級の段階を自動検出し、
SEMでの階層的スキル推薦に使用します。

このアプローチは恣意的なドメイン分類を排除し、
実際のデータから学習する完全にデータドリブンな手法です。
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class AcquisitionOrderHierarchy:
    """
    スキル取得順序ベースの階層構造を定義するクラス

    各スキルの「平均取得順序」を計算し、それに基づいて3段階に分類:
    - Stage 1: 初級（早期に取得されるスキル）
    - Stage 2: 中級（中期に取得されるスキル）
    - Stage 3: 上級（後期に取得されるスキル）
    """

    def __init__(
        self,
        member_competence: pd.DataFrame,
        competence_master: pd.DataFrame,
        n_stages: int = 3,
        min_acquisition_count: int = 3,
    ):
        """
        Args:
            member_competence: メンバー力量データ（取得日を含む）
            competence_master: 力量マスタ
            n_stages: ステージ数（デフォルト: 3）
            min_acquisition_count: 分析対象とする最小取得人数
        """
        self.member_competence = member_competence
        self.competence_master = competence_master
        self.n_stages = n_stages
        self.min_acquisition_count = min_acquisition_count

        # カラム名
        self.member_code_column = "メンバーコード"
        self.competence_code_column = "力量コード"
        self.acquired_date_column = "取得日"

        # スキルの平均取得順序を計算
        self.skill_acquisition_stats = self._calculate_acquisition_order_statistics()

        # ステージごとのスキル分類
        self.stage_classifications = self._classify_skills_by_acquisition_order()

        logger.info("\nAcquisition Order Hierarchy 構築完了")
        logger.info("  ステージ数: %d", self.n_stages)
        logger.info("  分析されたスキル数: %d", len(self.skill_acquisition_stats))

    def _calculate_acquisition_order_statistics(self) -> Dict[str, Dict]:
        """
        各スキルの取得順序統計を計算

        Returns:
            {スキルコード: {
                'avg_order': float,      # 平均取得順序
                'std_order': float,      # 標準偏差
                'count': int,            # 取得人数
                'competence_name': str   # 力量名
            }}
        """
        skill_orders: Dict[str, List[int]] = {}

        # メンバーごとに取得順序を計算
        for member_code in self.member_competence[self.member_code_column].unique():
            # このメンバーのスキルを取得
            member_skills = self.member_competence[
                self.member_competence[self.member_code_column] == member_code
            ].copy()

            # 取得日が存在するデータのみ使用
            member_skills = member_skills[
                member_skills[self.acquired_date_column].notna()
            ].copy()

            if member_skills.empty:
                continue

            # 取得日でソート
            member_skills[self.acquired_date_column] = pd.to_datetime(
                member_skills[self.acquired_date_column], errors="coerce"
            )
            member_skills = member_skills.sort_values(self.acquired_date_column)

            # 取得順序を付与（0始まり）
            for order, (_, row) in enumerate(member_skills.iterrows()):
                competence_code = row[self.competence_code_column]

                if competence_code not in skill_orders:
                    skill_orders[competence_code] = []

                skill_orders[competence_code].append(order)

        # 統計値を計算
        skill_stats = {}

        for competence_code, orders in skill_orders.items():
            # 最小取得人数のフィルタ
            if len(orders) < self.min_acquisition_count:
                continue

            # 力量名を取得
            comp_info = self.competence_master[
                self.competence_master["力量コード"] == competence_code
            ]

            if comp_info.empty:
                continue

            competence_name = comp_info.iloc[0]["力量名"]

            skill_stats[competence_code] = {
                "avg_order": np.mean(orders),
                "std_order": np.std(orders) if len(orders) > 1 else 0.0,
                "count": len(orders),
                "competence_name": competence_name,
            }

        return skill_stats

    def _classify_skills_by_acquisition_order(self) -> Dict[int, List[str]]:
        """
        スキルを平均取得順序でステージ分割

        Returns:
            {stage_id: [スキルコードリスト]}
        """
        if not self.skill_acquisition_stats:
            return {}

        # 平均取得順序でソート
        sorted_skills = sorted(
            self.skill_acquisition_stats.items(), key=lambda x: x[1]["avg_order"]
        )

        # ステージごとに分割
        total_skills = len(sorted_skills)
        skills_per_stage = total_skills // self.n_stages

        stage_classifications = {}

        for stage_id in range(self.n_stages):
            start_idx = stage_id * skills_per_stage

            if stage_id == self.n_stages - 1:
                # 最後のステージは残り全部
                end_idx = total_skills
            else:
                end_idx = (stage_id + 1) * skills_per_stage

            stage_skills = sorted_skills[start_idx:end_idx]
            stage_classifications[stage_id + 1] = [skill[0] for skill in stage_skills]

            # ステージ情報をログ出力
            avg_orders = [skill[1]["avg_order"] for skill in stage_skills]
            logger.info(
                f"  Stage {stage_id + 1}: {len(stage_skills)}個のスキル "
                f"(平均取得順序: {min(avg_orders):.1f}～{max(avg_orders):.1f})"
            )

        return stage_classifications

    def get_stage(self, competence_code: str) -> Optional[int]:
        """
        スキルのステージを取得

        Args:
            competence_code: 力量コード

        Returns:
            ステージ番号（1～n_stages）、該当なしの場合はNone
        """
        for stage_id, skill_codes in self.stage_classifications.items():
            if competence_code in skill_codes:
                return stage_id

        return None

    def get_skills_by_stage(self, stage_id: int) -> List[str]:
        """
        特定ステージのスキルコードリストを取得

        Args:
            stage_id: ステージ番号（1～n_stages）

        Returns:
            スキルコードのリスト
        """
        return self.stage_classifications.get(stage_id, [])

    def get_stage_name(self, stage_id: int) -> str:
        """
        ステージ名を取得

        Args:
            stage_id: ステージ番号（1～n_stages）

        Returns:
            ステージ名
        """
        stage_names = {
            1: "初級（早期習得スキル）",
            2: "中級（中期習得スキル）",
            3: "上級（後期習得スキル）",
            4: "エキスパート（専門スキル）",
            5: "マスター（最上級スキル）",
        }

        return stage_names.get(stage_id, f"Stage {stage_id}")

    def get_statistics(self) -> pd.DataFrame:
        """
        階層構造の統計情報を取得

        Returns:
            DataFrame with columns: [Stage, Stage_Name, Skill_Count, Avg_Order_Range]
        """
        stats = []

        for stage_id, skill_codes in self.stage_classifications.items():
            # このステージのスキルの平均取得順序範囲
            avg_orders = [
                self.skill_acquisition_stats[code]["avg_order"]
                for code in skill_codes
                if code in self.skill_acquisition_stats
            ]

            if not avg_orders:
                continue

            stats.append(
                {
                    "Stage": stage_id,
                    "Stage_Name": self.get_stage_name(stage_id),
                    "Skill_Count": len(skill_codes),
                    "Avg_Order_Min": min(avg_orders),
                    "Avg_Order_Max": max(avg_orders),
                    "Avg_Order_Mean": np.mean(avg_orders),
                }
            )

        return pd.DataFrame(stats)

    def estimate_member_stage(
        self, member_code: str
    ) -> Tuple[int, float, List[str]]:
        """
        メンバーの現在のステージを推定

        Args:
            member_code: メンバーコード

        Returns:
            (ステージ番号, 進捗率, 習得済みスキルリスト)
        """
        # メンバーの習得スキルを取得
        member_skills = self.member_competence[
            self.member_competence[self.member_code_column] == member_code
        ][self.competence_code_column].unique()

        member_skills_set = set(member_skills)

        # 各ステージの習得率を計算
        stage_progress = {}

        for stage_id, skill_codes in self.stage_classifications.items():
            if not skill_codes:
                stage_progress[stage_id] = 0.0
                continue

            acquired_count = sum(1 for code in skill_codes if code in member_skills_set)
            stage_progress[stage_id] = acquired_count / len(skill_codes)

        # 最も進捗率が高いステージを現在のステージとする
        if not stage_progress:
            return (1, 0.0, [])

        current_stage = max(stage_progress, key=stage_progress.get)
        progress = stage_progress[current_stage]

        # 習得済みスキルリスト（現在のステージ）
        acquired_skills = [
            code
            for code in self.stage_classifications[current_stage]
            if code in member_skills_set
        ]

        return (current_stage, progress, acquired_skills)

    def get_next_stage_skills(
        self, member_code: str, top_n: int = 10
    ) -> List[Dict]:
        """
        メンバーの次に習得すべきスキルを取得

        Args:
            member_code: メンバーコード
            top_n: 取得数

        Returns:
            推奨スキルのリスト
        """
        current_stage, progress, acquired_skills = self.estimate_member_stage(
            member_code
        )

        acquired_skills_set = set(acquired_skills)

        # 進捗率が80%以上なら次のステージを推薦
        if progress >= 0.8 and current_stage < self.n_stages:
            next_stage = current_stage + 1
        else:
            next_stage = current_stage

        # 次のステージのスキルを取得
        next_stage_skills = self.get_skills_by_stage(next_stage)

        # メンバーの全習得スキル
        all_member_skills = self.member_competence[
            self.member_competence[self.member_code_column] == member_code
        ][self.competence_code_column].unique()

        all_member_skills_set = set(all_member_skills)

        # 未習得スキルを抽出
        recommendations = []

        for competence_code in next_stage_skills:
            # 既に習得済みはスキップ
            if competence_code in all_member_skills_set:
                continue

            # スキル情報を取得
            if competence_code not in self.skill_acquisition_stats:
                continue

            skill_info = self.skill_acquisition_stats[competence_code]
            comp_info = self.competence_master[
                self.competence_master["力量コード"] == competence_code
            ]

            if comp_info.empty:
                continue

            comp_info = comp_info.iloc[0]

            # 優先度スコア: 平均取得順序が早いほど高スコア
            priority_score = 1.0 / (skill_info["avg_order"] + 1)

            recommendations.append(
                {
                    "competence_code": competence_code,
                    "competence_name": skill_info["competence_name"],
                    "competence_type": comp_info.get("力量タイプ", "UNKNOWN"),
                    "category": comp_info.get("力量カテゴリー名", ""),
                    "stage": next_stage,
                    "stage_name": self.get_stage_name(next_stage),
                    "avg_acquisition_order": skill_info["avg_order"],
                    "acquisition_count": skill_info["count"],
                    "priority_score": priority_score,
                }
            )

        # 優先度スコアでソート
        recommendations.sort(key=lambda x: x["priority_score"], reverse=True)

        return recommendations[:top_n]
