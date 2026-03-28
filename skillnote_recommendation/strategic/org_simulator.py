"""
組織シミュレーション

職種ベースのメンバー異動シミュレーションと前後比較
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

from skillnote_recommendation.organizational import org_metrics

logger = logging.getLogger(__name__)


class OrganizationSimulator:
    """組織シミュレーションエンジン"""
    
    def __init__(self):
        """初期化"""
        self.current_state = None
        self.simulated_state = None
        self.transfers = []
    
    def capture_current_state(
        self,
        members_df: pd.DataFrame,
        member_competence_df: pd.DataFrame,
        competence_master_df: pd.DataFrame,
        group_by: str = "職種"
    ) -> Dict:
        """
        現在の組織状態をキャプチャ
        
        Args:
            members_df: メンバーマスタ
            member_competence_df: メンバー習得力量データ
            competence_master_df: 力量マスタ
            group_by: グループ化カラム（デフォルト: 職種）
            
        Returns:
            組織状態の辞書
        """
        # グループ別メトリクスを計算
        group_summary = org_metrics.calculate_group_skill_summary(
            member_competence_df,
            members_df,
            group_by=group_by
        )
        
        # スキルカバレッジ
        coverage = org_metrics.calculate_skill_coverage(
            member_competence_df,
            competence_master_df
        )
        
        # スキル多様性
        diversity = org_metrics.calculate_skill_diversity_index(
            member_competence_df
        )
        
        # スキル集中度
        concentration = org_metrics.calculate_skill_concentration(
            member_competence_df,
            threshold=3
        )
        
        state = {
            "group_summary": group_summary,
            "coverage": coverage,
            "diversity": diversity,
            "concentration": concentration,
            "members_df": members_df.copy(),
            "member_competence_df": member_competence_df.copy()
        }
        
        self.current_state = state
        return state
    
    def simulate_transfer(
        self,
        member_code: str,
        from_group: str,
        to_group: str,
        group_column: str = "職種"
    ) -> None:
        """
        メンバー異動をシミュレーション（状態に追加）
        
        Args:
            member_code: メンバーコード
            from_group: 異動元グループ
            to_group: 異動先グループ
            group_column: グループカラム名
        """
        transfer = {
            "member_code": member_code,
            "from_group": from_group,
            "to_group": to_group,
            "group_column": group_column
        }
        self.transfers.append(transfer)
        logger.info(f"異動シミュレーション追加: {member_code} ({from_group} → {to_group})")
    
    def execute_simulation(
        self,
        competence_master_df: pd.DataFrame
    ) -> Dict:
        """
        シミュレーションを実行
        
        Args:
            competence_master_df: 力量マスタ
            
        Returns:
            シミュレーション後の組織状態
        """
        if self.current_state is None:
            raise ValueError("現在の状態がキャプチャされていません。先にcapture_current_stateを実行してください。")
        
        if len(self.transfers) == 0:
            logger.warning("異動が設定されていません")
            return self.current_state
        
        # 現在の状態をコピー
        simulated_members_df = self.current_state["members_df"].copy()
        
        # 異動を適用
        for transfer in self.transfers:
            member_code = transfer["member_code"]
            to_group = transfer["to_group"]
            group_column = transfer["group_column"]
            
            # メンバーマスタを更新
            simulated_members_df.loc[
                simulated_members_df["メンバーコード"] == member_code,
                group_column
            ] = to_group
        
        # シミュレーション後のメトリクスを計算
        member_competence_df = self.current_state["member_competence_df"]
        group_column = self.transfers[0]["group_column"] if self.transfers else "職種"
        
        group_summary = org_metrics.calculate_group_skill_summary(
            member_competence_df,
            simulated_members_df,
            group_by=group_column
        )
        
        coverage = org_metrics.calculate_skill_coverage(
            member_competence_df,
            competence_master_df
        )
        
        diversity = org_metrics.calculate_skill_diversity_index(
            member_competence_df
        )
        
        concentration = org_metrics.calculate_skill_concentration(
            member_competence_df,
            threshold=3
        )
        
        simulated_state = {
            "group_summary": group_summary,
            "coverage": coverage,
            "diversity": diversity,
            "concentration": concentration,
            "members_df": simulated_members_df,
            "member_competence_df": member_competence_df
        }
        
        self.simulated_state = simulated_state
        return simulated_state
    
    def compare_states(self) -> pd.DataFrame:
        """
        現状とシミュレーション後を比較
        
        Returns:
            比較結果DataFrame
        """
        if self.current_state is None or self.simulated_state is None:
            raise ValueError("現在の状態とシミュレーション後の状態が必要です")
        
        # グループサマリーを比較
        current_summary = self.current_state["group_summary"]
        simulated_summary = self.simulated_state["group_summary"]
        
        # マージして比較
        comparison = current_summary.merge(
            simulated_summary,
            on="グループ",
            how="outer",
            suffixes=("_現在", "_シミュレーション後")
        ).fillna(0)
        
        # 変化量を計算
        comparison["メンバー数_変化"] = (
            comparison["メンバー数_シミュレーション後"] - comparison["メンバー数_現在"]
        )
        comparison["平均スキル数/人_変化"] = (
            comparison["平均スキル数/人_シミュレーション後"] - comparison["平均スキル数/人_現在"]
        ).round(1)
        comparison["ユニークスキル数_変化"] = (
            comparison["ユニークスキル数_シミュレーション後"] - comparison["ユニークスキル数_現在"]
        )
        
        return comparison
    
    def calculate_balance_score(self, state: Dict) -> float:
        """
        組織バランススコアを計算
        
        Args:
            state: 組織状態
            
        Returns:
            バランススコア（0.0-1.0）
        """
        # 職種間のスキル偏差を計算
        group_summary = state["group_summary"]
        
        if len(group_summary) == 0:
            return 0.0
        
        # 平均スキル数の標準偏差（小さいほうが良い）
        avg_skills = group_summary["平均スキル数/人"]
        std_dev = avg_skills.std()
        mean_skills = avg_skills.mean()
        
        # 変動係数（CV）を計算
        cv = std_dev / mean_skills if mean_skills > 0 else 1.0
        
        # バランススコア（CVが小さいほど高スコア）
        balance_score = 1.0 / (1.0 + cv)
        
        return round(balance_score, 3)
    
    def reset_simulation(self) -> None:
        """シミュレーションをリセット"""
        self.transfers = []
        self.simulated_state = None
        logger.info("シミュレーションをリセットしました")
