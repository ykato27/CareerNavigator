"""
組織レベルのメトリクス計算

スキルカバレッジ、集中度、多様性などの組織KPIを計算
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from collections import Counter
import logging

logger = logging.getLogger(__name__)


def calculate_skill_coverage(
    member_competence_df: pd.DataFrame, 
    competence_master_df: pd.DataFrame
) -> Dict[str, float]:
    """
    スキルカバレッジ率を計算
    
    組織全体で何%のスキルを保有しているかを計算
    
    Args:
        member_competence_df: メンバー習得力量データ
        competence_master_df: 力量マスタデータ
        
    Returns:
        カバレッジ情報の辞書
        - coverage_rate: カバレッジ率 (0.0-1.0)
        - covered_skills: 保有スキル数
        - total_skills: 全スキル数
    """
    # 全スキル数
    total_skills = len(competence_master_df)
    
    # 組織で保有しているスキル（1人以上が習得している）
    covered_skills = member_competence_df["力量コード"].nunique()
    
    coverage_rate = covered_skills / total_skills if total_skills > 0 else 0.0
    
    return {
        "coverage_rate": coverage_rate,
        "covered_skills": covered_skills,
        "total_skills": total_skills
    }


def calculate_skill_concentration(
    member_competence_df: pd.DataFrame,
    threshold: int = 3
) -> Dict[str, any]:
    """
    スキル集中度を計算
    
    特定のスキルが少数のメンバーに偏っているかを測定
    
    Args:
        member_competence_df: メンバー習得力量データ
        threshold: 集中とみなす人数の閾値（この人数以下を集中とする）
        
    Returns:
        集中度情報の辞書
        - concentration_rate: 集中スキルの割合 (0.0-1.0)
        - concentrated_skills: 集中しているスキル数
        - total_skills: 全保有スキル数
        - skill_holder_counts: スキル別保有者数
    """
    # スキルごとの保有者数をカウント
    skill_counts = member_competence_df.groupby("力量コード")["メンバーコード"].nunique()
    
    # 閾値以下の人数しか持っていないスキル（集中スキル）
    concentrated_skills = (skill_counts <= threshold).sum()
    total_skills = len(skill_counts)
    
    concentration_rate = concentrated_skills / total_skills if total_skills > 0 else 0.0
    
    return {
        "concentration_rate": concentration_rate,
        "concentrated_skills": concentrated_skills,
        "total_skills": total_skills,
        "skill_holder_counts": skill_counts.to_dict()
    }


def calculate_skill_diversity_index(
    member_competence_df: pd.DataFrame,
    by_category: bool = False,
    category_column: Optional[str] = None
) -> float:
    """
    スキル多様性指標（Shannon Entropy）を計算
    
    Args:
        member_competence_df: メンバー習得力量データ
        by_category: カテゴリ別に計算するか
        category_column: カテゴリカラム名（by_category=Trueの場合）
        
    Returns:
        Shannon Entropy（値が大きいほど多様性が高い）
    """
    if by_category and category_column:
        # カテゴリ別の保有数で計算
        counts = member_competence_df[category_column].value_counts()
    else:
        # スキル別の保有数で計算
        counts = member_competence_df["力量コード"].value_counts()
    
    # 確率分布に変換
    probabilities = counts / counts.sum()
    
    # Shannon Entropy計算
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    
    return entropy


def get_skill_distribution_by_level(
    member_competence_df: pd.DataFrame,
    competence_code: str
) -> Dict[int, int]:
    """
    特定スキルのレベル分布を取得
    
    Args:
        member_competence_df: メンバー習得力量データ
        competence_code: 力量コード
        
    Returns:
        レベル別人数の辞書 {レベル: 人数}
    """
    skill_data = member_competence_df[
        member_competence_df["力量コード"] == competence_code
    ]
    
    if "レベル" in skill_data.columns:
        level_dist = skill_data["レベル"].value_counts().to_dict()
    else:
        # レベル情報がない場合は、保有者数のみ
        level_dist = {1: len(skill_data)}
    
    return level_dist


def calculate_group_skill_summary(
    member_competence_df: pd.DataFrame,
    members_df: pd.DataFrame,
    group_by: str
) -> pd.DataFrame:
    """
    グループ別のスキル集計サマリーを作成
    
    Args:
        member_competence_df: メンバー習得力量データ
        members_df: メンバーマスタ（グループ情報を含む）
        group_by: グループ化するカラム名（例: "役職", "職種", "職能・等級"）
        
    Returns:
        グループ別集計DataFrame
        - group: グループ名
        - member_count: メンバー数
        - avg_skills_per_member: 1人あたり平均スキル数
        - unique_skills: グループ内のユニークスキル数
    """
    # メンバーマスタとマージ
    merged = member_competence_df.merge(
        members_df[["メンバーコード", group_by]],
        on="メンバーコード",
        how="left"
    )
    
    # グループ別に集計
    summary = merged.groupby(group_by).agg(
        member_count=("メンバーコード", "nunique"),
        total_skill_records=("力量コード", "count"),
        unique_skills=("力量コード", "nunique")
    ).reset_index()
    
    # 1人あたり平均スキル数を計算
    summary["avg_skills_per_member"] = (
        summary["total_skill_records"] / summary["member_count"]
    ).round(1)
    
    # カラム名を日本語に
    summary = summary.rename(columns={
        group_by: "グループ",
        "member_count": "メンバー数",
        "unique_skills": "ユニークスキル数",
        "avg_skills_per_member": "平均スキル数/人"
    })
    
    return summary[["グループ", "メンバー数", "平均スキル数/人", "ユニークスキル数"]]


def calculate_cross_group_summary(
    member_competence_df: pd.DataFrame,
    members_df: pd.DataFrame,
    group_by_1: str,
    group_by_2: str
) -> pd.DataFrame:
    """
    2軸でのクロス集計を作成
    
    Args:
        member_competence_df: メンバー習得力量データ
        members_df: メンバーマスタ
        group_by_1: 1つ目のグループ化カラム（例: "職種"）
        group_by_2: 2つ目のグループ化カラム（例: "役職"）
        
    Returns:
        クロス集計DataFrame（ピボットテーブル形式）
    """
    # メンバーマスタとマージ
    merged = member_competence_df.merge(
        members_df[["メンバーコード", group_by_1, group_by_2]],
        on="メンバーコード",
        how="left"
    )
    
    # グループ別に集計
    cross_summary = merged.groupby([group_by_1, group_by_2]).agg(
        member_count=("メンバーコード", "nunique"),
        avg_skills=("力量コード", "count")
    ).reset_index()
    
    # 1人あたり平均スキル数を計算
    cross_summary["avg_skills_per_member"] = (
        cross_summary["avg_skills"] / cross_summary["member_count"]
    ).round(1)
    
    # ピボットテーブル化（平均スキル数）
    pivot_table = cross_summary.pivot_table(
        index=group_by_1,
        columns=group_by_2,
        values="avg_skills_per_member",
        fill_value=0
    )
    
    return pivot_table
