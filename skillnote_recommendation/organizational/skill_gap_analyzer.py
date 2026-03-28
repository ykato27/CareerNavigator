"""
スキルギャップ分析エンジン

組織の現状スキルと目標スキルのギャップを分析
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class SkillGapAnalyzer:
    """組織のスキルギャップを分析"""
    
    def __init__(self):
        """初期化"""
        self.current_profile = None
        self.target_profile = None
        self.gap_df = None
    
    def calculate_current_profile(
        self,
        member_competence_df: pd.DataFrame,
        competence_master_df: pd.DataFrame,
        group_by: Optional[str] = None,
        members_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        現在の組織スキルプロファイルを計算
        
        Args:
            member_competence_df: メンバー習得力量データ
            competence_master_df: 力量マスタデータ
            group_by: グループ化するカラム名（例: "職種", "役職"）
            members_df: メンバーマスタ（group_byを使用する場合必須）
            
        Returns:
            スキルプロファイルDataFrame
            - 力量コード
            - 力量名
            - 保有者数
            - 保有率 (0.0-1.0)
            - 平均レベル
        """
        if group_by and members_df is not None:
            # グループ別に計算
            merged = member_competence_df.merge(
                members_df[["メンバーコード", group_by]],
                on="メンバーコード",
                how="left"
            )
            total_members = members_df[group_by].notna().sum()
        else:
            # 組織全体で計算
            merged = member_competence_df.copy()
            total_members = merged["メンバーコード"].nunique()
        
        # レベル情報の処理（数値型に変換）
        has_level = False
        if "レベル" in merged.columns:
            try:
                # レベルを数値型に変換
                merged["レベル_数値"] = pd.to_numeric(merged["レベル"], errors='coerce')
                # 変換後にNaNでない値があれば有効
                if merged["レベル_数値"].notna().any():
                    has_level = True
                    logger.info("レベル情報が数値型に変換されました")
                else:
                    logger.warning("レベル情報が数値に変換できませんでした")
            except Exception as e:
                logger.warning(f"レベル情報の変換中にエラー: {e}")
        
        # スキルごとに集計
        if has_level:
            skill_stats = merged.groupby("力量コード").agg(
                holder_count=("メンバーコード", "nunique"),
                avg_level=("レベル_数値", "mean")
            ).reset_index()
        else:
            skill_stats = merged.groupby("力量コード").agg(
                holder_count=("メンバーコード", "nunique")
            ).reset_index()
            skill_stats["avg_level"] = 1.0  # デフォルト値
        
        # 力量マスタとマージして力量名を取得
        skill_stats = skill_stats.merge(
            competence_master_df[["力量コード", "力量名"]],
            on="力量コード",
            how="left"
        )
        
        # 保有率を計算
        skill_stats["保有率"] = skill_stats["holder_count"] / total_members
        
        # レベルがない場合は1として扱う
        if "avg_level" not in skill_stats.columns or skill_stats["avg_level"].isna().all():
            skill_stats["平均レベル"] = 1.0
        else:
            skill_stats["平均レベル"] = skill_stats["avg_level"].fillna(1.0).round(2)
        
        # カラムを整理
        profile = skill_stats[[
            "力量コード", "力量名", "holder_count", "保有率", "平均レベル"
        ]].rename(columns={"holder_count": "保有者数"})
        
        # 保有率順でソート
        profile = profile.sort_values("保有率", ascending=False).reset_index(drop=True)
        
        self.current_profile = profile
        return profile
    
    def calculate_target_profile_top_percentile(
        self,
        member_competence_df: pd.DataFrame,
        competence_master_df: pd.DataFrame,
        percentile: float = 0.2
    ) -> pd.DataFrame:
        """
        上位N%のメンバーの平均スキルを目標プロファイルとして計算
        
        Args:
            member_competence_df: メンバー習得力量データ
            competence_master_df: 力量マスタデータ
            percentile: 上位パーセンタイル（0.0-1.0、例: 0.2 = 上位20%）
            
        Returns:
            目標スキルプロファイルDataFrame
            - 力量コード
            - 力量名
            - 目標レベル
            - 目標保有率
        """
        # メンバーごとのスキル保有数を計算
        member_skill_counts = member_competence_df.groupby("メンバーコード").size()
        
        # 上位N%のメンバーを特定
        threshold = member_skill_counts.quantile(1 - percentile)
        top_members = member_skill_counts[member_skill_counts >= threshold].index.tolist()
        
        logger.info(f"上位{percentile*100:.0f}%のメンバー数: {len(top_members)}人 (閾値: {threshold:.1f}スキル)")
        
        # 上位メンバーのスキルデータのみ抽出
        top_member_data = member_competence_df[
            member_competence_df["メンバーコード"].isin(top_members)
        ].copy()
        
        # レベル情報の処理（数値型に変換）
        has_level = False
        if "レベル" in top_member_data.columns:
            try:
                top_member_data["レベル_数値"] = pd.to_numeric(top_member_data["レベル"], errors='coerce')
                if top_member_data["レベル_数値"].notna().any():
                    has_level = True
            except Exception as e:
                logger.warning(f"ターゲットプロファイルのレベル変換中にエラー: {e}")
        
        # 上位メンバーのスキルプロファイルを計算
        if has_level:
            target_stats = top_member_data.groupby("力量コード").agg(
                holder_count=("メンバーコード", "nunique"),
                avg_level=("レベル_数値", "mean")
            ).reset_index()
        else:
            target_stats = top_member_data.groupby("力量コード").agg(
                holder_count=("メンバーコード", "nunique")
            ).reset_index()
            target_stats["avg_level"] = 1.0
        
        # 力量マスタとマージ
        target_stats = target_stats.merge(
            competence_master_df[["力量コード", "力量名"]],
            on="力量コード",
            how="left"
        )
        
        # 目標保有率を計算（上位メンバー内での保有率）
        target_stats["目標保有率"] = target_stats["holder_count"] / len(top_members)
        
        # 目標レベル
        if "avg_level" not in target_stats.columns or target_stats["avg_level"].isna().all():
            target_stats["目標レベル"] = 1.0
        else:
            target_stats["目標レベル"] = target_stats["avg_level"].fillna(1.0).round(2)
        
        # カラムを整理
        target_profile = target_stats[[
            "力量コード", "力量名", "目標レベル", "目標保有率"
        ]].sort_values("目標保有率", ascending=False).reset_index(drop=True)
        
        self.target_profile = target_profile
        return target_profile
    
    def calculate_gap(
        self,
        current_profile: Optional[pd.DataFrame] = None,
        target_profile: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        現状とターゲットのギャップを計算
        
        Args:
            current_profile: 現在のプロファイル（省略時は保存済みを使用）
            target_profile: 目標プロファイル（省略時は保存済みを使用）
            
        Returns:
            ギャップDataFrame
            - 力量コード
            - 力量名
            - 現在保有率
            - 目標保有率
            - ギャップ（目標 - 現在）
            - ギャップ率（ギャップ / 目標）
            - 現在平均レベル
            - 目標レベル
            - レベルギャップ
        """
        if current_profile is None:
            current_profile = self.current_profile
        if target_profile is None:
            target_profile = self.target_profile
            
        if current_profile is None or target_profile is None:
            raise ValueError("プロファイルが計算されていません。先にcalculate_current_profileとcalculate_target_profileを実行してください。")
        
        # 現在と目標をマージ
        gap_df = target_profile.merge(
            current_profile[["力量コード", "保有率", "平均レベル"]],
            on="力量コード",
            how="left"
        )
        
        # 現在保有していないスキルは0で埋める
        gap_df["保有率"] = gap_df["保有率"].fillna(0)
        gap_df["平均レベル"] = gap_df["平均レベル"].fillna(0)
        
        # ギャップを計算
        gap_df["保有率ギャップ"] = gap_df["目標保有率"] - gap_df["保有率"]
        gap_df["保有率ギャップ率"] = (
            gap_df["保有率ギャップ"] / gap_df["目標保有率"]
        ).fillna(0).round(3)
        
        gap_df["レベルギャップ"] = gap_df["目標レベル"] - gap_df["平均レベル"]
        
        # カラム名を整理
        gap_df = gap_df.rename(columns={
            "保有率": "現在保有率",
            "平均レベル": "現在平均レベル"
        })
        
        # ギャップの大きい順にソート
        gap_df = gap_df.sort_values("保有率ギャップ", ascending=False).reset_index(drop=True)
        
        self.gap_df = gap_df
        return gap_df
    
    def identify_critical_skills(
        self,
        gap_df: Optional[pd.DataFrame] = None,
        threshold: float = 0.3
    ) -> pd.DataFrame:
        """
        クリティカルスキルを特定
        
        ギャップ率が閾値以上のスキルをクリティカルとして抽出
        
        Args:
            gap_df: ギャップDataFrame（省略時は保存済みを使用）
            threshold: クリティカルとみなすギャップ率の閾値（0.0-1.0）
            
        Returns:
            クリティカルスキルDataFrame
        """
        if gap_df is None:
            gap_df = self.gap_df
            
        if gap_df is None:
            raise ValueError("ギャップが計算されていません。先にcalculate_gapを実行してください。")
        
        # 閾値以上のギャップを持つスキルを抽出
        critical_skills = gap_df[gap_df["保有率ギャップ率"] >= threshold].copy()
        
        logger.info(f"クリティカルスキル数: {len(critical_skills)}個 (閾値: {threshold*100:.0f}%)")
        
        return critical_skills
    
    def generate_gap_recommendations(
        self,
        gap_df: Optional[pd.DataFrame] = None,
        top_n: int = 10
    ) -> List[Dict]:
        """
        ギャップを埋めるための推奨事項を生成
        
        Args:
            gap_df: ギャップDataFrame（省略時は保存済みを使用）
            top_n: 推奨するスキル数
            
        Returns:
            推奨事項のリスト
        """
        if gap_df is None:
            gap_df = self.gap_df
            
        if gap_df is None:
            raise ValueError("ギャップが計算されていません。先にcalculate_gapを実行してください。")
        
        # 上位N件を抽出
        top_gaps = gap_df.head(top_n)
        
        recommendations = []
        for _, row in top_gaps.iterrows():
            recommendation = {
                "competence_code": row["力量コード"],
                "competence_name": row["力量名"],
                "gap": row["保有率ギャップ"],
                "gap_rate": row["保有率ギャップ率"],
                "current_rate": row["現在保有率"],
                "target_rate": row["目標保有率"],
                "priority": "高" if row["保有率ギャップ率"] >= 0.5 else "中" if row["保有率ギャップ率"] >= 0.3 else "低",
                "recommendation": self._generate_recommendation_text(row)
            }
            recommendations.append(recommendation)
        
        return recommendations
    
    def _generate_recommendation_text(self, row: pd.Series) -> str:
        """推奨テキストを生成"""
        gap_pct = row["保有率ギャップ率"] * 100
        current_pct = row["現在保有率"] * 100
        target_pct = row["目標保有率"] * 100
        
        text = f"現在{current_pct:.1f}%の保有率を{target_pct:.1f}%まで引き上げる必要があります"
        
        if row["レベルギャップ"] > 0:
            text += f"（レベルも{row['レベルギャップ']:.1f}向上が必要）"
        
        return text
