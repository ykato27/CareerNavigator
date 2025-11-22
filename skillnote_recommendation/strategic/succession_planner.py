"""
後継者計画（サクセッションプラン）

役職ベースで後継者候補を特定し、準備度を評価
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SuccessionPlanner:
    """後継者計画エンジン"""
    
    def __init__(self):
        """初期化"""
        self.target_position_profile = None
        self.candidates = None
    
    def identify_critical_positions(
        self,
        members_df: pd.DataFrame,
        position_column: str = "役職",
        critical_keywords: List[str] = None
    ) -> List[str]:
        """
        重要ポジション（役職）を特定
        
        Args:
            members_df: メンバーマスタ
            position_column: 役職カラム名
            critical_keywords: 重要ポジションのキーワードリスト
            
        Returns:
            重要ポジションのリスト
        """
        if critical_keywords is None:
            # デフォルトの重要ポジションキーワード
            critical_keywords = ["マネージャー", "部長", "課長", "リーダー", "責任者", "GM"]
        
        if position_column not in members_df.columns:
            logger.warning(f"{position_column}カラムが見つかりません")
            return []
        
        # ユニークな役職リストを取得
        positions = members_df[position_column].dropna().unique().tolist()
        
        # キーワードに一致する役職を抽出
        critical_positions = []
        for pos in positions:
            if any(keyword in str(pos) for keyword in critical_keywords):
                critical_positions.append(pos)
        
        logger.info(f"重要ポジション数: {len(critical_positions)}")
        return critical_positions
    
    def calculate_position_skill_profile(
        self,
        target_position: str,
        members_df: pd.DataFrame,
        member_competence_df: pd.DataFrame,
        competence_master_df: pd.DataFrame,
        position_column: str = "役職"
    ) -> pd.DataFrame:
        """
        特定役職の現保有者のスキルプロファイルを計算
        
        Args:
            target_position: 対象役職
            members_df: メンバーマスタ
            member_competence_df: メンバー習得力量データ
            competence_master_df: 力量マスタ
            position_column: 役職カラム名
            
        Returns:
            スキルプロファイルDataFrame
            - 力量コード
            - 力量名
            - 平均レベル
            - 保有率（その役職内での）
        """
        # 対象役職のメンバーを抽出
        target_members = members_df[members_df[position_column] == target_position]["メンバーコード"].tolist()
        
        if len(target_members) == 0:
            logger.warning(f"{target_position}のメンバーが見つかりません")
            return pd.DataFrame()
        
        logger.info(f"{target_position}のメンバー数: {len(target_members)}人")
        
        # 対象メンバーのスキルデータを抽出
        target_skills = member_competence_df[
            member_competence_df["メンバーコード"].isin(target_members)
        ]
        
        # スキルごとに集計
        skill_profile = target_skills.groupby("力量コード").agg(
            holder_count=("メンバーコード", "nunique"),
            total_records=("メンバーコード", "count")
        ).reset_index()
        
        # レベル情報がある場合は平均レベルを計算
        if "レベル" in target_skills.columns:
            # レベルを数値に変換
            target_skills_clean = target_skills.copy()
            target_skills_clean["レベル_数値"] = pd.to_numeric(
                target_skills_clean["レベル"], errors='coerce'
            ).fillna(1.0)
            
            avg_levels = target_skills_clean.groupby("力量コード")["レベル_数値"].mean()
            skill_profile["平均レベル"] = skill_profile["力量コード"].map(avg_levels).fillna(1.0).round(2)
        else:
            skill_profile["平均レベル"] = 1.0
        
        # 力量マスタとマージして力量名を取得
        skill_profile = skill_profile.merge(
            competence_master_df[["力量コード", "力量名"]],
            on="力量コード",
            how="left"
        )
        
        # 保有率を計算（その役職内での保有率）
        skill_profile["保有率"] = skill_profile["holder_count"] / len(target_members)
        
        # カラムを整理
        profile = skill_profile[[
            "力量コード", "力量名", "平均レベル", "保有率"
        ]].sort_values("保有率", ascending=False).reset_index(drop=True)
        
        self.target_position_profile = profile
        return profile
    
    def find_succession_candidates(
        self,
        target_position: str,
        members_df: pd.DataFrame,
        member_competence_df: pd.DataFrame,
        competence_master_df: pd.DataFrame,
        position_column: str = "役職",
        grade_column: str = "職能・等級",
        exclude_current_holders: bool = True,
        max_candidates: int = 20
    ) -> pd.DataFrame:
        """
        後継者候補を検索
        
        Args:
            target_position: 対象役職
            members_df: メンバーマスタ
            member_competence_df: メンバー習得力量データ
            competence_master_df: 力量マスタ
            position_column: 役職カラム名
            grade_column: 等級カラム名
            exclude_current_holders: 現在の保有者を除外するか
            max_candidates: 最大候補者数
            
        Returns:
            候補者DataFrame
            - メンバーコード
            - メンバー名
            - 現在の役職
            - 現在の等級
            - 準備度スコア
            - スキルマッチ度
            - 保有スキル数
            - 不足スキル数
        """
        # 対象役職のスキルプロファイルを計算（まだない場合）
        if self.target_position_profile is None:
            self.calculate_position_skill_profile(
                target_position, members_df, member_competence_df, competence_master_df, position_column
            )
        
        target_profile = self.target_position_profile
        
        if target_profile.empty:
            return pd.DataFrame()
        
        # 候補者プールを作成
        if exclude_current_holders:
            # 現在の対象役職保有者を除外
            candidate_pool = members_df[members_df[position_column] != target_position].copy()
        else:
            candidate_pool = members_df.copy()
        
        # 各候補者のスコアを計算
        candidates_scores = []
        
        for _, member in candidate_pool.iterrows():
            member_code = member["メンバーコード"]
            member_name = member.get("メンバー名", member_code)
            current_position = member.get(position_column, "")
            current_grade = member.get(grade_column, "")
            
            # メンバーのスキルを取得
            member_skills = member_competence_df[
                member_competence_df["メンバーコード"] == member_code
            ]
            
            if member_skills.empty:
                continue
            
            # スキルマッチング計算
            match_result = self._calculate_skill_match(
                member_skills, target_profile, competence_master_df
            )
            
            # 準備度スコア計算
            readiness_score = self._calculate_readiness_score(
                match_result, current_position, current_grade
            )
            
            candidates_scores.append({
                "メンバーコード": member_code,
                "メンバー名": member_name,
                "現在の役職": current_position,
                "現在の等級": current_grade,
                "準備度スコア": readiness_score,
                "スキルマッチ度": match_result["match_rate"],
                "保有スキル数": match_result["matched_skills"],
                "不足スキル数": match_result["missing_skills"],
                "総合スコア詳細": match_result
            })
        
        # DataFrameに変換
        candidates_df = pd.DataFrame(candidates_scores)
        
        if candidates_df.empty:
            return pd.DataFrame()
        
        # 準備度スコア順にソート
        candidates_df = candidates_df.sort_values("準備度スコア", ascending=False).reset_index(drop=True)
        
        # 上位N件のみ返す
        candidates_df = candidates_df.head(max_candidates)
        
        self.candidates = candidates_df
        return candidates_df
    
    def _calculate_skill_match(
        self,
        member_skills: pd.DataFrame,
        target_profile: pd.DataFrame,
        competence_master_df: pd.DataFrame
    ) -> Dict:
        """
        スキルマッチングを計算
        
        Returns:
            マッチング結果の辞書
        """
        # メンバーが保有しているスキルコードのセット
        member_skill_codes = set(member_skills["力量コード"].unique())
        
        # ターゲットプロファイルのスキルコードのセット（保有率30%以上を必要スキルとする）
        required_skills = set(target_profile[target_profile["保有率"] >= 0.3]["力量コード"].unique())
        
        # マッチしたスキル数
        matched = member_skill_codes.intersection(required_skills)
        matched_count = len(matched)
        
        # 不足スキル数
        missing = required_skills - member_skill_codes
        missing_count = len(missing)
        
        # マッチ率
        total_required = len(required_skills)
        match_rate = matched_count / total_required if total_required > 0 else 0.0
        
        return {
            "match_rate": round(match_rate, 3),
            "matched_skills": matched_count,
            "missing_skills": missing_count,
            "total_required": total_required,
            "matched_skill_codes": list(matched),
            "missing_skill_codes": list(missing)
        }
    
    def _calculate_readiness_score(
        self,
        match_result: Dict,
        current_position: str,
        current_grade: str
    ) -> float:
        """
        準備度スコアを計算
        
        スキルマッチ度: 60%
        等級の近さ: 20%
        スキル数: 20%
        
        Returns:
            0.0-1.0のスコア
        """
        # スキルマッチ度（60%）
        skill_score = match_result["match_rate"] * 0.6
        
        # 等級スコア（20%） - 簡易的に等級文字列の類似度で計算
        # TODO: より詳細な等級階層があれば活用
        grade_score = 0.1  # デフォルト値
        
        # スキル数スコア（20%） - 多いほうが良い
        total_skills = match_result["matched_skills"] + match_result["missing_skills"]
        skill_count_score = min(match_result["matched_skills"] / max(total_skills, 1), 1.0) * 0.2
        
        # 総合スコア
        readiness_score = skill_score + grade_score + skill_count_score
        
        return round(readiness_score, 3)
    
    def estimate_development_timeline(
        self,
        missing_skills: int
    ) -> str:
        """
        育成期間を推定
        
        Args:
            missing_skills: 不足スキル数
            
        Returns:
            推定期間の文字列
        """
        if missing_skills == 0:
            return "即座"
        elif missing_skills <= 3:
            return "3-6ヶ月"
        elif missing_skills <= 6:
            return "6-12ヶ月"
        elif missing_skills <= 10:
            return "1-2年"
        else:
            return "2年以上"
    
    def generate_development_roadmap(
        self,
        candidate: pd.Series,
        target_profile: pd.DataFrame,
        competence_master_df: pd.DataFrame,
        member_competence_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        候補者向けの育成ロードマップを生成
        
        Args:
            candidate: 候補者のSeries
            target_profile: 目標スキルプロファイル
            competence_master_df: 力量マスタ
            member_competence_df: メンバー習得力量データ
            
        Returns:
            育成ロードマップDataFrame
            - スキル名
            - 現在レベル
            - 目標レベル
            - 優先度（High/Medium/Low）
            - 推定習得期間（月）
            - マイルストーン（3ヶ月/6ヶ月/1年後）
        """
        if "総合スコア詳細" not in candidate:
            return pd.DataFrame()
        
        detail = candidate["総合スコア詳細"]
        missing_skill_codes = detail.get("missing_skill_codes", [])
        
        if not missing_skill_codes:
            return pd.DataFrame()
        
        # 不足スキルの情報を取得
        missing_skills_info = target_profile[
            target_profile["力量コード"].isin(missing_skill_codes)
        ].copy()
        
        # 力量名を追加
        missing_skills_info = missing_skills_info.merge(
            competence_master_df[["力量コード", "力量名", "力量タイプ"]],
            on="力量コード",
            how="left",
            suffixes=("", "_master")
        )
        
        # 力量名の重複を解決
        if "力量名_master" in missing_skills_info.columns:
            missing_skills_info["力量名"] = missing_skills_info["力量名_master"].fillna(
                missing_skills_info["力量名"]
            )
            missing_skills_info = missing_skills_info.drop(columns=["力量名_master"])
        
        # メンバーの現在のスキルレベルを取得
        member_code = candidate["メンバーコード"]
        member_skills = member_competence_df[
            member_competence_df["メンバーコード"] == member_code
        ].set_index("力量コード")["レベル"].to_dict()
        
        # ロードマップを作成
        roadmap_data = []
        
        for _, skill_row in missing_skills_info.iterrows():
            skill_code = skill_row["力量コード"]
            skill_name = skill_row["力量名"]
            target_level = skill_row.get("平均レベル", 3)  # デフォルト3
            coverage = skill_row.get("保有率", 0.5)  # 保有率
            
            # 現在のレベル（保有していない場合は0）
            current_level = member_skills.get(skill_code, 0)
            if pd.notna(current_level):
                current_level = pd.to_numeric(current_level, errors='coerce')
                if pd.isna(current_level):
                    current_level = 0
            else:
                current_level = 0
            
            # 優先度を計算（保有率が高いほど優先度が高い）
            if coverage >= 0.7:
                priority = "High"
                priority_score = 3
            elif coverage >= 0.4:
                priority = "Medium"
                priority_score = 2
            else:
                priority = "Low"
                priority_score = 1
            
            # 推定習得期間（月） - レベル差と優先度に基づく
            level_gap = max(target_level - current_level, 1)
            estimated_months = int(level_gap * 2)  # 1レベルあたり2ヶ月
            
            # マイルストーンを決定
            if estimated_months <= 3:
                milestone = "3ヶ月後"
            elif estimated_months <= 6:
                milestone = "6ヶ月後"
            elif estimated_months <= 12:
                milestone = "1年後"
            else:
                milestone = "1年以降"
            
            roadmap_data.append({
                "力量コード": skill_code,
                "力量名": skill_name,
                "力量タイプ": skill_row.get("力量タイプ", ""),
                "現在レベル": current_level,
                "目標レベル": round(target_level, 1),
                "レベルギャップ": round(target_level - current_level, 1),
                "優先度": priority,
                "優先度スコア": priority_score,
                "推定習得期間（月）": estimated_months,
                "マイルストーン": milestone,
                "保有率": round(coverage * 100, 1)
            })
        
        roadmap_df = pd.DataFrame(roadmap_data)
        
        # 優先度スコア順、次に保有率順にソート
        roadmap_df = roadmap_df.sort_values(
            ["優先度スコア", "保有率"],
            ascending=[False, False]
        ).reset_index(drop=True)
        
        return roadmap_df
    
    def analyze_candidate_strengths(
        self,
        candidate: pd.Series,
        member_competence_df: pd.DataFrame,
        competence_master_df: pd.DataFrame,
        members_df: pd.DataFrame
    ) -> Dict:
        """
        候補者の強みを分析
        
        Args:
            candidate: 候補者のSeries
            member_competence_df: メンバー習得力量データ
            competence_master_df: 力量マスタ
            members_df: メンバーマスタ
            
        Returns:
            強み分析の辞書
        """
        member_code = candidate["メンバーコード"]
        
        # メンバーのスキルを取得
        member_skills = member_competence_df[
            member_competence_df["メンバーコード"] == member_code
        ]
        
        # 力量タイプ別にスキル数を集計
        type_counts = {}
        
        if "力量タイプ" in competence_master_df.columns:
            skill_by_type = member_skills.merge(
                competence_master_df[["力量コード", "力量タイプ"]],
                on="力量コード",
                how="left"
            )
            type_counts = skill_by_type["力量タイプ"].value_counts().to_dict()
        else:
            # 力量タイプカラムが存在しない場合は全てを「全スキル」としてカウント
            type_counts = {"全スキル": len(member_skills)}
        
        # 最も多いタイプを特定
        if type_counts:
            top_type = max(type_counts, key=type_counts.get)
            top_type_count = type_counts[top_type]
        else:
            top_type = "不明"
            top_type_count = 0
        
        # 全体の中での位置（スキル数の百分位数）
        all_member_skill_counts = member_competence_df.groupby("メンバーコード").size()
        member_skill_count = len(member_skills)
        percentile = (all_member_skill_counts < member_skill_count).sum() / len(all_member_skill_counts) * 100
        
        return {
            "総スキル数": member_skill_count,
            "スキル数百分位": round(percentile, 1),
            "最強カテゴリ": top_type,
            "最強カテゴリスキル数": top_type_count,
            "カテゴリ別内訳": type_counts
        }
    
    def simulate_succession_impact(
        self,
        candidate: pd.Series,
        members_df: pd.DataFrame,
        member_competence_df: pd.DataFrame,
        competence_master_df: pd.DataFrame,
        position_column: str = "役職"
    ) -> Dict:
        """
        後継者選択の組織への影響をシミュレーション（What-If分析）
        
        Args:
            candidate: 選択された候補者のSeries
            members_df: メンバーマスタ
            member_competence_df: メンバー習得力量データ
            competence_master_df: 力量マスタ
            position_column: 役職カラム名
            
        Returns:
            影響分析の辞書
        """
        member_code = candidate["メンバーコード"]
        member_name = candidate["メンバー名"]
        current_position = candidate.get("現在の役職", "")
        
        # 現在のポジションが空くことによる影響
        current_position_holders = members_df[
            members_df[position_column] == current_position
        ]
        
        # 連鎖分析: 現在のポジションの後継者候補も検索
        cascade_candidates = None
        cascade_profile = None
        
        if current_position and len(current_position_holders) > 0:
            try:
                # 現在のポジションのスキルプロファイル計算
                cascade_profile = self.calculate_position_skill_profile(
                    current_position,
                    members_df,
                    member_competence_df,
                    competence_master_df,
                    position_column
                )
                
                # 候補者自身を除外して後継者検索
                cascade_candidates = self.find_succession_candidates(
                    current_position,
                    members_df,
                    member_competence_df,
                    competence_master_df,
                    position_column=position_column,
                    exclude_current_holders=True,
                    max_candidates=5
                )
                
                # 選択した候補者自身も結果から除外
                if cascade_candidates is not None and not cascade_candidates.empty:
                    cascade_candidates = cascade_candidates[
                        cascade_candidates["メンバーコード"] != member_code
                    ]
                
            except Exception as e:
                logger.warning(f"連鎖分析でエラー: {e}")
                cascade_candidates = pd.DataFrame()
        
        # 候補者のスキル分析
        member_skills = member_competence_df[
            member_competence_df["メンバーコード"] == member_code
        ]
        
        # スキルタイプ別集計
        skill_type_counts = {}
        
        if "力量タイプ" in competence_master_df.columns:
            skill_by_type = member_skills.merge(
                competence_master_df[["力量コード", "力量タイプ"]],
                on="力量コード",
                how="left"
            )
            skill_type_counts = skill_by_type["力量タイプ"].value_counts().to_dict()
        else:
            # 力量タイプカラムが存在しない場合はスキル数のみ
            skill_type_counts = {"全スキル": len(member_skills)}
        
        # 影響サマリー
        impact_summary = {
            "候補者名": member_name,
            "現在のポジション": current_position,
            "現在のポジション保有者数": len(current_position_holders),
            "連鎖的な影響": {
                "空くポジション": current_position,
                "後継者候補数": len(cascade_candidates) if cascade_candidates is not None else 0,
                "後継者候補": cascade_candidates.head(3)["メンバー名"].tolist() if cascade_candidates is not None and not cascade_candidates.empty else [],
                "連鎖候補詳細": cascade_candidates
            },
            "移動するスキル": {
                "総スキル数": len(member_skills),
                "スキルタイプ別": skill_type_counts
            }
        }
        
        return impact_summary
    
    def calculate_organization_balance_score(
        self,
        members_df: pd.DataFrame,
        member_competence_df: pd.DataFrame,
        group_column: str = "職種"
    ) -> Dict:
        """
        組織全体のスキルバランススコアを計算
        
        Args:
            members_df: メンバーマスタ
            member_competence_df: メンバー習得力量データ
            group_column: グループカラム（職種など）
            
        Returns:
            バランススコアの辞書
        """
        # グループ別のメンバー数とスキル数を集計
        group_stats = []
        
        for group in members_df[group_column].dropna().unique():
            group_members = members_df[members_df[group_column] == group]["メンバーコード"].tolist()
            
            group_skills = member_competence_df[
                member_competence_df["メンバーコード"].isin(group_members)
            ]
            
            total_skills = len(group_skills)
            unique_skills = group_skills["力量コード"].nunique()
            avg_skills_per_member = total_skills / len(group_members) if len(group_members) > 0 else 0
            
            group_stats.append({
                "グループ": group,
                "メンバー数": len(group_members),
                "総スキル数": total_skills,
                "ユニークスキル数": unique_skills,
                "平均スキル数/人": round(avg_skills_per_member, 2)
            })
        
        group_stats_df = pd.DataFrame(group_stats)
        
        # バランススコアを計算（標準偏差が小さいほど良い）
        if len(group_stats_df) > 0:
            skill_variance = group_stats_df["平均スキル数/人"].std()
            # スコアは0-100で、標準偏差が小さいほど高い
            balance_score = max(0, 100 - skill_variance * 10)
        else:
            balance_score = 0
        
        return {
            "バランススコア": round(balance_score, 1),
            "グループ別統計": group_stats_df,
            "スキル分散": round(skill_variance, 2) if len(group_stats_df) > 0 else 0
        }
