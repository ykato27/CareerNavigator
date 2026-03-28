"""
コンテンツベース推薦システム

メンバーの属性（職種・等級）と力量の属性（カテゴリ・タイプ）を活用した
コンテンツベースの推薦を実施
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .feature_engineering import FeatureEngineer
from ..core.models import Recommendation


class ContentBasedRecommender:
    """コンテンツベース推薦システム

    メンバーと力量の属性情報を活用し、
    特徴量の類似度に基づいて推薦を行う
    """

    def __init__(
        self,
        feature_engineer: FeatureEngineer,
        member_master: pd.DataFrame,
        competence_master: pd.DataFrame,
        member_competence: pd.DataFrame,
    ):
        """
        Args:
            feature_engineer: 特徴量エンジニア
            member_master: メンバーマスタ
            competence_master: 力量マスタ
            member_competence: メンバー習得力量
        """
        self.fe = feature_engineer
        self.member_master = member_master
        self.competence_master = competence_master
        self.member_competence = member_competence

        # カラム名のマッピングを設定
        self._setup_column_mappings()

        # 同じ職種・等級のメンバープロファイルを作成
        self._build_role_grade_profiles()

        print("\nコンテンツベース推薦システム初期化完了")

    def _setup_column_mappings(self):
        """カラム名のマッピングを設定（日本語と英語の両方に対応）"""
        # メンバーマスタのカラム
        self.member_code_col = (
            "メンバーコード" if "メンバーコード" in self.member_master.columns else "member_code"
        )

        # メンバー習得力量のカラム
        self.mc_member_code_col = (
            "メンバーコード"
            if "メンバーコード" in self.member_competence.columns
            else "member_code"
        )
        self.mc_competence_code_col = (
            "力量コード" if "力量コード" in self.member_competence.columns else "competence_code"
        )
        self.mc_level_col = "レベル" if "レベル" in self.member_competence.columns else "level"

        # 力量マスタのカラム
        self.comp_code_col = (
            "力量コード" if "力量コード" in self.competence_master.columns else "competence_code"
        )
        self.comp_type_col = (
            "力量タイプ" if "力量タイプ" in self.competence_master.columns else "competence_type"
        )
        self.comp_category_col = (
            "力量カテゴリー名"
            if "力量カテゴリー名" in self.competence_master.columns
            else "category"
        )
        self.comp_name_col = "力量名" if "力量名" in self.competence_master.columns else "name"

    def _build_role_grade_profiles(self):
        """職種・等級ごとのプロファイルを構築

        同じ職種・等級のメンバーが習得している力量の傾向を分析
        """
        self.role_grade_profiles = {}

        # 職種と等級の組み合わせごとにプロファイル作成
        if "role" in self.member_master.columns and "grade" in self.member_master.columns:
            for _, member_row in self.member_master.iterrows():
                role = member_row.get("role", "UNKNOWN")
                grade = member_row.get("grade", "UNKNOWN")
                key = f"{role}_{grade}"

                if key not in self.role_grade_profiles:
                    # この職種・等級のメンバーを取得
                    similar_members = self.member_master[
                        (self.member_master["role"] == role)
                        & (self.member_master["grade"] == grade)
                    ][self.member_code_col].tolist()

                    # 彼らが習得している力量を集計
                    profile_comps = self.member_competence[
                        self.member_competence[self.mc_member_code_col].isin(similar_members)
                    ]

                    # 力量ごとの習得率と平均レベルを計算
                    comp_stats = (
                        profile_comps.groupby(self.mc_competence_code_col)
                        .agg({self.mc_level_col: ["mean", "count"]})
                        .reset_index()
                    )
                    comp_stats.columns = [self.mc_competence_code_col, "avg_level", "count"]
                    comp_stats["acquisition_rate"] = comp_stats["count"] / len(similar_members)

                    self.role_grade_profiles[key] = comp_stats

    def recommend(
        self,
        member_code: str,
        top_n: int = 10,
        competence_type: Optional[List[str]] = None,
        category_filter: Optional[str] = None,
    ) -> List[Recommendation]:
        """コンテンツベース推薦を実行

        Args:
            member_code: 対象メンバーコード
            top_n: 推薦件数
            competence_type: 力量タイプフィルタ
            category_filter: カテゴリーフィルタ

        Returns:
            推薦結果のリスト
        """
        # メンバー情報を取得
        member_info = self.member_master[self.member_master[self.member_code_col] == member_code]

        if member_info.empty:
            print(f"警告: メンバー {member_code} が見つかりません")
            return []

        member_row = member_info.iloc[0]
        role = member_row.get("role", "UNKNOWN")
        grade = member_row.get("grade", "UNKNOWN")

        # 既習得力量を取得
        acquired_comps = self.member_competence[
            self.member_competence[self.mc_member_code_col] == member_code
        ][self.mc_competence_code_col].tolist()

        # 候補力量を取得（未習得のもの）
        all_comps = self.competence_master[self.comp_code_col].tolist()
        candidate_comps = [c for c in all_comps if c not in acquired_comps]

        # フィルタリング
        if competence_type:
            candidate_comps = [
                c
                for c in candidate_comps
                if self.competence_master[self.competence_master[self.comp_code_col] == c][
                    self.comp_type_col
                ].iloc[0]
                in competence_type
            ]

        if category_filter:
            candidate_comps = [
                c
                for c in candidate_comps
                if self.competence_master[self.competence_master[self.comp_code_col] == c][
                    self.comp_category_col
                ].iloc[0]
                == category_filter
            ]

        # 親和性スコアを計算
        affinity_scores = self.fe.compute_batch_affinity(member_code, candidate_comps)

        # 同じ職種・等級のプロファイルスコアを計算
        profile_key = f"{role}_{grade}"
        profile_scores = self._compute_profile_scores(candidate_comps, profile_key)

        # 力量の人気度を計算
        popularity_scores = self._compute_popularity_scores(candidate_comps)

        # スコアを統合
        final_scores = {}
        for comp_code in candidate_comps:
            affinity = affinity_scores.get(comp_code, 0.0)
            profile = profile_scores.get(comp_code, 0.0)
            popularity = popularity_scores.get(comp_code, 0.0)

            # 重み付き平均
            final_score = (
                0.5 * affinity  # 個人の親和性
                + 0.3 * profile  # 職種・等級プロファイル適合度
                + 0.2 * popularity  # 人気度
            )

            final_scores[comp_code] = {
                "score": final_score,
                "affinity": affinity,
                "profile": profile,
                "popularity": popularity,
            }

        # スコア順にソート
        sorted_comps = sorted(final_scores.items(), key=lambda x: x[1]["score"], reverse=True)

        # Recommendationオブジェクトを生成
        recommendations = []
        for comp_code, scores in sorted_comps[:top_n]:
            comp_info = self.competence_master[
                self.competence_master[self.comp_code_col] == comp_code
            ].iloc[0]

            reason = self._generate_reason(comp_code, scores, role, grade)

            rec = Recommendation(
                competence_code=comp_code,
                competence_name=comp_info[self.comp_name_col],
                competence_type=comp_info[self.comp_type_col],
                category=comp_info.get(self.comp_category_col, "UNKNOWN"),
                priority_score=scores["score"],
                category_importance=scores["profile"],
                acquisition_ease=scores["affinity"],
                popularity=scores["popularity"],
                reason=reason,
            )

            recommendations.append(rec)

        return recommendations

    def _compute_profile_scores(
        self, competence_codes: List[str], profile_key: str
    ) -> Dict[str, float]:
        """職種・等級プロファイルに基づくスコアを計算

        Args:
            competence_codes: 力量コードのリスト
            profile_key: プロファイルキー（role_grade）

        Returns:
            {competence_code: profile_score}
        """
        if profile_key not in self.role_grade_profiles:
            return {c: 0.0 for c in competence_codes}

        profile = self.role_grade_profiles[profile_key]

        scores = {}
        for comp_code in competence_codes:
            profile_row = profile[profile[self.mc_competence_code_col] == comp_code]

            if not profile_row.empty:
                # 習得率を使用
                acquisition_rate = profile_row.iloc[0]["acquisition_rate"]
                scores[comp_code] = float(acquisition_rate)
            else:
                scores[comp_code] = 0.0

        return scores

    def _compute_popularity_scores(self, competence_codes: List[str]) -> Dict[str, float]:
        """力量の人気度スコアを計算

        Args:
            competence_codes: 力量コードのリスト

        Returns:
            {competence_code: popularity_score}
        """
        # 各力量を習得しているメンバー数をカウント
        comp_counts = self.member_competence.groupby(self.mc_competence_code_col).size()

        max_count = comp_counts.max() if not comp_counts.empty else 1

        scores = {}
        for comp_code in competence_codes:
            count = comp_counts.get(comp_code, 0)
            scores[comp_code] = float(count / max_count)

        return scores

    def _generate_reason(
        self, competence_code: str, scores: Dict[str, float], role: str, grade: str
    ) -> str:
        """推薦理由を生成

        Args:
            competence_code: 力量コード
            scores: スコア辞書
            role: 職種
            grade: 等級

        Returns:
            推薦理由
        """
        reasons = []

        if scores["affinity"] > 0.5:
            reasons.append("あなたの習得済み力量と高い親和性があります")

        if scores["profile"] > 0.3:
            reasons.append(f"{role}・{grade}のメンバーによく習得されています")

        if scores["popularity"] > 0.5:
            reasons.append("多くのメンバーが習得している人気の力量です")

        if not reasons:
            reasons.append("コンテンツベース推薦による提案")

        return "、".join(reasons)

    def explain_recommendation(self, member_code: str, competence_code: str) -> Dict:
        """推薦の詳細な説明を生成

        Args:
            member_code: メンバーコード
            competence_code: 力量コード

        Returns:
            説明情報の辞書
        """
        # メンバー情報
        member_info = self.member_master[self.member_master["member_code"] == member_code]

        if member_info.empty:
            return {"error": "メンバーが見つかりません"}

        member_row = member_info.iloc[0]
        role = member_row.get("role", "UNKNOWN")
        grade = member_row.get("grade", "UNKNOWN")

        # 親和性スコア
        affinity = self.fe.compute_member_competence_affinity(member_code, competence_code)

        # プロファイルスコア
        profile_key = f"{role}_{grade}"
        profile_scores = self._compute_profile_scores([competence_code], profile_key)
        profile_score = profile_scores.get(competence_code, 0.0)

        # 人気度スコア
        popularity_scores = self._compute_popularity_scores([competence_code])
        popularity = popularity_scores.get(competence_code, 0.0)

        # 時系列パターン
        temporal_features = self.fe.extract_temporal_patterns(member_code)

        # 関連力量
        related_comps = self.fe.get_related_competences(competence_code, top_k=5)

        return {
            "member_code": member_code,
            "competence_code": competence_code,
            "role": role,
            "grade": grade,
            "affinity_score": affinity,
            "profile_score": profile_score,
            "popularity_score": popularity,
            "temporal_features": temporal_features,
            "related_competences": related_comps,
        }
