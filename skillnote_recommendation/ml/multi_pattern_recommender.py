"""
マルチパターン推薦器

3つのキャリアパターン（類似、異なる1、異なる2）それぞれから
力量を推薦する。各パターンの参考人物の平均プロファイルを使用。
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass

from skillnote_recommendation.core.models import Recommendation
from skillnote_recommendation.ml.career_pattern_classifier import (
    CareerPatternClassifier,
    CareerPatternGroup,
)


@dataclass
class PatternRecommendation:
    """キャリアパターン別推薦結果"""

    pattern_name: str  # 'similar', 'different1', 'different2'
    pattern_label: str  # 表示用ラベル
    reference_persons: List[
        Dict[str, str]
    ]  # 参考人物情報 [{'code': ..., 'name': ..., 'similarity': ...}]
    recommendations: List[Recommendation]  # 推薦力量リスト
    avg_profile_used: bool  # 平均プロファイルを使用したか
    message: str  # メッセージ（参考人物が少ない場合など）
    filtered_count: int = 0  # フィルタリング後の参考人物数（自分より優秀な人）
    total_count: int = 0  # フィルタリング前の総メンバー数


class MultiPatternRecommender:
    """マルチパターン推薦器"""

    def __init__(
        self,
        classifier: CareerPatternClassifier,
        competence_master: pd.DataFrame,
        member_competence: pd.DataFrame,
        mf_model,
    ):
        """
        初期化

        Args:
            classifier: キャリアパターン分類器
            competence_master: 力量マスタ
            member_competence: メンバー習得力量データ
            mf_model: Matrix Factorizationモデル
        """
        self.classifier = classifier
        self.competence_master = competence_master
        self.member_competence = member_competence
        self.mf_model = mf_model

    def recommend_by_patterns(
        self,
        target_member_code: str,
        top_k_per_pattern: Dict[str, int] = None,
        competence_type: List[str] = None,
    ) -> Dict[str, PatternRecommendation]:
        """
        キャリアパターンごとに推薦を生成

        Args:
            target_member_code: 対象メンバーコード
            top_k_per_pattern: 各パターンでの推薦件数
                {'similar': 5, 'different1': 5, 'different2': 5}
            competence_type: 推薦する力量タイプのリスト（例: ['SKILL', 'EDUCATION']）
                Noneの場合は全てのタイプを推薦

        Returns:
            {'similar': PatternRecommendation, 'different1': ..., 'different2': ...}
        """
        if top_k_per_pattern is None:
            top_k_per_pattern = {"similar": 5, "different1": 5, "different2": 5}

        # キャリアパターンに分類
        groups = self.classifier.classify_career_patterns(target_member_code)

        # 対象メンバーの既習得力量を取得
        acquired_competences = self._get_acquired_competences(target_member_code)

        # 各パターンで推薦を生成
        results = {}
        for pattern_name, group in groups.items():
            top_k = top_k_per_pattern.get(pattern_name, 5)

            recommendation = self._recommend_for_group(
                target_member_code=target_member_code,
                group=group,
                acquired_competences=acquired_competences,
                top_k=top_k,
                competence_type=competence_type,
            )

            results[pattern_name] = recommendation

        return results

    def _recommend_for_group(
        self,
        target_member_code: str,
        group: CareerPatternGroup,
        acquired_competences: List[str],
        top_k: int,
        competence_type: List[str] = None,
    ) -> PatternRecommendation:
        """
        特定のキャリアパターングループに対して推薦を生成

        Args:
            target_member_code: 対象メンバーコード
            group: キャリアパターングループ
            acquired_competences: 既習得力量のリスト
            top_k: 推薦件数
            competence_type: 推薦する力量タイプのリスト（例: ['SKILL', 'EDUCATION']）
                Noneの場合は全てのタイプを推薦

        Returns:
            PatternRecommendation
        """
        # 参考人物が少ない場合
        if len(group.member_codes) < self.classifier.min_persons_per_group:
            return PatternRecommendation(
                pattern_name=group.pattern_name,
                pattern_label=group.pattern_label,
                reference_persons=[],
                recommendations=[],
                avg_profile_used=False,
                message=f"参考人物が{self.classifier.min_persons_per_group}名未満のため、推薦をスキップしました。",
                filtered_count=0,
                total_count=len(group.member_codes),
            )

        # 対象メンバーの総合スキルレベルを取得
        target_total_level = self._get_member_total_skill_level(target_member_code)

        # グループ内で対象者より総合スキルレベルが高いメンバーのみをフィルタリング
        filtered_codes = []
        filtered_names = []
        filtered_sims = []

        total_count = len(group.member_codes)

        for code, name, sim in zip(group.member_codes, group.member_names, group.similarities):
            member_total_level = self._get_member_total_skill_level(code)
            # 対象者より総合スキルレベルが高いメンバーのみを選定
            if member_total_level > target_total_level:
                filtered_codes.append(code)
                filtered_names.append(name)
                filtered_sims.append(sim)

        # フィルタリング後のメンバーが少ない場合
        if len(filtered_codes) < self.classifier.min_persons_per_group:
            return PatternRecommendation(
                pattern_name=group.pattern_name,
                pattern_label=group.pattern_label,
                reference_persons=[],
                recommendations=[],
                avg_profile_used=False,
                message=f"あなたより総合スキルレベルが高い参考人物が{self.classifier.min_persons_per_group}名未満のため、推薦をスキップしました。",
                filtered_count=len(filtered_codes),
                total_count=total_count,
            )

        # フィルタリング後の参考人物情報を整形
        reference_persons = [
            {"code": code, "name": name, "similarity": f"{sim:.2f}"}
            for code, name, sim in zip(filtered_codes, filtered_names, filtered_sims)
        ]

        # フィルタリング後のグループの平均プロファイルを計算
        avg_profile = self._calculate_average_profile(filtered_codes)

        # 平均プロファイルを使って力量スコアを予測
        competence_scores = self._predict_competences_from_profile(
            avg_profile, exclude_competences=acquired_competences
        )

        # Recommendationオブジェクトに変換
        # 力量タイプでフィルタリングする場合、十分な数を確保するため多めに候補を取得
        recommendations = []
        for competence_code, score in competence_scores:
            # すでに十分な数の推薦を取得した場合は終了
            if len(recommendations) >= top_k:
                break

            comp_info = self.competence_master[
                self.competence_master["力量コード"] == competence_code
            ]

            if len(comp_info) > 0:
                comp_info = comp_info.iloc[0]

                # 力量タイプでフィルタリング
                if competence_type is not None:
                    comp_type = comp_info["力量タイプ"]
                    if comp_type not in competence_type:
                        continue  # 指定されたタイプでない場合はスキップ

                # スコアを0-10に正規化
                priority_score = self._normalize_score(score, [s for _, s in competence_scores])

                rec = Recommendation(
                    competence_code=competence_code,
                    competence_name=comp_info["力量名"],
                    competence_type=comp_info["力量タイプ"],
                    category=comp_info.get("力量カテゴリー名", ""),
                    priority_score=priority_score,
                    category_importance=0.0,
                    acquisition_ease=0.0,
                    popularity=0.0,
                    reason=self._generate_reason(
                        comp_info,
                        group,
                        score,
                        filtered_count=len(filtered_codes),
                        total_count=total_count,
                    ),
                    reference_persons=[],  # 個別の参考人物は使用しない
                )
                recommendations.append(rec)

        return PatternRecommendation(
            pattern_name=group.pattern_name,
            pattern_label=group.pattern_label,
            reference_persons=reference_persons,
            recommendations=recommendations,
            avg_profile_used=True,
            message="",
            filtered_count=len(filtered_codes),
            total_count=total_count,
        )

    def _calculate_average_profile(self, member_codes: List[str]) -> np.ndarray:
        """
        複数メンバーの潜在因子の平均を計算

        Args:
            member_codes: メンバーコードのリスト

        Returns:
            平均潜在因子ベクトル
        """
        profiles = []

        for member_code in member_codes:
            if member_code in self.mf_model.member_index:
                profile = self.mf_model.get_member_factors(member_code)
                profiles.append(profile)

        if len(profiles) == 0:
            # デフォルト：ゼロベクトル
            return np.zeros(self.mf_model.n_components)

        # 平均を計算
        avg_profile = np.mean(profiles, axis=0)
        return avg_profile

    def _predict_competences_from_profile(
        self, profile: np.ndarray, exclude_competences: List[str]
    ) -> List[Tuple[str, float]]:
        """
        プロファイル（潜在因子ベクトル）から力量スコアを予測

        Args:
            profile: 潜在因子ベクトル (n_components,)
            exclude_competences: 除外する力量コードのリスト

        Returns:
            (力量コード, スコア) のリスト（スコア降順）
        """
        # profile × H で力量スコアを計算
        # profile: (n_components,)
        # H: (n_components, n_competences)
        # scores: (n_competences,)
        scores = profile @ self.mf_model.H

        # 力量コードと対応付け
        competence_scores = list(zip(self.mf_model.competence_codes, scores))

        # 除外する力量をフィルタ
        competence_scores = [
            (code, score) for code, score in competence_scores if code not in exclude_competences
        ]

        # スコアでソート（降順）
        competence_scores.sort(key=lambda x: x[1], reverse=True)

        return competence_scores

    def _get_acquired_competences(self, member_code: str) -> List[str]:
        """対象メンバーの既習得力量を取得"""
        acquired = (
            self.member_competence[self.member_competence["メンバーコード"] == member_code][
                "力量コード"
            ]
            .unique()
            .tolist()
        )
        return acquired

    def _get_member_total_skill_level(self, member_code: str) -> float:
        """メンバーの総合スキルレベル（全保有力量の正規化レベルの合計）を取得"""
        member_comp = self.member_competence[
            self.member_competence["メンバーコード"] == member_code
        ]
        if len(member_comp) == 0:
            return 0.0
        return float(member_comp["正規化レベル"].sum())

    def _normalize_score(self, score: float, all_scores: List[float]) -> float:
        """スコアを0-10に正規化"""
        if not all_scores:
            return 5.0

        min_s, max_s = min(all_scores), max(all_scores)
        if max_s == min_s:
            return 5.0

        return round(((score - min_s) / (max_s - min_s)) * 10, 2)

    def _generate_reason(
        self,
        competence_info: pd.Series,
        group: CareerPatternGroup,
        score: float,
        filtered_count: int = None,
        total_count: int = None,
    ) -> str:
        """推薦理由を生成"""
        comp_name = competence_info["力量名"]
        comp_type = competence_info["力量タイプ"]
        pattern_label = group.pattern_label

        # フィルタリング後のメンバー数を使用（指定がない場合は全体）
        member_count = filtered_count if filtered_count is not None else len(group.member_codes)

        if group.pattern_name == "similar":
            reason = (
                f"**{comp_name}** は、あなたと類似したキャリアパスを持ち、かつあなたより総合スキルレベルが高い"
                f"{member_count}名のメンバーが習得している力量です。あなたのキャリアにも適合する可能性が高いです。"
            )
        elif group.pattern_name == "different1":
            reason = (
                f"**{comp_name}** は、あなたとやや異なるキャリアパスを持ち、かつあなたより総合スキルレベルが高い"
                f"{member_count}名のメンバーが習得している力量です。キャリアの幅を広げるのに適しています。"
            )
        else:  # different2
            reason = (
                f"**{comp_name}** は、あなたと大きく異なるキャリアパスを持ち、かつあなたより総合スキルレベルが高い"
                f"{member_count}名のメンバーが習得している力量です。新しい専門領域への挑戦に適しています。"
            )

        return reason


def create_multi_pattern_recommender(
    classifier: CareerPatternClassifier,
    competence_master: pd.DataFrame,
    member_competence: pd.DataFrame,
    mf_model,
) -> MultiPatternRecommender:
    """
    マルチパターン推薦器を作成

    Args:
        classifier: キャリアパターン分類器
        competence_master: 力量マスタ
        member_competence: メンバー習得力量データ
        mf_model: Matrix Factorizationモデル

    Returns:
        MultiPatternRecommender
    """
    return MultiPatternRecommender(
        classifier=classifier,
        competence_master=competence_master,
        member_competence=member_competence,
        mf_model=mf_model,
    )
