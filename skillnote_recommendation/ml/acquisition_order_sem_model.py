"""
スキル取得順序ベースSEMモデル

スキルの取得順序（初級→中級→上級）に基づいたSEMモデルを構築し、
メンバーの潜在的なスキルレベルを推定、推薦に活用します。

このモデルは実際の取得時刻データから学習する完全にデータドリブンな手法です。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

from skillnote_recommendation.ml.acquisition_order_hierarchy import (
    AcquisitionOrderHierarchy,
)
from skillnote_recommendation.ml.unified_sem_estimator import (
    UnifiedSEMEstimator,
    MeasurementModelSpec,
    StructuralModelSpec,
)

logger = logging.getLogger(__name__)


class AcquisitionOrderSEMModel:
    """
    スキル取得順序ベースの階層的SEMモデル

    スキルの平均取得順序から、初級→中級→上級の潜在変数を推定し、
    スキル推薦に活用します。

    例:
        Stage 1（初級・平均取得順序0-10）
            初級スキル（潜在変数） → [Python基礎, Git, HTML基礎]
                ↓ パス係数 β=0.65
        Stage 2（中級・平均取得順序11-20）
            中級スキル（潜在変数） → [Web開発, API設計, テスト]
                ↓ パス係数 β=0.58
        Stage 3（上級・平均取得順序21以降）
            上級スキル（潜在変数） → [システム設計, アーキテクチャ]
    """

    def __init__(
        self,
        member_competence: pd.DataFrame,
        competence_master: pd.DataFrame,
        acquisition_hierarchy: Optional[AcquisitionOrderHierarchy] = None,
        n_stages: int = 3,
    ):
        """
        Args:
            member_competence: メンバー力量データ（メンバーコード、力量コード、正規化レベル、取得日）
            competence_master: 力量マスタ（力量コード、力量名）
            acquisition_hierarchy: スキル取得順序階層（Noneの場合は自動生成）
            n_stages: ステージ数（デフォルト: 3）
        """
        self.member_competence = member_competence
        self.competence_master = competence_master
        self.n_stages = n_stages

        # 取得順序階層を構築
        if acquisition_hierarchy is None:
            self.acquisition_hierarchy = AcquisitionOrderHierarchy(
                member_competence, competence_master, n_stages=n_stages
            )
        else:
            self.acquisition_hierarchy = acquisition_hierarchy

        # SEMモデル
        self.sem_model: Optional[UnifiedSEMEstimator] = None

        # パス係数
        self.path_coefficients: List[float] = []

        # 潜在変数スコア（メンバーごと）
        self.latent_scores: Dict[str, Dict[int, float]] = {}

        self.is_fitted = False

    def fit(self, min_competences_per_stage: int = 3):
        """
        SEMモデルを学習

        Args:
            min_competences_per_stage: 各ステージで最低限必要な力量数
        """
        logger.info("=" * 60)
        logger.info("スキル取得順序SEMモデルの学習開始")
        logger.info("=" * 60)

        try:
            # SEMモデルを構築・学習
            sem_model, path_coefs = self._fit_sem(min_competences_per_stage)

            if sem_model is not None:
                self.sem_model = sem_model
                self.path_coefficients = path_coefs
                logger.info("✅ SEMモデル学習完了")
                logger.info(f"   パス係数: {[f'{c:.3f}' for c in path_coefs]}")
            else:
                logger.warning("⚠️ SEMモデル学習失敗（データ不足）")
                return

            # 潜在変数スコアを推定
            self._estimate_latent_scores()

            self.is_fitted = True

            logger.info("\n" + "=" * 60)
            logger.info("✅ スキル取得順序SEMモデルの学習完了")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"❌ SEMモデル学習エラー: {e}", exc_info=True)

    def _fit_sem(
        self, min_competences_per_stage: int
    ) -> Tuple[Optional[UnifiedSEMEstimator], List[float]]:
        """
        SEMモデルを構築・学習

        Args:
            min_competences_per_stage: 各ステージで最低限必要な力量数

        Returns:
            (学習済みSEMモデル, パス係数リスト)
        """
        # 測定モデルを定義（各ステージの潜在変数）
        measurement_models = []

        for stage_id in range(1, self.n_stages + 1):
            stage_name = self.acquisition_hierarchy.get_stage_name(stage_id)
            stage_skills = self.acquisition_hierarchy.get_skills_by_stage(stage_id)

            if len(stage_skills) < min_competences_per_stage:
                logger.warning(
                    f"  ⚠️ Stage {stage_id}のスキル数不足: "
                    f"{len(stage_skills)}個 < {min_competences_per_stage}個"
                )
                continue

            # 力量名に変換
            skill_names = []
            for code in stage_skills:
                comp_info = self.competence_master[
                    self.competence_master["力量コード"] == code
                ]
                if not comp_info.empty:
                    skill_names.append(comp_info.iloc[0]["力量名"])

            if len(skill_names) >= 2:
                measurement_models.append(
                    MeasurementModelSpec(
                        latent_variable=f"Stage_{stage_id}_{stage_name}",
                        indicators=skill_names,
                        reference_indicator=skill_names[0],
                    )
                )

                logger.info(
                    f"  Stage {stage_id} ({stage_name}): {len(skill_names)}個のスキル"
                )

        if len(measurement_models) < 2:
            logger.warning(
                f"  ⚠️ 測定モデルが2個未満: {len(measurement_models)}個 "
                f"（SEMには最低2個必要）"
            )
            return None, []

        # 構造モデルを定義（ステージ間の因果関係）
        structural_models = []

        for i in range(len(measurement_models) - 1):
            from_var = measurement_models[i].latent_variable
            to_var = measurement_models[i + 1].latent_variable

            structural_models.append(
                StructuralModelSpec(from_variable=from_var, to_variable=to_var)
            )

        # メンバー×力量のデータフレームを作成
        skill_matrix = self.member_competence.pivot_table(
            index="メンバーコード",
            columns="力量コード",
            values="正規化レベル",
            fill_value=0.0,
        )

        # 力量名でリネーム
        competence_name_map = dict(
            zip(
                self.competence_master["力量コード"],
                self.competence_master["力量名"],
            )
        )

        renamed_columns = {
            code: competence_name_map.get(code, code) for code in skill_matrix.columns
        }
        skill_matrix_renamed = skill_matrix.rename(columns=renamed_columns)

        # SEMモデルを学習
        try:
            sem_model = UnifiedSEMEstimator(
                measurement_model=measurement_models, structural_model=structural_models
            )

            sem_model.fit(skill_matrix_renamed)

            # パス係数を抽出
            path_coefs = []
            for structural_spec in structural_models:
                param_name = (
                    f"β_{structural_spec.from_variable}→{structural_spec.to_variable}"
                )
                if param_name in sem_model.params:
                    path_coefs.append(sem_model.params[param_name].value)
                else:
                    path_coefs.append(0.0)

            return sem_model, path_coefs

        except Exception as e:
            logger.error(f"  ❌ SEMモデル学習エラー: {e}")
            return None, []

    def _estimate_latent_scores(self):
        """
        各メンバーの潜在変数スコアを推定
        """
        if self.sem_model is None:
            return

        logger.info("\nメンバーの潜在変数スコアを推定中...")

        # 全メンバーの潜在変数スコアを計算
        try:
            # メンバー×力量のデータフレームを作成（学習時と同じデータで推定）
            skill_matrix = self.member_competence.pivot_table(
                index="メンバーコード",
                columns="力量コード",
                values="正規化レベル",
                fill_value=0.0,
            )

            # 力量名でリネーム
            competence_name_map = dict(
                zip(
                    self.competence_master["力量コード"],
                    self.competence_master["力量名"],
                )
            )

            renamed_columns = {
                code: competence_name_map.get(code, code) for code in skill_matrix.columns
            }
            skill_matrix_renamed = skill_matrix.rename(columns=renamed_columns)

            # predict_latent_scores()を使用して潜在変数スコアを推定
            latent_scores_df = self.sem_model.predict_latent_scores(skill_matrix_renamed)

            # メンバーごとにステージ別スコアを格納
            for member_code in latent_scores_df.index:
                self.latent_scores[member_code] = {}

                for stage_id in range(1, self.n_stages + 1):
                    stage_name = self.acquisition_hierarchy.get_stage_name(stage_id)
                    col_name = f"Stage_{stage_id}_{stage_name}"

                    if col_name in latent_scores_df.columns:
                        self.latent_scores[member_code][stage_id] = latent_scores_df.loc[
                            member_code, col_name
                        ]
                    else:
                        self.latent_scores[member_code][stage_id] = 0.0

            logger.info(
                f"✅ {len(self.latent_scores)}名の潜在変数スコア推定完了"
            )

        except Exception as e:
            logger.error(f"❌ 潜在変数スコア推定エラー: {e}")

    def get_member_latent_scores(self, member_code: str) -> Optional[Dict[int, float]]:
        """
        メンバーの潜在変数スコアを取得

        Args:
            member_code: メンバーコード

        Returns:
            {stage_id: latent_score}
        """
        return self.latent_scores.get(member_code)

    def recommend_next_skills(
        self, member_code: str, top_n: int = 10
    ) -> List[Dict]:
        """
        メンバーに次に習得すべきスキルを推薦

        Args:
            member_code: メンバーコード
            top_n: 推薦数

        Returns:
            推薦スキルのリスト
        """
        if not self.is_fitted:
            logger.warning("⚠️ モデルが学習されていません")
            return []

        # 基本推薦を取得（取得順序ベース）
        recommendations = self.acquisition_hierarchy.get_next_stage_skills(
            member_code, top_n
        )

        # 潜在変数スコアで調整
        latent_scores = self.get_member_latent_scores(member_code)

        if latent_scores:
            for rec in recommendations:
                stage_id = rec["stage"]
                latent_score = latent_scores.get(stage_id, 0.0)

                # 潜在変数スコアを考慮した優先度調整
                rec["latent_score"] = latent_score
                rec["adjusted_priority_score"] = rec["priority_score"] * (
                    1 + latent_score * 0.2
                )

            # 調整後のスコアでソート
            recommendations.sort(
                key=lambda x: x["adjusted_priority_score"], reverse=True
            )

        return recommendations

    def get_statistics(self) -> pd.DataFrame:
        """
        モデルの統計情報を取得

        Returns:
            DataFrame with statistics
        """
        return self.acquisition_hierarchy.get_statistics()
