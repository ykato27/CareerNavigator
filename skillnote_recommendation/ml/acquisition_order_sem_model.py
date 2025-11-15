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
            logger.info("_fit_sem()を実行中...")
            sem_model, path_coefs = self._fit_sem(min_competences_per_stage)
            logger.info(f"_fit_sem()完了: sem_model={sem_model is not None}, path_coefs={path_coefs}")

            if sem_model is not None:
                self.sem_model = sem_model
                self.path_coefficients = path_coefs
                logger.info("✅ SEMモデル学習完了")
                logger.info(f"   パス係数: {[f'{c:.3f}' for c in path_coefs]}")

                # 潜在変数スコアを推定
                logger.info("_estimate_latent_scores()を実行中...")
                self._estimate_latent_scores()
                logger.info("_estimate_latent_scores()完了")

                self.is_fitted = True
                logger.info("is_fitted = True を設定")
            else:
                logger.warning("⚠️ SEMモデル学習失敗（データ不足）")
                return

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
        # まずデータマッピングを準備（力量コード → 力量名）
        competence_name_map = dict(
            zip(
                self.competence_master["力量コード"],
                self.competence_master["力量名"],
            )
        )

        # メンバー×力量のデータフレームを作成
        skill_matrix = self.member_competence.pivot_table(
            index="メンバーコード",
            columns="力量コード",
            values="正規化レベル",
            fill_value=0.0,
        )

        # 力量名でリネーム
        renamed_columns = {
            code: competence_name_map.get(code, code) for code in skill_matrix.columns
        }
        skill_matrix_renamed = skill_matrix.rename(columns=renamed_columns)

        # 利用可能な力量名（リネーム後のカラム名）を記録
        available_skill_names = set(skill_matrix_renamed.columns)
        logger.info(f"  利用可能な力量数: {len(available_skill_names)}")

        # 測定モデルを定義（各ステージの潜在変数）
        measurement_models = []
        used_skill_names = set()  # 既に使用した「力量名」を記録（重要：コードではなく名前）

        for stage_id in range(1, self.n_stages + 1):
            stage_name = self.acquisition_hierarchy.get_stage_name(stage_id)
            stage_skills = self.acquisition_hierarchy.get_skills_by_stage(stage_id)

            logger.info(f"  Stage {stage_id}の処理開始: {len(stage_skills)}個のスキル")

            if len(stage_skills) < min_competences_per_stage:
                logger.warning(
                    f"  ⚠️ Stage {stage_id}のスキル数不足: "
                    f"{len(stage_skills)}個 < {min_competences_per_stage}個"
                )
                continue

            # 力量名に変換（既に使用された「力量名」は除外、かつデータに存在するものだけ）
            skill_names = []
            failed_codes = []
            duplicated_names = []  # 力量名重複
            unavailable_codes = []

            for code in stage_skills:
                skill_name = competence_name_map.get(code)
                if skill_name is None:
                    failed_codes.append(code)
                    continue

                # **絶対に重要：力量名で重複チェック**
                if skill_name in used_skill_names:
                    duplicated_names.append(skill_name)
                    logger.warning(f"  ⚠️ Stage {stage_id}: '{skill_name}'は既に使用済み（前のStageで使用）")
                    continue

                # データに実際に存在するか確認
                if skill_name not in available_skill_names:
                    unavailable_codes.append((code, skill_name))
                    continue

                skill_names.append(skill_name)
                used_skill_names.add(skill_name)  # 力量名を記録

            logger.info(
                f"  Stage {stage_id}: 利用可能={len(skill_names)}個, "
                f"力量マスタ不在={len(failed_codes)}個, "
                f"データ不在={len(unavailable_codes)}個, "
                f"重複除外={len(duplicated_names)}個"
            )
            if failed_codes:
                logger.warning(
                    f"  ⚠️ Stage {stage_id}で力量マスタに不在: {failed_codes[:5]}"
                )
            if unavailable_codes:
                logger.warning(
                    f"  ⚠️ Stage {stage_id}でメンバーデータに不在: {unavailable_codes[:5]}"
                )
            if duplicated_names:
                logger.info(
                    f"  ℹ️ Stage {stage_id}で重複スキル除外: {duplicated_names[:5]}"
                )

            if len(skill_names) >= 2:
                # 重要：skill_names内の重複チェック
                unique_skill_names = list(dict.fromkeys(skill_names))  # 順序保持しながら重複除外

                logger.info(f"  Stage {stage_id}: 処理前={len(skill_names)}個, 重複除外後={len(unique_skill_names)}個")
                logger.info(f"  Stage {stage_id}の力量名: {unique_skill_names[:10]}")

                measurement_models.append(
                    MeasurementModelSpec(
                        latent_name=f"Stage_{stage_id}_{stage_name}",
                        observed_vars=unique_skill_names,
                        reference_indicator=unique_skill_names[0],
                    )
                )

                logger.info(
                    f"  Stage {stage_id} ({stage_name}): {len(unique_skill_names)}個のスキル → 測定モデル追加"
                )
            else:
                logger.warning(
                    f"  ⚠️ Stage {stage_id}: スキル数不足（{len(skill_names)}個 < 2個）"
                )

        # 全measurement_modelsの観測変数の重複チェック（最後の砦）
        all_observed_vars = []
        for mm in measurement_models:
            all_observed_vars.extend(mm.observed_vars)

        unique_vars = set(all_observed_vars)
        logger.info(f"  全Stage合計: 観測変数数={len(all_observed_vars)}, ユニーク数={len(unique_vars)}")

        if len(all_observed_vars) > len(unique_vars):
            duplicated_vars = [v for v in all_observed_vars if all_observed_vars.count(v) > 1]
            logger.error(f"  ❌ 観測変数重複検出: {set(duplicated_vars)}")
            logger.error(f"  測定モデルの詳細:")
            for i, mm in enumerate(measurement_models):
                logger.error(f"    Stage {i + 1}: {mm.observed_vars}")
            return None, []

        if len(measurement_models) < 2:
            logger.warning(
                f"  ⚠️ 測定モデルが2個未満: {len(measurement_models)}個 "
                f"（SEMには最低2個必要）"
            )
            return None, []

        # 構造モデルを定義（ステージ間の因果関係）
        structural_models = []

        for i in range(len(measurement_models) - 1):
            from_var = measurement_models[i].latent_name
            to_var = measurement_models[i + 1].latent_name

            structural_models.append(
                StructuralModelSpec(from_latent=from_var, to_latent=to_var)
            )

        # SEMモデルを学習
        try:
            logger.info(f"  UnifiedSEMEstimatorを初期化中...")
            logger.info(f"  - 測定モデル数: {len(measurement_models)}")
            logger.info(f"  - 構造モデル数: {len(structural_models)}")
            logger.info(f"  - データ形状: {skill_matrix_renamed.shape}")

            sem_model = UnifiedSEMEstimator(
                measurement_specs=measurement_models, structural_specs=structural_models
            )
            logger.info(f"  UnifiedSEMEstimator初期化完了")

            logger.info(f"  SEMモデルをfit中...")
            sem_model.fit(skill_matrix_renamed)
            logger.info(f"  SEMモデルfitは成功しました: is_fitted={sem_model.is_fitted}")

            # パス係数を抽出
            path_coefs = []
            for structural_spec in structural_models:
                param_name = (
                    f"β_{structural_spec.from_latent}→{structural_spec.to_latent}"
                )
                if param_name in sem_model.params:
                    path_coefs.append(sem_model.params[param_name].value)
                else:
                    path_coefs.append(0.0)

            logger.info(f"  パス係数抽出完了: {path_coefs}")
            return sem_model, path_coefs

        except Exception as e:
            logger.error(f"  ❌ SEMモデル学習エラー: {e}", exc_info=True)
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
