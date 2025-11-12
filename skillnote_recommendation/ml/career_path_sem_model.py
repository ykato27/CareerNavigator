"""
キャリアパスSEMモデル

役職ごとの標準的なスキル習得パスをSEMでモデル化し、
メンバーの現在位置を推定して次のステップを推薦します。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

from skillnote_recommendation.ml.career_path_hierarchy import CareerPathHierarchy
from skillnote_recommendation.ml.unified_sem_estimator import (
    UnifiedSEMEstimator,
    MeasurementModelSpec,
    StructuralModelSpec,
)

logger = logging.getLogger(__name__)


class CareerPathSEMModel:
    """
    キャリアパスの因果構造SEMモデル

    役職ごとに、キャリアステージの進行を因果構造としてモデル化:

    例（主任の場合）:
        入門期（潜在変数） → [リーダーシップ, チーム運営, 進捗管理]
            ↓ (β=0.65)
        成長期（潜在変数） → [プロジェクト管理, リスク管理, 目標設定]
            ↓ (β=0.58)
        熟達期（潜在変数） → [複数PJ統括, リソース配分, 優先順位]
    """

    def __init__(
        self,
        member_master: pd.DataFrame,
        member_competence: pd.DataFrame,
        competence_master: pd.DataFrame,
        career_path_hierarchy: Optional[CareerPathHierarchy] = None,
    ):
        """
        Args:
            member_master: メンバーマスタ（役職情報を含む）
            member_competence: メンバー力量データ
            competence_master: 力量マスタ
            career_path_hierarchy: キャリアパス階層（Noneの場合は自動生成）
        """
        self.member_master = member_master
        self.member_competence = member_competence
        self.competence_master = competence_master

        # キャリアパス階層を構築
        if career_path_hierarchy is None:
            self.career_path_hierarchy = CareerPathHierarchy(
                member_master, member_competence, competence_master
            )
        else:
            self.career_path_hierarchy = career_path_hierarchy

        # SEMモデル（役職ごと）
        self.sem_models: Dict[str, UnifiedSEMEstimator] = {}

        # パス係数（役職ごと）
        self.path_coefficients: Dict[str, List[float]] = {}

        # メンバーの現在位置（役職ごと）
        self.member_positions: Dict[str, Dict[str, Tuple[int, float]]] = {}

        self.is_fitted = False

    def fit(
        self,
        roles: Optional[List[str]] = None,
        min_members_per_role: int = 5,
        min_skills_per_stage: int = 3
    ):
        """
        各役職のキャリアパスSEMモデルを学習

        Args:
            roles: 学習する役職リスト（Noneの場合は全役職）
            min_members_per_role: 役職ごとの最低メンバー数
            min_skills_per_stage: ステージごとの最低スキル数
        """
        if roles is None:
            roles = list(self.career_path_hierarchy.career_paths.keys())

        logger.info("=" * 60)
        logger.info("キャリアパスSEMモデルの学習開始")
        logger.info("=" * 60)

        for role in roles:
            logger.info(f"\n【{role}】の学習中...")

            try:
                # 役職のメンバー数を確認
                role_members = self.member_master[
                    self.member_master['役職'] == role
                ]['メンバーコード'].tolist()

                if len(role_members) < min_members_per_role:
                    logger.warning(
                        f"  ⚠️ メンバー数不足: {len(role_members)}名 < {min_members_per_role}名"
                    )
                    continue

                # 役職のSEMモデルを構築・学習
                sem_model, path_coefs = self._fit_role_sem(
                    role, role_members, min_skills_per_stage
                )

                if sem_model is not None:
                    self.sem_models[role] = sem_model
                    self.path_coefficients[role] = path_coefs
                    logger.info(f"✅ {role}のSEMモデル学習完了")
                else:
                    logger.warning(f"⚠️ {role}のSEMモデル学習スキップ（データ不足）")

            except Exception as e:
                logger.error(f"❌ {role}のSEMモデル学習失敗: {e}")

        # メンバーの現在位置を推定
        self._estimate_member_positions()

        self.is_fitted = True

        logger.info("\n" + "=" * 60)
        logger.info(f"✅ キャリアパスSEMモデルの学習完了（{len(self.sem_models)}役職）")
        logger.info("=" * 60)

    def _fit_role_sem(
        self,
        role: str,
        role_members: List[str],
        min_skills_per_stage: int
    ) -> Tuple[Optional[UnifiedSEMEstimator], List[float]]:
        """
        特定の役職のSEMモデルを構築・学習

        Args:
            role: 役職名
            role_members: 役職のメンバーコードリスト
            min_skills_per_stage: ステージごとの最低スキル数

        Returns:
            (学習済みSEMモデル, パス係数リスト)
        """
        stages = self.career_path_hierarchy.get_role_stages(role)

        if len(stages) < 2:
            logger.warning(f"  ⚠️ ステージ数不足: {len(stages)}個 < 2個")
            return None, []

        logger.info(f"  ステージ数: {len(stages)}")

        # 役職メンバーのスキルデータを取得
        role_skills = self.member_competence[
            self.member_competence['メンバーコード'].isin(role_members)
        ]

        # 力量名マッピング
        competence_name_map = dict(zip(
            self.competence_master['力量コード'],
            self.competence_master['力量名']
        ))

        # 測定モデルを定義（各ステージの潜在変数）
        measurement_models = []

        for stage_info in stages:
            stage_num = stage_info['stage']
            stage_name = stage_info['name']

            # このステージのスキルを取得
            stage_skills = self.career_path_hierarchy.get_skills_by_stage(
                role, stage_num
            )

            if len(stage_skills) < min_skills_per_stage:
                logger.warning(
                    f"  ⚠️ Stage {stage_num}のスキル数不足: {len(stage_skills)}個"
                )
                continue

            # 力量名に変換
            stage_skill_names = [
                competence_name_map.get(code, code)
                for code in stage_skills
                if code in competence_name_map
            ]

            if len(stage_skill_names) >= 2:
                measurement_models.append(
                    MeasurementModelSpec(
                        latent_variable=f'{role}_{stage_name}',
                        indicators=stage_skill_names,
                        reference_indicator=stage_skill_names[0]
                    )
                )

        if len(measurement_models) < 2:
            logger.warning(f"  ⚠️ 測定モデルが2個未満: {len(measurement_models)}個")
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
        skill_matrix = role_skills.pivot_table(
            index='メンバーコード',
            columns='力量コード',
            values='正規化レベル',
            fill_value=0.0
        )

        # 力量名でリネーム
        renamed_columns = {
            code: competence_name_map.get(code, code)
            for code in skill_matrix.columns
        }
        skill_matrix_renamed = skill_matrix.rename(columns=renamed_columns)

        # SEMモデルを学習
        try:
            sem_model = UnifiedSEMEstimator(
                measurement_model=measurement_models,
                structural_model=structural_models
            )

            sem_model.fit(skill_matrix_renamed)

            # パス係数を抽出
            path_coefs = []
            for structural_spec in structural_models:
                param_name = f"β_{structural_spec.from_variable}→{structural_spec.to_variable}"
                if param_name in sem_model.params:
                    path_coefs.append(sem_model.params[param_name].value)
                else:
                    path_coefs.append(0.0)

            logger.info(f"  ✅ SEMモデル学習完了（パス係数: {[f'{c:.3f}' for c in path_coefs]}）")

            return sem_model, path_coefs

        except Exception as e:
            logger.error(f"  ❌ SEMモデル学習エラー: {e}")
            return None, []

    def _estimate_member_positions(self):
        """
        各メンバーの現在のキャリア位置を推定

        self.member_positions = {
            '主任': {
                'M001': (stage=1, progress=0.65),
                'M002': (stage=2, progress=0.80),
            },
            ...
        }
        """
        logger.info("\nメンバーの現在位置を推定中...")

        # 役職ごとに初期化
        for role in self.sem_models.keys():
            self.member_positions[role] = {}

        # メンバーごとに推定
        for _, member_row in self.member_master.iterrows():
            member_code = member_row['メンバーコード']
            role = member_row.get('役職', None)

            if role is None or role not in self.sem_models:
                continue

            # キャリア位置を推定
            stage, progress = self.career_path_hierarchy.estimate_member_stage(
                member_code, role
            )

            self.member_positions[role][member_code] = (stage, progress)

        logger.info(f"✅ {sum(len(v) for v in self.member_positions.values())}名の位置推定完了")

    def get_member_position(
        self,
        member_code: str
    ) -> Tuple[Optional[str], int, float]:
        """
        メンバーの現在のキャリア位置を取得

        Args:
            member_code: メンバーコード

        Returns:
            (役職, ステージ, 進捗率)
        """
        member_info = self.member_master[
            self.member_master['メンバーコード'] == member_code
        ]

        if len(member_info) == 0:
            return (None, 0, 0.0)

        role = member_info.iloc[0].get('役職', None)

        if role is None or role not in self.member_positions:
            return (role, 0, 0.0)

        if member_code not in self.member_positions[role]:
            return (role, 0, 0.0)

        stage, progress = self.member_positions[role][member_code]

        return (role, stage, progress)

    def recommend_next_steps(
        self,
        member_code: str,
        top_n: int = 10
    ) -> List[Dict]:
        """
        メンバーの次のキャリアステップを推薦

        Args:
            member_code: メンバーコード
            top_n: 推薦数

        Returns:
            推薦スキルのリスト
        """
        role, current_stage, progress = self.get_member_position(member_code)

        if role is None or role not in self.sem_models:
            return []

        # 次のステージのスキルを取得
        recommendations = self.career_path_hierarchy.get_next_stage_skills(
            member_code, role, top_n
        )

        # パス係数情報を付加
        if role in self.path_coefficients:
            path_coefs = self.path_coefficients[role]

            for rec in recommendations:
                rec_stage = rec['stage']

                # このステージへのパス係数
                if rec_stage > 0 and rec_stage <= len(path_coefs):
                    rec['path_coefficient'] = path_coefs[rec_stage - 1]
                else:
                    rec['path_coefficient'] = 0.0

        return recommendations

    def get_career_progression_summary(
        self,
        member_code: str
    ) -> Dict:
        """
        メンバーのキャリア進捗サマリーを取得

        Returns:
            {
                'role': str,
                'current_stage': int,
                'current_stage_name': str,
                'progress': float,
                'next_stage_name': str,
                'estimated_completion_months': int,
            }
        """
        role, current_stage, progress = self.get_member_position(member_code)

        if role is None:
            return {}

        current_stage_info = self.career_path_hierarchy.get_stage_info(
            role, current_stage
        )

        stages = self.career_path_hierarchy.get_role_stages(role)

        next_stage = current_stage + 1 if current_stage < len(stages) - 1 else current_stage
        next_stage_info = self.career_path_hierarchy.get_stage_info(role, next_stage)

        # 完了までの推定月数
        if current_stage_info:
            remaining_progress = 1.0 - progress
            estimated_months = int(
                current_stage_info['typical_duration_months'] * remaining_progress
            )
        else:
            estimated_months = 0

        return {
            'role': role,
            'current_stage': current_stage,
            'current_stage_name': current_stage_info['name'] if current_stage_info else f'Stage {current_stage}',
            'progress': progress,
            'next_stage_name': next_stage_info['name'] if next_stage_info else f'Stage {next_stage}',
            'estimated_completion_months': estimated_months,
        }

    def get_role_path_summary(self, role: str) -> pd.DataFrame:
        """
        役職のキャリアパス全体のサマリーを取得

        Returns:
            DataFrame with columns: [Stage, Stage_Name, Skills, Path_Coefficient]
        """
        stages = self.career_path_hierarchy.get_role_stages(role)

        if not stages:
            return pd.DataFrame()

        path_data = []

        path_coefs = self.path_coefficients.get(role, [])

        for i, stage_info in enumerate(stages):
            stage_num = stage_info['stage']
            stage_name = stage_info['name']

            # このステージのスキルを取得
            stage_skills = self.career_path_hierarchy.get_skills_by_stage(
                role, stage_num
            )

            # 力量名に変換
            stage_skill_names = []
            for code in stage_skills[:5]:  # 最大5個表示
                comp_info = self.competence_master[
                    self.competence_master['力量コード'] == code
                ]
                if len(comp_info) > 0:
                    stage_skill_names.append(comp_info.iloc[0]['力量名'])

            # パス係数
            if i > 0 and i - 1 < len(path_coefs):
                path_coef = path_coefs[i - 1]
            else:
                path_coef = None

            path_data.append({
                'Stage': stage_num,
                'Stage_Name': stage_name,
                'Skills': ', '.join(stage_skill_names) if stage_skill_names else 'N/A',
                'Path_Coefficient': f'{path_coef:.3f}' if path_coef is not None else 'N/A',
            })

        return pd.DataFrame(path_data)

    def calculate_path_alignment_score(
        self,
        member_code: str,
        competence_code: str
    ) -> float:
        """
        推薦スキルがパスに沿っているかを評価するスコアを計算

        Path Alignment Score:
        - 現在のステージのスキル: 1.0（最高優先度）
        - 次のステージのスキル: 0.8（高優先度）
        - 2段階先のスキル: 0.5（中優先度）
        - 過去のステージのスキル: 0.2（低優先度）
        - パス上にないスキル: 0.0

        Args:
            member_code: メンバーコード
            competence_code: 力量コード

        Returns:
            Path Alignment Score（0.0～1.0）
        """
        role, current_stage, progress = self.get_member_position(member_code)

        if role is None or role not in self.sem_models:
            return 0.0

        # スキルがどのステージに属するか判定
        skill_stage = None
        stages = self.career_path_hierarchy.get_role_stages(role)

        for stage_info in stages:
            stage_num = stage_info['stage']
            stage_skills = self.career_path_hierarchy.get_skills_by_stage(role, stage_num)

            if competence_code in stage_skills:
                skill_stage = stage_num
                break

        if skill_stage is None:
            # パス上にないスキル
            return 0.0

        # 現在の段階とスキルの段階の距離
        stage_distance = skill_stage - current_stage

        # 距離に基づいてスコアを計算
        if stage_distance == 0:
            # 現在の段階のスキル → 高スコア
            base_score = 1.0
        elif stage_distance == 1:
            # 次の段階のスキル → 中程度のスコア
            base_score = 0.8
        elif stage_distance == 2:
            # 2段階先のスキル → 低めのスコア
            base_score = 0.5
        elif stage_distance < 0:
            # 過去の段階のスキル（既に通過した段階）→ 最低スコア
            base_score = 0.2
        else:
            # 3段階以上先のスキル → 最低スコア
            base_score = 0.1

        # 現在の段階の進度で調整
        # 進度が高いほど、次の段階のスキルが推奨されやすい
        if stage_distance == 1:
            progress_adjustment = progress * 0.2
            base_score = min(1.0, base_score + progress_adjustment)

        # パス係数で調整（次のステージへの因果効果が強い場合、スコアを上げる）
        if role in self.path_coefficients and stage_distance == 1:
            path_coefs = self.path_coefficients[role]
            if current_stage < len(path_coefs):
                # 現在→次ステージのパス係数
                path_coef = path_coefs[current_stage]
                # パス係数が高いほど、次のステージのスキルを推奨
                path_adjustment = path_coef * 0.1
                base_score = min(1.0, base_score + path_adjustment)

        return base_score

    def generate_path_explanation(
        self,
        member_code: str,
        competence_code: str
    ) -> str:
        """
        推薦理由を生成

        Args:
            member_code: メンバーコード
            competence_code: 力量コード

        Returns:
            推薦理由の文字列
        """
        role, current_stage, progress = self.get_member_position(member_code)

        if role is None:
            return "推薦理由を生成できませんでした（役職情報なし）。"

        if role not in self.sem_models:
            return f"推薦理由を生成できませんでした（役職「{role}」のSEMモデルなし）。"

        # スキル情報を取得
        comp_info = self.competence_master[
            self.competence_master['力量コード'] == competence_code
        ]

        if len(comp_info) == 0:
            skill_name = competence_code
        else:
            skill_name = comp_info.iloc[0]['力量名']

        # スキルがどのステージに属するか判定
        skill_stage = None
        stages = self.career_path_hierarchy.get_role_stages(role)

        for stage_info in stages:
            stage_num = stage_info['stage']
            stage_skills = self.career_path_hierarchy.get_skills_by_stage(role, stage_num)

            if competence_code in stage_skills:
                skill_stage = stage_num
                break

        if skill_stage is None:
            return f"「{skill_name}」はキャリアパス上にありません。"

        # 現在のステージ情報
        current_stage_info = self.career_path_hierarchy.get_stage_info(role, current_stage)
        current_stage_name = current_stage_info['name'] if current_stage_info else f'Stage {current_stage}'

        # スキルのステージ情報
        skill_stage_info = self.career_path_hierarchy.get_stage_info(role, skill_stage)
        skill_stage_name = skill_stage_info['name'] if skill_stage_info else f'Stage {skill_stage}'

        # 進度パーセント
        progress_pct = progress * 100

        # 段階間の距離
        stage_distance = skill_stage - current_stage

        # Path Alignment Score
        path_score = self.calculate_path_alignment_score(member_code, competence_code)

        # 推薦理由を生成
        if stage_distance == 0:
            # 現在の段階のスキル
            explanation = (
                f"【キャリアパス因果構造モデル推薦】\n\n"
                f"あなたは現在、役職「{role}」の{current_stage_name}（進度: {progress_pct:.1f}%）にいます。\n\n"
                f"「{skill_name}」は{current_stage_name}で習得すべきスキルです。\n"
                f"この段階を完了することで、次の段階への進出が可能になります。"
            )
        elif stage_distance == 1:
            # 次の段階のスキル
            # パス係数を取得
            path_coef = None
            if role in self.path_coefficients:
                path_coefs = self.path_coefficients[role]
                if current_stage < len(path_coefs):
                    path_coef = path_coefs[current_stage]

            explanation = (
                f"【キャリアパス因果構造モデル推薦】\n\n"
                f"あなたは現在、役職「{role}」の{current_stage_name}（進度: {progress_pct:.1f}%）にいます。\n\n"
                f"「{skill_name}」は次のステップである{skill_stage_name}のスキルです。\n"
                f"現在の段階の進度が{progress_pct:.1f}%に達しているため、次の段階への準備として推奨します。"
            )

            if path_coef is not None and path_coef > 0.5:
                explanation += (
                    f"\n\n【因果効果】\n"
                    f"{current_stage_name} → {skill_stage_name}のパス係数: β={path_coef:.3f}\n"
                    f"現在の段階のスキル習得が、次の段階のスキル習得に強い因果効果を持っています。"
                )

        elif stage_distance == 2:
            # 2段階先のスキル
            explanation = (
                f"【キャリアパス因果構造モデル推薦】\n\n"
                f"あなたは現在、役職「{role}」の{current_stage_name}（進度: {progress_pct:.1f}%）にいます。\n\n"
                f"「{skill_name}」は{skill_stage_name}のスキルで、2段階先のスキルです。\n"
                f"将来のキャリアを見据えた先行学習として推奨します。"
            )
        elif stage_distance > 2:
            # 3段階以上先のスキル
            explanation = (
                f"【キャリアパス因果構造モデル推薦】\n\n"
                f"あなたは現在、役職「{role}」の{current_stage_name}（進度: {progress_pct:.1f}%）にいます。\n\n"
                f"「{skill_name}」は{skill_stage_name}のスキルで、{stage_distance}段階先のスキルです。\n"
                f"長期的なキャリア開発の観点から、先を見据えた学習として推奨します。"
            )
        else:
            # 過去の段階のスキル
            explanation = (
                f"【キャリアパス因果構造モデル推薦】\n\n"
                f"あなたは現在、役職「{role}」の{current_stage_name}（進度: {progress_pct:.1f}%）にいます。\n\n"
                f"「{skill_name}」は{skill_stage_name}のスキルで、基礎固めに役立ちます。\n"
                f"基本スキルの補強により、現在の段階での成長が加速します。"
            )

        # Path Alignment Scoreを追加
        explanation += f"\n\nPath Alignment Score: {path_score:.2f}（パス親和性: {path_score*100:.0f}%）"

        return explanation
