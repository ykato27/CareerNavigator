"""
スキル依存関係分析モジュール

時系列データから学習順序パターンを抽出し、
推奨される学習パスを提供します。
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import pandas as pd
import numpy as np

from skillnote_recommendation.core.config import Config

logger = logging.getLogger(__name__)


@dataclass
class SkillTransition:
    """スキル遷移情報"""
    prerequisite_code: str
    prerequisite_name: str
    dependent_code: str
    dependent_name: str
    transition_count: int
    median_time_gap_days: float
    confidence: float
    dependency_strength: str  # '強', '中', '弱', 'なし'
    reverse_transition_count: int
    evidence: str


@dataclass
class LearningPath:
    """推奨学習パス"""
    competence_code: str
    competence_name: str
    competence_type: str
    category: str
    recommended_prerequisites: List[Dict]
    can_learn_in_parallel: List[Dict]
    unlocks: List[Dict]
    estimated_difficulty: str
    estimated_learning_hours: Optional[int]
    success_rate: float


class SkillDependencyAnalyzer:
    """
    スキル依存関係分析クラス

    時系列データから学習順序パターンを抽出し、
    スキル間の依存関係を推定します。
    """

    def __init__(
        self,
        member_competence: pd.DataFrame,
        competence_master: pd.DataFrame,
        time_window_days: int = 180,
        min_transition_count: int = 3,
        confidence_threshold: float = 0.3
    ):
        """
        初期化

        Args:
            member_competence: メンバー習得力量データ（取得日必須）
            competence_master: 力量マスタ
            time_window_days: 遷移とみなす最大期間（日数）
            min_transition_count: 遷移として認識する最小人数
            confidence_threshold: 依存関係とみなす信頼度閾値
        """
        self.member_competence = member_competence.copy()
        self.competence_master = competence_master.copy()
        self.time_window_days = time_window_days
        self.min_transition_count = min_transition_count
        self.confidence_threshold = confidence_threshold

        # データ検証
        self._validate_data()

        # 取得日をdatetime型に変換
        self._prepare_data()

        logger.info("SkillDependencyAnalyzer initialized with %d members, %d competences",
                   self.member_competence['メンバーコード'].nunique(),
                   self.competence_master.shape[0])

    def _validate_data(self):
        """データの妥当性を検証"""
        if '取得日' not in self.member_competence.columns:
            raise ValueError("member_competenceに'取得日'カラムが必要です")

        required_cols = ['力量コード', '力量名', '力量タイプ']
        missing_cols = [col for col in required_cols if col not in self.competence_master.columns]
        if missing_cols:
            raise ValueError(f"competence_masterに必要なカラムがありません: {missing_cols}")

    def _prepare_data(self):
        """データを準備"""
        # 取得日を変換
        self.member_competence['取得日_dt'] = pd.to_datetime(
            self.member_competence['取得日'],
            errors='coerce'
        )

        # 無効な日付を除外
        valid_dates = self.member_competence['取得日_dt'].notna()
        if not valid_dates.any():
            raise ValueError("有効な取得日が1つもありません")

        self.member_competence = self.member_competence[valid_dates].copy()

        logger.info("Data prepared: %d valid acquisition records", len(self.member_competence))

    def extract_temporal_transitions(self) -> pd.DataFrame:
        """
        時系列ベースでスキル遷移パターンを抽出

        Returns:
            スキル遷移データフレーム
        """
        logger.info("Extracting temporal skill transitions...")

        transitions = []

        # メンバーごとに取得順序を分析
        for member_code in self.member_competence['メンバーコード'].unique():
            member_data = self.member_competence[
                self.member_competence['メンバーコード'] == member_code
            ].sort_values('取得日_dt')

            if len(member_data) < 2:
                continue

            # 連続するスキルペアを抽出
            for i in range(len(member_data) - 1):
                skill_a = member_data.iloc[i]
                skill_b = member_data.iloc[i + 1]

                time_diff = (skill_b['取得日_dt'] - skill_a['取得日_dt']).days

                # 時間窓内の遷移のみ記録
                if 0 < time_diff <= self.time_window_days:
                    transitions.append({
                        'member_code': member_code,
                        'prerequisite_code': skill_a['力量コード'],
                        'dependent_code': skill_b['力量コード'],
                        'time_gap_days': time_diff,
                        'acquisition_date_a': skill_a['取得日_dt'],
                        'acquisition_date_b': skill_b['取得日_dt']
                    })

        if not transitions:
            logger.warning("No transitions found within time window")
            return pd.DataFrame()

        transition_df = pd.DataFrame(transitions)
        logger.info("Extracted %d skill transitions from %d members",
                   len(transition_df), transition_df['member_code'].nunique())

        return transition_df

    def calculate_transition_confidence(
        self,
        transition_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        遷移の信頼度を計算

        Args:
            transition_df: 遷移データフレーム

        Returns:
            信頼度付き遷移集計データフレーム
        """
        if transition_df.empty:
            return pd.DataFrame()

        # スキルペアごとに集計
        pattern_counts = transition_df.groupby(
            ['prerequisite_code', 'dependent_code']
        ).agg({
            'member_code': 'count',
            'time_gap_days': 'median'
        }).reset_index()

        pattern_counts.columns = [
            'prerequisite_code', 'dependent_code',
            'transition_count', 'median_time_gap'
        ]

        # 最小遷移数でフィルタ
        pattern_counts = pattern_counts[
            pattern_counts['transition_count'] >= self.min_transition_count
        ].copy()

        # 信頼度を計算: prerequisiteを習得した人のうち、dependentも習得した割合
        confidences = []
        for _, row in pattern_counts.iterrows():
            prerequisite_code = row['prerequisite_code']
            dependent_code = row['dependent_code']

            # prerequisiteを習得した総人数
            total_with_prerequisite = self.member_competence[
                self.member_competence['力量コード'] == prerequisite_code
            ]['メンバーコード'].nunique()

            # dependentも習得した人数（遷移数）
            transition_count = row['transition_count']

            # 信頼度 = 遷移率
            confidence = transition_count / total_with_prerequisite if total_with_prerequisite > 0 else 0
            confidences.append(confidence)

        pattern_counts['confidence'] = confidences

        logger.info("Calculated confidence for %d transition patterns", len(pattern_counts))

        return pattern_counts

    def infer_dependency_direction(
        self,
        transition_confidences: pd.DataFrame
    ) -> List[SkillTransition]:
        """
        双方向遷移を比較して因果の方向を推定

        Args:
            transition_confidences: 信頼度付き遷移データ

        Returns:
            SkillTransitionオブジェクトのリスト
        """
        if transition_confidences.empty:
            return []

        logger.info("Inferring dependency direction from bidirectional transitions...")

        skill_transitions = []
        processed_pairs = set()

        for _, row in transition_confidences.iterrows():
            skill_a = row['prerequisite_code']
            skill_b = row['dependent_code']

            # すでに処理済みのペアはスキップ
            if (skill_a, skill_b) in processed_pairs or (skill_b, skill_a) in processed_pairs:
                continue

            # A→Bの遷移
            a_to_b_count = row['transition_count']
            a_to_b_confidence = row['confidence']
            a_to_b_time_gap = row['median_time_gap']

            # B→Aの遷移を探す
            reverse_row = transition_confidences[
                (transition_confidences['prerequisite_code'] == skill_b) &
                (transition_confidences['dependent_code'] == skill_a)
            ]

            if not reverse_row.empty:
                b_to_a_count = reverse_row.iloc[0]['transition_count']
                b_to_a_confidence = reverse_row.iloc[0]['confidence']
            else:
                b_to_a_count = 0
                b_to_a_confidence = 0

            # 依存関係の方向と強度を判定
            if a_to_b_count > b_to_a_count * 2 and a_to_b_confidence >= self.confidence_threshold:
                # A→Bの方が明らかに多い → Aが前提
                prerequisite = skill_a
                dependent = skill_b
                confidence = a_to_b_confidence
                transition_count = a_to_b_count
                reverse_count = b_to_a_count
                time_gap = a_to_b_time_gap
                strength = self._get_dependency_strength(confidence)

            elif b_to_a_count > a_to_b_count * 2 and b_to_a_confidence >= self.confidence_threshold:
                # B→Aの方が明らかに多い → Bが前提
                prerequisite = skill_b
                dependent = skill_a
                confidence = b_to_a_confidence
                transition_count = b_to_a_count
                reverse_count = a_to_b_count
                time_gap = reverse_row.iloc[0]['median_time_gap']
                strength = self._get_dependency_strength(confidence)

            else:
                # 双方向がほぼ同じ or 信頼度が低い → 依存関係なし（並列学習可能）
                processed_pairs.add((skill_a, skill_b))
                continue

            # 力量情報を取得
            prerequisite_info = self.competence_master[
                self.competence_master['力量コード'] == prerequisite
            ]
            dependent_info = self.competence_master[
                self.competence_master['力量コード'] == dependent
            ]

            if prerequisite_info.empty or dependent_info.empty:
                continue

            prerequisite_name = prerequisite_info.iloc[0]['力量名']
            dependent_name = dependent_info.iloc[0]['力量名']

            evidence = f"{transition_count}人が{prerequisite_name}→{dependent_name}の順序で学習、逆方向は{reverse_count}人"

            skill_transition = SkillTransition(
                prerequisite_code=prerequisite,
                prerequisite_name=prerequisite_name,
                dependent_code=dependent,
                dependent_name=dependent_name,
                transition_count=transition_count,
                median_time_gap_days=time_gap,
                confidence=confidence,
                dependency_strength=strength,
                reverse_transition_count=reverse_count,
                evidence=evidence
            )

            skill_transitions.append(skill_transition)
            processed_pairs.add((skill_a, skill_b))

        logger.info("Identified %d skill dependencies", len(skill_transitions))

        return skill_transitions

    def _get_dependency_strength(self, confidence: float) -> str:
        """信頼度から依存関係の強度を判定"""
        if confidence >= 0.7:
            return '強'
        elif confidence >= 0.5:
            return '中'
        elif confidence >= self.confidence_threshold:
            return '弱'
        else:
            return 'なし'

    def find_parallel_learnable_skills(
        self,
        transition_confidences: pd.DataFrame
    ) -> List[Tuple[str, str, str]]:
        """
        並列学習可能なスキルペアを特定

        Args:
            transition_confidences: 信頼度付き遷移データ

        Returns:
            (skill_a_code, skill_b_code, reason) のリスト
        """
        if transition_confidences.empty:
            return []

        parallel_skills = []
        processed_pairs = set()

        for _, row in transition_confidences.iterrows():
            skill_a = row['prerequisite_code']
            skill_b = row['dependent_code']

            if (skill_a, skill_b) in processed_pairs or (skill_b, skill_a) in processed_pairs:
                continue

            # A→Bの遷移
            a_to_b_count = row['transition_count']

            # B→Aの遷移を探す
            reverse_row = transition_confidences[
                (transition_confidences['prerequisite_code'] == skill_b) &
                (transition_confidences['dependent_code'] == skill_a)
            ]

            if not reverse_row.empty:
                b_to_a_count = reverse_row.iloc[0]['transition_count']

                # 双方向がほぼ同数（比率が1.5倍未満）→ 並列学習可能
                ratio = max(a_to_b_count, b_to_a_count) / min(a_to_b_count, b_to_a_count)
                if ratio < 1.5:
                    reason = f"どちらの順序で学んでも良い（{a_to_b_count}人 vs {b_to_a_count}人）"
                    parallel_skills.append((skill_a, skill_b, reason))
                    processed_pairs.add((skill_a, skill_b))

        return parallel_skills

    def generate_learning_paths(
        self,
        target_competence_codes: Optional[List[str]] = None
    ) -> Dict[str, LearningPath]:
        """
        推奨学習パスを生成

        Args:
            target_competence_codes: 対象力量コードのリスト（Noneの場合は全力量）

        Returns:
            力量コードをキーとしたLearningPathの辞書
        """
        logger.info("Generating learning paths...")

        # 遷移パターンを抽出
        transitions_df = self.extract_temporal_transitions()
        if transitions_df.empty:
            logger.warning("No transitions found, cannot generate learning paths")
            return {}

        # 信頼度を計算
        confidences_df = self.calculate_transition_confidence(transitions_df)

        # 依存関係を推定
        skill_transitions = self.infer_dependency_direction(confidences_df)

        # 並列学習可能なスキルを特定
        parallel_skills = self.find_parallel_learnable_skills(confidences_df)

        # 対象力量を決定
        if target_competence_codes is None:
            target_competence_codes = self.competence_master['力量コード'].tolist()

        # 各力量の学習パスを生成
        learning_paths = {}

        for comp_code in target_competence_codes:
            comp_info = self.competence_master[
                self.competence_master['力量コード'] == comp_code
            ]

            if comp_info.empty:
                continue

            comp_name = comp_info.iloc[0]['力量名']
            comp_type = comp_info.iloc[0]['力量タイプ']
            comp_category = comp_info.iloc[0].get('力量カテゴリー名', '')

            # 前提スキルを探す
            prerequisites = [
                {
                    'skill_code': t.prerequisite_code,
                    'skill_name': t.prerequisite_name,
                    'reason': f"この力量を学んだ人の{int(t.confidence * 100)}%が、事前に習得しています",
                    'evidence': t.evidence,
                    'average_time_gap_days': int(t.median_time_gap_days),
                    'confidence': t.confidence,
                    'dependency_strength': t.dependency_strength
                }
                for t in skill_transitions if t.dependent_code == comp_code
            ]

            # このスキルを習得後に学べるスキル（unlock）
            unlocks = [
                {
                    'skill_code': t.dependent_code,
                    'skill_name': t.dependent_name,
                    'reason': f"{t.dependent_name}を学んだ人の{int(t.confidence * 100)}%が、この力量を事前に習得しています"
                }
                for t in skill_transitions if t.prerequisite_code == comp_code
            ]

            # 並列学習可能なスキル
            parallel = [
                {
                    'skill_code': p[1] if p[0] == comp_code else p[0],
                    'reason': p[2]
                }
                for p in parallel_skills if comp_code in (p[0], p[1])
            ]

            # 難易度を推定（前提スキル数に基づく）
            if len(prerequisites) == 0:
                difficulty = '初級'
            elif len(prerequisites) <= 2:
                difficulty = '中級'
            else:
                difficulty = '上級'

            # 成功率を推定（前提スキルの信頼度平均）
            if prerequisites:
                avg_confidence = np.mean([p['confidence'] for p in prerequisites])
                success_rate = avg_confidence
            else:
                success_rate = 0.8  # デフォルト

            learning_path = LearningPath(
                competence_code=comp_code,
                competence_name=comp_name,
                competence_type=comp_type,
                category=comp_category,
                recommended_prerequisites=prerequisites,
                can_learn_in_parallel=parallel,
                unlocks=unlocks,
                estimated_difficulty=difficulty,
                estimated_learning_hours=None,  # TODO: 将来的に実装
                success_rate=success_rate
            )

            learning_paths[comp_code] = learning_path

        logger.info("Generated learning paths for %d competences", len(learning_paths))

        return learning_paths

    def get_dependency_graph_data(self) -> Dict:
        """
        依存関係グラフ用のデータを取得（可視化用）

        Returns:
            ノードとエッジの情報を含む辞書
        """
        transitions_df = self.extract_temporal_transitions()
        if transitions_df.empty:
            return {'nodes': [], 'edges': []}

        confidences_df = self.calculate_transition_confidence(transitions_df)
        skill_transitions = self.infer_dependency_direction(confidences_df)

        # ノード（スキル）
        nodes = []
        skill_codes = set()
        for t in skill_transitions:
            skill_codes.add(t.prerequisite_code)
            skill_codes.add(t.dependent_code)

        for code in skill_codes:
            comp_info = self.competence_master[
                self.competence_master['力量コード'] == code
            ]
            if not comp_info.empty:
                nodes.append({
                    'id': code,
                    'label': comp_info.iloc[0]['力量名'],
                    'type': comp_info.iloc[0]['力量タイプ']
                })

        # エッジ（依存関係）
        edges = []
        for t in skill_transitions:
            edges.append({
                'source': t.prerequisite_code,
                'target': t.dependent_code,
                'weight': t.confidence,
                'strength': t.dependency_strength,
                'evidence': t.evidence,
                'time_gap_days': int(t.median_time_gap_days)
            })

        return {
            'nodes': nodes,
            'edges': edges
        }
