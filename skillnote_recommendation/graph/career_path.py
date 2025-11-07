"""
Career Path Recommendation System

目標メンバーへのキャリアパス推薦機能
力量ギャップ分析と学習パスの生成を提供
"""

from typing import List, Dict, Tuple, Optional, Set
import pandas as pd
import numpy as np
from dataclasses import dataclass
from .knowledge_graph import CompetenceKnowledgeGraph
from .category_hierarchy import CategoryHierarchy


# 定数
PHASE_BASIC = "基礎固め"
PHASE_INTERMEDIATE = "専門性構築"
PHASE_EXPERT = "エキスパート"

# スコアリング重み
WEIGHT_HIERARCHY = 0.3  # カテゴリー階層の重み
WEIGHT_EASE = 0.3       # 習得容易性の重み
WEIGHT_IMPORTANCE = 0.4  # 重要度の重み


@dataclass
class CompetenceGap:
    """力量ギャップ情報

    Attributes:
        competence_code: 力量コード
        competence_name: 力量名
        competence_type: 力量タイプ
        category: カテゴリー
        importance_score: 重要度スコア（目標メンバーにとっての）
        ease_score: 習得容易性スコア（自分にとっての）
        hierarchy_level: 階層レベル（基礎=1, 応用=2, 専門=3）
        priority_score: 総合優先度スコア
        phase: 推奨習得フェーズ
        prerequisites: 前提となる力量のリスト
    """
    competence_code: str
    competence_name: str
    competence_type: str
    category: str
    importance_score: float
    ease_score: float
    hierarchy_level: int
    priority_score: float
    phase: str
    prerequisites: List[str]


@dataclass
class CareerPathAnalysis:
    """キャリアパス分析結果

    Attributes:
        source_member_code: 分析対象メンバーコード
        target_member_code: 目標メンバーコード
        gap_score: ギャップスコア（0-1、0=完全一致、1=完全相違）
        common_competences: 共通力量のリスト
        missing_competences: 不足力量のリスト（CompetenceGap）
        phase_1_competences: Phase 1の力量リスト
        phase_2_competences: Phase 2の力量リスト
        phase_3_competences: Phase 3の力量リスト
        estimated_completion_rate: 現在の到達度（0-1）
    """
    source_member_code: str
    target_member_code: str
    gap_score: float
    common_competences: List[Dict]
    missing_competences: List[CompetenceGap]
    phase_1_competences: List[CompetenceGap]
    phase_2_competences: List[CompetenceGap]
    phase_3_competences: List[CompetenceGap]
    estimated_completion_rate: float


class CareerGapAnalyzer:
    """キャリアギャップ分析エンジン

    目標メンバーと現在のメンバーの力量差分を分析し、
    ギャップを定量化する。
    """

    def __init__(self,
                 knowledge_graph: CompetenceKnowledgeGraph,
                 member_competence_df: pd.DataFrame,
                 competence_master_df: pd.DataFrame):
        """
        Args:
            knowledge_graph: ナレッジグラフ
            member_competence_df: メンバー保有力量データ
            competence_master_df: 力量マスタ
        """
        self.kg = knowledge_graph
        self.member_competence_df = member_competence_df
        self.competence_master_df = competence_master_df

    def analyze_gap(self,
                    source_member_code: str,
                    target_member_code: str) -> Dict:
        """
        2人のメンバー間の力量ギャップを分析

        Args:
            source_member_code: 分析対象メンバーコード
            target_member_code: 目標メンバーコード

        Returns:
            ギャップ分析結果の辞書
        """
        print(f"\n{'='*80}")
        print(f"キャリアギャップ分析: {source_member_code} → {target_member_code}")
        print(f"{'='*80}")

        # 各メンバーの保有力量を取得
        source_competences = self._get_member_competences(source_member_code)
        target_competences = self._get_member_competences(target_member_code)

        # 共通力量と不足力量を計算
        common_codes = set(source_competences.keys()) & set(target_competences.keys())
        missing_codes = set(target_competences.keys()) - set(source_competences.keys())

        print(f"\n分析結果:")
        print(f"  共通力量: {len(common_codes)}個")
        print(f"  不足力量: {len(missing_codes)}個")
        print(f"  到達度: {len(common_codes) / len(target_competences) * 100:.1f}%")

        # ギャップスコアを計算（Jaccard距離）
        union = len(set(source_competences.keys()) | set(target_competences.keys()))
        gap_score = 1.0 - (len(common_codes) / union) if union > 0 else 0.0

        # 共通力量の詳細情報を取得
        common_competences = []
        for code in common_codes:
            comp_info = self._get_competence_info(code)
            if comp_info:
                common_competences.append(comp_info)

        # 不足力量に重要度スコアを付与
        missing_competences_info = []
        for code in missing_codes:
            comp_info = self._get_competence_info(code)
            if comp_info:
                # 目標メンバーのレベルを重要度とする
                importance = target_competences.get(code, 1.0) / 5.0  # 正規化
                comp_info['importance_score'] = importance
                missing_competences_info.append(comp_info)

        return {
            'source_member_code': source_member_code,
            'target_member_code': target_member_code,
            'gap_score': gap_score,
            'common_competences': common_competences,
            'missing_competences': missing_competences_info,
            'source_competences': source_competences,
            'target_competences': target_competences,
            'estimated_completion_rate': len(common_codes) / len(target_competences) if len(target_competences) > 0 else 0.0
        }

    def _get_member_competences(self, member_code: str) -> Dict[str, float]:
        """
        メンバーの保有力量を取得

        Args:
            member_code: メンバーコード

        Returns:
            {力量コード: レベル} の辞書
        """
        member_data = self.member_competence_df[
            self.member_competence_df['メンバーコード'] == member_code
        ]

        competences = {}
        for _, row in member_data.iterrows():
            competences[row['力量コード']] = row['正規化レベル']

        return competences

    def _get_competence_info(self, competence_code: str) -> Optional[Dict]:
        """
        力量の詳細情報を取得

        Args:
            competence_code: 力量コード

        Returns:
            力量情報の辞書（存在しない場合はNone）
        """
        comp_data = self.competence_master_df[
            self.competence_master_df['力量コード'] == competence_code
        ]

        if len(comp_data) == 0:
            return None

        row = comp_data.iloc[0]
        return {
            'competence_code': row['力量コード'],
            'competence_name': row['力量名'],
            'competence_type': row['力量タイプ'],
            'category': row.get('力量カテゴリー名', 'その他'),
        }


class LearningPathGenerator:
    """学習パス生成エンジン

    ギャップ分析結果から、最適な学習パスを生成する。
    バランス型スコアリング: 階層 + 容易性 + 重要度を総合的に考慮
    """

    def __init__(self,
                 knowledge_graph: CompetenceKnowledgeGraph,
                 category_hierarchy: Optional[CategoryHierarchy] = None):
        """
        Args:
            knowledge_graph: ナレッジグラフ
            category_hierarchy: カテゴリー階層（オプション）
        """
        self.kg = knowledge_graph
        self.category_hierarchy = category_hierarchy

    def generate_learning_path(self,
                                gap_analysis: Dict,
                                max_per_phase: int = 5) -> CareerPathAnalysis:
        """
        学習パスを生成

        Args:
            gap_analysis: CareerGapAnalyzerのanalyze_gap結果
            max_per_phase: 各フェーズの最大推薦数

        Returns:
            CareerPathAnalysis オブジェクト
        """
        print(f"\n{'='*80}")
        print(f"学習パス生成")
        print(f"{'='*80}")

        missing_competences = gap_analysis['missing_competences']
        source_competences = gap_analysis['source_competences']

        # 各力量にスコアを付与
        scored_gaps = []
        for comp_info in missing_competences:
            gap = self._score_competence(
                comp_info,
                source_competences
            )
            scored_gaps.append(gap)

        # 優先度順にソート
        scored_gaps.sort(key=lambda x: x.priority_score, reverse=True)

        # フェーズに分類
        phase_1 = [g for g in scored_gaps if g.phase == PHASE_BASIC][:max_per_phase]
        phase_2 = [g for g in scored_gaps if g.phase == PHASE_INTERMEDIATE][:max_per_phase]
        phase_3 = [g for g in scored_gaps if g.phase == PHASE_EXPERT][:max_per_phase]

        print(f"\n学習パス:")
        print(f"  Phase 1（{PHASE_BASIC}）: {len(phase_1)}個")
        print(f"  Phase 2（{PHASE_INTERMEDIATE}）: {len(phase_2)}個")
        print(f"  Phase 3（{PHASE_EXPERT}）: {len(phase_3)}個")

        return CareerPathAnalysis(
            source_member_code=gap_analysis['source_member_code'],
            target_member_code=gap_analysis['target_member_code'],
            gap_score=gap_analysis['gap_score'],
            common_competences=gap_analysis['common_competences'],
            missing_competences=scored_gaps,
            phase_1_competences=phase_1,
            phase_2_competences=phase_2,
            phase_3_competences=phase_3,
            estimated_completion_rate=gap_analysis['estimated_completion_rate']
        )

    def _score_competence(self,
                          comp_info: Dict,
                          source_competences: Dict[str, float]) -> CompetenceGap:
        """
        力量をスコアリング（バランス型）

        Args:
            comp_info: 力量情報
            source_competences: 分析対象メンバーの保有力量

        Returns:
            CompetenceGap オブジェクト
        """
        # 1. 階層レベルを判定（基礎=1, 応用=2, 専門=3）
        hierarchy_level = self._determine_hierarchy_level(comp_info)

        # 2. 習得容易性を計算（既存スキルとの関連性）
        ease_score = self._calculate_ease_score(comp_info, source_competences)

        # 3. 重要度スコア（目標メンバーのレベル）
        importance_score = comp_info.get('importance_score', 0.5)

        # 4. 総合優先度スコア（バランス型）
        priority_score = (
            WEIGHT_HIERARCHY * (1.0 - (hierarchy_level - 1) / 2.0) +  # 基礎ほど高スコア
            WEIGHT_EASE * ease_score +
            WEIGHT_IMPORTANCE * importance_score
        )

        # 5. フェーズを決定
        phase = self._determine_phase(hierarchy_level, priority_score)

        # 6. 前提条件を特定
        prerequisites = self._identify_prerequisites(comp_info, source_competences)

        return CompetenceGap(
            competence_code=comp_info['competence_code'],
            competence_name=comp_info['competence_name'],
            competence_type=comp_info['competence_type'],
            category=comp_info['category'],
            importance_score=importance_score,
            ease_score=ease_score,
            hierarchy_level=hierarchy_level,
            priority_score=priority_score,
            phase=phase,
            prerequisites=prerequisites
        )

    def _determine_hierarchy_level(self, comp_info: Dict) -> int:
        """
        カテゴリー階層から力量レベルを判定

        Returns:
            1=基礎, 2=応用, 3=専門
        """
        # カテゴリー名から推定（簡易実装）
        category = comp_info.get('category', '').lower()
        comp_name = comp_info.get('competence_name', '').lower()

        # キーワードベースの判定
        if '基礎' in comp_name or '入門' in comp_name or '初級' in comp_name:
            return 1
        elif '応用' in comp_name or '中級' in comp_name or '実践' in comp_name:
            return 2
        elif '専門' in comp_name or '上級' in comp_name or 'エキスパート' in comp_name:
            return 3

        # デフォルトは中間レベル
        return 2

    def _calculate_ease_score(self,
                               comp_info: Dict,
                               source_competences: Dict[str, float]) -> float:
        """
        習得容易性を計算

        同じカテゴリーの力量を既に持っていれば習得しやすい

        Returns:
            0.0-1.0 のスコア（1.0=非常に習得しやすい）
        """
        category = comp_info.get('category', '')

        # 同じカテゴリーの保有力量数をカウント
        same_category_count = 0
        for comp_code in source_competences.keys():
            # Knowledge Graphから力量情報を取得
            comp_node = f"competence_{comp_code}"
            if self.kg.G.has_node(comp_node):
                node_data = self.kg.get_node_info(comp_node)
                if node_data.get('category') == category:
                    same_category_count += 1

        # 保有数に応じたスコア（最大10個で飽和）
        ease_score = min(same_category_count / 10.0, 1.0)

        return ease_score

    def _determine_phase(self, hierarchy_level: int, priority_score: float) -> str:
        """
        習得フェーズを決定

        階層レベルと優先度スコアを組み合わせて判定
        """
        if hierarchy_level == 1:
            return PHASE_BASIC
        elif hierarchy_level == 2:
            return PHASE_INTERMEDIATE
        else:
            return PHASE_EXPERT

    def _identify_prerequisites(self,
                                 comp_info: Dict,
                                 source_competences: Dict[str, float]) -> List[str]:
        """
        前提条件となる力量を特定

        同じカテゴリーの基礎レベル力量で、まだ習得していないものを返す
        """
        # 簡易実装: 空リストを返す
        # 実際には、カテゴリー階層やグラフ構造から前提関係を抽出
        return []


# ===================================================================
# グラフベース推薦専用の学習パス生成
# ===================================================================

@dataclass
class RecommendationLearningPath:
    """グラフベース推薦の学習パス

    Attributes:
        phase_1_basic: Phase 1（基礎固め）の力量リスト
        phase_2_intermediate: Phase 2（専門性構築）の力量リスト
        phase_3_expert: Phase 3（エキスパート）の力量リスト
        all_recommendations: 全推薦力量（元のスコア順）
    """
    phase_1_basic: List[Dict]
    phase_2_intermediate: List[Dict]
    phase_3_expert: List[Dict]
    all_recommendations: List[Dict]


def generate_learning_path_from_recommendations(
    recommendations: List[Tuple[str, float, List]],
    knowledge_graph: CompetenceKnowledgeGraph,
    member_code: str,
    competence_master_df: pd.DataFrame,
    member_competence_df: pd.DataFrame
) -> RecommendationLearningPath:
    """
    グラフベース推薦結果から段階的な学習パスを生成

    Args:
        recommendations: RandomWalkRecommenderのrecommend()結果
            [(力量コード, スコア, パス), ...]
        knowledge_graph: ナレッジグラフ
        member_code: 対象メンバーコード
        competence_master_df: 力量マスタ
        member_competence_df: メンバー保有力量データ

    Returns:
        RecommendationLearningPath オブジェクト
    """
    # メンバーの既習得力量を取得
    member_competences = _get_member_competence_dict(member_code, member_competence_df)

    # 推薦力量を分析してスコアリング
    scored_recommendations = []
    for comp_code, rwr_score, paths in recommendations:
        comp_info = _get_competence_info_dict(comp_code, competence_master_df)
        if not comp_info:
            continue

        # 階層レベルを判定
        hierarchy_level = _determine_hierarchy_level_simple(comp_info)

        # 習得容易性を計算
        ease_score = _calculate_ease_score_simple(
            comp_info, member_competences, knowledge_graph
        )

        # 総合優先度スコア
        # RWRスコアを重視しつつ、階層と容易性も考慮
        priority_score = (
            0.5 * rwr_score +  # RWRスコアを50%
            0.3 * (1.0 - (hierarchy_level - 1) / 2.0) +  # 基礎ほど高スコア（30%）
            0.2 * ease_score  # 習得容易性（20%）
        )

        # フェーズを決定
        if hierarchy_level == 1:
            phase = PHASE_BASIC
        elif hierarchy_level == 2:
            phase = PHASE_INTERMEDIATE
        else:
            phase = PHASE_EXPERT

        scored_recommendations.append({
            'competence_code': comp_code,
            'competence_name': comp_info['competence_name'],
            'competence_type': comp_info['competence_type'],
            'category': comp_info['category'],
            'rwr_score': rwr_score,
            'hierarchy_level': hierarchy_level,
            'ease_score': ease_score,
            'priority_score': priority_score,
            'phase': phase,
            'paths': paths
        })

    # フェーズごとに分類（各フェーズ内では優先度順）
    phase_1 = sorted(
        [r for r in scored_recommendations if r['phase'] == PHASE_BASIC],
        key=lambda x: x['priority_score'],
        reverse=True
    )

    phase_2 = sorted(
        [r for r in scored_recommendations if r['phase'] == PHASE_INTERMEDIATE],
        key=lambda x: x['priority_score'],
        reverse=True
    )

    phase_3 = sorted(
        [r for r in scored_recommendations if r['phase'] == PHASE_EXPERT],
        key=lambda x: x['priority_score'],
        reverse=True
    )

    return RecommendationLearningPath(
        phase_1_basic=phase_1,
        phase_2_intermediate=phase_2,
        phase_3_expert=phase_3,
        all_recommendations=scored_recommendations
    )


def _get_member_competence_dict(
    member_code: str,
    member_competence_df: pd.DataFrame
) -> Dict[str, float]:
    """メンバーの保有力量を辞書形式で取得"""
    member_data = member_competence_df[
        member_competence_df['メンバーコード'] == member_code
    ]

    competences = {}
    for _, row in member_data.iterrows():
        competences[row['力量コード']] = row['正規化レベル']

    return competences


def _get_competence_info_dict(
    competence_code: str,
    competence_master_df: pd.DataFrame
) -> Optional[Dict]:
    """力量の詳細情報を辞書形式で取得"""
    comp_data = competence_master_df[
        competence_master_df['力量コード'] == competence_code
    ]

    if len(comp_data) == 0:
        return None

    row = comp_data.iloc[0]
    return {
        'competence_code': row['力量コード'],
        'competence_name': row['力量名'],
        'competence_type': row['力量タイプ'],
        'category': row.get('力量カテゴリー名', 'その他'),
    }


def _determine_hierarchy_level_simple(comp_info: Dict) -> int:
    """
    力量の階層レベルを簡易判定

    Returns:
        1=基礎, 2=応用, 3=専門
    """
    comp_name = comp_info.get('competence_name', '').lower()
    category = comp_info.get('category', '').lower()

    # キーワードベースの判定
    basic_keywords = ['基礎', '入門', '初級', '基本', '概論', 'basic', 'intro']
    intermediate_keywords = ['応用', '中級', '実践', '活用', 'intermediate', 'practical']
    expert_keywords = ['専門', '上級', 'エキスパート', '高度', 'advanced', 'expert']

    # 力量名でチェック
    if any(keyword in comp_name for keyword in basic_keywords):
        return 1
    elif any(keyword in comp_name for keyword in expert_keywords):
        return 3
    elif any(keyword in comp_name for keyword in intermediate_keywords):
        return 2

    # カテゴリー名でチェック
    if any(keyword in category for keyword in basic_keywords):
        return 1
    elif any(keyword in category for keyword in expert_keywords):
        return 3

    # デフォルトは中間レベル
    return 2


def _calculate_ease_score_simple(
    comp_info: Dict,
    member_competences: Dict[str, float],
    knowledge_graph: CompetenceKnowledgeGraph
) -> float:
    """
    習得容易性を簡易計算

    同じカテゴリーの力量を既に持っていれば習得しやすい

    Returns:
        0.0-1.0 のスコア（1.0=非常に習得しやすい）
    """
    category = comp_info.get('category', '')

    # 同じカテゴリーの保有力量数をカウント
    same_category_count = 0
    for comp_code in member_competences.keys():
        comp_node = f"competence_{comp_code}"
        if knowledge_graph.G.has_node(comp_node):
            node_data = knowledge_graph.get_node_info(comp_node)
            if node_data.get('category') == category:
                same_category_count += 1

    # 保有数に応じたスコア（最大10個で飽和）
    ease_score = min(same_category_count / 10.0, 1.0)

    return ease_score
