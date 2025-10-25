"""
Hybrid Recommender: RWR + NMF

Random Walk with Restartとmatrix factorization (NMF)を組み合わせた
ハイブリッド推薦システム
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass

from .knowledge_graph import CompetenceKnowledgeGraph
from .random_walk import RandomWalkRecommender
from ..ml.ml_recommender import MLRecommender
from ..core.models import Recommendation


@dataclass
class HybridRecommendation:
    """ハイブリッド推薦結果

    Attributes:
        competence_code: 力量コード
        score: 最終スコア（ハイブリッド）
        rwr_score: RWRスコア
        nmf_score: NMFスコア
        paths: 推薦パス
        reasons: 推薦理由
        competence_info: 力量情報
    """
    competence_code: str
    score: float
    rwr_score: float
    nmf_score: float
    paths: List[List[Dict]]
    reasons: List[str]
    competence_info: Dict


class HybridGraphRecommender:
    """RWRとNMFを融合したハイブリッド推薦エンジン

    グラフ構造（RWR）と協調フィルタリング（NMF）の強みを組み合わせて、
    より精度の高い推薦を実現する。

    推薦プロセス:
        1. RWRでグラフベースのスコアを計算
        2. NMFで協調フィルタリングベースのスコアを計算
        3. 両スコアを融合して最終ランキングを生成
        4. 推薦パスと理由を付与
    """

    def __init__(self,
                 knowledge_graph: CompetenceKnowledgeGraph,
                 ml_recommender: MLRecommender,
                 rwr_weight: float = 0.5,
                 restart_prob: float = 0.15):
        """
        Args:
            knowledge_graph: ナレッジグラフ
            ml_recommender: NMFベースのML推薦エンジン
            rwr_weight: RWRスコアの重み（0-1）、残りがNMFの重み
            restart_prob: RWRの再スタート確率
        """
        self.kg = knowledge_graph
        self.ml_recommender = ml_recommender
        self.rwr = RandomWalkRecommender(
            knowledge_graph=knowledge_graph,
            restart_prob=restart_prob
        )
        self.rwr_weight = rwr_weight
        self.nmf_weight = 1.0 - rwr_weight

        print(f"\nHybrid Graph Recommender 初期化完了")
        print(f"  RWR重み: {self.rwr_weight:.2f}")
        print(f"  NMF重み: {self.nmf_weight:.2f}")

    def recommend(self,
                  member_code: str,
                  top_n: int = 10,
                  competence_type: Optional[List[str]] = None,
                  category_filter: Optional[str] = None,
                  use_diversity: bool = True) -> List[HybridRecommendation]:
        """
        ハイブリッド推薦を実行

        Args:
            member_code: 対象メンバーコード
            top_n: 推薦件数
            competence_type: 力量タイプフィルタ
            category_filter: カテゴリーフィルタ
            use_diversity: 多様性を考慮するか

        Returns:
            ハイブリッド推薦結果のリスト
        """
        print(f"\n{'='*80}")
        print(f"Hybrid推薦開始: {member_code}")
        print(f"{'='*80}")

        # 1. RWRで推薦（パス付き）
        print("\n[1/4] RWR推薦...")
        rwr_results = self.rwr.recommend(
            member_code=member_code,
            top_n=top_n * 3,  # 多めに取得してフィルタ
            return_paths=True
        )
        rwr_dict = {comp: (score, paths) for comp, score, paths in rwr_results}

        # 2. NMFで推薦
        print("\n[2/4] NMF推薦...")
        nmf_results = self.ml_recommender.recommend(
            member_code=member_code,
            top_n=top_n * 3,
            competence_type=competence_type,
            category_filter=category_filter,
            use_diversity=use_diversity
        )
        nmf_dict = {rec.competence_code: rec.priority_score for rec in nmf_results}

        # 3. スコアを正規化して融合
        print("\n[3/4] スコア融合...")
        hybrid_scores = self._fuse_scores(rwr_dict, nmf_dict)

        # 4. フィルタリングとTop-N選択
        print(f"\n[4/4] Top-{top_n}選択...")
        recommendations = self._select_top_n(
            hybrid_scores=hybrid_scores,
            nmf_results=nmf_results,
            top_n=top_n,
            competence_type=competence_type,
            category_filter=category_filter
        )

        print(f"\n完了: {len(recommendations)}件の推薦を生成")
        return recommendations

    def _fuse_scores(self,
                     rwr_dict: Dict[str, Tuple[float, List]],
                     nmf_dict: Dict[str, float]) -> Dict[str, Dict]:
        """
        RWRとNMFのスコアを融合

        Args:
            rwr_dict: {力量コード: (RWRスコア, パス)}
            nmf_dict: {力量コード: NMFスコア}

        Returns:
            {力量コード: {'score': float, 'rwr_score': float, 'nmf_score': float, 'paths': list}}
        """
        # スコアを正規化（0-1）
        rwr_scores_only = {k: v[0] for k, v in rwr_dict.items()}
        rwr_normalized = self._normalize_scores(rwr_scores_only)
        nmf_normalized = self._normalize_scores(nmf_dict)

        # 全力量コードを取得
        all_competences = set(rwr_dict.keys()) | set(nmf_dict.keys())

        # ハイブリッドスコア計算
        hybrid_scores = {}
        for comp_code in all_competences:
            rwr_score = rwr_normalized.get(comp_code, 0.0)
            nmf_score = nmf_normalized.get(comp_code, 0.0)

            # 重み付き平均
            hybrid_score = (
                self.rwr_weight * rwr_score +
                self.nmf_weight * nmf_score
            )

            # 両方で推薦された場合はブーストを与える
            if comp_code in rwr_dict and comp_code in nmf_dict:
                hybrid_score *= 1.1  # 10%ブースト

            hybrid_scores[comp_code] = {
                'score': hybrid_score,
                'rwr_score': rwr_score,
                'nmf_score': nmf_score,
                'paths': rwr_dict.get(comp_code, (0, []))[1]
            }

        return hybrid_scores

    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """スコアを0-1に正規化（Min-Max正規化）"""
        if not scores:
            return {}

        values = np.array(list(scores.values()))
        min_val = values.min()
        max_val = values.max()

        if max_val == min_val:
            return {k: 1.0 for k in scores.keys()}

        return {
            k: float((v - min_val) / (max_val - min_val))
            for k, v in scores.items()
        }

    def _select_top_n(self,
                      hybrid_scores: Dict[str, Dict],
                      nmf_results: List[Recommendation],
                      top_n: int,
                      competence_type: Optional[List[str]],
                      category_filter: Optional[str]) -> List[HybridRecommendation]:
        """
        Top-N推薦を選択し、HybridRecommendationオブジェクトを生成

        Args:
            hybrid_scores: ハイブリッドスコア辞書
            nmf_results: NMF推薦結果（力量情報を取得するため）
            top_n: 推薦件数
            competence_type: 力量タイプフィルタ
            category_filter: カテゴリーフィルタ

        Returns:
            HybridRecommendationのリスト
        """
        # NMF結果から力量情報を取得
        competence_info_map = {
            rec.competence_code: {
                '力量名': rec.competence_name,
                '力量タイプ': rec.competence_type,
                'カテゴリー': rec.category,
                '概要': None,  # Recommendationクラスにdescriptionフィールドはない
            }
            for rec in nmf_results
        }

        # スコア順にソート
        sorted_scores = sorted(
            hybrid_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )

        # Top-Nを選択
        recommendations = []
        for comp_code, data in sorted_scores:
            # 力量情報を取得（NMF結果にない場合はマスタから取得）
            comp_info = competence_info_map.get(comp_code)
            if comp_info is None:
                comp_info = self._get_competence_info_from_master(comp_code)

            # フィルタリング
            if competence_type and comp_info.get('力量タイプ') not in competence_type:
                continue
            if category_filter and comp_info.get('カテゴリー') != category_filter:
                continue

            # 推薦理由を生成
            reasons = self._generate_reasons(comp_code, data)

            # HybridRecommendationを作成
            hybrid_rec = HybridRecommendation(
                competence_code=comp_code,
                score=data['score'],
                rwr_score=data['rwr_score'],
                nmf_score=data['nmf_score'],
                paths=self._convert_paths_to_readable(data['paths']),
                reasons=reasons,
                competence_info=comp_info
            )

            recommendations.append(hybrid_rec)

            if len(recommendations) >= top_n:
                break

        return recommendations

    def _get_competence_info_from_master(self, comp_code: str) -> Dict:
        """力量マスタから力量情報を取得"""
        comp_node = f"competence_{comp_code}"
        if self.kg.G.has_node(comp_node):
            node_data = self.kg.get_node_info(comp_node)
            return {
                '力量名': node_data.get('name', comp_code),
                '力量タイプ': node_data.get('type', 'UNKNOWN'),
                'カテゴリー': node_data.get('category', None),
                '概要': node_data.get('description', None),
            }
        return {
            '力量名': comp_code,
            '力量タイプ': 'UNKNOWN',
            'カテゴリー': None,
            '概要': None,
        }

    def _convert_paths_to_readable(self, paths: List[List[str]]) -> List[List[Dict]]:
        """パスを人間が読める形式に変換"""
        readable_paths = []
        for path in paths:
            readable_path = []
            for node in path:
                node_info = self.kg.get_node_info(node)
                readable_path.append({
                    'id': node,
                    'type': node_info.get('node_type', 'unknown'),
                    'name': node_info.get('name', node),
                })
            readable_paths.append(readable_path)
        return readable_paths

    def _generate_reasons(self, comp_code: str, data: Dict) -> List[str]:
        """推薦理由を生成"""
        reasons = []

        # RWRスコアが高い場合
        if data['rwr_score'] > 0.5:
            # パスから理由を生成
            paths = data['paths']
            if paths:
                path_reasons = self.rwr._generate_reasons(paths)
                reasons.extend(path_reasons)

        # NMFスコアが高い場合
        if data['nmf_score'] > 0.5:
            reasons.append("類似メンバーの習得パターンから推薦")

        # 両方高い場合
        if data['rwr_score'] > 0.3 and data['nmf_score'] > 0.3:
            reasons.append("グラフ構造と協調フィルタリングの両方で高評価")

        return reasons if reasons else ["推薦システムによる提案"]

    def explain_recommendation(self,
                               member_code: str,
                               competence_code: str) -> Dict:
        """
        推薦の詳細な説明を生成

        Args:
            member_code: メンバーコード
            competence_code: 力量コード

        Returns:
            説明情報の辞書
        """
        # RWRの説明を取得
        rwr_explanation = self.rwr.explain_recommendation(
            member_code=member_code,
            competence_code=competence_code
        )

        # スコアを再計算
        # （実装は省略、実際の推薦時のスコアをキャッシュする方が効率的）

        return {
            'member_code': member_code,
            'competence_code': competence_code,
            'rwr_explanation': rwr_explanation,
            'hybrid_score': None,  # 実装省略
        }
