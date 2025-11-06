"""
Hybrid Recommender: RWR + NMF + Content-Based

Random Walk with Restart、matrix factorization (NMF)、
コンテンツベース推薦を組み合わせたハイブリッド推薦システム
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass

from .knowledge_graph import CompetenceKnowledgeGraph
from .random_walk import RandomWalkRecommender
from ..ml.ml_recommender import MLRecommender
from ..ml.content_based_recommender import ContentBasedRecommender
from ..ml.feature_engineering import FeatureEngineer
from ..core.models import Recommendation


# デフォルト設定値
DEFAULT_GRAPH_WEIGHT = 0.4    # グラフベース（RWR）の重み
DEFAULT_CF_WEIGHT = 0.3       # 協調フィルタリング（NMF）の重み
DEFAULT_CONTENT_WEIGHT = 0.3  # コンテンツベースの重み
DEFAULT_RESTART_PROB = 0.15
CANDIDATE_MULTIPLIER = 3  # 候補を多めに取得する倍率
HYBRID_BOOST_FACTOR = 1.15  # 複数手法で推薦された場合のブースト
RWR_SCORE_THRESHOLD_HIGH = 0.5  # 推薦理由生成の閾値（高）
RWR_SCORE_THRESHOLD_LOW = 0.3  # 推薦理由生成の閾値（低）
NMF_SCORE_THRESHOLD_HIGH = 0.5
NMF_SCORE_THRESHOLD_LOW = 0.3
CONTENT_SCORE_THRESHOLD_HIGH = 0.5
CONTENT_SCORE_THRESHOLD_LOW = 0.3


@dataclass
class HybridRecommendation:
    """ハイブリッド推薦結果

    Attributes:
        competence_code: 力量コード
        score: 最終スコア（ハイブリッド）
        graph_score: グラフベーススコア（RWR）
        cf_score: 協調フィルタリングスコア（NMF）
        content_score: コンテンツベーススコア
        paths: 推薦パス
        reasons: 推薦理由
        competence_info: 力量情報
    """
    competence_code: str
    score: float
    graph_score: float
    cf_score: float
    content_score: float
    paths: List[List[Dict]]
    reasons: List[str]
    competence_info: Dict


class HybridGraphRecommender:
    """RWR + NMF + Content-Basedを融合したハイブリッド推薦エンジン

    グラフ構造（RWR）、協調フィルタリング（NMF）、
    コンテンツベース（属性情報）の強みを組み合わせて、
    より精度の高い推薦を実現する。

    推薦プロセス:
        1. RWRでグラフベースのスコアを計算
        2. NMFで協調フィルタリングベースのスコアを計算
        3. コンテンツベースで属性ベースのスコアを計算
        4. 3つのスコアを融合して最終ランキングを生成
        5. 推薦パスと理由を付与
    """

    def __init__(self,
                 knowledge_graph: CompetenceKnowledgeGraph,
                 ml_recommender: MLRecommender,
                 content_recommender: ContentBasedRecommender,
                 feature_engineer: FeatureEngineer,
                 graph_weight: float = DEFAULT_GRAPH_WEIGHT,
                 cf_weight: float = DEFAULT_CF_WEIGHT,
                 content_weight: float = DEFAULT_CONTENT_WEIGHT,
                 restart_prob: float = DEFAULT_RESTART_PROB,
                 max_path_length: int = 10,
                 max_paths: int = 10,
                 enable_cache: bool = True):
        """
        Args:
            knowledge_graph: ナレッジグラフ
            ml_recommender: NMFベースのML推薦エンジン
            content_recommender: コンテンツベース推薦エンジン
            feature_engineer: 特徴量エンジニア
            graph_weight: グラフベーススコアの重み
            cf_weight: 協調フィルタリングスコアの重み
            content_weight: コンテンツベーススコアの重み
            restart_prob: RWRの再スタート確率
            max_path_length: 推薦パスの最大長さ（ステップ数）
            max_paths: 各力量に対して抽出する推薦パスの最大数
            enable_cache: RWRのPageRankキャッシュを有効にするか
        """
        self.kg = knowledge_graph
        self.ml_recommender = ml_recommender
        self.content_recommender = content_recommender
        self.fe = feature_engineer

        self.rwr = RandomWalkRecommender(
            knowledge_graph=knowledge_graph,
            restart_prob=restart_prob,
            max_path_length=max_path_length,
            max_paths=max_paths,
            enable_cache=enable_cache
        )

        # 重みの正規化
        total_weight = graph_weight + cf_weight + content_weight
        self.graph_weight = graph_weight / total_weight
        self.cf_weight = cf_weight / total_weight
        self.content_weight = content_weight / total_weight

        print(f"\nHybrid Recommender 初期化完了")
        print(f"  グラフベース重み: {self.graph_weight:.2f}")
        print(f"  協調フィルタリング重み: {self.cf_weight:.2f}")
        print(f"  コンテンツベース重み: {self.content_weight:.2f}")
        print(f"  キャッシュ: {'有効' if enable_cache else '無効'}")

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
        print("\n[1/5] グラフベース推薦（RWR）...")
        rwr_results = self.rwr.recommend(
            member_code=member_code,
            top_n=top_n * CANDIDATE_MULTIPLIER,  # 多めに取得してフィルタ
            return_paths=True
        )
        rwr_dict = {comp: (score, paths) for comp, score, paths in rwr_results}

        # 2. NMFで推薦
        print("\n[2/5] 協調フィルタリング推薦（NMF）...")
        nmf_results = self.ml_recommender.recommend(
            member_code=member_code,
            top_n=top_n * CANDIDATE_MULTIPLIER,
            competence_type=competence_type,
            category_filter=category_filter,
            use_diversity=use_diversity
        )
        nmf_dict = {rec.competence_code: rec.priority_score for rec in nmf_results}

        # 3. コンテンツベースで推薦
        print("\n[3/5] コンテンツベース推薦...")
        content_results = self.content_recommender.recommend(
            member_code=member_code,
            top_n=top_n * CANDIDATE_MULTIPLIER,
            competence_type=competence_type,
            category_filter=category_filter
        )
        content_dict = {rec.competence_code: rec.priority_score for rec in content_results}

        # 4. スコアを正規化して融合
        print("\n[4/5] スコア融合...")
        hybrid_scores = self._fuse_scores(rwr_dict, nmf_dict, content_dict)

        # 5. フィルタリングとTop-N選択
        print(f"\n[5/5] Top-{top_n}選択...")
        recommendations = self._select_top_n(
            hybrid_scores=hybrid_scores,
            nmf_results=nmf_results,
            content_results=content_results,
            top_n=top_n,
            competence_type=competence_type,
            category_filter=category_filter
        )

        print(f"\n完了: {len(recommendations)}件の推薦を生成")
        return recommendations

    def _fuse_scores(self,
                     rwr_dict: Dict[str, Tuple[float, List]],
                     nmf_dict: Dict[str, float],
                     content_dict: Dict[str, float]) -> Dict[str, Dict]:
        """
        RWR、NMF、コンテンツベースのスコアを融合

        Args:
            rwr_dict: {力量コード: (RWRスコア, パス)}
            nmf_dict: {力量コード: NMFスコア}
            content_dict: {力量コード: コンテンツベーススコア}

        Returns:
            {力量コード: {'score': float, 'graph_score': float, 'cf_score': float,
                        'content_score': float, 'paths': list}}
        """
        # スコアを正規化（0-1）
        rwr_scores_only = {k: v[0] for k, v in rwr_dict.items()}
        rwr_normalized = self._normalize_scores(rwr_scores_only)
        nmf_normalized = self._normalize_scores(nmf_dict)
        content_normalized = self._normalize_scores(content_dict)

        # 全力量コードを取得
        all_competences = set(rwr_dict.keys()) | set(nmf_dict.keys()) | set(content_dict.keys())

        # ハイブリッドスコア計算
        hybrid_scores = {}
        for comp_code in all_competences:
            graph_score = rwr_normalized.get(comp_code, 0.0)
            cf_score = nmf_normalized.get(comp_code, 0.0)
            content_score = content_normalized.get(comp_code, 0.0)

            # 重み付き平均
            hybrid_score = (
                self.graph_weight * graph_score +
                self.cf_weight * cf_score +
                self.content_weight * content_score
            )

            # 複数手法で推薦された場合はブーストを与える
            recommendation_count = sum([
                comp_code in rwr_dict,
                comp_code in nmf_dict,
                comp_code in content_dict
            ])

            if recommendation_count >= 2:
                boost = 1.0 + (recommendation_count - 1) * (HYBRID_BOOST_FACTOR - 1.0)
                hybrid_score *= boost

            hybrid_scores[comp_code] = {
                'score': hybrid_score,
                'graph_score': graph_score,
                'cf_score': cf_score,
                'content_score': content_score,
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
                      content_results: List[Recommendation],
                      top_n: int,
                      competence_type: Optional[List[str]],
                      category_filter: Optional[str]) -> List[HybridRecommendation]:
        """
        Top-N推薦を選択し、HybridRecommendationオブジェクトを生成

        Args:
            hybrid_scores: ハイブリッドスコア辞書
            nmf_results: NMF推薦結果（力量情報を取得するため）
            content_results: コンテンツベース推薦結果
            top_n: 推薦件数
            competence_type: 力量タイプフィルタ
            category_filter: カテゴリーフィルタ

        Returns:
            HybridRecommendationのリスト
        """
        # 推薦結果から力量情報を取得（NMFとコンテンツベースから）
        competence_info_map = {}

        for rec in nmf_results:
            competence_info_map[rec.competence_code] = {
                '力量名': rec.competence_name,
                '力量タイプ': rec.competence_type,
                'カテゴリー': rec.category,
                '概要': None,
            }

        for rec in content_results:
            if rec.competence_code not in competence_info_map:
                competence_info_map[rec.competence_code] = {
                    '力量名': rec.competence_name,
                    '力量タイプ': rec.competence_type,
                    'カテゴリー': rec.category,
                    '概要': None,
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
                graph_score=data['graph_score'],
                cf_score=data['cf_score'],
                content_score=data['content_score'],
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

        # グラフベーススコアが高い場合
        if data['graph_score'] > RWR_SCORE_THRESHOLD_HIGH:
            # パスから理由を生成
            paths = data['paths']
            if paths:
                path_reasons = self.rwr._generate_reasons(paths)
                reasons.extend(path_reasons)

        # 協調フィルタリングスコアが高い場合
        if data['cf_score'] > NMF_SCORE_THRESHOLD_HIGH:
            reasons.append("類似メンバーの習得パターンから推薦")

        # コンテンツベーススコアが高い場合
        if data['content_score'] > CONTENT_SCORE_THRESHOLD_HIGH:
            reasons.append("あなたの職種・等級・習得履歴と親和性が高い")

        # 複数手法で高評価の場合
        high_scores = sum([
            data['graph_score'] > RWR_SCORE_THRESHOLD_LOW,
            data['cf_score'] > NMF_SCORE_THRESHOLD_LOW,
            data['content_score'] > CONTENT_SCORE_THRESHOLD_LOW
        ])

        if high_scores >= 2:
            methods = []
            if data['graph_score'] > RWR_SCORE_THRESHOLD_LOW:
                methods.append("グラフ構造")
            if data['cf_score'] > NMF_SCORE_THRESHOLD_LOW:
                methods.append("協調フィルタリング")
            if data['content_score'] > CONTENT_SCORE_THRESHOLD_LOW:
                methods.append("コンテンツベース")

            reasons.append(f"複数の推薦手法（{', '.join(methods)}）で高評価")

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
