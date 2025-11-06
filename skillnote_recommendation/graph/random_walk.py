"""
Random Walk with Restart (RWR) Recommender

グラフ上のランダムウォークに基づく推薦アルゴリズム。
メンバーノードから開始して、力量・カテゴリーを経由して新しい力量を発見する。

主な機能:
- PageRankベースのグラフ推薦
- カテゴリーベースのフォールバック
- 類似メンバーベースのフォールバック
- 推薦パスの可視化
"""

import logging
import networkx as nx
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass

from .knowledge_graph import CompetenceKnowledgeGraph


# ===================================================================
# Constants
# ===================================================================

DEFAULT_RESTART_PROB = 0.15
DEFAULT_MAX_ITER = 100
DEFAULT_TOLERANCE = 1e-6
DEFAULT_MAX_PATHS = 10
DEFAULT_MAX_PATH_LENGTH = 10
MIN_SCORE_THRESHOLD = 1e-10
CANDIDATE_MULTIPLIER = 3

logger = logging.getLogger(__name__)


# ===================================================================
# Data Models
# ===================================================================

@dataclass
class RecommendationStats:
    """推薦統計情報"""
    total_nodes: int
    acquired_count: int
    all_competence_count: int
    excluded_acquired: int
    excluded_type: int
    candidate_count: int
    fallback_used: bool = False


# ===================================================================
# Random Walk Recommender
# ===================================================================

class RandomWalkRecommender:
    """RWRベースの推薦エンジン

    Random Walk with Restart (RWR) アルゴリズムを使用して、
    グラフ構造に基づく力量推薦を行う。

    Attributes:
        graph: NetworkXグラフオブジェクト
        kg: CompetenceKnowledgeGraphインスタンス
        restart_prob: 再スタート確率
        max_iter: PageRankの最大反復回数
        tolerance: PageRank収束判定の閾値
        max_path_length: 推薦パスの最大長さ
        max_paths: 各力量に対して抽出するパスの最大数
        enable_cache: PageRank結果のキャッシュを有効にするか
    """

    def __init__(
        self,
        knowledge_graph: CompetenceKnowledgeGraph,
        restart_prob: float = DEFAULT_RESTART_PROB,
        max_iter: int = DEFAULT_MAX_ITER,
        tolerance: float = DEFAULT_TOLERANCE,
        max_path_length: int = DEFAULT_MAX_PATH_LENGTH,
        max_paths: int = DEFAULT_MAX_PATHS,
        enable_cache: bool = True
    ):
        """
        Args:
            knowledge_graph: ナレッジグラフ
            restart_prob: 再スタート確率（PageRankのダンピング係数相当）
            max_iter: 最大反復回数
            tolerance: 収束判定の閾値
            max_path_length: 推薦パスの最大長さ（ステップ数）
            max_paths: 各力量に対して抽出する推薦パスの最大数
            enable_cache: PageRank結果のキャッシュを有効にするか
        """
        self.graph = knowledge_graph.G
        self.kg = knowledge_graph
        self.restart_prob = restart_prob
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.max_path_length = max_path_length
        self.max_paths = max_paths
        self.enable_cache = enable_cache

        # PageRank結果のキャッシュ
        self._pagerank_cache: Dict[str, Dict[str, float]] = {}

    # ===============================================================
    # Public Methods
    # ===============================================================

    def recommend(
        self,
        member_code: str,
        top_n: int = 10,
        return_paths: bool = True,
        competence_type: Optional[List[str]] = None
    ) -> List[Tuple[str, float, List[List[str]]]]:
        """RWRで力量を推薦

        Args:
            member_code: 対象メンバーコード
            top_n: 推薦件数
            return_paths: 推薦パスを返すかどうか
            competence_type: フィルタする力量タイプのリスト

        Returns:
            [(力量コード, スコア, 推薦パス), ...]

        Raises:
            ValueError: メンバーがグラフに存在しない場合
        """
        member_node = f"member_{member_code}"

        if not self.graph.has_node(member_node):
            raise ValueError(f"メンバー {member_code} がグラフに存在しません")

        logger.info(f"RWR推薦開始: member={member_code}, top_n={top_n}")

        # 1. RWRスコアを計算
        scores = self._calculate_rwr_scores(member_node)

        # 2. 力量候補を抽出
        acquired_competences = self.kg.get_member_acquired_competences(member_code)
        competence_scores, stats = self._extract_competence_candidates(
            scores, acquired_competences, competence_type
        )

        # 3. フォールバックが必要な場合
        if len(competence_scores) == 0:
            logger.info("候補が0件のため、フォールバック推薦を実行")
            competence_scores = self._apply_fallback(
                member_code, acquired_competences, competence_type
            )
            stats.fallback_used = True

        # 4. Top-N選択とパス抽出
        recommendations = self._select_top_n_with_paths(
            member_node, competence_scores, top_n, return_paths
        )

        logger.info(
            f"RWR推薦完了: recommendations={len(recommendations)}, "
            f"fallback={stats.fallback_used}"
        )

        return recommendations

    def clear_cache(self):
        """PageRankキャッシュをクリア"""
        self._pagerank_cache.clear()
        logger.info("PageRankキャッシュをクリアしました")

    def get_cache_stats(self) -> Dict[str, int]:
        """キャッシュ統計を取得"""
        return {
            'cached_members': len(self._pagerank_cache),
            'total_nodes': len(self.graph.nodes())
        }

    def explain_recommendation(
        self,
        member_code: str,
        competence_code: str,
        max_paths: int = 3
    ) -> Dict:
        """推薦の説明を生成

        Args:
            member_code: メンバーコード
            competence_code: 力量コード
            max_paths: 説明するパス数

        Returns:
            説明情報の辞書
        """
        member_node = f"member_{member_code}"
        competence_node = f"competence_{competence_code}"

        # パスを抽出
        paths = self._extract_paths(member_node, competence_node, max_paths=max_paths)

        # パスを人間が読める形式に変換
        readable_paths = self._convert_paths_to_readable(paths)

        # 推薦理由を生成
        reasons = self._generate_reasons(paths)

        return {
            'member_code': member_code,
            'competence_code': competence_code,
            'paths': readable_paths,
            'reasons': reasons,
        }

    # ===============================================================
    # Private Methods - Core Algorithm
    # ===============================================================

    def _calculate_rwr_scores(self, start_node: str) -> Dict[str, float]:
        """RWRスコアを計算（PageRank使用）

        Args:
            start_node: 開始ノード（メンバーノード）

        Returns:
            {ノードID: 訪問確率スコア}
        """
        # キャッシュチェック
        if self.enable_cache and start_node in self._pagerank_cache:
            logger.debug(f"キャッシュヒット: {start_node}")
            return self._pagerank_cache[start_node]

        # NetworkXのPersonalized PageRankを使用
        scores = nx.pagerank(
            self.graph,
            alpha=1 - self.restart_prob,
            personalization={start_node: 1.0},
            max_iter=self.max_iter,
            tol=self.tolerance,
            weight='weight'
        )

        # 非常に小さいスコアは除外
        filtered_scores = {
            node: score for node, score in scores.items()
            if score > MIN_SCORE_THRESHOLD
        }

        # キャッシュに保存
        if self.enable_cache:
            self._pagerank_cache[start_node] = filtered_scores

        logger.debug(f"PageRank完了: {len(filtered_scores)}ノード")

        return filtered_scores

    def _extract_competence_candidates(
        self,
        scores: Dict[str, float],
        acquired_competences: Set[str],
        competence_type: Optional[List[str]]
    ) -> Tuple[List[Tuple[str, float]], RecommendationStats]:
        """力量候補を抽出してフィルタリング

        Args:
            scores: RWRスコア
            acquired_competences: 既習得力量コード
            competence_type: 力量タイプフィルタ

        Returns:
            (候補リスト, 統計情報)
        """
        # 統計情報を初期化
        stats = RecommendationStats(
            total_nodes=len(scores),
            acquired_count=len(acquired_competences),
            all_competence_count=0,
            excluded_acquired=0,
            excluded_type=0,
            candidate_count=0
        )

        # 力量ノードを抽出
        all_competence_scores = [
            (node.replace("competence_", ""), score, node)
            for node, score in scores.items()
            if node.startswith("competence_")
        ]
        stats.all_competence_count = len(all_competence_scores)

        # フィルタリング
        competence_scores = []
        for comp_code, score, node in all_competence_scores:
            # 既習得力量を除外
            if comp_code in acquired_competences:
                stats.excluded_acquired += 1
                continue

            # 力量タイプフィルタ
            if competence_type is not None:
                comp_info = self.kg.get_node_info(node)
                comp_type = comp_info.get('type', comp_info.get('competence_type', 'UNKNOWN'))
                if comp_type not in competence_type:
                    stats.excluded_type += 1
                    continue

            competence_scores.append((comp_code, score))

        stats.candidate_count = len(competence_scores)

        logger.info(
            f"力量候補抽出: 全{stats.all_competence_count}個 → "
            f"除外({stats.excluded_acquired}+{stats.excluded_type}) → "
            f"候補{stats.candidate_count}個"
        )

        return competence_scores, stats

    def _select_top_n_with_paths(
        self,
        member_node: str,
        competence_scores: List[Tuple[str, float]],
        top_n: int,
        return_paths: bool
    ) -> List[Tuple[str, float, List[List[str]]]]:
        """Top-N選択とパス抽出

        Args:
            member_node: メンバーノード
            competence_scores: [(力量コード, スコア), ...]
            top_n: 推薦件数
            return_paths: パスを返すかどうか

        Returns:
            [(力量コード, スコア, 推薦パス), ...]
        """
        # スコアでソート
        sorted_scores = sorted(competence_scores, key=lambda x: x[1], reverse=True)[:top_n]

        # パス抽出
        recommendations = []
        for comp_code, score in sorted_scores:
            paths = []
            if return_paths:
                paths = self._extract_paths(
                    member_node,
                    f"competence_{comp_code}",
                    max_paths=self.max_paths,
                    max_length=self.max_path_length
                )
            recommendations.append((comp_code, score, paths))

        return recommendations

    # ===============================================================
    # Private Methods - Fallback Logic
    # ===============================================================

    def _apply_fallback(
        self,
        member_code: str,
        acquired_competences: Set[str],
        competence_type: Optional[List[str]]
    ) -> List[Tuple[str, float]]:
        """フォールバック推薦を適用

        Args:
            member_code: メンバーコード
            acquired_competences: 既習得力量
            competence_type: 力量タイプフィルタ

        Returns:
            [(力量コード, スコア), ...]
        """
        # 1. カテゴリーベース推薦
        category_scores = self._category_based_fallback(
            acquired_competences, competence_type
        )

        # 2. 類似メンバーベース推薦（候補が少ない場合）
        if len(category_scores) < 5:
            logger.info("候補が少ないため、類似メンバーベース推薦を追加")
            similar_scores = self._similar_member_fallback(
                member_code, acquired_competences, competence_type
            )
            category_scores.extend(similar_scores)

        # 重複除去
        return self._deduplicate_scores(category_scores)

    def _category_based_fallback(
        self,
        acquired_competences: Set[str],
        competence_type: Optional[List[str]]
    ) -> List[Tuple[str, float]]:
        """カテゴリーベースのフォールバック推薦

        既習得力量と同じカテゴリーの未習得力量を推薦

        Args:
            acquired_competences: 既習得力量コードのセット
            competence_type: 力量タイプフィルタ

        Returns:
            [(力量コード, スコア), ...]
        """
        # 既習得力量のカテゴリーを取得
        acquired_categories = {
            self.kg.get_competence_category(comp_code)
            for comp_code in acquired_competences
        }
        acquired_categories.discard(None)

        logger.debug(f"既習得力量のカテゴリー: {len(acquired_categories)}個")

        # 同じカテゴリーの未習得力量を探す
        fallback_scores = []
        for category in acquired_categories:
            category_node = f"category_{category}"
            if not self.graph.has_node(category_node):
                continue

            # カテゴリーに属する力量を取得
            for neighbor in self.graph.neighbors(category_node):
                if not neighbor.startswith("competence_"):
                    continue

                comp_code = neighbor.replace("competence_", "")

                # フィルタリング
                if self._should_exclude_competence(
                    comp_code, neighbor, acquired_competences, competence_type
                ):
                    continue

                # カテゴリーベーススコア（一定値）
                fallback_scores.append((comp_code, 0.001))

        logger.info(f"カテゴリーベース推薦: {len(fallback_scores)}個")
        return fallback_scores

    def _similar_member_fallback(
        self,
        member_code: str,
        acquired_competences: Set[str],
        competence_type: Optional[List[str]]
    ) -> List[Tuple[str, float]]:
        """類似メンバーベースのフォールバック推薦

        Args:
            member_code: メンバーコード
            acquired_competences: 既習得力量コードのセット
            competence_type: 力量タイプフィルタ

        Returns:
            [(力量コード, スコア), ...]
        """
        member_node = f"member_{member_code}"

        # 類似メンバーを探す
        similar_members = self._find_similar_members(member_node)
        logger.debug(f"類似メンバー: {len(similar_members)}名")

        # 類似メンバーの保有力量を推薦
        fallback_scores = []
        for similar_member_node, similarity in similar_members[:3]:  # 上位3名
            for neighbor in self.graph.neighbors(similar_member_node):
                if not neighbor.startswith("competence_"):
                    continue

                comp_code = neighbor.replace("competence_", "")

                # フィルタリング
                if self._should_exclude_competence(
                    comp_code, neighbor, acquired_competences, competence_type
                ):
                    continue

                # 類似度ベーススコア
                score = 0.0005 * similarity
                fallback_scores.append((comp_code, score))

        logger.info(f"類似メンバーベース推薦: {len(fallback_scores)}個")
        return fallback_scores

    # ===============================================================
    # Private Methods - Path Extraction
    # ===============================================================

    def _extract_paths(
        self,
        source: str,
        target: str,
        max_paths: int = DEFAULT_MAX_PATHS,
        max_length: int = DEFAULT_MAX_PATH_LENGTH
    ) -> List[List[str]]:
        """推薦パスを抽出

        Args:
            source: 開始ノード（メンバー）
            target: 終了ノード（力量）
            max_paths: 抽出する最大パス数
            max_length: 最大パス長

        Returns:
            [[node1, node2, node3], ...] のパスリスト
        """
        try:
            # k-shortest pathsアルゴリズム使用
            path_generator = nx.shortest_simple_paths(
                self.graph, source, target, weight=None
            )

            # パス抽出
            paths = []
            rejected_count = 0
            for path in path_generator:
                if len(path) - 1 <= max_length:
                    paths.append(path)
                    if len(paths) >= max_paths:
                        break
                else:
                    rejected_count += 1
                    if rejected_count >= max_paths * 3:
                        break

            if paths:
                logger.debug(f"パス抽出成功: {len(paths)}個 (除外: {rejected_count}個)")
                return paths

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass

        # フォールバック: 代替パスを生成
        return self._generate_fallback_paths(source, target, max_paths)

    def _generate_fallback_paths(
        self,
        source: str,
        target: str,
        max_paths: int
    ) -> List[List[str]]:
        """代替パスを生成

        Args:
            source: 開始ノード
            target: 終了ノード
            max_paths: 最大パス数

        Returns:
            代替パスのリスト
        """
        fallback_paths = []

        # 1. カテゴリー経由パス
        category_path = self._create_category_path(target)
        if category_path:
            fallback_paths.append([source] + category_path)
            logger.debug("フォールバック: カテゴリー経由パス生成")

        # 2. 既習得力量経由パス
        if len(fallback_paths) < max_paths:
            competence_paths = self._create_competence_paths(source, target, max_paths)
            fallback_paths.extend(competence_paths)

        # 3. 類似メンバー経由パス
        if len(fallback_paths) < max_paths:
            member_paths = self._create_similar_member_paths(source, target, max_paths)
            fallback_paths.extend(member_paths)

        # 4. 最低限のパス（直接パス）
        if not fallback_paths:
            fallback_paths.append([source, target])
            logger.debug("フォールバック: 直接パス生成")

        return fallback_paths[:max_paths]

    def _create_category_path(self, target: str) -> Optional[List[str]]:
        """カテゴリー経由パスを作成"""
        if not self.graph.has_node(target):
            return None

        target_data = self.kg.get_node_info(target)
        target_category = target_data.get('category')

        if target_category:
            category_node = f"category_{target_category}"
            if self.graph.has_node(category_node):
                return [category_node, target]

        return None

    def _create_competence_paths(
        self,
        source: str,
        target: str,
        max_paths: int
    ) -> List[List[str]]:
        """既習得力量経由パスを作成"""
        paths = []

        try:
            member_neighbors = list(self.graph.neighbors(source))
            target_data = self.kg.get_node_info(target)
            target_category = target_data.get('category')

            for neighbor in member_neighbors[:3]:
                if not neighbor.startswith("competence_"):
                    continue

                neighbor_data = self.kg.get_node_info(neighbor)
                neighbor_category = neighbor_data.get('category')

                if neighbor_category == target_category and neighbor_category:
                    category_node = f"category_{target_category}"
                    if self.graph.has_node(category_node):
                        paths.append([source, neighbor, category_node, target])
                        logger.debug(f"フォールバック: 既習得力量経由パス生成")
                        if len(paths) >= max_paths:
                            break
        except Exception as e:
            logger.warning(f"既習得力量経由パス生成エラー: {e}")

        return paths

    def _create_similar_member_paths(
        self,
        source: str,
        target: str,
        max_paths: int
    ) -> List[List[str]]:
        """類似メンバー経由パスを作成"""
        paths = []

        try:
            similar_members = self._find_similar_members(source)

            for similar_member, _ in similar_members[:2]:
                if self.graph.has_edge(similar_member, target):
                    paths.append([source, similar_member, target])
                    logger.debug("フォールバック: 類似メンバー経由パス生成")
                    if len(paths) >= max_paths:
                        break
        except Exception as e:
            logger.warning(f"類似メンバー経由パス生成エラー: {e}")

        return paths

    # ===============================================================
    # Private Methods - Utilities
    # ===============================================================

    def _find_similar_members(self, member_node: str) -> List[Tuple[str, float]]:
        """類似メンバーを検索

        Args:
            member_node: メンバーノード

        Returns:
            [(類似メンバーノード, 類似度), ...]（類似度降順）
        """
        similar_members = []
        for neighbor in self.graph.neighbors(member_node):
            if neighbor.startswith("member_"):
                edge_data = self.graph[member_node][neighbor]
                if edge_data.get('edge_type') == 'similar':
                    similarity = edge_data.get('similarity', 0.5)
                    similar_members.append((neighbor, similarity))

        return sorted(similar_members, key=lambda x: x[1], reverse=True)

    def _should_exclude_competence(
        self,
        comp_code: str,
        comp_node: str,
        acquired_competences: Set[str],
        competence_type: Optional[List[str]]
    ) -> bool:
        """力量を除外すべきか判定

        Args:
            comp_code: 力量コード
            comp_node: 力量ノード
            acquired_competences: 既習得力量
            competence_type: 力量タイプフィルタ

        Returns:
            True: 除外すべき, False: 含めるべき
        """
        # 既習得チェック
        if comp_code in acquired_competences:
            return True

        # 力量タイプチェック
        if competence_type is not None:
            comp_info = self.kg.get_node_info(comp_node)
            comp_type = comp_info.get('type', comp_info.get('competence_type', 'UNKNOWN'))
            if comp_type not in competence_type:
                return True

        return False

    def _deduplicate_scores(
        self,
        scores: List[Tuple[str, float]]
    ) -> List[Tuple[str, float]]:
        """スコアリストから重複を除去

        Args:
            scores: [(力量コード, スコア), ...]

        Returns:
            重複除去されたリスト
        """
        seen = set()
        unique_scores = []
        for comp_code, score in scores:
            if comp_code not in seen:
                seen.add(comp_code)
                unique_scores.append((comp_code, score))
        return unique_scores

    def _convert_paths_to_readable(
        self,
        paths: List[List[str]]
    ) -> List[List[Dict]]:
        """パスを人間が読める形式に変換

        Args:
            paths: パスのリスト

        Returns:
            読みやすい形式のパスリスト
        """
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

    def _generate_reasons(self, paths: List[List[str]]) -> List[str]:
        """パスから推薦理由を生成

        Args:
            paths: パスのリスト

        Returns:
            推薦理由のリスト
        """
        reasons = []

        for path in paths:
            if len(path) < 3:
                continue

            # パスのパターンを解析
            path_types = [
                self.graph.nodes[node].get('node_type', 'unknown')
                for node in path
            ]

            # パターンごとに説明を生成
            if 'category' in path_types:
                category_node = next(
                    (n for n in path if n.startswith('category_')), None
                )
                if category_node:
                    category_name = self.graph.nodes[category_node].get('name', '')
                    reasons.append(f"同じカテゴリー「{category_name}」の力量として推薦")

            elif path_types.count('member') > 1:
                similar_member_node = (
                    path[1] if len(path) > 1 and path[1].startswith('member_') else None
                )
                if similar_member_node:
                    similar_name = self.graph.nodes[similar_member_node].get('name', '')
                    reasons.append(
                        f"類似メンバー「{similar_name}」が保有している力量として推薦"
                    )

            else:
                reasons.append("グラフ構造に基づく推薦")

        # 重複を削除
        return list(set(reasons)) if reasons else ["グラフ構造に基づく推薦"]
