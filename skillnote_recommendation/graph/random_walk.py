"""
Random Walk with Restart (RWR) アルゴリズム

グラフ上のランダムウォークに基づく推薦アルゴリズム。
メンバーノードから開始して、力量・カテゴリーを経由して
新しい力量を発見する。
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Set, Optional
from .knowledge_graph import CompetenceKnowledgeGraph


class RandomWalkRecommender:
    """RWRベースの推薦エンジン

    Random Walk with Restart (RWR) アルゴリズムを使用して、
    グラフ構造に基づく力量推薦を行う。
    """

    def __init__(self,
                 knowledge_graph: CompetenceKnowledgeGraph,
                 restart_prob: float = 0.15,
                 max_iter: int = 100,
                 tolerance: float = 1e-6,
                 enable_cache: bool = True):
        """
        Args:
            knowledge_graph: ナレッジグラフ
            restart_prob: 再スタート確率（PageRankのダンピング係数相当）
                         0.15 = スタート地点に15%の確率で戻る
            max_iter: 最大反復回数
            tolerance: 収束判定の閾値
            enable_cache: PageRank結果のキャッシュを有効にするか
        """
        self.graph = knowledge_graph.G
        self.kg = knowledge_graph
        self.restart_prob = restart_prob
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.enable_cache = enable_cache

        # Plan 3: PageRank結果のキャッシュ
        self._pagerank_cache: Dict[str, Dict[str, float]] = {}

    def recommend(self,
                  member_code: str,
                  top_n: int = 10,
                  return_paths: bool = True) -> List[Tuple[str, float, List[List[str]]]]:
        """
        RWRで力量を推薦

        Args:
            member_code: 対象メンバーコード
            top_n: 推薦件数
            return_paths: 推薦パスを返すかどうか

        Returns:
            [(力量コード, スコア, 推薦パス), ...]
            推薦パスは [[node1, node2, ...], ...] の形式
        """
        member_node = f"member_{member_code}"

        if not self.graph.has_node(member_node):
            raise ValueError(f"メンバー {member_code} がグラフに存在しません")

        print(f"\n{'='*80}")
        print(f"RWR推薦: {member_code}")
        print(f"{'='*80}")

        # 1. RWRスコアを計算
        print("  [1/3] RWRスコア計算中...")
        scores = self._random_walk_with_restart(member_node)

        # 2. 力量ノードのみ抽出してソート
        print("  [2/3] 力量スコア抽出中...")
        competence_scores = []
        acquired_competences = self.kg.get_member_acquired_competences(member_code)

        for node, score in scores.items():
            if node.startswith("competence_"):
                comp_code = node.replace("competence_", "")

                # 既に習得済みの力量は除外
                if comp_code not in acquired_competences:
                    competence_scores.append((comp_code, score))

        competence_scores.sort(key=lambda x: x[1], reverse=True)

        # 3. Top-N推薦と推薦パスを抽出
        print(f"  [3/3] Top-{top_n}推薦とパス抽出中...")
        recommendations = []

        for comp_code, score in competence_scores[:top_n]:
            paths = []
            if return_paths:
                paths = self._extract_paths(
                    member_node,
                    f"competence_{comp_code}",
                    max_paths=3,
                    max_length=5
                )
            recommendations.append((comp_code, score, paths))

        print(f"  完了: {len(recommendations)}件の推薦を生成")

        return recommendations

    def _random_walk_with_restart(self, start_node: str) -> Dict[str, float]:
        """
        RWRアルゴリズムの実装（NetworkX PageRank最適化版 + キャッシング）

        Args:
            start_node: 開始ノード（メンバーノード）

        Returns:
            {ノードID: 訪問確率スコア}
        """
        # Plan 3: キャッシュチェック
        if self.enable_cache and start_node in self._pagerank_cache:
            print(f"    キャッシュヒット: {start_node}")
            return self._pagerank_cache[start_node]

        # NetworkXのPersonalized PageRankを使用（高速化）
        # alpha = 1 - restart_prob (PageRankのダンピング係数)
        # personalization = {start_node: 1.0} (RWRの開始ノード)
        scores = nx.pagerank(
            self.graph,
            alpha=1 - self.restart_prob,
            personalization={start_node: 1.0},
            max_iter=self.max_iter,
            tol=self.tolerance,
            weight='weight'  # エッジ重みを考慮
        )

        # 非常に小さいスコアは除外
        filtered_scores = {
            node: score for node, score in scores.items()
            if score > 1e-10
        }

        # Plan 3: キャッシュに保存
        if self.enable_cache:
            self._pagerank_cache[start_node] = filtered_scores

        print(f"    完了: NetworkX PageRank使用（高速化版）")

        return filtered_scores

    # _build_transition_matrix メソッドは削除
    # NetworkXのPageRankが内部で効率的に遷移行列を処理するため不要になりました

    def clear_cache(self):
        """PageRankキャッシュをクリア"""
        self._pagerank_cache.clear()
        print("PageRankキャッシュをクリアしました")

    def get_cache_stats(self) -> Dict[str, int]:
        """キャッシュ統計を取得"""
        return {
            'cached_members': len(self._pagerank_cache),
            'total_nodes': len(self.graph.nodes())
        }

    def _extract_paths(self,
                       source: str,
                       target: str,
                       max_paths: int = 3,
                       max_length: int = 5) -> List[List[str]]:
        """
        推薦パスを抽出（Plan 4: k-shortest paths最適化版）

        Args:
            source: 開始ノード（メンバー）
            target: 終了ノード（力量）
            max_paths: 抽出する最大パス数
            max_length: 最大パス長

        Returns:
            [[node1, node2, node3], ...] のパスリスト
        """
        try:
            # Plan 4: k-shortest paths アルゴリズム（短い順に生成）
            # all_simple_pathsよりも効率的（全パス列挙を避ける）
            path_generator = nx.shortest_simple_paths(
                self.graph,
                source,
                target,
                weight=None  # ホップ数で最短（重み無視）
            )

            # 最初のmax_paths個のパスのみ取得（長さ制限付き）
            paths = []
            for path in path_generator:
                if len(path) - 1 <= max_length:  # パス長チェック
                    paths.append(path)
                    if len(paths) >= max_paths:
                        break

            return paths

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def _score_path(self, path: List[str]) -> float:
        """
        パスのスコア計算

        エッジ重みの積と、パスの長さを考慮してスコアを計算

        Args:
            path: ノードのリスト

        Returns:
            パススコア
        """
        if len(path) < 2:
            return 0.0

        score = 1.0

        # エッジ重みの積
        for i in range(len(path) - 1):
            edge_data = self.graph[path[i]][path[i+1]]
            weight = edge_data.get('weight', 1.0)
            score *= weight

        # 短いパスを優先（ペナルティ）
        length_penalty = 0.9 ** (len(path) - 2)
        score *= length_penalty

        return score

    def explain_recommendation(self,
                               member_code: str,
                               competence_code: str,
                               max_paths: int = 3) -> Dict:
        """
        推薦の説明を生成

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

        # 推薦理由を生成
        reasons = self._generate_reasons(paths)

        return {
            'member_code': member_code,
            'competence_code': competence_code,
            'paths': readable_paths,
            'reasons': reasons,
        }

    def _generate_reasons(self, paths: List[List[str]]) -> List[str]:
        """
        パスから推薦理由を生成

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
            path_types = [self.graph.nodes[node].get('node_type', 'unknown')
                          for node in path]

            # パターンごとに説明を生成
            if 'category' in path_types:
                # メンバー → 力量 → カテゴリー → 力量 のパターン
                category_node = next((n for n in path if n.startswith('category_')), None)
                if category_node:
                    category_name = self.graph.nodes[category_node].get('name', '')
                    reasons.append(f"同じカテゴリー「{category_name}」の力量として推薦")

            elif path_types.count('member') > 1:
                # メンバー → メンバー → 力量 のパターン（類似メンバー経由）
                similar_member_node = path[1] if len(path) > 1 and path[1].startswith('member_') else None
                if similar_member_node:
                    similar_name = self.graph.nodes[similar_member_node].get('name', '')
                    reasons.append(f"類似メンバー「{similar_name}」が保有している力量として推薦")

            else:
                # その他のパターン
                reasons.append("グラフ構造に基づく推薦")

        # 重複を削除
        return list(set(reasons)) if reasons else ["グラフ構造に基づく推薦"]
