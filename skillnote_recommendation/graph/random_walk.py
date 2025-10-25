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
                 tolerance: float = 1e-6):
        """
        Args:
            knowledge_graph: ナレッジグラフ
            restart_prob: 再スタート確率（PageRankのダンピング係数相当）
                         0.15 = スタート地点に15%の確率で戻る
            max_iter: 最大反復回数
            tolerance: 収束判定の閾値
        """
        self.graph = knowledge_graph.G
        self.kg = knowledge_graph
        self.restart_prob = restart_prob
        self.max_iter = max_iter
        self.tolerance = tolerance

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
        RWRアルゴリズムの実装

        Args:
            start_node: 開始ノード（メンバーノード）

        Returns:
            {ノードID: 訪問確率スコア}
        """
        # ノードリストとインデックスマッピング
        nodes = list(self.graph.nodes())
        n_nodes = len(nodes)
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        idx_to_node = {i: node for i, node in enumerate(nodes)}

        # 遷移行列を構築
        transition_matrix = self._build_transition_matrix(nodes, node_to_idx)

        # 初期ベクトル（スタートノードのみ1）
        start_vector = np.zeros(n_nodes)
        start_idx = node_to_idx[start_node]
        start_vector[start_idx] = 1.0

        # RWR反復計算
        current_vector = start_vector.copy()

        for iteration in range(self.max_iter):
            # 次のステップの確率ベクトルを計算
            # RWRの式: r(t+1) = (1-c) * P^T * r(t) + c * r(0)
            # c = restart_prob, P = 遷移行列, r(0) = 初期ベクトル
            next_vector = (
                (1 - self.restart_prob) * (transition_matrix.T @ current_vector) +
                self.restart_prob * start_vector
            )

            # 収束判定
            diff = np.linalg.norm(next_vector - current_vector, ord=1)
            if diff < self.tolerance:
                print(f"    収束: {iteration + 1}回反復")
                break

            current_vector = next_vector

        else:
            print(f"    警告: 最大反復回数 {self.max_iter} に達しました")

        # スコア辞書に変換
        scores = {}
        for i in range(n_nodes):
            if current_vector[i] > 1e-10:  # 非常に小さいスコアは除外
                scores[idx_to_node[i]] = float(current_vector[i])

        return scores

    def _build_transition_matrix(self,
                                  nodes: List[str],
                                  node_to_idx: Dict[str, int]) -> np.ndarray:
        """
        重み付き遷移行列を構築

        Args:
            nodes: ノードリスト
            node_to_idx: ノードID → インデックスのマッピング

        Returns:
            遷移行列 (n_nodes × n_nodes)
        """
        n = len(nodes)
        matrix = np.zeros((n, n))

        for node in nodes:
            i = node_to_idx[node]
            neighbors = list(self.graph.neighbors(node))

            if len(neighbors) == 0:
                # 孤立ノードの場合、自己ループ
                matrix[i][i] = 1.0
                continue

            # 重み付きで正規化
            # エッジタイプによって重みを調整することも可能
            total_weight = 0.0
            neighbor_weights = []

            for neighbor in neighbors:
                edge_data = self.graph[node][neighbor]
                weight = edge_data.get('weight', 1.0)

                # エッジタイプによる重み調整（オプション）
                edge_type = edge_data.get('edge_type', 'unknown')
                if edge_type == 'acquired':
                    weight *= 1.5  # 習得関係を強調
                elif edge_type == 'belongs_to':
                    weight *= 1.0  # カテゴリー関係は標準
                elif edge_type == 'similar':
                    weight *= 1.2  # 類似関係をやや強調

                neighbor_weights.append((neighbor, weight))
                total_weight += weight

            # 正規化して遷移確率を設定
            for neighbor, weight in neighbor_weights:
                j = node_to_idx[neighbor]
                matrix[i][j] = weight / total_weight

        return matrix

    def _extract_paths(self,
                       source: str,
                       target: str,
                       max_paths: int = 3,
                       max_length: int = 5) -> List[List[str]]:
        """
        推薦パスを抽出（重要！可視化に使用）

        Args:
            source: 開始ノード（メンバー）
            target: 終了ノード（力量）
            max_paths: 抽出する最大パス数
            max_length: 最大パス長

        Returns:
            [[node1, node2, node3], ...] のパスリスト
        """
        try:
            # 単純パス列挙（長さ制限付き）
            all_paths = nx.all_simple_paths(
                self.graph,
                source,
                target,
                cutoff=max_length
            )

            # パスをスコアリング（エッジ重みの積）
            scored_paths = []
            for path in all_paths:
                score = self._score_path(path)
                scored_paths.append((path, score))

            # スコア順にソート
            scored_paths.sort(key=lambda x: x[1], reverse=True)

            # Top-N パスを返す
            return [path for path, score in scored_paths[:max_paths]]

        except nx.NetworkXNoPath:
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
