"""
Knowledge Graph構築モジュール

メンバー、力量、カテゴリーのヘテロジニアスグラフを構築し、
グラフベースの推薦アルゴリズムの基盤を提供
"""

import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from sklearn.metrics.pairwise import cosine_similarity


class CompetenceKnowledgeGraph:
    """力量推薦のためのナレッジグラフ

    ノードタイプ:
        - member: メンバーノード
        - competence: 力量ノード
        - category: カテゴリーノード

    エッジタイプ:
        - acquired: メンバー → 力量（習得関係）
        - belongs_to: 力量 → カテゴリー（所属関係）
        - similar: メンバー → メンバー（類似関係）
    """

    def __init__(self,
                 member_competence: pd.DataFrame,
                 member_master: pd.DataFrame,
                 competence_master: pd.DataFrame):
        """
        Args:
            member_competence: メンバー保有力量データ
                必須列: メンバーコード, 力量コード, 正規化レベル
            member_master: メンバーマスタ
                必須列: メンバーコード, メンバー名
            competence_master: 力量マスタ
                必須列: 力量コード, 力量名, 力量タイプ
                任意列: 力量カテゴリー名
        """
        self.G = nx.Graph()
        self.member_competence_df = member_competence
        self.member_master_df = member_master
        self.competence_master_df = competence_master

        # グラフ構築
        print("\n" + "=" * 80)
        print("Knowledge Graph 構築開始")
        print("=" * 80)
        self._build_graph()
        print("\nKnowledge Graph 構築完了")
        print(f"  ノード数: {self.G.number_of_nodes()}")
        print(f"  エッジ数: {self.G.number_of_edges()}")
        self._print_graph_stats()

    def _build_graph(self):
        """グラフ構築のメインロジック"""
        # 1. メンバーノード追加
        print("\n[1/6] メンバーノードを追加中...")
        self._add_member_nodes()

        # 2. 力量ノード追加
        print("[2/6] 力量ノードを追加中...")
        self._add_competence_nodes()

        # 3. カテゴリーノード追加
        print("[3/6] カテゴリーノードを追加中...")
        self._add_category_nodes()

        # 4. メンバー-力量エッジ（習得関係）
        print("[4/6] メンバー-力量エッジを追加中...")
        self._add_member_competence_edges()

        # 5. 力量-カテゴリーエッジ（所属関係）
        print("[5/6] 力量-カテゴリーエッジを追加中...")
        self._add_competence_category_edges()

        # 6. メンバー間類似度エッジ
        print("[6/6] メンバー間類似度エッジを追加中...")
        self._add_member_similarity_edges()

    def _add_member_nodes(self):
        """メンバーノードを追加"""
        for _, row in self.member_master_df.iterrows():
            node_id = f"member_{row['メンバーコード']}"
            self.G.add_node(
                node_id,
                node_type="member",
                code=row['メンバーコード'],
                name=row['メンバー名'],
                grade=row.get('職能等級', None),
                position=row.get('役職', None),
            )
        print(f"  追加: {len(self.member_master_df)}個のメンバーノード")

    def _add_competence_nodes(self):
        """力量ノードを追加"""
        for _, row in self.competence_master_df.iterrows():
            node_id = f"competence_{row['力量コード']}"
            self.G.add_node(
                node_id,
                node_type="competence",
                code=row['力量コード'],
                name=row['力量名'],
                type=row['力量タイプ'],
                category=row.get('力量カテゴリー名', None),
                description=row.get('概要', None),
            )
        print(f"  追加: {len(self.competence_master_df)}個の力量ノード")

    def _add_category_nodes(self):
        """カテゴリーノードを追加"""
        # 力量マスタからユニークなカテゴリーを抽出
        categories = self.competence_master_df['力量カテゴリー名'].dropna().unique()

        for category in categories:
            if category and str(category).strip():
                node_id = f"category_{category}"
                self.G.add_node(
                    node_id,
                    node_type="category",
                    name=category
                )
        print(f"  追加: {len(categories)}個のカテゴリーノード")

    def _add_member_competence_edges(self):
        """メンバー-力量エッジ（習得関係）を追加"""
        edge_count = 0
        for _, row in self.member_competence_df.iterrows():
            member_node = f"member_{row['メンバーコード']}"
            competence_node = f"competence_{row['力量コード']}"

            # 両方のノードが存在する場合のみエッジを追加
            if self.G.has_node(member_node) and self.G.has_node(competence_node):
                self.G.add_edge(
                    member_node,
                    competence_node,
                    edge_type="acquired",
                    weight=float(row['正規化レベル']),
                    level=float(row['正規化レベル'])
                )
                edge_count += 1
        print(f"  追加: {edge_count}本のメンバー-力量エッジ")

    def _add_competence_category_edges(self):
        """力量-カテゴリーエッジ（所属関係）を追加"""
        edge_count = 0
        for _, row in self.competence_master_df.iterrows():
            category = row.get('力量カテゴリー名')
            if pd.notna(category) and str(category).strip():
                competence_node = f"competence_{row['力量コード']}"
                category_node = f"category_{category}"

                # 両方のノードが存在する場合のみエッジを追加
                if self.G.has_node(competence_node) and self.G.has_node(category_node):
                    self.G.add_edge(
                        competence_node,
                        category_node,
                        edge_type="belongs_to",
                        weight=1.0
                    )
                    edge_count += 1
        print(f"  追加: {edge_count}本の力量-カテゴリーエッジ")

    def _add_member_similarity_edges(self, threshold: float = 0.3, top_k: int = 5):
        """メンバー間の類似度エッジを追加

        Args:
            threshold: 類似度の閾値（この値以上のペアにエッジを張る）
            top_k: 各メンバーに対して上位K人までエッジを張る
        """
        # メンバー×力量マトリクスを作成
        member_comp_matrix = self.member_competence_df.pivot_table(
            index='メンバーコード',
            columns='力量コード',
            values='正規化レベル',
            fill_value=0
        )

        # コサイン類似度を計算
        similarity_matrix = cosine_similarity(member_comp_matrix.values)

        # 類似度エッジを追加
        edge_count = 0
        member_codes = member_comp_matrix.index.tolist()

        for i, member_i in enumerate(member_codes):
            # 自分自身を除いて類似度が高い順にソート
            similarities = []
            for j, member_j in enumerate(member_codes):
                if i != j:
                    similarities.append((member_j, similarity_matrix[i][j]))

            # 類似度でソート
            similarities.sort(key=lambda x: x[1], reverse=True)

            # 上位K人、かつ閾値以上のメンバーとエッジを張る
            for member_j, sim_score in similarities[:top_k]:
                if sim_score >= threshold:
                    member_node_i = f"member_{member_i}"
                    member_node_j = f"member_{member_j}"

                    # 重複エッジを避けるため、既に存在するかチェック
                    if not self.G.has_edge(member_node_i, member_node_j):
                        self.G.add_edge(
                            member_node_i,
                            member_node_j,
                            edge_type="similar",
                            weight=float(sim_score),
                            similarity=float(sim_score)
                        )
                        edge_count += 1

        print(f"  追加: {edge_count}本のメンバー類似度エッジ（閾値={threshold}, top_k={top_k}）")

    def get_neighbors(self, node_id: str, edge_type: Optional[str] = None) -> List[str]:
        """指定ノードの隣接ノードを取得

        Args:
            node_id: ノードID
            edge_type: エッジタイプでフィルタ（None=全て）

        Returns:
            隣接ノードIDのリスト
        """
        if not self.G.has_node(node_id):
            return []

        neighbors = []
        for neighbor in self.G.neighbors(node_id):
            edge_data = self.G[node_id][neighbor]
            if edge_type is None or edge_data.get('edge_type') == edge_type:
                neighbors.append(neighbor)
        return neighbors

    def get_node_info(self, node_id: str) -> Dict:
        """ノードの情報を取得

        Args:
            node_id: ノードID

        Returns:
            ノードの属性辞書
        """
        if not self.G.has_node(node_id):
            return {}
        return dict(self.G.nodes[node_id])

    def get_member_acquired_competences(self, member_code: str) -> Set[str]:
        """メンバーが習得済みの力量コードを取得

        Args:
            member_code: メンバーコード

        Returns:
            習得済み力量コードのセット
        """
        member_node = f"member_{member_code}"
        acquired = set()

        for neighbor in self.get_neighbors(member_node, edge_type="acquired"):
            if neighbor.startswith("competence_"):
                comp_code = neighbor.replace("competence_", "")
                acquired.add(comp_code)

        return acquired

    def get_competence_category(self, competence_code: str) -> Optional[str]:
        """力量が所属するカテゴリー名を取得

        Args:
            competence_code: 力量コード

        Returns:
            カテゴリー名（存在しない場合はNone）
        """
        competence_node = f"competence_{competence_code}"

        for neighbor in self.get_neighbors(competence_node, edge_type="belongs_to"):
            if neighbor.startswith("category_"):
                return neighbor.replace("category_", "")

        return None

    def _print_graph_stats(self):
        """グラフの統計情報を出力"""
        # ノードタイプ別の数
        node_types = {}
        for node in self.G.nodes():
            node_type = self.G.nodes[node].get('node_type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1

        print("\n  ノードタイプ別:")
        for node_type, count in sorted(node_types.items()):
            print(f"    - {node_type}: {count}")

        # エッジタイプ別の数
        edge_types = {}
        for u, v in self.G.edges():
            edge_type = self.G[u][v].get('edge_type', 'unknown')
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1

        print("\n  エッジタイプ別:")
        for edge_type, count in sorted(edge_types.items()):
            print(f"    - {edge_type}: {count}")

        # 平均次数
        avg_degree = sum(dict(self.G.degree()).values()) / self.G.number_of_nodes()
        print(f"\n  平均次数: {avg_degree:.2f}")

    def export_to_gexf(self, filename: str):
        """グラフをGEXF形式でエクスポート（Gephi等で可視化可能）

        Args:
            filename: 出力ファイル名
        """
        nx.write_gexf(self.G, filename)
        print(f"\nグラフをエクスポートしました: {filename}")
