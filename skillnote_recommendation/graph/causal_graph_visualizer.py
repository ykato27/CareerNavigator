"""
因果グラフ可視化モジュール

学習された因果構造をGraphvizまたはPyVisを用いて可視化します。

改善内容:
1. PyVisによるインタラクティブなHTML可視化
2. ノードフィルタリング（PageRank、次数中心性）
3. クライアントサイドレンダリングによる高速化
"""

import graphviz
import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple, Union
import logging
import tempfile
import os

# Optional dependency
try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False
    Network = None

logger = logging.getLogger(__name__)

class CausalGraphVisualizer:
    """
    因果グラフ可視化クラス
    """
    
    def __init__(self, adjacency_matrix: pd.DataFrame):
        """
        Args:
            adjacency_matrix: 隣接行列 (行=From, 列=To)
        """
        self.adj_matrix = adjacency_matrix
        
    def visualize(
        self, 
        threshold: float = 0.1, 
        highlight_nodes: Optional[List[str]] = None,
        highlight_color: str = 'lightblue',
        output_format: str = 'png'
    ) -> graphviz.Digraph:
        """
        グラフを可視化
        
        Args:
            threshold: エッジを表示する最小の係数（絶対値）
            highlight_nodes: ハイライトするノード名のリスト
            highlight_color: ハイライト色
            output_format: 出力フォーマット
            
        Returns:
            graphviz.Digraphオブジェクト
        """
        dot = graphviz.Digraph(comment='Causal Graph')
        dot.attr(rankdir='LR') # 左から右へ
        dot.format = output_format
        
        nodes = self.adj_matrix.columns.tolist()
        highlight_set = set(highlight_nodes) if highlight_nodes else set()
        
        # ノードの追加
        for node in nodes:
            attrs = {'style': 'filled', 'fillcolor': 'white', 'shape': 'box'}
            if node in highlight_set:
                attrs['fillcolor'] = highlight_color
            
            dot.node(node, node, **attrs)
            
        # エッジの追加
        # adj_matrix は From -> To
        for from_node in nodes:
            for to_node in nodes:
                if from_node == to_node:
                    continue
                    
                weight = self.adj_matrix.loc[from_node, to_node]
                
                if abs(weight) >= threshold:
                    # 係数の大きさで線の太さを変える
                    penwidth = str(max(1, abs(weight) * 5))
                    
                    # 正の因果は黒、負の因果は赤
                    color = 'black' if weight > 0 else 'red'
                    
                    label = f"{weight:.2f}"
                    
                    dot.edge(
                        from_node, 
                        to_node, 
                        label=label, 
                        penwidth=penwidth, 
                        color=color,
                        fontcolor=color
                    )
                    
        return dot
        
    def visualize_ego_network(
        self,
        center_node: str,
        radius: int = 1,
        threshold: float = 0.1
    ) -> graphviz.Digraph:
        """
        特定のノードを中心としたサブグラフ（エゴネットワーク）を可視化
        """
        # 簡易実装: 関連するノードを抽出して visualize を呼ぶ
        related_nodes = {center_node}

        # radius=1 の範囲 (親と子)
        # Parents (From -> Center)
        parents = self.adj_matrix.index[self.adj_matrix[center_node].abs() >= threshold].tolist()
        related_nodes.update(parents)

        # Children (Center -> To)
        children = self.adj_matrix.columns[self.adj_matrix.loc[center_node].abs() >= threshold].tolist()
        related_nodes.update(children)

        # サブセット作成
        sub_matrix = self.adj_matrix.loc[list(related_nodes), list(related_nodes)]

        # サブクラス作成して可視化
        sub_visualizer = CausalGraphVisualizer(sub_matrix)
        return sub_visualizer.visualize(threshold=threshold, highlight_nodes=[center_node])

    def visualize_interactive(
        self,
        output_path: str = "temp_graph.html",
        threshold: float = 0.1,
        top_n: int = 50,
        height: str = "600px",
        width: str = "100%",
        notebook: bool = False,
        show_negative: bool = False
    ) -> str:
        """
        PyVisを用いてインタラクティブなHTMLグラフを生成し、パスを返す。
        
        Args:
            output_path: 出力HTMLファイルのパス
            threshold: エッジを表示する最小の係数（絶対値）
            top_n: 表示する最大ノード数（中心性に基づく）
            height: グラフの高さ
            width: グラフの幅
            notebook: Notebook環境かどうか
            show_negative: 負の因果関係も表示するか（False=正のみ、True=正負両方）
            
        Returns:
            str: 生成されたHTMLファイルのパス
        """
        if not PYVIS_AVAILABLE:
            raise ImportError("pyvis is not installed. Please install it with: pip install pyvis>=0.3.2")
        
        # PyVisネットワークの初期化
        net = Network(height=height, width=width, bgcolor="#ffffff", font_color="#333333", notebook=notebook)
        # 物理演算の調整（安定化のため）
        net.force_atlas_2based()
        
        # 1. ノードのフィルタリング（次数中心性で上位N個を選択）
        # 隣接行列の絶対値を使用
        abs_adj = self.adj_matrix.abs()
        
        # 次数（入次数 + 出次数）を計算
        degrees = abs_adj.sum(axis=0) + abs_adj.sum(axis=1)
        
        # 上位N個のノードを取得
        top_nodes = degrees.sort_values(ascending=False).head(top_n).index.tolist()
        
        # 2. ノードの追加
        for node in top_nodes:
            # ノードのサイズを次数に比例させる
            size = 10 + degrees[node] * 2
            net.add_node(node, label=node, title=node, size=size, color="#2E7D32") # Green theme
            
        # 3. エッジの追加
        # 上位ノード間のエッジのみ追加
        for from_node in top_nodes:
            for to_node in top_nodes:
                if from_node == to_node:
                    continue
                
                weight = self.adj_matrix.loc[from_node, to_node]
                
                if abs(weight) >= threshold:
                    # 負の因果関係をスキップする場合
                    if not show_negative and weight < 0:
                        continue
                    
                    # エッジの太さと色
                    width_val = max(1, abs(weight) * 5)
                    color = "#333333" if weight > 0 else "#DC3545" # Black for positive, Red for negative
                    title = f"{from_node} -> {to_node}: {weight:.2f}"
                    
                    net.add_edge(
                        from_node, 
                        to_node, 
                        value=abs(weight), 
                        title=title, 
                        color=color,
                        width=width_val,
                        arrowStrikethrough=False
                    )
        
        # オプション設定（操作パネルを表示しない、物理演算を少し落ち着かせる）
        net.set_options("""
        var options = {
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 100,
              "springConstant": 0.08
            },
            "maxVelocity": 50,
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
            "stabilization": {
              "enabled": true,
              "iterations": 150
            }
          },
          "nodes": {
            "font": {
              "size": 14,
              "face": "sans-serif"
            }
          },
          "edges": {
            "arrows": {
              "to": {
                "enabled": true,
                "scaleFactor": 0.5
              }
            },
            "smooth": {
              "type": "continuous"
            }
          }
        }
        """)
        
        # 保存
        # Streamlit Cloudなど書き込み権限の問題を回避するため、絶対パスを使用することを推奨
        try:
            net.save_graph(output_path)
            return output_path
        except Exception as e:
            logger.error(f"グラフの保存に失敗しました: {e}")
            # フォールバック: カレントディレクトリ
            fallback_path = "temp_network.html"
            net.save_graph(fallback_path)
            return fallback_path

    def _filter_nodes_by_centrality(
        self,
        top_n: int = 50,
        method: str = "pagerank",
        threshold: float = 0.1
    ) -> List[str]:
        """
        中心性指標に基づいてノードをフィルタリング

        Args:
            top_n: 上位N個のノードを選択
            method: 中心性の計算方法 ("pagerank" または "degree")
            threshold: エッジを考慮する最小係数

        Returns:
            選択されたノード名のリスト
        """
        # NetworkXグラフに変換
        G = nx.DiGraph()

        nodes = self.adj_matrix.columns.tolist()

        for from_node in nodes:
            for to_node in nodes:
                if from_node == to_node:
                    continue

                weight = self.adj_matrix.loc[from_node, to_node]

                if abs(weight) >= threshold:
                    G.add_edge(from_node, to_node, weight=abs(weight))

        # 中心性を計算
        if method == "pagerank":
            centrality = nx.pagerank(G, weight='weight')
        elif method == "degree":
            # 次数中心性（重み付き）
            centrality = dict(G.degree(weight='weight'))
        else:
            raise ValueError(f"Unknown method: {method}")

        # 中心性でソートして上位N個を取得
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        top_nodes = [node for node, _ in sorted_nodes[:top_n]]

        logger.info(f"Selected {len(top_nodes)} nodes using {method} centrality")

        return top_nodes

    def visualize_interactive(
        self,
        output_path: str = None,
        threshold: float = 0.1,
        top_n: int = 50,
        centrality_method: str = "pagerank",
        highlight_nodes: Optional[List[str]] = None,
        physics_enabled: bool = True,
        height: str = "750px",
        width: str = "100%"
    ) -> str:
        """
        PyVisを用いてインタラクティブなHTMLグラフを生成

        Args:
            output_path: 出力先HTMLファイルパス（Noneの場合は一時ファイル）
            threshold: エッジを表示する最小の係数（絶対値）
            top_n: 表示する最大ノード数（中心性でフィルタ）
            centrality_method: 中心性の計算方法 ("pagerank" または "degree")
            highlight_nodes: ハイライトするノード名のリスト
            physics_enabled: 物理演算を有効にするか
            height: グラフの高さ
            width: グラフの幅

        Returns:
            生成されたHTMLファイルのパス
        """
        try:
            from pyvis.network import Network
        except ImportError:
            raise ImportError(
                "pyvis is not installed. Please install it with: pip install pyvis>=0.3.2"
            )

        # ノードをフィルタリング
        if top_n < len(self.adj_matrix.columns):
            selected_nodes = self._filter_nodes_by_centrality(
                top_n=top_n,
                method=centrality_method,
                threshold=threshold
            )
        else:
            selected_nodes = self.adj_matrix.columns.tolist()

        # PyVisネットワークを作成
        net = Network(
            height=height,
            width=width,
            directed=True,
            notebook=False
        )

        # 物理演算の設定
        if physics_enabled:
            net.set_options("""
            {
                "physics": {
                    "enabled": true,
                    "barnesHut": {
                        "gravitationalConstant": -30000,
                        "centralGravity": 0.3,
                        "springLength": 200,
                        "springConstant": 0.04,
                        "damping": 0.09,
                        "avoidOverlap": 0.5
                    },
                    "stabilization": {
                        "enabled": true,
                        "iterations": 100
                    }
                }
            }
            """)
        else:
            net.toggle_physics(False)

        highlight_set = set(highlight_nodes) if highlight_nodes else set()

        # ノードを追加
        for node in selected_nodes:
            color = "#97C2FC" if node in highlight_set else "#DDDDDD"
            title = f"<b>{node}</b>"  # ホバー時の情報

            net.add_node(
                node,
                label=node,
                color=color,
                title=title,
                size=20 if node in highlight_set else 15,
                font={"size": 14}
            )

        # エッジを追加
        selected_nodes_set = set(selected_nodes)

        for from_node in selected_nodes:
            for to_node in selected_nodes:
                if from_node == to_node:
                    continue

                weight = self.adj_matrix.loc[from_node, to_node]

                if abs(weight) >= threshold:
                    # 係数の大きさで線の太さを変える
                    edge_width = max(1, abs(weight) * 5)

                    # 正の因果は青、負の因果は赤
                    color = "#4169E1" if weight > 0 else "#DC143C"

                    # ホバー時の情報
                    title = f"{from_node} → {to_node}<br>係数: {weight:.3f}"

                    net.add_edge(
                        from_node,
                        to_node,
                        value=edge_width,
                        color=color,
                        title=title,
                        label=f"{weight:.2f}",
                        arrows="to"
                    )

        # 出力パスの決定
        if output_path is None:
            # 一時ファイルを作成
            fd, output_path = tempfile.mkstemp(suffix=".html", prefix="causal_graph_")
            os.close(fd)

        # HTMLファイルを生成
        net.save_graph(output_path)

        logger.info(f"Interactive graph saved to: {output_path}")
        logger.info(f"Nodes: {len(selected_nodes)}, Edges: {net.num_edges()}")

        return output_path

    def visualize_interactive_ego_network(
        self,
        center_node: str,
        radius: int = 1,
        threshold: float = 0.1,
        output_path: str = None,
        physics_enabled: bool = True,
        height: str = "750px",
        width: str = "100%"
    ) -> str:
        """
        特定のノードを中心としたエゴネットワークをインタラクティブに可視化

        Args:
            center_node: 中心ノード
            radius: ネットワークの半径（深さ）
            threshold: エッジを表示する最小の係数
            output_path: 出力先HTMLファイルパス
            physics_enabled: 物理演算を有効にするか
            height: グラフの高さ
            width: グラフの幅

        Returns:
            生成されたHTMLファイルのパス
        """
        # エゴネットワークのノードを抽出
        related_nodes = {center_node}

        # NetworkXグラフに変換してBFS
        G = nx.DiGraph()
        nodes = self.adj_matrix.columns.tolist()

        for from_node in nodes:
            for to_node in nodes:
                if from_node == to_node:
                    continue

                weight = self.adj_matrix.loc[from_node, to_node]
                if abs(weight) >= threshold:
                    G.add_edge(from_node, to_node, weight=weight)

        # エゴネットワーク抽出（両方向）
        if center_node in G:
            # 外向き（子孫）
            descendants = set()
            for _ in range(radius):
                new_nodes = set()
                for node in list(related_nodes):
                    if node in G:
                        new_nodes.update(G.successors(node))
                related_nodes.update(new_nodes)
                descendants.update(new_nodes)

            # 内向き（祖先）
            ancestors = set()
            for _ in range(radius):
                new_nodes = set()
                for node in list(related_nodes):
                    if node in G:
                        new_nodes.update(G.predecessors(node))
                related_nodes.update(new_nodes)
                ancestors.update(new_nodes)

        # サブ隣接行列を作成
        related_nodes_list = list(related_nodes)
        sub_matrix = self.adj_matrix.loc[related_nodes_list, related_nodes_list]

        # サブビジュアライザーを作成してインタラクティブ可視化
        sub_visualizer = CausalGraphVisualizer(sub_matrix)

        return sub_visualizer.visualize_interactive(
            output_path=output_path,
            threshold=threshold,
            top_n=len(related_nodes_list),  # 全ノード表示
            highlight_nodes=[center_node],
            physics_enabled=physics_enabled,
            height=height,
            width=width
        )
