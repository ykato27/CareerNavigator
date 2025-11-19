"""
因果グラフ可視化モジュール

学習された因果構造をGraphvizを用いて可視化します。
"""

import graphviz
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging

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
