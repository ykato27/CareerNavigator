"""
Recommendation Path Visualizer

推薦パスをPlotlyを使ってインタラクティブに可視化する
"""

import plotly.graph_objects as go
import networkx as nx
from typing import List, Dict, Tuple, Optional
import numpy as np


class RecommendationPathVisualizer:
    """推薦パスの可視化クラス

    Plotlyを使用して、推薦パスをインタラクティブなネットワーク図として表示する。
    """

    def __init__(self):
        """初期化"""
        # ノードタイプ別の色設定
        self.node_colors = {
            'member': '#FF6B6B',      # 赤系（対象メンバー）
            'competence': '#4ECDC4',  # 青緑系（推薦力量）
            'category': '#95E1D3',    # 緑系（カテゴリー）
            'similar_member': '#FFA07A',  # オレンジ系（類似メンバー）
        }

        # ノードタイプ別のサイズ
        self.node_sizes = {
            'member': 20,
            'competence': 15,
            'category': 12,
            'similar_member': 15,
        }

    def visualize_recommendation_path(self,
                                      paths: List[List[Dict]],
                                      target_member_name: str,
                                      target_competence_name: str,
                                      scores: Optional[List[float]] = None) -> go.Figure:
        """
        推薦パスを可視化

        Args:
            paths: パスのリスト。各パスは [{'id': str, 'type': str, 'name': str}, ...] の形式
            target_member_name: 対象メンバー名
            target_competence_name: 推薦力量名
            scores: 各パスのスコア（オプション）

        Returns:
            Plotly Figure オブジェクト
        """
        if not paths:
            return self._create_empty_figure("推薦パスが見つかりませんでした")

        # NetworkXグラフを構築
        G = self._build_graph_from_paths(paths)

        # レイアウトを計算
        pos = self._calculate_layout(G, paths)

        # Plotly Figure を作成
        fig = self._create_plotly_figure(G, pos, paths, scores)

        # レイアウト設定
        fig.update_layout(
            title=dict(
                text=f"推薦パス: {target_member_name} → {target_competence_name}",
                x=0.5,
                xanchor='center',
                font=dict(size=20)
            ),
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=60),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            width=1000,
            height=600,
        )

        return fig

    def _build_graph_from_paths(self, paths: List[List[Dict]]) -> nx.DiGraph:
        """パスからNetworkXグラフを構築"""
        G = nx.DiGraph()

        for path_idx, path in enumerate(paths):
            for i, node in enumerate(path):
                node_id = node['id']

                # ノードを追加
                if not G.has_node(node_id):
                    # ノードタイプを調整（類似メンバーの場合）
                    node_type = node['type']
                    if node_type == 'member' and i > 0:  # 最初以外のメンバーノードは類似メンバー
                        node_type = 'similar_member'

                    G.add_node(
                        node_id,
                        name=node['name'],
                        type=node_type,
                        path_indices={path_idx}
                    )
                else:
                    # 既存ノードの場合、パスインデックスを追加
                    G.nodes[node_id]['path_indices'].add(path_idx)

                # エッジを追加
                if i > 0:
                    prev_node_id = path[i-1]['id']
                    if G.has_edge(prev_node_id, node_id):
                        # 既存エッジの場合、パスインデックスを追加
                        G[prev_node_id][node_id]['path_indices'].add(path_idx)
                    else:
                        G.add_edge(
                            prev_node_id,
                            node_id,
                            path_indices={path_idx}
                        )

        return G

    def _calculate_layout(self, G: nx.DiGraph, paths: List[List[Dict]]) -> Dict:
        """レイアウトを計算（階層レイアウト）"""
        # 各ノードの階層を計算
        node_layers = {}

        for path in paths:
            for i, node in enumerate(path):
                node_id = node['id']
                if node_id not in node_layers:
                    node_layers[node_id] = i
                else:
                    # 最小の階層を採用
                    node_layers[node_id] = min(node_layers[node_id], i)

        # 階層ごとにノードをグループ化
        layers = {}
        for node_id, layer in node_layers.items():
            if layer not in layers:
                layers[layer] = []
            layers[layer].append(node_id)

        # 位置を計算
        pos = {}
        max_layer = max(layers.keys())

        for layer, nodes in layers.items():
            x = layer / max_layer if max_layer > 0 else 0.5
            n_nodes = len(nodes)

            for i, node_id in enumerate(nodes):
                # Y座標を計算（中央に配置）
                if n_nodes == 1:
                    y = 0.5
                else:
                    y = i / (n_nodes - 1)

                pos[node_id] = (x, y)

        return pos

    def _create_plotly_figure(self,
                              G: nx.DiGraph,
                              pos: Dict,
                              paths: List[List[Dict]],
                              scores: Optional[List[float]]) -> go.Figure:
        """Plotly Figureを作成"""
        fig = go.Figure()

        # エッジを描画
        self._add_edges_to_figure(fig, G, pos, paths, scores)

        # ノードを描画
        self._add_nodes_to_figure(fig, G, pos)

        return fig

    def _add_edges_to_figure(self,
                             fig: go.Figure,
                             G: nx.DiGraph,
                             pos: Dict,
                             paths: List[List[Dict]],
                             scores: Optional[List[float]]):
        """エッジを描画"""
        # パスごとに色を割り当て
        path_colors = self._generate_path_colors(len(paths))

        for u, v, data in G.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]

            # このエッジが含まれるパス
            path_indices = data['path_indices']

            # 複数のパスに含まれる場合は太くする
            width = 1 + len(path_indices) * 0.5

            # パスの色を使用（複数ある場合は最初のパスの色）
            path_idx = min(path_indices)
            color = path_colors[path_idx]

            # 矢印を描画
            fig.add_trace(go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode='lines',
                line=dict(
                    color=color,
                    width=width,
                ),
                hoverinfo='none',
                showlegend=False,
                opacity=0.6,
            ))

            # 矢印の頭を追加
            self._add_arrow_head(fig, x0, y0, x1, y1, color)

    def _add_arrow_head(self, fig: go.Figure, x0: float, y0: float, x1: float, y1: float, color: str):
        """矢印の頭を追加"""
        # 矢印のサイズ
        arrow_length = 0.02

        # 方向ベクトル
        dx = x1 - x0
        dy = y1 - y0
        length = np.sqrt(dx**2 + dy**2)

        if length > 0:
            # 正規化
            dx /= length
            dy /= length

            # 矢印の頭の位置
            arrow_x = x1 - arrow_length * dx
            arrow_y = y1 - arrow_length * dy

            # 矢印を追加
            fig.add_trace(go.Scatter(
                x=[arrow_x, x1],
                y=[arrow_y, y1],
                mode='markers',
                marker=dict(
                    size=8,
                    color=color,
                    symbol='arrow',
                    angleref='previous',
                ),
                hoverinfo='none',
                showlegend=False,
            ))

    def _add_nodes_to_figure(self, fig: go.Figure, G: nx.DiGraph, pos: Dict):
        """ノードを描画"""
        # ノードタイプごとにグループ化
        node_groups = {}
        for node_id in G.nodes():
            node_data = G.nodes[node_id]
            node_type = node_data['type']

            if node_type not in node_groups:
                node_groups[node_type] = {
                    'ids': [],
                    'x': [],
                    'y': [],
                    'text': [],
                    'hovertext': [],
                }

            node_groups[node_type]['ids'].append(node_id)
            x, y = pos[node_id]
            node_groups[node_type]['x'].append(x)
            node_groups[node_type]['y'].append(y)
            node_groups[node_type]['text'].append(node_data['name'])
            node_groups[node_type]['hovertext'].append(
                f"{node_data['name']}<br>タイプ: {node_type}"
            )

        # タイプごとに描画
        type_labels = {
            'member': '対象メンバー',
            'competence': '推薦力量',
            'category': 'カテゴリー',
            'similar_member': '類似メンバー',
        }

        for node_type, group in node_groups.items():
            fig.add_trace(go.Scatter(
                x=group['x'],
                y=group['y'],
                mode='markers+text',
                marker=dict(
                    size=self.node_sizes.get(node_type, 15),
                    color=self.node_colors.get(node_type, '#999999'),
                    line=dict(color='white', width=2),
                ),
                text=group['text'],
                textposition='top center',
                textfont=dict(size=10),
                hovertext=group['hovertext'],
                hoverinfo='text',
                name=type_labels.get(node_type, node_type),
                showlegend=True,
            ))

    def _generate_path_colors(self, n_paths: int) -> List[str]:
        """パスごとの色を生成"""
        if n_paths == 1:
            return ['#3498db']
        elif n_paths == 2:
            return ['#3498db', '#e74c3c']
        elif n_paths == 3:
            return ['#3498db', '#e74c3c', '#2ecc71']
        else:
            # より多くのパスの場合は色相を変えて生成
            colors = []
            for i in range(n_paths):
                hue = i / n_paths
                # HSVからRGBに変換（簡易版）
                rgb = self._hsv_to_rgb(hue, 0.7, 0.9)
                colors.append(f'rgb({rgb[0]},{rgb[1]},{rgb[2]})')
            return colors

    def _hsv_to_rgb(self, h: float, s: float, v: float) -> Tuple[int, int, int]:
        """HSVからRGBに変換"""
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return int(r * 255), int(g * 255), int(b * 255)

    def _create_empty_figure(self, message: str) -> go.Figure:
        """空のFigureを作成"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16),
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
        )
        return fig


def visualize_multiple_recommendations(
    recommendations: List[Dict],
    top_n: int = 3
) -> Dict[str, go.Figure]:
    """
    複数の推薦のパスを可視化

    Args:
        recommendations: 推薦結果のリスト
            各要素は {'competence_code', 'competence_name', 'paths', 'score'} を含む
        top_n: 可視化する推薦の数

    Returns:
        {力量コード: Figure} の辞書
    """
    visualizer = RecommendationPathVisualizer()
    figures = {}

    for i, rec in enumerate(recommendations[:top_n]):
        if rec.get('paths'):
            fig = visualizer.visualize_recommendation_path(
                paths=rec['paths'],
                target_member_name="対象メンバー",
                target_competence_name=rec.get('competence_name', rec['competence_code']),
                scores=None
            )
            figures[rec['competence_code']] = fig

    return figures
