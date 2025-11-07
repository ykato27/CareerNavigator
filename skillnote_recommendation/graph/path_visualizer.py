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

        # フェーズ別の色設定（力量ノード用）
        self.phase_colors = {
            1: '#28a745',  # Phase 1: 緑（基礎固め）
            2: '#ffc107',  # Phase 2: 黄（専門性構築）
            3: '#dc3545',  # Phase 3: 赤（エキスパート）
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
                                      scores: Optional[List[float]] = None,
                                      phase_info: Optional[Dict[str, int]] = None) -> go.Figure:
        """
        推薦パスを可視化

        Args:
            paths: パスのリスト。各パスは [{'id': str, 'type': str, 'name': str}, ...] の形式
            target_member_name: 対象メンバー名
            target_competence_name: 推薦力量名
            scores: 各パスのスコア（オプション）
            phase_info: 力量コード → フェーズ番号(1/2/3)のマッピング（オプション）

        Returns:
            Plotly Figure オブジェクト
        """
        if not paths:
            return self._create_empty_figure("推薦パスが見つかりませんでした")

        # NetworkXグラフを構築
        G = self._build_graph_from_paths(paths, phase_info)

        # レイアウトを計算（フェーズ情報を考慮）
        pos = self._calculate_layout(G, paths, phase_info)

        # Plotly Figure を作成
        fig = self._create_plotly_figure(G, pos, paths, scores, phase_info)

        # レイアウト設定
        if phase_info:
            title_text = (
                f"<b>推薦パス: {target_member_name} → {target_competence_name}</b><br>"
                f"<sub>📊 推薦ロジック: RWRパス + 段階的学習パス（Phase 1 → 2 → 3）</sub><br>"
                f"<sub style='font-size:10px'>💡 各ノードにカーソルを合わせると詳しい説明が表示されます</sub>"
            )
        else:
            title_text = (
                f"<b>推薦パス: {target_member_name} → {target_competence_name}</b><br>"
                f"<sub>📊 推薦ロジック: あなたの既習得力量 → 類似メンバー → 推薦力量</sub><br>"
                f"<sub style='font-size:10px'>💡 各ノードにカーソルを合わせると詳しい説明が表示されます</sub>"
            )

        fig.update_layout(
            title=dict(
                text=title_text,
                x=0.5,
                xanchor='center',
                font=dict(size=18)
            ),
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=120),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            width=1000,
            height=600,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                title=dict(text="<b>ノードの種類</b>", font=dict(size=12)),
                font=dict(size=11)
            )
        )

        return fig

    def _build_graph_from_paths(self, paths: List[List[Dict]], phase_info: Optional[Dict[str, int]] = None) -> nx.DiGraph:
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

                    # フェーズ情報を取得（力量ノードの場合）
                    phase = None
                    if node_type == 'competence' and phase_info:
                        # node_idから力量コードを抽出（例: "competence_C001" -> "C001"）
                        comp_code = node_id.replace('competence_', '')
                        phase = phase_info.get(comp_code)

                    G.add_node(
                        node_id,
                        name=node['name'],
                        type=node_type,
                        phase=phase,
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

    def _calculate_layout(self, G: nx.DiGraph, paths: List[List[Dict]],
                          phase_info: Optional[Dict[str, int]] = None) -> Dict:
        """レイアウトを計算（階層レイアウト + フェーズベース）"""
        # 各ノードの階層とフェーズを計算
        node_layers = {}
        node_phases = {}

        for path in paths:
            for i, node in enumerate(path):
                node_id = node['id']
                node_type = node.get('type', '')

                # 階層を計算
                if node_id not in node_layers:
                    node_layers[node_id] = i
                else:
                    # 最小の階層を採用
                    node_layers[node_id] = min(node_layers[node_id], i)

                # フェーズ情報を取得
                if phase_info and node_type == 'competence':
                    comp_code = node_id.replace('competence_', '')
                    if comp_code in phase_info:
                        node_phases[node_id] = phase_info[comp_code]

        # フェーズ情報がある場合は、フェーズベースでグループ化
        if node_phases:
            # フェーズごとにノードをグループ化
            phase_groups = {1: [], 2: [], 3: []}
            member_nodes = []
            other_nodes = []

            for node_id in G.nodes():
                node_data = G.nodes[node_id]
                node_type = node_data.get('type', '')

                if node_type == 'member':
                    member_nodes.append(node_id)
                elif node_id in node_phases:
                    phase = node_phases[node_id]
                    phase_groups[phase].append(node_id)
                else:
                    other_nodes.append(node_id)

            # 位置を計算
            pos = {}

            # メンバーノードを左端に配置
            for i, node_id in enumerate(member_nodes):
                y = 0.5 if len(member_nodes) == 1 else i / (len(member_nodes) - 1)
                pos[node_id] = (0.0, y)

            # Phase 1のノードを配置（x=0.25）
            if phase_groups[1]:
                n_nodes = len(phase_groups[1])
                for i, node_id in enumerate(phase_groups[1]):
                    y = 0.5 if n_nodes == 1 else i / (n_nodes - 1)
                    pos[node_id] = (0.25, y)

            # Phase 2のノードを配置（x=0.5）
            if phase_groups[2]:
                n_nodes = len(phase_groups[2])
                for i, node_id in enumerate(phase_groups[2]):
                    y = 0.5 if n_nodes == 1 else i / (n_nodes - 1)
                    pos[node_id] = (0.5, y)

            # Phase 3のノードを配置（x=0.75）
            if phase_groups[3]:
                n_nodes = len(phase_groups[3])
                for i, node_id in enumerate(phase_groups[3]):
                    y = 0.5 if n_nodes == 1 else i / (n_nodes - 1)
                    pos[node_id] = (0.75, y)

            # その他のノード（類似メンバーなど）を配置
            if other_nodes:
                n_nodes = len(other_nodes)
                for i, node_id in enumerate(other_nodes):
                    # 階層に基づいて配置
                    layer = node_layers.get(node_id, 1)
                    x = 0.15 + layer * 0.1  # 適切な位置に配置
                    y = 0.5 if n_nodes == 1 else i / (n_nodes - 1)
                    pos[node_id] = (x, y)

            return pos

        # フェーズ情報がない場合は従来の階層レイアウト
        else:
            # 階層ごとにノードをグループ化
            layers = {}
            for node_id, layer in node_layers.items():
                if layer not in layers:
                    layers[layer] = []
                layers[layer].append(node_id)

            # 位置を計算
            pos = {}
            max_layer = max(layers.keys()) if layers else 0

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
                              scores: Optional[List[float]],
                              phase_info: Optional[Dict[str, int]] = None) -> go.Figure:
        """Plotly Figureを作成"""
        fig = go.Figure()

        # 各パスを個別に描画（エッジ）
        self._add_paths_as_traces(fig, pos, paths, scores)

        # ノードを描画
        self._add_nodes_to_figure(fig, G, pos, phase_info)

        return fig

    def _add_paths_as_traces(self,
                             fig: go.Figure,
                             pos: Dict,
                             paths: List[List[Dict]],
                             scores: Optional[List[float]]):
        """各パスを個別のトレースとして描画"""
        path_colors = self._generate_path_colors(len(paths))

        for path_idx, path in enumerate(paths):
            if len(path) < 2:
                continue

            # このパスのエッジを収集
            x_coords = []
            y_coords = []

            for i in range(len(path) - 1):
                node_id = path[i]['id']
                next_node_id = path[i + 1]['id']

                if node_id in pos and next_node_id in pos:
                    x0, y0 = pos[node_id]
                    x1, y1 = pos[next_node_id]

                    # エッジの座標を追加（Noneで区切って複数セグメントを描画）
                    x_coords.extend([x0, x1, None])
                    y_coords.extend([y0, y1, None])

            # パス全体を1つのトレースとして追加
            if x_coords:
                fig.add_trace(go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='lines',
                    line=dict(
                        color=path_colors[path_idx],
                        width=2,
                    ),
                    hoverinfo='skip',
                    showlegend=True,
                    name=f'パス{path_idx + 1}',
                    opacity=0.7,
                ))

                # 矢印を追加（各エッジの終点に）
                for i in range(len(path) - 1):
                    node_id = path[i]['id']
                    next_node_id = path[i + 1]['id']

                    if node_id in pos and next_node_id in pos:
                        x0, y0 = pos[node_id]
                        x1, y1 = pos[next_node_id]

                        # 矢印を追加
                        fig.add_annotation(
                            x=x1,
                            y=y1,
                            ax=x0,
                            ay=y0,
                            xref='x',
                            yref='y',
                            axref='x',
                            ayref='y',
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1,
                            arrowwidth=1.5,
                            arrowcolor=path_colors[path_idx],
                            opacity=0.7,
                        )

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

    def _get_node_role_explanation(self, node_type: str, is_start_path: bool = False) -> str:
        """ノードタイプごとの役割説明を取得"""
        explanations = {
            'member': '推薦を受ける対象者',
            'competence': 'パスの起点となる既習得力量、または<br>パスの終点となる推薦力量',
            'category': '力量が属するカテゴリー。<br>同じカテゴリーの力量を探すのに使用',
            'similar_member': 'あなたと同じ力量を持つメンバー。<br>このメンバーが習得している力量が推薦されます',
        }
        return explanations.get(node_type, 'グラフのノード')

    def _add_nodes_to_figure(self, fig: go.Figure, G: nx.DiGraph, pos: Dict, phase_info: Optional[Dict[str, int]] = None):
        """ノードを描画"""
        # ノードタイプごとの説明
        type_descriptions = {
            'member': '👤 あなた（対象メンバー）',
            'competence': '📚 あなたの既習得力量',
            'category': '📁 カテゴリー',
            'similar_member': '🤝 類似メンバー（あなたと似たスキルを持つ人）',
        }

        # フェーズ情報がある場合は、力量ノードをフェーズごとに分類
        # それ以外のノードタイプは通常通り分類
        node_groups = {}
        for node_id in G.nodes():
            node_data = G.nodes[node_id]
            node_type = node_data['type']
            phase = node_data.get('phase')

            # グループキーを決定（力量ノードでフェーズ情報がある場合は"competence_phase_X"、それ以外は通常のタイプ）
            if node_type == 'competence' and phase is not None:
                group_key = f'competence_phase_{phase}'
            else:
                group_key = node_type

            if group_key not in node_groups:
                node_groups[group_key] = {
                    'ids': [],
                    'x': [],
                    'y': [],
                    'text': [],
                    'hovertext': [],
                    'type': node_type,
                    'phase': phase if node_type == 'competence' else None,
                }

            node_groups[group_key]['ids'].append(node_id)
            x, y = pos[node_id]
            node_groups[group_key]['x'].append(x)
            node_groups[group_key]['y'].append(y)
            node_groups[group_key]['text'].append(node_data['name'])

            # ホバーテキストに役割の説明を追加（フェーズ情報も含める）
            role_description = type_descriptions.get(node_type, f'タイプ: {node_type}')
            hover_text = f"<b>{node_data['name']}</b><br><br>{role_description}"

            # フェーズ情報がある場合は追加
            if phase is not None:
                phase_names = {1: '🌱 Phase 1: 基礎固め', 2: '🌿 Phase 2: 専門性構築', 3: '🌳 Phase 3: エキスパート'}
                hover_text += f"<br><br>📚 {phase_names.get(phase, f'Phase {phase}')}"

            hover_text += f"<br><br>💡 このノードの役割:<br>{self._get_node_role_explanation(node_type)}"

            node_groups[group_key]['hovertext'].append(hover_text)

        # タイプ/フェーズごとに描画
        type_labels = {
            'member': '👤 あなた（対象メンバー）',
            'competence': '📚 既習得力量',
            'category': '📁 カテゴリー',
            'similar_member': '🤝 類似メンバー',
            'competence_phase_1': '🌱 Phase 1: 基礎固め',
            'competence_phase_2': '🌿 Phase 2: 専門性構築',
            'competence_phase_3': '🌳 Phase 3: エキスパート',
        }

        for group_key, group in node_groups.items():
            node_type = group['type']
            phase = group['phase']

            # 色を決定（フェーズ情報がある力量ノードの場合はフェーズ色、それ以外は通常色）
            if node_type == 'competence' and phase is not None:
                color = self.phase_colors.get(phase, self.node_colors['competence'])
            else:
                color = self.node_colors.get(node_type, '#999999')

            fig.add_trace(go.Scatter(
                x=group['x'],
                y=group['y'],
                mode='markers+text',
                marker=dict(
                    size=self.node_sizes.get(node_type, 15),
                    color=color,
                    line=dict(color='white', width=2),
                ),
                text=group['text'],
                textposition='top center',
                textfont=dict(size=10),
                hovertext=group['hovertext'],
                hoverinfo='text',
                name=type_labels.get(group_key, group_key),
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
