"""
Enhanced Recommendation Path Visualizer

高度なインタラクティブ機能を持つ推薦パス可視化システム

Features:
- 力学的レイアウト（Fruchterman-Reingold）
- パスごとのトグル表示
- ノード・エッジの詳細情報表示
- 時間情報の統合（遷移期間の可視化）
- フィルタリング機能
- 色覚多様性対応カラーパレット
- アニメーション対応
"""

import plotly.graph_objects as go
import networkx as nx
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass
import colorsys


# 色覚多様性対応カラーパレット（Okabe-Ito palette）
COLORBLIND_SAFE_PALETTE = [
    "#0173B2",  # Blue
    "#DE8F05",  # Orange
    "#029E73",  # Green
    "#CC78BC",  # Purple
    "#CA9161",  # Brown
    "#FBAFE4",  # Pink
    "#949494",  # Gray
    "#ECE133",  # Yellow
]


@dataclass
class PathStatistics:
    """パスの統計情報"""

    path_id: int
    length: int
    total_transitions: int  # パス内の総遷移人数
    avg_transition_days: float  # 平均遷移期間
    quality_score: float  # パス品質スコア（0-1）


@dataclass
class EdgeStatistics:
    """エッジの統計情報"""

    source_name: str
    target_name: str
    transition_count: int  # 遷移人数
    avg_days: float  # 平均日数
    median_days: float  # 中央値日数
    success_rate: float  # 成功率（オプション）


class EnhancedPathVisualizer:
    """高度なインタラクティブ機能を持つ推薦パス可視化クラス

    主な機能:
    - 力学的レイアウトによる見やすい配置
    - パスごとのトグル表示
    - ノード・エッジの詳細情報のインタラクティブ表示
    - 時間情報の可視化（エッジの太さ・色）
    - パス品質に基づくフィルタリング
    """

    def __init__(
        self,
        layout_algorithm: str = "fruchterman_reingold",
        colorblind_safe: bool = True,
        show_edge_statistics: bool = True,
        animate_paths: bool = False,
    ):
        """
        Args:
            layout_algorithm: レイアウトアルゴリズム
                - 'fruchterman_reingold': 力学的レイアウト（推奨）
                - 'spring': Springレイアウト
                - 'hierarchical': 階層レイアウト
            colorblind_safe: 色覚多様性対応カラーパレットを使用
            show_edge_statistics: エッジの統計情報を表示
            animate_paths: パスをアニメーション表示
        """
        self.layout_algorithm = layout_algorithm
        self.colorblind_safe = colorblind_safe
        self.show_edge_statistics = show_edge_statistics
        self.animate_paths = animate_paths

        # ノードタイプ別の設定
        self.node_config = {
            "member": {"color": "#E74C3C", "size": 25, "symbol": "circle", "label": "対象メンバー"},
            "competence": {"color": "#3498DB", "size": 20, "symbol": "square", "label": "推薦力量"},
            "category": {
                "color": "#2ECC71",
                "size": 18,
                "symbol": "diamond",
                "label": "カテゴリー",
            },
            "similar_member": {
                "color": "#F39C12",
                "size": 20,
                "symbol": "circle",
                "label": "類似メンバー",
            },
        }

    def visualize_paths(
        self,
        paths: List[List[Dict]],
        target_member_name: str,
        target_competence_name: str,
        edge_statistics: Optional[Dict[Tuple[str, str], EdgeStatistics]] = None,
        path_scores: Optional[List[float]] = None,
        min_quality_score: float = 0.0,
    ) -> go.Figure:
        """
        推薦パスを高度な可視化で表示

        Args:
            paths: パスのリスト。各パスは [{'id': str, 'type': str, 'name': str}, ...] の形式
            target_member_name: 対象メンバー名
            target_competence_name: 推薦力量名
            edge_statistics: エッジごとの統計情報
            path_scores: 各パスのスコア
            min_quality_score: 表示する最小品質スコア（フィルタリング）

        Returns:
            Plotly Figure オブジェクト
        """
        if not paths:
            return self._create_empty_figure("推薦パスが見つかりませんでした")

        # パス品質スコアを計算
        path_statistics = self._calculate_path_statistics(paths, edge_statistics, path_scores)

        # フィルタリング
        filtered_paths = [
            (path, stat)
            for path, stat in zip(paths, path_statistics)
            if stat.quality_score >= min_quality_score
        ]

        if not filtered_paths:
            return self._create_empty_figure(
                f"品質スコア{min_quality_score}以上のパスが見つかりませんでした"
            )

        paths, path_statistics = zip(*filtered_paths)

        # NetworkXグラフを構築
        G = self._build_graph_from_paths(paths)

        # レイアウトを計算
        pos = self._calculate_layout(G, paths)

        # Plotly Figureを作成
        fig = self._create_interactive_figure(G, pos, paths, path_statistics, edge_statistics)

        # レイアウト設定
        fig.update_layout(
            title=dict(
                text=f"🎯 推薦パス可視化: {target_member_name} → {target_competence_name}",
                x=0.5,
                xanchor="center",
                font=dict(size=22, family="Arial, sans-serif"),
            ),
            showlegend=True,
            hovermode="closest",
            margin=dict(b=40, l=40, r=40, t=80),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            plot_bgcolor="#F8F9FA",
            paper_bgcolor="white",
            width=1200,
            height=700,
            font=dict(family="Arial, sans-serif"),
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.01,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="gray",
                borderwidth=1,
            ),
            # インタラクティブ機能の設定
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=[
                        dict(
                            args=[{"visible": [True] * len(fig.data)}],
                            label="全パス表示",
                            method="restyle",
                        ),
                        dict(
                            args=[{"visible": self._create_top3_visibility(len(paths))}],
                            label="Top3のみ",
                            method="restyle",
                        ),
                    ],
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.0,
                    xanchor="left",
                    y=1.15,
                    yanchor="top",
                ),
            ],
        )

        return fig

    def _calculate_path_statistics(
        self,
        paths: List[List[Dict]],
        edge_statistics: Optional[Dict[Tuple[str, str], EdgeStatistics]],
        path_scores: Optional[List[float]],
    ) -> List[PathStatistics]:
        """パスごとの統計情報を計算"""
        statistics = []

        for i, path in enumerate(paths):
            total_transitions = 0
            total_days = 0
            edge_count = 0

            # エッジごとの統計を集計
            for j in range(len(path) - 1):
                source_id = path[j]["id"]
                target_id = path[j + 1]["id"]

                if edge_statistics and (source_id, target_id) in edge_statistics:
                    stat = edge_statistics[(source_id, target_id)]
                    total_transitions += stat.transition_count
                    total_days += stat.avg_days
                    edge_count += 1

            # 平均を計算
            avg_days = total_days / edge_count if edge_count > 0 else 0

            # 品質スコアを計算（複数要素を考慮）
            quality_score = self._calculate_quality_score(
                path_length=len(path),
                total_transitions=total_transitions,
                avg_days=avg_days,
                path_score=path_scores[i] if path_scores and i < len(path_scores) else 0.5,
            )

            statistics.append(
                PathStatistics(
                    path_id=i,
                    length=len(path),
                    total_transitions=total_transitions,
                    avg_transition_days=avg_days,
                    quality_score=quality_score,
                )
            )

        return statistics

    def _calculate_quality_score(
        self, path_length: int, total_transitions: int, avg_days: float, path_score: float
    ) -> float:
        """パス品質スコアを計算（0-1）

        考慮要素:
        - パス長（短いほど良い）
        - 遷移人数（多いほど良い）
        - 平均日数（適度な期間が良い: 30-180日）
        - パススコア（RWRスコアなど）
        """
        # パス長スコア（2-5が最適）
        if path_length < 2:
            length_score = 0.0
        elif path_length <= 5:
            length_score = 1.0 - (path_length - 2) * 0.15
        else:
            length_score = max(0.0, 1.0 - (path_length - 5) * 0.2)

        # 遷移人数スコア（対数スケール）
        transition_score = min(1.0, np.log1p(total_transitions) / np.log1p(100))

        # 日数スコア（30-180日が最適）
        if avg_days <= 0:
            days_score = 0.5
        elif 30 <= avg_days <= 180:
            days_score = 1.0
        elif avg_days < 30:
            days_score = 0.5 + (avg_days / 30) * 0.5
        else:
            days_score = max(0.3, 1.0 - (avg_days - 180) / 365 * 0.7)

        # 重み付き平均
        quality = 0.3 * length_score + 0.3 * transition_score + 0.2 * days_score + 0.2 * path_score

        return quality

    def _build_graph_from_paths(self, paths: List[List[Dict]]) -> nx.DiGraph:
        """パスからNetworkXグラフを構築（パス情報を保持）"""
        G = nx.DiGraph()

        for path_idx, path in enumerate(paths):
            for i, node in enumerate(path):
                node_id = node["id"]

                if not G.has_node(node_id):
                    # ノードタイプを調整
                    node_type = node["type"]
                    if node_type == "member" and i > 0:
                        node_type = "similar_member"

                    G.add_node(
                        node_id,
                        name=node["name"],
                        type=node_type,
                        path_indices=set([path_idx]),
                        position_in_paths={path_idx: i},
                    )
                else:
                    G.nodes[node_id]["path_indices"].add(path_idx)
                    G.nodes[node_id]["position_in_paths"][path_idx] = i

                # エッジを追加
                if i > 0:
                    prev_node_id = path[i - 1]["id"]
                    if G.has_edge(prev_node_id, node_id):
                        G[prev_node_id][node_id]["path_indices"].add(path_idx)
                    else:
                        G.add_edge(prev_node_id, node_id, path_indices=set([path_idx]), weight=1.0)

        return G

    def _calculate_layout(self, G: nx.DiGraph, paths: List[List[Dict]]) -> Dict:
        """レイアウトを計算（力学的レイアウト推奨）"""
        if self.layout_algorithm == "fruchterman_reingold":
            # Fruchterman-Reingold力学的レイアウト
            pos = nx.spring_layout(
                G, k=1.0 / np.sqrt(len(G.nodes())), iterations=50, seed=42, scale=2.0  # 最適距離
            )
        elif self.layout_algorithm == "spring":
            pos = nx.spring_layout(G, iterations=50, seed=42)
        elif self.layout_algorithm == "hierarchical":
            # 階層レイアウト（従来の方法）
            pos = self._hierarchical_layout(G, paths)
        else:
            # デフォルト: spring layout
            pos = nx.spring_layout(G, iterations=50, seed=42)

        return pos

    def _hierarchical_layout(self, G: nx.DiGraph, paths: List[List[Dict]]) -> Dict:
        """階層レイアウト（従来の方法を改良）"""
        node_layers = {}

        for path in paths:
            for i, node in enumerate(path):
                node_id = node["id"]
                if node_id not in node_layers:
                    node_layers[node_id] = i
                else:
                    node_layers[node_id] = min(node_layers[node_id], i)

        layers = {}
        for node_id, layer in node_layers.items():
            if layer not in layers:
                layers[layer] = []
            layers[layer].append(node_id)

        pos = {}
        max_layer = max(layers.keys()) if layers else 0

        for layer, nodes in layers.items():
            x = layer / max_layer if max_layer > 0 else 0.5
            n_nodes = len(nodes)

            for i, node_id in enumerate(nodes):
                if n_nodes == 1:
                    y = 0.5
                else:
                    y = 0.2 + (i / (n_nodes - 1)) * 0.6  # 0.2-0.8の範囲に配置

                pos[node_id] = (x, y)

        return pos

    def _create_interactive_figure(
        self,
        G: nx.DiGraph,
        pos: Dict,
        paths: List[List[Dict]],
        path_statistics: List[PathStatistics],
        edge_statistics: Optional[Dict[Tuple[str, str], EdgeStatistics]],
    ) -> go.Figure:
        """インタラクティブなFigureを作成"""
        fig = go.Figure()

        # パス色を生成
        path_colors = self._generate_path_colors(len(paths))

        # 各パスをトレースとして追加（個別にトグル可能）
        for path_idx, (path, stat) in enumerate(zip(paths, path_statistics)):
            self._add_path_trace(
                fig, path, pos, path_idx, path_colors[path_idx], stat, edge_statistics
            )

        # ノードを追加
        self._add_node_traces(fig, G, pos)

        return fig

    def _add_path_trace(
        self,
        fig: go.Figure,
        path: List[Dict],
        pos: Dict,
        path_idx: int,
        color: str,
        stat: PathStatistics,
        edge_statistics: Optional[Dict[Tuple[str, str], EdgeStatistics]],
    ):
        """パスのトレースを追加"""
        x_coords = []
        y_coords = []
        hover_texts = []

        for i in range(len(path) - 1):
            node_id = path[i]["id"]
            next_node_id = path[i + 1]["id"]

            if node_id in pos and next_node_id in pos:
                x0, y0 = pos[node_id]
                x1, y1 = pos[next_node_id]

                x_coords.extend([x0, x1, None])
                y_coords.extend([y0, y1, None])

                # ホバーテキストを生成
                hover_text = self._generate_edge_hover_text(path[i], path[i + 1], edge_statistics)
                hover_texts.extend([hover_text, hover_text, None])

        # エッジのトレースを追加
        if x_coords:
            fig.add_trace(
                go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode="lines",
                    line=dict(
                        color=color,
                        width=3,
                    ),
                    hovertext=hover_texts,
                    hoverinfo="text",
                    showlegend=True,
                    name=f"パス{path_idx + 1} (品質: {stat.quality_score:.2f})",
                    legendgroup=f"path{path_idx}",
                    opacity=0.7,
                )
            )

            # 矢印を追加
            self._add_arrows_to_path(fig, path, pos, color)

    def _generate_edge_hover_text(
        self,
        source_node: Dict,
        target_node: Dict,
        edge_statistics: Optional[Dict[Tuple[str, str], EdgeStatistics]],
    ) -> str:
        """エッジのホバーテキストを生成"""
        source_id = source_node["id"]
        target_id = target_node["id"]

        text = f"<b>{source_node['name']}</b> → <b>{target_node['name']}</b><br>"

        if edge_statistics and (source_id, target_id) in edge_statistics:
            stat = edge_statistics[(source_id, target_id)]
            text += f"遷移人数: {stat.transition_count}人<br>"
            text += f"平均期間: {stat.avg_days:.1f}日<br>"
            text += f"中央値: {stat.median_days:.1f}日"
            if stat.success_rate > 0:
                text += f"<br>成功率: {stat.success_rate:.1%}"

        return text

    def _add_arrows_to_path(self, fig: go.Figure, path: List[Dict], pos: Dict, color: str):
        """パスに矢印を追加"""
        for i in range(len(path) - 1):
            node_id = path[i]["id"]
            next_node_id = path[i + 1]["id"]

            if node_id in pos and next_node_id in pos:
                x0, y0 = pos[node_id]
                x1, y1 = pos[next_node_id]

                fig.add_annotation(
                    x=x1,
                    y=y1,
                    ax=x0,
                    ay=y0,
                    xref="x",
                    yref="y",
                    axref="x",
                    ayref="y",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.2,
                    arrowwidth=2,
                    arrowcolor=color,
                    opacity=0.7,
                )

    def _add_node_traces(self, fig: go.Figure, G: nx.DiGraph, pos: Dict):
        """ノードのトレースを追加（タイプ別）"""
        node_groups = {}

        for node_id in G.nodes():
            node_data = G.nodes[node_id]
            node_type = node_data["type"]

            if node_type not in node_groups:
                node_groups[node_type] = {
                    "x": [],
                    "y": [],
                    "text": [],
                    "hovertext": [],
                    "customdata": [],
                }

            x, y = pos[node_id]
            node_groups[node_type]["x"].append(x)
            node_groups[node_type]["y"].append(y)
            node_groups[node_type]["text"].append(node_data["name"])

            # 詳細なホバーテキスト
            hover_text = self._generate_node_hover_text(node_data)
            node_groups[node_type]["hovertext"].append(hover_text)
            node_groups[node_type]["customdata"].append(node_id)

        # タイプごとにトレースを追加
        for node_type, group in node_groups.items():
            config = self.node_config.get(node_type, self.node_config["competence"])

            fig.add_trace(
                go.Scatter(
                    x=group["x"],
                    y=group["y"],
                    mode="markers+text",
                    marker=dict(
                        size=config["size"],
                        color=config["color"],
                        symbol=config["symbol"],
                        line=dict(color="white", width=2),
                    ),
                    text=group["text"],
                    textposition="top center",
                    textfont=dict(size=11, family="Arial, sans-serif"),
                    hovertext=group["hovertext"],
                    hoverinfo="text",
                    customdata=group["customdata"],
                    name=config["label"],
                    showlegend=True,
                    legendgroup="nodes",
                )
            )

    def _generate_node_hover_text(self, node_data: Dict) -> str:
        """ノードのホバーテキストを生成"""
        text = f"<b>{node_data['name']}</b><br>"
        text += f"タイプ: {node_data['type']}<br>"
        text += f"経由パス数: {len(node_data['path_indices'])}個"
        return text

    def _generate_path_colors(self, n_paths: int) -> List[str]:
        """パスごとの色を生成（色覚多様性対応）"""
        if self.colorblind_safe:
            # Okabe-Itoパレットを使用
            colors = []
            for i in range(n_paths):
                colors.append(COLORBLIND_SAFE_PALETTE[i % len(COLORBLIND_SAFE_PALETTE)])
            return colors
        else:
            # HSV色空間で生成
            colors = []
            for i in range(n_paths):
                hue = i / max(n_paths, 1)
                r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
                colors.append(f"rgb({int(r*255)},{int(g*255)},{int(b*255)})")
            return colors

    def _create_top3_visibility(self, n_paths: int) -> List[bool]:
        """Top3パスのみ表示する可視性リストを作成"""
        visibility = []
        # パスのトレース（最初のn_paths個）
        for i in range(n_paths):
            visibility.append(i < 3)
        # ノードのトレース（常に表示）
        for _ in range(10):  # ノードタイプの最大数（余裕を持たせる）
            visibility.append(True)
        return visibility

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
            font=dict(size=18, family="Arial, sans-serif"),
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="#F8F9FA",
            paper_bgcolor="white",
        )
        return fig


def create_comparison_view(visualizers: List[Tuple[str, go.Figure]]) -> go.Figure:
    """
    複数の可視化を並べて比較表示

    Args:
        visualizers: [(タイトル, Figure), ...] のリスト

    Returns:
        サブプロットを含むFigure
    """
    from plotly.subplots import make_subplots

    n_figs = len(visualizers)
    rows = (n_figs + 1) // 2
    cols = min(n_figs, 2)

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[title for title, _ in visualizers],
        horizontal_spacing=0.1,
        vertical_spacing=0.15,
    )

    for idx, (title, vis_fig) in enumerate(visualizers):
        row = idx // 2 + 1
        col = idx % 2 + 1

        for trace in vis_fig.data:
            fig.add_trace(trace, row=row, col=col)

    fig.update_layout(
        height=400 * rows,
        width=1400,
        showlegend=True,
        title_text="推薦パス比較ビュー",
        title_font_size=20,
    )

    return fig
