"""
SEM（構造方程式モデリング）結果のネットワークグラフ可視化

UnifiedSEM と HierarchicalSEM の結果をネットワークグラフ（ノード・エッジ図）として
インタラクティブに可視化するモジュール。

特徴：
- 潜在変数をノード、関係をエッジで表現
- ファクターローディング（観測変数→潜在変数）も表示可能
- 統計的有意性を色・太さで表現
- Plotly による対話的可視化
- NetworkX による自動レイアウト
"""

import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class SEMNetworkVisualizer:
    """
    SEM結果をネットワークグラフで可視化するクラス
    """

    def __init__(self):
        """初期化"""
        # ノードタイプ別の色設定
        self.node_colors = {
            "latent": "#667eea",  # 青系（潜在変数）
            "observed": "#764ba2",  # 紫系（観測変数）
        }

        # ノードサイズ
        self.node_sizes = {
            "latent": 30,
            "observed": 15,
        }

        # 有意性別の色設定（エッジ用）
        self.edge_colors = {
            "significant": "#2ecc71",  # 緑（有意）
            "non_significant": "#bdc3c7",  # グレー（非有意）
        }

    def visualize_measurement_model(
        self,
        lambda_matrix: np.ndarray,
        latent_vars: List[str],
        observed_vars: List[str],
        loading_threshold: float = 0.3,
    ) -> go.Figure:
        """
        測定モデル（観測変数→潜在変数）を可視化

        Args:
            lambda_matrix: ファクターローディング行列 (shape: n_observed × n_latent)
            latent_vars: 潜在変数名のリスト
            observed_vars: 観測変数名（スキルコード）のリスト
            loading_threshold: ローディング強度の表示閾値（デフォルト: 0.3）

        Returns:
            Plotly Figure オブジェクト
        """
        # NetworkXグラフを構築
        G = nx.DiGraph()

        # ノード追加：潜在変数
        for latent in latent_vars:
            G.add_node(latent, node_type="latent")

        # ノード追加：観測変数
        for observed in observed_vars:
            G.add_node(observed, node_type="observed")

        # エッジ追加：ローディング
        edge_traces = []
        loading_values = []

        for i, observed in enumerate(observed_vars):
            for j, latent in enumerate(latent_vars):
                loading = abs(lambda_matrix[i, j])

                if loading > loading_threshold:
                    G.add_edge(observed, latent, weight=loading)
                    loading_values.append(loading)

        if not G.edges():
            return self._create_empty_figure("有意なローディングが見つかりませんでした")

        # レイアウト計算（二部グラフレイアウト）
        pos = self._calculate_bipartite_layout(latent_vars, observed_vars)

        # Plotly Figure を作成
        fig = self._create_measurement_figure(
            G, pos, lambda_matrix, latent_vars, observed_vars, loading_threshold
        )

        return fig

    def visualize_structural_model(
        self,
        b_matrix: np.ndarray,
        latent_vars: List[str],
        path_significance: Optional[Dict[Tuple[str, str], bool]] = None,
    ) -> go.Figure:
        """
        構造モデル（潜在変数→潜在変数）を可視化

        Args:
            b_matrix: 構造係数行列 B (shape: n_latent × n_latent)
            latent_vars: 潜在変数名のリスト
            path_significance: パス係数の統計的有意性

        Returns:
            Plotly Figure オブジェクト
        """
        # NetworkXグラフを構築
        G = nx.DiGraph()

        # ノード追加
        for latent in latent_vars:
            G.add_node(latent, node_type="latent")

        # エッジ追加：構造係数（0でない場合のみ）
        for i, from_var in enumerate(latent_vars):
            for j, to_var in enumerate(latent_vars):
                coeff = b_matrix[j, i]
                if abs(coeff) > 0.001:  # 数値誤差を考慮
                    is_sig = True
                    if path_significance:
                        is_sig = path_significance.get((from_var, to_var), False)
                    G.add_edge(
                        from_var,
                        to_var,
                        weight=abs(coeff),
                        coefficient=coeff,
                        is_significant=is_sig,
                    )

        if not G.edges():
            return self._create_empty_figure("構造パスが見つかりませんでした")

        # レイアウト計算（階層型）
        pos = self._calculate_hierarchical_layout(G)

        # Plotly Figure を作成
        fig = self._create_structural_figure(G, pos, latent_vars)

        return fig

    def visualize_combined_model(
        self,
        lambda_matrix: np.ndarray,
        b_matrix: np.ndarray,
        latent_vars: List[str],
        observed_vars: List[str],
        loading_threshold: float = 0.3,
        path_significance: Optional[Dict[Tuple[str, str], bool]] = None,
    ) -> go.Figure:
        """
        完全なSEMモデル（測定+構造）を統合可視化

        Args:
            lambda_matrix: ファクターローディング行列
            b_matrix: 構造係数行列
            latent_vars: 潜在変数名
            observed_vars: 観測変数名
            loading_threshold: ローディング表示閾値
            path_significance: パス係数の有意性

        Returns:
            Plotly Figure オブジェクト
        """
        # NetworkXグラフを構築
        G = nx.DiGraph()

        # ノード追加：潜在変数
        for latent in latent_vars:
            G.add_node(latent, node_type="latent", level=1)

        # ノード追加：観測変数
        for observed in observed_vars:
            G.add_node(observed, node_type="observed", level=0)

        # エッジ追加：測定モデル
        for i, observed in enumerate(observed_vars):
            for j, latent in enumerate(latent_vars):
                loading = abs(lambda_matrix[i, j])
                if loading > loading_threshold:
                    G.add_edge(
                        observed, latent, edge_type="measurement", weight=loading
                    )

        # エッジ追加：構造モデル
        for i, from_var in enumerate(latent_vars):
            for j, to_var in enumerate(latent_vars):
                coeff = b_matrix[j, i]
                if abs(coeff) > 0.001:
                    is_sig = True
                    if path_significance:
                        is_sig = path_significance.get((from_var, to_var), False)
                    G.add_edge(
                        from_var,
                        to_var,
                        edge_type="structural",
                        weight=abs(coeff),
                        coefficient=coeff,
                        is_significant=is_sig,
                    )

        if not G.edges():
            return self._create_empty_figure("エッジが見つかりませんでした")

        # レイアウト計算
        pos = self._calculate_combined_layout(G, latent_vars, observed_vars)

        # Plotly Figure を作成
        fig = self._create_combined_figure(G, pos, lambda_matrix, b_matrix)

        return fig

    # ============================================================
    # 内部メソッド
    # ============================================================

    def _calculate_bipartite_layout(
        self, latent_vars: List[str], observed_vars: List[str]
    ) -> Dict[str, Tuple[float, float]]:
        """
        二部グラフレイアウトを計算（左側：観測変数、右側：潜在変数）
        """
        pos = {}

        # 潜在変数を右側に配置
        n_latent = len(latent_vars)
        for i, var in enumerate(latent_vars):
            pos[var] = (1, (i - (n_latent - 1) / 2) * 2)

        # 観測変数を左側に配置
        n_observed = len(observed_vars)
        for i, var in enumerate(observed_vars):
            pos[var] = (0, (i - (n_observed - 1) / 2) * 1.5)

        return pos

    def _calculate_hierarchical_layout(self, G: nx.DiGraph) -> Dict[str, Tuple[float, float]]:
        """
        階層型レイアウトを計算（トップダウン）
        """
        pos = nx.spring_layout(
            G, k=2, iterations=50, seed=42, weight="weight"
        )
        return pos

    def _calculate_combined_layout(
        self,
        G: nx.DiGraph,
        latent_vars: List[str],
        observed_vars: List[str],
    ) -> Dict[str, Tuple[float, float]]:
        """
        統合モデル用レイアウト（3層構造）
        """
        pos = {}

        # 層0：観測変数（下側）
        n_observed = len(observed_vars)
        for i, var in enumerate(observed_vars):
            pos[var] = (i - (n_observed - 1) / 2, -2)

        # 層1：潜在変数（上側）
        n_latent = len(latent_vars)
        for i, var in enumerate(latent_vars):
            pos[var] = (i - (n_latent - 1) / 2, 0)

        return pos

    def _create_measurement_figure(
        self,
        G: nx.DiGraph,
        pos: Dict[str, Tuple[float, float]],
        lambda_matrix: np.ndarray,
        latent_vars: List[str],
        observed_vars: List[str],
        loading_threshold: float,
    ) -> go.Figure:
        """
        測定モデルのFigureを作成
        """
        fig = go.Figure()

        # エッジを描画
        for edge in G.edges(data=True):
            from_node, to_node, data = edge
            x0, y0 = pos[from_node]
            x1, y1 = pos[to_node]

            loading = data["weight"]
            line_width = 1 + loading * 3  # 線の太さをローディングで変化

            fig.add_trace(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode="lines",
                    line=dict(
                        width=line_width,
                        color="#667eea",
                    ),
                    hovertemplate=f"{from_node} → {to_node}<br>ローディング: {loading:.3f}<extra></extra>",
                    showlegend=False,
                )
            )

        # ノードを描画：観測変数
        observed_x = [pos[node][0] for node in observed_vars]
        observed_y = [pos[node][1] for node in observed_vars]

        fig.add_trace(
            go.Scatter(
                x=observed_x,
                y=observed_y,
                mode="markers+text",
                marker=dict(
                    size=self.node_sizes["observed"],
                    color=self.node_colors["observed"],
                    line=dict(color="white", width=2),
                ),
                text=observed_vars,
                textposition="middle center",
                textfont=dict(size=10, color="white"),
                hovertemplate="%{text}<extra></extra>",
                showlegend=True,
                name="観測変数（スキル）",
            )
        )

        # ノードを描画：潜在変数
        latent_x = [pos[node][0] for node in latent_vars]
        latent_y = [pos[node][1] for node in latent_vars]

        fig.add_trace(
            go.Scatter(
                x=latent_x,
                y=latent_y,
                mode="markers+text",
                marker=dict(
                    size=self.node_sizes["latent"],
                    color=self.node_colors["latent"],
                    line=dict(color="white", width=2),
                ),
                text=latent_vars,
                textposition="middle center",
                textfont=dict(size=11, color="white", weight="bold"),
                hovertemplate="%{text}<extra></extra>",
                showlegend=True,
                name="潜在変数",
            )
        )

        fig.update_layout(
            title="📊 測定モデル：スキル→潜在変数の関係<br><sub>矢印の太さはローディング強度</sub>",
            showlegend=True,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=100),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white",
            width=1000,
            height=600,
        )

        return fig

    def _create_structural_figure(
        self,
        G: nx.DiGraph,
        pos: Dict[str, Tuple[float, float]],
        latent_vars: List[str],
    ) -> go.Figure:
        """
        構造モデルのFigureを作成
        """
        fig = go.Figure()

        # エッジを描画
        for edge in G.edges(data=True):
            from_node, to_node, data = edge
            x0, y0 = pos[from_node]
            x1, y1 = pos[to_node]

            coefficient = data["coefficient"]
            is_significant = data.get("is_significant", True)
            line_width = 1 + abs(coefficient) * 3
            color = (
                self.edge_colors["significant"]
                if is_significant
                else self.edge_colors["non_significant"]
            )

            fig.add_trace(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode="lines",
                    line=dict(width=line_width, color=color),
                    hovertemplate=f"{from_node} → {to_node}<br>係数: {coefficient:.3f}<extra></extra>",
                    showlegend=False,
                )
            )

        # ノードを描画
        node_x = [pos[node][0] for node in latent_vars]
        node_y = [pos[node][1] for node in latent_vars]

        fig.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                marker=dict(
                    size=self.node_sizes["latent"],
                    color=self.node_colors["latent"],
                    line=dict(color="white", width=2),
                ),
                text=latent_vars,
                textposition="middle center",
                textfont=dict(size=11, color="white", weight="bold"),
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            )
        )

        fig.update_layout(
            title="📊 構造モデル：潜在変数間の因果関係<br><sub>緑：有意、グレー：非有意</sub>",
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=100),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white",
            width=1000,
            height=600,
        )

        return fig

    def _create_combined_figure(
        self,
        G: nx.DiGraph,
        pos: Dict[str, Tuple[float, float]],
        lambda_matrix: np.ndarray,
        b_matrix: np.ndarray,
    ) -> go.Figure:
        """
        統合モデルのFigureを作成（測定+構造）
        """
        fig = go.Figure()

        # エッジを描画
        for edge in G.edges(data=True):
            from_node, to_node, data = edge
            x0, y0 = pos[from_node]
            x1, y1 = pos[to_node]

            edge_type = data["edge_type"]
            line_width = 1 + data["weight"] * 3

            if edge_type == "measurement":
                color = "#667eea"
                line_dash = "solid"
            else:  # structural
                is_significant = data.get("is_significant", True)
                color = (
                    self.edge_colors["significant"]
                    if is_significant
                    else self.edge_colors["non_significant"]
                )
                line_dash = "dash" if not is_significant else "solid"

            fig.add_trace(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode="lines",
                    line=dict(width=line_width, color=color, dash=line_dash),
                    hovertemplate=f"{from_node} → {to_node}<br>タイプ: {edge_type}<extra></extra>",
                    showlegend=False,
                )
            )

        # ノード情報を取得
        latent_nodes = [node for node, attr in G.nodes(data=True) if attr.get("node_type") == "latent"]
        observed_nodes = [
            node for node, attr in G.nodes(data=True) if attr.get("node_type") == "observed"
        ]

        # 観測変数ノード
        if observed_nodes:
            obs_x = [pos[node][0] for node in observed_nodes]
            obs_y = [pos[node][1] for node in observed_nodes]

            fig.add_trace(
                go.Scatter(
                    x=obs_x,
                    y=obs_y,
                    mode="markers+text",
                    marker=dict(
                        size=self.node_sizes["observed"],
                        color=self.node_colors["observed"],
                        line=dict(color="white", width=2),
                    ),
                    text=observed_nodes,
                    textposition="middle center",
                    textfont=dict(size=9, color="white"),
                    hovertemplate="%{text}<extra></extra>",
                    showlegend=True,
                    name="スキル（観測変数）",
                )
            )

        # 潜在変数ノード
        if latent_nodes:
            lat_x = [pos[node][0] for node in latent_nodes]
            lat_y = [pos[node][1] for node in latent_nodes]

            fig.add_trace(
                go.Scatter(
                    x=lat_x,
                    y=lat_y,
                    mode="markers+text",
                    marker=dict(
                        size=self.node_sizes["latent"],
                        color=self.node_colors["latent"],
                        line=dict(color="white", width=2),
                    ),
                    text=latent_nodes,
                    textposition="middle center",
                    textfont=dict(size=10, color="white", weight="bold"),
                    hovertemplate="%{text}<extra></extra>",
                    showlegend=True,
                    name="潜在変数（力量カテゴリー）",
                )
            )

        fig.update_layout(
            title="🧬 統合SEM構造<br><sub>実線：有意なパス | 点線：非有意なパス | 下→上：測定 | 横：構造</sub>",
            showlegend=True,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=120),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white",
            width=1200,
            height=700,
        )

        return fig

    def _create_empty_figure(self, message: str) -> go.Figure:
        """
        メッセージを表示するFigureを作成
        """
        fig = go.Figure()
        fig.add_annotation(text=message, showarrow=False, font=dict(size=18))
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
        return fig
