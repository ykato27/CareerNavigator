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
            "latent": "#2E86DE",  # 濃い青（潜在変数）
            "observed": "#A23B72",  # 濃いマゼンタ（観測変数）
        }

        # ノードサイズ
        self.node_sizes = {
            "latent": 40,
            "observed": 20,
        }

        # 有意性別の色設定（エッジ用）
        self.edge_colors = {
            "significant": "#27AE60",  # 濃い緑（有意）
            "non_significant": "#95A5A6",  # 濃いグレー（非有意）
            "loading": "#3498DB",  # 明るい青（ローディング）
            "skill_connection": "#E74C3C",  # 赤（スキル間連結）
        }

    def visualize_measurement_model(
        self,
        lambda_matrix: np.ndarray,
        latent_vars: List[str],
        observed_vars: List[str],
        loading_threshold: float = 0.3,
        skill_name_mapping: Optional[Dict[str, str]] = None,
    ) -> go.Figure:
        """
        測定モデル（観測変数→潜在変数）を可視化

        Args:
            lambda_matrix: ファクターローディング行列 (shape: n_observed × n_latent)
            latent_vars: 潜在変数名のリスト
            observed_vars: 観測変数名（スキルコード）のリスト
            loading_threshold: ローディング強度の表示閾値（デフォルト: 0.3）
            skill_name_mapping: スキルコード → スキル名（日本語）のマッピング

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
            G, pos, lambda_matrix, latent_vars, observed_vars, loading_threshold, skill_name_mapping
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

    def visualize_skill_network(
        self,
        lambda_matrix: np.ndarray,
        latent_vars: List[str],
        observed_vars: List[str],
        skill_name_mapping: Optional[Dict[str, str]] = None,
        loading_threshold: float = 0.3,
        edge_limit: Optional[int] = None,
    ) -> go.Figure:
        """
        スキル間のネットワークグラフを可視化

        同じ潜在変数に統話するスキル同士を連結。
        ローディング強度に基づいて接続。

        Args:
            lambda_matrix: ファクターローディング行列
            latent_vars: 潜在変数名
            observed_vars: 観測変数名（スキルコード）
            skill_name_mapping: スキルコード → スキル名（日本語）のマッピング
            loading_threshold: 接続判定閾値
            edge_limit: 表示するエッジの最大本数（Noneの場合は全て表示）

        Returns:
            Plotly Figure オブジェクト
        """
        # NetworkXグラフを構築
        G = nx.Graph()

        # ノード追加：スキルのみ
        for skill in observed_vars:
            # スキル名マッピングがあれば使用、なければコードを使用
            display_name = skill_name_mapping.get(skill, skill) if skill_name_mapping else skill
            G.add_node(skill, node_type="skill", display_name=display_name)

        # エッジ追加：同じ潜在変数に統話するスキル同士
        all_edges = []  # 強度順でソート用

        for j, latent in enumerate(latent_vars):
            # この潜在変数に統話するスキルを検出
            contributing_skills = []
            for i, skill in enumerate(observed_vars):
                loading = abs(lambda_matrix[i, j])
                if loading > loading_threshold:
                    contributing_skills.append((skill, loading))

            # スキル同士を接続
            for k1 in range(len(contributing_skills)):
                for k2 in range(k1 + 1, len(contributing_skills)):
                    skill1, loading1 = contributing_skills[k1]
                    skill2, loading2 = contributing_skills[k2]

                    # ローディングの平均を接続強度として使用
                    weight = (loading1 + loading2) / 2
                    latent_context = latent

                    all_edges.append({
                        'from': skill1,
                        'to': skill2,
                        'weight': weight,
                        'latent': latent_context,
                    })

        if not all_edges:
            return self._create_empty_figure("スキル間の接続が見つかりませんでした")

        # エッジを強度でソート（強い順）
        all_edges.sort(key=lambda x: x['weight'], reverse=True)

        # edge_limit が指定されていれば、上位のみを使用
        if edge_limit is not None:
            all_edges = all_edges[:edge_limit]

        # グラフにエッジを追加
        for edge in all_edges:
            G.add_edge(edge['from'], edge['to'], weight=edge['weight'], latent=edge['latent'])

        # レイアウト計算
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42, weight="weight")

        # Plotly Figure を作成
        fig = self._create_skill_network_figure(G, pos, latent_vars)

        return fig

    def visualize_combined_model(
        self,
        lambda_matrix: np.ndarray,
        b_matrix: np.ndarray,
        latent_vars: List[str],
        observed_vars: List[str],
        loading_threshold: float = 0.3,
        path_significance: Optional[Dict[Tuple[str, str], bool]] = None,
        skill_name_mapping: Optional[Dict[str, str]] = None,
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
            skill_name_mapping: スキルコード → スキル名（日本語）のマッピング

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
        fig = self._create_combined_figure(G, pos, lambda_matrix, b_matrix, skill_name_mapping)

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
        skill_name_mapping: Optional[Dict[str, str]] = None,
    ) -> go.Figure:
        """
        測定モデルのFigureを作成
        """
        fig = go.Figure()

        # スキル表示用のマッピング
        def get_display_name(code: str, mapping: Optional[Dict[str, str]] = None) -> str:
            if mapping and code in mapping:
                return mapping[code]
            return code

        # エッジを描画
        for edge in G.edges(data=True):
            from_node, to_node, data = edge
            x0, y0 = pos[from_node]
            x1, y1 = pos[to_node]

            loading = data["weight"]
            line_width = 1 + loading * 3  # 線の太さをローディングで変化

            from_display = get_display_name(from_node, skill_name_mapping)
            to_display = get_display_name(to_node, skill_name_mapping)

            fig.add_trace(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode="lines",
                    line=dict(
                        width=line_width,
                        color="#667eea",
                    ),
                    hovertemplate=f"{from_display} → {to_display}<br>ローディング: {loading:.3f}<extra></extra>",
                    showlegend=False,
                )
            )

        # ノードを描画：観測変数
        observed_x = [pos[node][0] for node in observed_vars]
        observed_y = [pos[node][1] for node in observed_vars]
        observed_display = [get_display_name(code, skill_name_mapping) for code in observed_vars]

        fig.add_trace(
            go.Scatter(
                x=observed_x,
                y=observed_y,
                mode="markers+text",
                marker=dict(
                    size=self.node_sizes["observed"] + 5,
                    color=self.node_colors["observed"],
                    line=dict(color="black", width=3),
                ),
                text=observed_display,
                textposition="middle center",
                textfont=dict(size=12, color="black", weight="bold"),
                hovertemplate="%{text}<extra></extra>",
                showlegend=True,
                name="スキル（観測変数）",
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
                    size=self.node_sizes["latent"] + 5,
                    color=self.node_colors["latent"],
                    line=dict(color="black", width=3),
                ),
                text=latent_vars,
                textposition="middle center",
                textfont=dict(size=13, color="black", weight="bold"),
                hovertemplate="%{text}<extra></extra>",
                showlegend=True,
                name="力量カテゴリー（潜在変数）",
            )
        )

        fig.update_layout(
            title="📊 測定モデル：スキル→力量カテゴリーの関係<br><sub>矢印の太さ = ローディング強度 | 赤い線：強い関係</sub>",
            showlegend=True,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=120),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="#F8F9FA",
            width=1100,
            height=650,
            font=dict(family="Arial, sans-serif", size=12),
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
                    size=self.node_sizes["latent"] + 5,
                    color=self.node_colors["latent"],
                    line=dict(color="black", width=3),
                ),
                text=latent_vars,
                textposition="middle center",
                textfont=dict(size=13, color="black", weight="bold"),
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            )
        )

        fig.update_layout(
            title="📊 構造モデル：力量カテゴリー間の因果関係<br><sub>濃い緑：有意 | 濃いグレー：非有意 | 線の太さ = 係数の大きさ</sub>",
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=130),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="#F8F9FA",
            width=1000,
            height=650,
            font=dict(family="Arial, sans-serif", size=12),
        )

        return fig

    def _create_combined_figure(
        self,
        G: nx.DiGraph,
        pos: Dict[str, Tuple[float, float]],
        lambda_matrix: np.ndarray,
        b_matrix: np.ndarray,
        skill_name_mapping: Optional[Dict[str, str]] = None,
    ) -> go.Figure:
        """
        統合モデルのFigureを作成（測定+構造）
        """
        fig = go.Figure()

        # スキル表示用のマッピング
        def get_display_name(code: str, mapping: Optional[Dict[str, str]] = None) -> str:
            if mapping and code in mapping:
                return mapping[code]
            return code

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

            from_display = get_display_name(from_node, skill_name_mapping)
            to_display = get_display_name(to_node, skill_name_mapping)

            fig.add_trace(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode="lines",
                    line=dict(width=line_width, color=color, dash=line_dash),
                    hovertemplate=f"{from_display} → {to_display}<br>タイプ: {edge_type}<extra></extra>",
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
            obs_display = [get_display_name(code, skill_name_mapping) for code in observed_nodes]

            fig.add_trace(
                go.Scatter(
                    x=obs_x,
                    y=obs_y,
                    mode="markers+text",
                    marker=dict(
                        size=self.node_sizes["observed"] + 3,
                        color=self.node_colors["observed"],
                        line=dict(color="black", width=3),
                    ),
                    text=obs_display,
                    textposition="middle center",
                    textfont=dict(size=11, color="black", weight="bold"),
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
                        size=self.node_sizes["latent"] + 3,
                        color=self.node_colors["latent"],
                        line=dict(color="black", width=3),
                    ),
                    text=latent_nodes,
                    textposition="middle center",
                    textfont=dict(size=12, color="black", weight="bold"),
                    hovertemplate="%{text}<extra></extra>",
                    showlegend=True,
                    name="力量カテゴリー（潜在変数）",
                )
            )

        fig.update_layout(
            title="🧬 統合SEM構造<br><sub>下→上：測定モデル | 横：構造モデル | 濃い緑：有意 | 濃いグレー：非有意</sub>",
            showlegend=True,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=140),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="#F8F9FA",
            width=1300,
            height=750,
            font=dict(family="Arial, sans-serif", size=12),
        )

        return fig

    def _create_skill_network_figure(
        self,
        G: nx.Graph,
        pos: Dict[str, Tuple[float, float]],
        latent_vars: List[str],
    ) -> go.Figure:
        """
        スキルネットワークのFigureを作成

        ノードに display_name 属性がある場合は日本語名を使用
        """
        fig = go.Figure()

        # エッジを描画
        for edge in G.edges(data=True):
            from_node, to_node, data = edge
            x0, y0 = pos[from_node]
            x1, y1 = pos[to_node]

            weight = data["weight"]
            line_width = 2 + weight * 3

            fig.add_trace(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode="lines",
                    line=dict(width=line_width, color="#E74C3C"),
                    hovertemplate=f"{from_node} ↔ {to_node}<br>接続強度: {weight:.3f}<extra></extra>",
                    showlegend=False,
                )
            )

        # ノードを描画
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]

        # ノードテキストを取得（display_name があればそれを使用）
        node_texts = []
        for node in G.nodes():
            node_attr = G.nodes[node]
            display_name = node_attr.get('display_name', node)
            node_texts.append(display_name)

        fig.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                marker=dict(
                    size=self.node_sizes["observed"] + 5,
                    color=self.node_colors["observed"],
                    line=dict(color="white", width=3),
                ),
                text=node_texts,
                textposition="top center",
                textfont=dict(size=12, color="black", weight="bold"),
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            )
        )

        fig.update_layout(
            title="📊 スキル間ネットワーク<br><sub>同じ力量カテゴリーに統話するスキル同士の関連性</sub>",
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=120),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="#F8F9FA",
            width=1100,
            height=750,
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
