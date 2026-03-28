"""
役職ベースの成長パス可視化

役職ごとの成長パスをグラフ構造として可視化する。
"""

import networkx as nx
import plotly.graph_objects as go
from typing import Dict, List, Optional
import logging

from .role_based_growth_path import RoleGrowthPath, SkillAcquisitionPattern

logger = logging.getLogger(__name__)


class RoleGrowthPathVisualizer:
    """
    役職の成長パスをグラフとして可視化するクラス
    """

    def __init__(self):
        """初期化"""
        # 段階ごとの色設定
        self.stage_colors = {
            "early": "#28a745",  # 緑（初期段階）
            "mid": "#ffc107",  # 黄色（中期段階）
            "late": "#dc3545",  # 赤（後期段階）
            "default": "#6c757d",  # グレー（その他）
        }

    def visualize_growth_path(
        self, growth_path: RoleGrowthPath, max_skills: int = 30, show_edges: bool = True
    ) -> go.Figure:
        """
        役職の成長パスをグラフとして可視化

        Args:
            growth_path: 成長パスオブジェクト
            max_skills: 表示する最大スキル数
            show_edges: エッジを表示するか

        Returns:
            Plotly Figure オブジェクト
        """
        if not growth_path.skills_in_order:
            return self._create_empty_figure("成長パスデータがありません")

        # 表示するスキルを制限
        skills = growth_path.skills_in_order[:max_skills]

        # NetworkXグラフを構築
        G = self._build_graph(skills, show_edges)

        # レイアウトを計算
        pos = self._calculate_layout(skills, len(growth_path.skills_in_order))

        # Plotly Figureを作成
        fig = self._create_plotly_figure(G, pos, skills, len(growth_path.skills_in_order))

        # タイトルとレイアウト設定
        title_text = (
            f"<b>役職「{growth_path.role_name}」の成長パス</b><br>"
            f"<sub>メンバー数: {growth_path.total_members}名 | "
            f"分析されたスキル数: {len(growth_path.skills_in_order)}個 "
            f"（表示: {len(skills)}個）</sub><br>"
            f"<sub style='font-size:10px'>💡 左から右へ：初期段階 → 中期段階 → 後期段階</sub>"
        )

        fig.update_layout(
            title=dict(text=title_text, x=0.5, xanchor="center", font=dict(size=16)),
            showlegend=True,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=100),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white",
            height=600,
        )

        return fig

    def visualize_multiple_roles(
        self, growth_paths: Dict[str, RoleGrowthPath], max_skills_per_role: int = 20
    ) -> Dict[str, go.Figure]:
        """
        複数の役職の成長パスを可視化

        Args:
            growth_paths: 役職名をキーとした成長パス辞書
            max_skills_per_role: 各役職で表示する最大スキル数

        Returns:
            役職名をキーとしたFigure辞書
        """
        figures = {}

        for role_name, growth_path in growth_paths.items():
            try:
                fig = self.visualize_growth_path(
                    growth_path=growth_path, max_skills=max_skills_per_role, show_edges=True
                )
                figures[role_name] = fig
            except Exception as e:
                logger.error(f"役職 '{role_name}' の可視化エラー: {e}")
                continue

        return figures

    def _build_graph(self, skills: List[SkillAcquisitionPattern], show_edges: bool) -> nx.DiGraph:
        """
        NetworkXグラフを構築

        Args:
            skills: スキルパターンのリスト
            show_edges: エッジを表示するか

        Returns:
            NetworkXの有向グラフ
        """
        G = nx.DiGraph()

        # ノードを追加
        for skill in skills:
            G.add_node(
                skill.competence_code,
                name=skill.competence_name,
                average_order=skill.average_order,
                acquisition_rate=skill.acquisition_rate,
                acquisition_count=skill.acquisition_count,
                total_members=skill.total_members,
                category=skill.category,
            )

        # エッジを追加（取得順序に基づく）
        if show_edges and len(skills) > 1:
            # 隣接するスキル間にエッジを追加
            for i in range(len(skills) - 1):
                G.add_edge(skills[i].competence_code, skills[i + 1].competence_code)

        return G

    def _calculate_layout(self, skills: List[SkillAcquisitionPattern], total_skills: int) -> Dict:
        """
        階層的なレイアウトを計算

        Args:
            skills: スキルパターンのリスト
            total_skills: 全体のスキル数

        Returns:
            ノードIDをキー、座標をバリューとした辞書
        """
        pos = {}

        # 段階の閾値
        early_threshold = 0.3
        late_threshold = 0.7

        for i, skill in enumerate(skills):
            # 全体に対する位置を計算（0.0 - 1.0）
            position_ratio = (skill.average_order + 1) / (total_skills + 1)

            # X座標：取得順序に基づく（0.0 - 1.0）
            x = position_ratio

            # Y座標：同じ段階内での順序
            # 各段階ごとにカウント
            stage_index = 0
            stage_total = 0

            if position_ratio < early_threshold:
                # 初期段階
                early_skills = [
                    s
                    for s in skills
                    if (s.average_order + 1) / (total_skills + 1) < early_threshold
                ]
                stage_index = early_skills.index(skill) if skill in early_skills else 0
                stage_total = len(early_skills)
            elif position_ratio < late_threshold:
                # 中期段階
                mid_skills = [
                    s
                    for s in skills
                    if early_threshold
                    <= (s.average_order + 1) / (total_skills + 1)
                    < late_threshold
                ]
                stage_index = mid_skills.index(skill) if skill in mid_skills else 0
                stage_total = len(mid_skills)
            else:
                # 後期段階
                late_skills = [
                    s
                    for s in skills
                    if (s.average_order + 1) / (total_skills + 1) >= late_threshold
                ]
                stage_index = late_skills.index(skill) if skill in late_skills else 0
                stage_total = len(late_skills)

            # Y座標を計算（中央揃え）
            if stage_total == 1:
                y = 0.5
            else:
                y = stage_index / (stage_total - 1)

            pos[skill.competence_code] = (x, y)

        return pos

    def _create_plotly_figure(
        self, G: nx.DiGraph, pos: Dict, skills: List[SkillAcquisitionPattern], total_skills: int
    ) -> go.Figure:
        """
        Plotly Figureを作成

        Args:
            G: NetworkXグラフ
            pos: ノード座標
            skills: スキルパターンのリスト
            total_skills: 全体のスキル数

        Returns:
            Plotly Figure
        """
        fig = go.Figure()

        # エッジを描画
        edge_trace = self._create_edge_trace(G, pos)
        if edge_trace:
            fig.add_trace(edge_trace)

        # ノードを段階ごとにグループ化して描画
        self._add_node_traces(fig, G, pos, skills, total_skills)

        return fig

    def _create_edge_trace(self, G: nx.DiGraph, pos: Dict) -> Optional[go.Scatter]:
        """
        エッジのトレースを作成

        Args:
            G: NetworkXグラフ
            pos: ノード座標

        Returns:
            エッジのScatterトレース
        """
        if not G.edges():
            return None

        edge_x = []
        edge_y = []

        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        return go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=1, color="#888"),
            hoverinfo="none",
            showlegend=False,
        )

    def _add_node_traces(
        self,
        fig: go.Figure,
        G: nx.DiGraph,
        pos: Dict,
        skills: List[SkillAcquisitionPattern],
        total_skills: int,
    ):
        """
        ノードのトレースを追加

        Args:
            fig: Plotly Figure
            G: NetworkXグラフ
            pos: ノード座標
            skills: スキルパターンのリスト
            total_skills: 全体のスキル数
        """
        # 段階の閾値
        early_threshold = 0.3
        late_threshold = 0.7

        # 段階ごとにノードをグループ化
        stages = {
            "early": {"skills": [], "label": "🌱 初期段階"},
            "mid": {"skills": [], "label": "🌿 中期段階"},
            "late": {"skills": [], "label": "🌳 後期段階"},
        }

        for skill in skills:
            position_ratio = (skill.average_order + 1) / (total_skills + 1)

            if position_ratio < early_threshold:
                stages["early"]["skills"].append(skill)
            elif position_ratio < late_threshold:
                stages["mid"]["skills"].append(skill)
            else:
                stages["late"]["skills"].append(skill)

        # 各段階のノードを描画
        for stage_key, stage_data in stages.items():
            if not stage_data["skills"]:
                continue

            node_x = []
            node_y = []
            node_text = []
            node_size = []
            node_hover = []

            for skill in stage_data["skills"]:
                x, y = pos[skill.competence_code]
                node_x.append(x)
                node_y.append(y)
                node_text.append(
                    skill.competence_name[:15] + "..."
                    if len(skill.competence_name) > 15
                    else skill.competence_name
                )

                # ノードサイズ：取得率に応じて（最小10、最大40）
                size = 10 + (skill.acquisition_rate * 30)
                node_size.append(size)

                # ホバーテキスト
                hover_text = (
                    f"<b>{skill.competence_name}</b><br>"
                    f"取得率: {skill.acquisition_rate*100:.1f}% ({skill.acquisition_count}/{skill.total_members}名)<br>"
                    f"平均取得順序: {skill.average_order:.1f}番目<br>"
                    f"カテゴリ: {skill.category}"
                )
                node_hover.append(hover_text)

            fig.add_trace(
                go.Scatter(
                    x=node_x,
                    y=node_y,
                    mode="markers+text",
                    marker=dict(
                        size=node_size,
                        color=self.stage_colors[stage_key],
                        line=dict(width=2, color="white"),
                    ),
                    text=node_text,
                    textposition="top center",
                    textfont=dict(size=9),
                    hovertext=node_hover,
                    hoverinfo="text",
                    name=stage_data["label"],
                    showlegend=True,
                )
            )

    def _create_empty_figure(self, message: str) -> go.Figure:
        """
        空のFigureを作成

        Args:
            message: 表示するメッセージ

        Returns:
            空のPlotly Figure
        """
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
            height=400,
        )
        return fig
