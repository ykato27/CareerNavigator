"""
スキル依存関係SEM（構造方程式モデリング）

実際のスキル（力量）間の因果関係を分析します。
- 各スキルを観測変数として扱う
- スキル間の相関から因果関係（パス係数）を推定
- 統計的有意性検定を実施
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import scipy.stats as stats

try:
    import networkx as nx
    import plotly.graph_objects as go
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False

logger = logging.getLogger(__name__)


@dataclass
class SkillPathCoefficient:
    """スキル間のパス係数"""
    from_skill: str  # 元のスキル
    from_skill_name: str  # 元のスキル名
    to_skill: str  # 先のスキル
    to_skill_name: str  # 先のスキル名
    coefficient: float  # パス係数
    p_value: float  # p値
    t_value: float  # t値
    is_significant: bool  # 有意か (p < 0.05)
    ci_lower: float  # 信頼区間下限
    ci_upper: float  # 信頼区間上限


class SkillDependencySEMModel:
    """
    スキル依存関係の構造方程式モデル

    実際のスキル間の因果関係を分析し、習得経路を推定します。
    """

    def __init__(
        self,
        member_competence_df: pd.DataFrame,
        competence_master_df: pd.DataFrame,
        min_members: int = 3,
        confidence_level: float = 0.95,
    ):
        """
        初期化

        Args:
            member_competence_df: メンバー習得力量データ
            competence_master_df: 力量マスタデータ
            min_members: パス係数を推定するために必要な最小メンバー数
            confidence_level: 信頼区間のレベル (0.95 = 95%)
        """
        self.member_competence_df = member_competence_df.copy()
        self.competence_master_df = competence_master_df.copy()
        self.min_members = min_members
        self.confidence_level = confidence_level

        # スキルマスターを整理
        self._prepare_skill_master()

        # スキル依存関係を分析
        self.skill_paths: List[SkillPathCoefficient] = []
        self.skill_network: Dict[str, List[str]] = {}
        self._analyze_skill_dependencies()

        logger.info(
            f"SkillDependencySEMModel initialized with {len(self.skill_paths)} path coefficients"
        )

    def _prepare_skill_master(self):
        """スキルマスターを準備"""
        self.skill_info = {}  # {skill_code: {'name': ..., 'type': ...}}

        for _, row in self.competence_master_df.iterrows():
            skill_code = row.get("力量コード")
            self.skill_info[skill_code] = {
                'name': row.get("力量名", skill_code),
                'type': row.get("力量タイプ", "SKILL"),
                'category': row.get("力量カテゴリー名", "その他"),
            }

    def _analyze_skill_dependencies(self):
        """スキル依存関係を分析"""
        # スキル間の相関行列を計算
        skill_correlation_matrix = self._compute_skill_correlation_matrix()

        if skill_correlation_matrix is None or skill_correlation_matrix.empty:
            logger.warning("Insufficient data to compute skill correlations")
            return

        # 相関のあるスキルペアを見つけ、パス係数を推定
        skills = skill_correlation_matrix.columns.tolist()

        for i, from_skill in enumerate(skills):
            self.skill_network[from_skill] = []

            for j, to_skill in enumerate(skills):
                if i != j:
                    # 単方向の因果関係を推定（from_skill → to_skill）
                    path_coeff = self._estimate_path_coefficient(
                        from_skill, to_skill, skill_correlation_matrix
                    )

                    if path_coeff and path_coeff.is_significant:
                        self.skill_paths.append(path_coeff)
                        self.skill_network[from_skill].append(to_skill)

    def _compute_skill_correlation_matrix(self) -> Optional[pd.DataFrame]:
        """スキル間の相関行列を計算"""
        # ピボットテーブルで各メンバーのスキルレベルを取得
        skill_levels = self.member_competence_df.pivot_table(
            index='メンバーコード',
            columns='力量コード',
            values='正規化レベル',
            fill_value=0
        )

        if skill_levels.empty or len(skill_levels) < self.min_members:
            return None

        # 相関行列を計算
        correlation_matrix = skill_levels.corr(method='pearson')

        return correlation_matrix

    def _estimate_path_coefficient(
        self,
        from_skill: str,
        to_skill: str,
        correlation_matrix: pd.DataFrame,
    ) -> Optional[SkillPathCoefficient]:
        """
        パス係数を推定（偏回帰係数）

        from_skill → to_skill の因果関係を推定
        """
        try:
            # メンバーのスキルレベル取得
            from_data = self.member_competence_df[
                self.member_competence_df['力量コード'] == from_skill
            ][['メンバーコード', '正規化レベル']].rename(
                columns={'正規化レベル': from_skill}
            ).set_index('メンバーコード')

            to_data = self.member_competence_df[
                self.member_competence_df['力量コード'] == to_skill
            ][['メンバーコード', '正規化レベル']].rename(
                columns={'正規化レベル': to_skill}
            ).set_index('メンバーコード')

            # 共通メンバーでマージ
            merged = pd.concat([from_data, to_data], axis=1).dropna()

            if len(merged) < self.min_members:
                return None

            from_levels = merged[from_skill].values
            to_levels = merged[to_skill].values

            # 単回帰でパス係数を推定
            # Y = a + b*X
            n = len(from_levels)
            mean_x = np.mean(from_levels)
            mean_y = np.mean(to_levels)

            numerator = np.sum((from_levels - mean_x) * (to_levels - mean_y))
            denominator = np.sum((from_levels - mean_x) ** 2)

            if denominator == 0:
                return None

            coefficient = numerator / denominator
            intercept = mean_y - coefficient * mean_x

            # 予測値と残差を計算
            y_pred = intercept + coefficient * from_levels
            residuals = to_levels - y_pred

            # 標準誤差とt値を計算
            mse = np.sum(residuals ** 2) / (n - 2)
            se_coefficient = np.sqrt(mse / denominator)
            t_value = coefficient / se_coefficient if se_coefficient > 0 else 0

            # p値を計算
            p_value = 2 * (1 - stats.t.cdf(abs(t_value), n - 2))

            # 信頼区間を計算
            t_critical = stats.t.ppf((1 + self.confidence_level) / 2, n - 2)
            ci_lower = coefficient - t_critical * se_coefficient
            ci_upper = coefficient + t_critical * se_coefficient

            # 有意性判定（p < 0.05）
            is_significant = p_value < 0.05 and abs(coefficient) > 0.1

            return SkillPathCoefficient(
                from_skill=from_skill,
                from_skill_name=self.skill_info.get(from_skill, {}).get('name', from_skill),
                to_skill=to_skill,
                to_skill_name=self.skill_info.get(to_skill, {}).get('name', to_skill),
                coefficient=coefficient,
                p_value=p_value,
                t_value=t_value,
                is_significant=is_significant,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
            )

        except Exception as e:
            logger.debug(f"Error estimating path coefficient {from_skill}->{to_skill}: {e}")
            return None

    def calculate_sem_score(self, member_code: str, skill_code: str) -> float:
        """
        メンバーのスキルに対するSEMスコアを計算

        このスキルへの入力パスの強度に基づいてスコアを計算
        """
        # このスキルへの入力パスを見つける
        incoming_paths = [p for p in self.skill_paths if p.to_skill == skill_code]

        if not incoming_paths:
            return 0.5  # デフォルト値

        # メンバーの習得度を取得
        member_data = self.member_competence_df[
            self.member_competence_df['メンバーコード'] == member_code
        ]

        total_score = 0.0
        total_weight = 0.0

        for path in incoming_paths:
            from_skill_data = member_data[
                member_data['力量コード'] == path.from_skill
            ]

            if not from_skill_data.empty:
                from_level = from_skill_data['正規化レベル'].values[0] / 5.0
                # パス係数を重みとして使用（正規化）
                weight = max(0, path.coefficient)
                total_score += from_level * weight
                total_weight += weight

        if total_weight == 0:
            return 0.5

        return min(1.0, total_score / total_weight)

    def get_skill_network_graph(self) -> Dict[str, Any]:
        """スキルネットワークグラフを取得"""
        nodes = []
        edges = []

        # ノード情報
        for skill_code, skill_info in self.skill_info.items():
            nodes.append({
                'id': skill_code,
                'label': skill_info['name'],
                'type': skill_info['type'],
                'category': skill_info['category'],
            })

        # エッジ情報
        for path in self.skill_paths:
            edges.append({
                'from': path.from_skill,
                'to': path.to_skill,
                'coefficient': path.coefficient,
                'p_value': path.p_value,
                'is_significant': path.is_significant,
            })

        return {
            'nodes': nodes,
            'edges': edges,
        }

    def visualize_skill_network(self) -> Optional[go.Figure]:
        """スキル依存関係ネットワークを可視化"""
        if not HAS_VISUALIZATION:
            logger.warning("networkx and plotly are required for visualization")
            return None

        graph_data = self.get_skill_network_graph()

        if not graph_data['edges']:
            return None

        # ネットワークグラフを作成
        G = nx.DiGraph()

        # ノードを追加
        for node in graph_data['nodes']:
            G.add_node(node['id'], label=node['label'], node_type=node['type'])

        # エッジを追加（有意なパスのみ）
        for edge in graph_data['edges']:
            if edge['is_significant']:
                G.add_edge(
                    edge['from'],
                    edge['to'],
                    weight=abs(edge['coefficient']),
                    coefficient=edge['coefficient'],
                )

        # レイアウトを計算
        try:
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        except Exception as e:
            logger.warning(f"Spring layout failed: {e}, using circular layout")
            pos = nx.circular_layout(G)

        # エッジを描画
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode='lines',
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            showlegend=False,
        )

        # ノードを描画
        node_x, node_y, node_text, node_color = [], [], [], []

        color_map = {
            'SKILL': '#1f77b4',
            'EDUCATION': '#ff7f0e',
            'LICENSE': '#2ca02c',
        }

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            # ノード情報を取得
            node_info = next((n for n in graph_data['nodes'] if n['id'] == node), {})
            node_text.append(node_info.get('label', node))

            node_type = node_info.get('type', 'SKILL')
            node_color.append(color_map.get(node_type, '#1f77b4'))

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=node_text,
            textposition='top center',
            hoverinfo='text',
            marker=dict(
                size=15,
                color=node_color,
                line_width=2,
                line_color='#ffffff',
            ),
        )

        # Figure を作成
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title='📊 スキル依存関係ネットワーク',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600,
        )

        return fig

    def get_skill_dependencies_for_skill(self, skill_code: str) -> Dict[str, Any]:
        """特定のスキルの依存関係を取得"""
        # 入力パス（このスキルへの依存）
        incoming = [p for p in self.skill_paths if p.to_skill == skill_code]
        # 出力パス（このスキルが依存するスキル）
        outgoing = [p for p in self.skill_paths if p.from_skill == skill_code]

        return {
            'skill_code': skill_code,
            'skill_name': self.skill_info.get(skill_code, {}).get('name', skill_code),
            'incoming': [
                {
                    'skill_code': p.from_skill,
                    'skill_name': p.from_skill_name,
                    'coefficient': p.coefficient,
                    'p_value': p.p_value,
                    'is_significant': p.is_significant,
                }
                for p in incoming
            ],
            'outgoing': [
                {
                    'skill_code': p.to_skill,
                    'skill_name': p.to_skill_name,
                    'coefficient': p.coefficient,
                    'p_value': p.p_value,
                    'is_significant': p.is_significant,
                }
                for p in outgoing
            ],
        }
