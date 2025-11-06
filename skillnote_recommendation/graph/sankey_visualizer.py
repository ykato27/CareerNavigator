"""
Sankey Diagram Visualizer for Skill Transitions

スキル遷移パターンをサンキーダイアグラムで可視化

サンキーダイアグラムの利点:
- 遷移フローが直感的に理解できる
- 遷移人数が線の太さで表現される
- 複数の推薦パスを同時に比較できる
"""

import plotly.graph_objects as go
from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import defaultdict


class SkillTransitionSankeyVisualizer:
    """スキル遷移をサンキーダイアグラムで可視化するクラス

    特徴:
    - 遷移人数をフローの太さで表現
    - ノードの重要度を自動計算
    - カテゴリーごとの色分け
    - インタラクティブなホバー情報
    """

    def __init__(self,
                 show_percentages: bool = True,
                 color_by_category: bool = True,
                 min_flow_threshold: int = 1):
        """
        Args:
            show_percentages: パーセンテージを表示
            color_by_category: カテゴリーごとに色分け
            min_flow_threshold: 表示する最小フロー数（遷移人数）
        """
        self.show_percentages = show_percentages
        self.color_by_category = color_by_category
        self.min_flow_threshold = min_flow_threshold

        # カテゴリー別の色設定
        self.category_colors = {
            'member': 'rgba(231, 76, 60, 0.6)',      # 赤
            'competence': 'rgba(52, 152, 219, 0.6)',  # 青
            'category': 'rgba(46, 204, 113, 0.6)',    # 緑
            'similar_member': 'rgba(243, 156, 18, 0.6)',  # オレンジ
            'unknown': 'rgba(149, 165, 166, 0.6)',    # グレー
        }

    def visualize_transition_flow(self,
                                  paths: List[List[Dict]],
                                  target_member_name: str,
                                  target_competence_name: str,
                                  transition_counts: Optional[Dict[Tuple[str, str], int]] = None) -> go.Figure:
        """
        スキル遷移フローをサンキーダイアグラムで可視化

        Args:
            paths: パスのリスト。各パスは [{'id': str, 'type': str, 'name': str}, ...] の形式
            target_member_name: 対象メンバー名
            target_competence_name: 推薦力量名
            transition_counts: エッジごとの遷移人数 {(source_id, target_id): count}

        Returns:
            Plotly Figure オブジェクト
        """
        if not paths:
            return self._create_empty_figure("推薦パスが見つかりませんでした")

        # ノードとリンクを抽出
        nodes, links = self._extract_nodes_and_links(paths, transition_counts)

        if not links:
            return self._create_empty_figure("表示可能な遷移フローがありません")

        # サンキーダイアグラムを作成
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=20,
                thickness=25,
                line=dict(color="white", width=2),
                label=nodes['labels'],
                color=nodes['colors'],
                customdata=nodes['customdata'],
                hovertemplate='<b>%{label}</b><br>' +
                             'タイプ: %{customdata[0]}<br>' +
                             '経由フロー数: %{customdata[1]}<br>' +
                             '<extra></extra>',
            ),
            link=dict(
                source=links['sources'],
                target=links['targets'],
                value=links['values'],
                color=links['colors'],
                customdata=links['customdata'],
                hovertemplate='<b>%{customdata[0]}</b> → <b>%{customdata[1]}</b><br>' +
                             '遷移人数: %{value}人<br>' +
                             '割合: %{customdata[2]:.1f}%<br>' +
                             '<extra></extra>',
            )
        )])

        # レイアウト設定
        fig.update_layout(
            title=dict(
                text=f"📊 スキル遷移フロー: {target_member_name} → {target_competence_name}",
                x=0.5,
                xanchor='center',
                font=dict(size=22, family='Arial, sans-serif')
            ),
            font=dict(size=12, family='Arial, sans-serif'),
            plot_bgcolor='#F8F9FA',
            paper_bgcolor='white',
            width=1200,
            height=700,
            margin=dict(l=20, r=20, t=80, b=20),
        )

        return fig

    def visualize_skill_matrix_heatmap(self,
                                       transition_matrix: Dict[Tuple[str, str], int],
                                       skill_names: Dict[str, str]) -> go.Figure:
        """
        スキル遷移マトリクスをヒートマップで可視化

        Args:
            transition_matrix: {(source_skill, target_skill): count}
            skill_names: {skill_code: skill_name}

        Returns:
            Plotly Figure オブジェクト
        """
        # スキルのリストを作成
        all_skills = sorted(set(
            [s for s, t in transition_matrix.keys()] +
            [t for s, t in transition_matrix.keys()]
        ))

        # マトリクスを作成
        n = len(all_skills)
        matrix = np.zeros((n, n))

        for i, source in enumerate(all_skills):
            for j, target in enumerate(all_skills):
                count = transition_matrix.get((source, target), 0)
                matrix[i, j] = count

        # スキル名のリスト
        skill_labels = [skill_names.get(s, s) for s in all_skills]

        # ホバーテキストを作成
        hover_texts = []
        for i, source in enumerate(all_skills):
            row_texts = []
            for j, target in enumerate(all_skills):
                count = int(matrix[i, j])
                if count > 0:
                    text = (f"<b>{skill_labels[i]}</b> → <b>{skill_labels[j]}</b><br>"
                           f"遷移人数: {count}人")
                else:
                    text = "遷移なし"
                row_texts.append(text)
            hover_texts.append(row_texts)

        # ヒートマップを作成
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=skill_labels,
            y=skill_labels,
            colorscale='Blues',
            hoverongaps=False,
            hovertext=hover_texts,
            hoverinfo='text',
            colorbar=dict(
                title='遷移人数',
                titleside='right',
            ),
        ))

        fig.update_layout(
            title=dict(
                text='🗺️ スキル遷移マトリクス（ヒートマップ）',
                x=0.5,
                xanchor='center',
                font=dict(size=22, family='Arial, sans-serif')
            ),
            xaxis=dict(
                title='遷移先スキル',
                tickangle=-45,
            ),
            yaxis=dict(
                title='遷移元スキル',
            ),
            width=1000,
            height=900,
            font=dict(size=10, family='Arial, sans-serif'),
            plot_bgcolor='white',
            paper_bgcolor='white',
        )

        return fig

    def _extract_nodes_and_links(self,
                                 paths: List[List[Dict]],
                                 transition_counts: Optional[Dict[Tuple[str, str], int]]) -> Tuple[Dict, Dict]:
        """ノードとリンクを抽出"""
        node_map = {}  # {node_id: index}
        node_data = []  # ノード情報のリスト

        link_data = defaultdict(int)  # {(source_idx, target_idx): count}
        link_details = {}  # {(source_idx, target_idx): (source_name, target_name)}

        # パスを走査してノードとリンクを収集
        for path in paths:
            for i, node in enumerate(path):
                node_id = node['id']

                # ノードを追加（初回のみ）
                if node_id not in node_map:
                    node_map[node_id] = len(node_data)
                    node_data.append({
                        'id': node_id,
                        'name': node['name'],
                        'type': node['type'] if i > 0 or not node['type'] == 'member' else 'member',
                        'flow_count': 0
                    })

                # リンクを追加
                if i > 0:
                    prev_node_id = path[i - 1]['id']
                    source_idx = node_map[prev_node_id]
                    target_idx = node_map[node_id]

                    # 遷移カウントを取得
                    if transition_counts and (prev_node_id, node_id) in transition_counts:
                        count = transition_counts[(prev_node_id, node_id)]
                    else:
                        count = 1  # デフォルト

                    link_data[(source_idx, target_idx)] += count
                    link_details[(source_idx, target_idx)] = (
                        path[i - 1]['name'],
                        node['name']
                    )

                    # ノードのフロー数を更新
                    node_data[source_idx]['flow_count'] += 1
                    node_data[target_idx]['flow_count'] += 1

        # ノード情報を整形
        nodes = {
            'labels': [n['name'] for n in node_data],
            'colors': [self._get_node_color(n['type']) for n in node_data],
            'customdata': [[n['type'], n['flow_count']] for n in node_data],
        }

        # リンク情報を整形
        total_flow = sum(link_data.values())
        links = {
            'sources': [],
            'targets': [],
            'values': [],
            'colors': [],
            'customdata': [],
        }

        for (source_idx, target_idx), count in link_data.items():
            if count >= self.min_flow_threshold:
                links['sources'].append(source_idx)
                links['targets'].append(target_idx)
                links['values'].append(count)

                # リンクの色（薄い色）
                source_type = node_data[source_idx]['type']
                color = self.category_colors.get(source_type, self.category_colors['unknown'])
                # 透明度を下げる
                color = color.replace('0.6', '0.3')
                links['colors'].append(color)

                # カスタムデータ
                source_name, target_name = link_details[(source_idx, target_idx)]
                percentage = (count / total_flow * 100) if total_flow > 0 else 0
                links['customdata'].append([source_name, target_name, percentage])

        return nodes, links

    def _get_node_color(self, node_type: str) -> str:
        """ノードタイプに応じた色を返す"""
        if self.color_by_category:
            return self.category_colors.get(node_type, self.category_colors['unknown'])
        else:
            # デフォルト色
            return 'rgba(100, 150, 200, 0.6)'

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
            font=dict(size=18, family='Arial, sans-serif'),
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='#F8F9FA',
            paper_bgcolor='white',
        )
        return fig


class TimeBasedSankeyVisualizer:
    """時間軸を考慮したサンキーダイアグラム可視化

    各遷移の時間情報を色で表現
    """

    def __init__(self):
        """初期化"""
        # 時間帯別の色設定（0-30日: 緑, 30-90日: 黄, 90-180日: オレンジ, 180日+: 赤）
        self.time_colors = {
            'fast': 'rgba(46, 204, 113, 0.4)',      # 緑（0-30日）
            'normal': 'rgba(241, 196, 15, 0.4)',    # 黄（30-90日）
            'slow': 'rgba(230, 126, 34, 0.4)',      # オレンジ（90-180日）
            'very_slow': 'rgba(231, 76, 60, 0.4)',  # 赤（180日+）
        }

    def visualize_with_time_info(self,
                                 paths: List[List[Dict]],
                                 target_member_name: str,
                                 target_competence_name: str,
                                 edge_time_info: Dict[Tuple[str, str], Dict[str, float]]) -> go.Figure:
        """
        時間情報を含むサンキーダイアグラム

        Args:
            paths: パスのリスト
            target_member_name: 対象メンバー名
            target_competence_name: 推薦力量名
            edge_time_info: エッジごとの時間情報
                {(source_id, target_id): {'avg_days': float, 'median_days': float, 'count': int}}

        Returns:
            Plotly Figure
        """
        if not paths:
            return self._create_empty_figure("推薦パスが見つかりませんでした")

        # ノードとリンクを抽出（時間情報付き）
        nodes, links = self._extract_nodes_and_links_with_time(paths, edge_time_info)

        if not links:
            return self._create_empty_figure("表示可能な遷移フローがありません")

        # サンキーダイアグラムを作成
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=20,
                thickness=25,
                line=dict(color="white", width=2),
                label=nodes['labels'],
                color=nodes['colors'],
            ),
            link=dict(
                source=links['sources'],
                target=links['targets'],
                value=links['values'],
                color=links['colors'],
                customdata=links['customdata'],
                hovertemplate='<b>%{customdata[0]}</b> → <b>%{customdata[1]}</b><br>' +
                             '遷移人数: %{value}人<br>' +
                             '平均期間: %{customdata[2]:.1f}日<br>' +
                             '中央値: %{customdata[3]:.1f}日<br>' +
                             '<extra></extra>',
            )
        )])

        # レイアウト設定
        fig.update_layout(
            title=dict(
                text=f"⏰ 時間考慮スキル遷移フロー: {target_member_name} → {target_competence_name}",
                x=0.5,
                xanchor='center',
                font=dict(size=22, family='Arial, sans-serif')
            ),
            font=dict(size=12, family='Arial, sans-serif'),
            plot_bgcolor='#F8F9FA',
            paper_bgcolor='white',
            width=1200,
            height=700,
            margin=dict(l=20, r=20, t=80, b=40),
            annotations=[
                dict(
                    text='色の意味: <span style="color:#2ECC71">■</span> 0-30日 | '
                         '<span style="color:#F1C40F">■</span> 30-90日 | '
                         '<span style="color:#E67E22">■</span> 90-180日 | '
                         '<span style="color:#E74C3C">■</span> 180日+',
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=-0.05,
                    showarrow=False,
                    font=dict(size=12),
                )
            ]
        )

        return fig

    def _extract_nodes_and_links_with_time(self,
                                           paths: List[List[Dict]],
                                           edge_time_info: Dict[Tuple[str, str], Dict[str, float]]) -> Tuple[Dict, Dict]:
        """時間情報を含むノードとリンクを抽出"""
        node_map = {}
        node_data = []
        link_data = defaultdict(lambda: {'count': 0, 'total_days': 0, 'median_days': []})

        # パスを走査
        for path in paths:
            for i, node in enumerate(path):
                node_id = node['id']

                if node_id not in node_map:
                    node_map[node_id] = len(node_data)
                    node_data.append({
                        'name': node['name'],
                        'type': node['type'],
                    })

                if i > 0:
                    prev_node_id = path[i - 1]['id']
                    source_idx = node_map[prev_node_id]
                    target_idx = node_map[node_id]

                    key = (source_idx, target_idx)
                    link_data[key]['count'] += 1
                    link_data[key]['source_name'] = path[i - 1]['name']
                    link_data[key]['target_name'] = node['name']

                    # 時間情報を追加
                    if edge_time_info and (prev_node_id, node_id) in edge_time_info:
                        time_info = edge_time_info[(prev_node_id, node_id)]
                        link_data[key]['total_days'] += time_info.get('avg_days', 0)
                        link_data[key]['median_days'].append(time_info.get('median_days', 0))

        # ノード情報を整形
        nodes = {
            'labels': [n['name'] for n in node_data],
            'colors': ['rgba(100, 150, 200, 0.6)' for _ in node_data],
        }

        # リンク情報を整形
        links = {
            'sources': [],
            'targets': [],
            'values': [],
            'colors': [],
            'customdata': [],
        }

        for (source_idx, target_idx), data in link_data.items():
            count = data['count']
            avg_days = data['total_days'] / count if count > 0 else 0
            median_days = np.median(data['median_days']) if data['median_days'] else 0

            links['sources'].append(source_idx)
            links['targets'].append(target_idx)
            links['values'].append(count)

            # 時間に応じた色
            color = self._get_time_color(avg_days)
            links['colors'].append(color)

            # カスタムデータ
            links['customdata'].append([
                data['source_name'],
                data['target_name'],
                avg_days,
                median_days
            ])

        return nodes, links

    def _get_time_color(self, avg_days: float) -> str:
        """平均日数に応じた色を返す"""
        if avg_days <= 30:
            return self.time_colors['fast']
        elif avg_days <= 90:
            return self.time_colors['normal']
        elif avg_days <= 180:
            return self.time_colors['slow']
        else:
            return self.time_colors['very_slow']

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
            font=dict(size=18, family='Arial, sans-serif'),
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='#F8F9FA',
            paper_bgcolor='white',
        )
        return fig
