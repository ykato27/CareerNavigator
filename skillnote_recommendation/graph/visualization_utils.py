"""
Visualization Utilities

グラフ可視化のエクスポートとカスタマイズ機能
"""

import plotly.graph_objects as go
from typing import Optional
import os


def export_figure_as_html(fig: go.Figure, filename: str, auto_open: bool = False) -> str:
    """
    Plotly FigureをHTMLファイルとしてエクスポート

    Args:
        fig: Plotly Figure
        filename: 出力ファイル名
        auto_open: ブラウザで自動的に開くか

    Returns:
        出力ファイルパス
    """
    # 出力ディレクトリを作成
    output_dir = "output/visualizations"
    os.makedirs(output_dir, exist_ok=True)

    # フルパスを生成
    filepath = os.path.join(output_dir, filename)
    if not filepath.endswith('.html'):
        filepath += '.html'

    # HTMLとして保存
    fig.write_html(
        filepath,
        config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
        },
        include_plotlyjs='cdn',
        auto_open=auto_open
    )

    return filepath


def export_figure_as_image(fig: go.Figure, filename: str, format: str = 'png',
                           width: int = 1200, height: int = 800, scale: int = 2) -> str:
    """
    Plotly Figureを画像ファイルとしてエクスポート

    Args:
        fig: Plotly Figure
        filename: 出力ファイル名
        format: 画像フォーマット ('png', 'jpg', 'svg', 'pdf')
        width: 画像幅（ピクセル）
        height: 画像高さ（ピクセル）
        scale: スケール倍率（高解像度用）

    Returns:
        出力ファイルパス

    Note:
        kaleido パッケージが必要です: pip install kaleido
    """
    # 出力ディレクトリを作成
    output_dir = "output/visualizations"
    os.makedirs(output_dir, exist_ok=True)

    # フルパスを生成
    filepath = os.path.join(output_dir, filename)
    if not filepath.endswith(f'.{format}'):
        filepath += f'.{format}'

    try:
        # 画像として保存
        fig.write_image(
            filepath,
            format=format,
            width=width,
            height=height,
            scale=scale
        )
        return filepath
    except Exception as e:
        # kaleido がインストールされていない場合のエラーハンドリング
        raise ImportError(
            f"画像エクスポートには kaleido が必要です: pip install kaleido\n"
            f"エラー詳細: {str(e)}"
        )


def customize_figure_layout(fig: go.Figure,
                            title: Optional[str] = None,
                            show_legend: bool = True,
                            theme: str = 'plotly',
                            width: Optional[int] = None,
                            height: Optional[int] = None) -> go.Figure:
    """
    Figureのレイアウトをカスタマイズ

    Args:
        fig: Plotly Figure
        title: タイトル（Noneの場合は変更しない）
        show_legend: 凡例を表示するか
        theme: テーマ ('plotly', 'plotly_white', 'plotly_dark', 'ggplot2', 'seaborn')
        width: 幅（ピクセル）
        height: 高さ（ピクセル）

    Returns:
        カスタマイズされたFigure
    """
    layout_updates = {
        'showlegend': show_legend,
        'template': theme,
    }

    if title:
        layout_updates['title'] = dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=20)
        )

    if width:
        layout_updates['width'] = width

    if height:
        layout_updates['height'] = height

    fig.update_layout(**layout_updates)
    return fig


class ExplanationGenerator:
    """推薦理由の詳細説明を生成するクラス"""

    def __init__(self, category_hierarchy=None):
        """
        Args:
            category_hierarchy: CategoryHierarchyインスタンス（オプション）
        """
        self.category_hierarchy = category_hierarchy

    def generate_detailed_explanation(self,
                                      paths: list,
                                      rwr_score: float,
                                      nmf_score: float,
                                      competence_info: dict) -> dict:
        """
        詳細な推薦説明を生成

        Args:
            paths: 推薦パスのリスト
            rwr_score: RWRスコア
            nmf_score: NMFスコア
            competence_info: 力量情報

        Returns:
            説明情報の辞書
        """
        explanations = {
            'summary': self._generate_summary(rwr_score, nmf_score),
            'path_based_reasons': self._generate_path_reasons(paths),
            'score_breakdown': self._generate_score_breakdown(rwr_score, nmf_score),
            'category_insights': self._generate_category_insights(paths, competence_info),
            'confidence_level': self._calculate_confidence(rwr_score, nmf_score),
        }

        return explanations

    def _generate_summary(self, rwr_score: float, nmf_score: float) -> str:
        """要約を生成"""
        if rwr_score > 0.7 and nmf_score > 0.7:
            return "グラフ構造と協調フィルタリングの両方で非常に高く評価されています。"
        elif rwr_score > nmf_score:
            return "主にグラフ構造（カテゴリーや類似メンバー）から推薦されています。"
        elif nmf_score > rwr_score:
            return "主に協調フィルタリング（類似パターン）から推薦されています。"
        else:
            return "グラフとフィルタリングの両アプローチからバランス良く推薦されています。"

    def _generate_path_reasons(self, paths: list) -> list:
        """パスベースの理由を生成"""
        reasons = []

        for path in paths:
            if len(path) < 2:
                continue

            # パス内のノードタイプを分析
            node_types = [node.get('type') for node in path]

            # パターンに応じた理由を生成
            if 'category' in node_types:
                category_names = [node['name'] for node in path if node.get('type') == 'category']
                if category_names:
                    reasons.append(f"カテゴリー「{category_names[0]}」を経由した関連性")

            if 'similar_member' in node_types or node_types.count('member') > 1:
                member_names = [node['name'] for node in path if node.get('type') in ['member', 'similar_member']]
                if len(member_names) > 1:
                    reasons.append(f"類似メンバー「{member_names[1]}」が保有")

        return reasons if reasons else ["グラフ構造に基づく推薦"]

    def _generate_score_breakdown(self, rwr_score: float, nmf_score: float) -> dict:
        """スコア内訳を生成"""
        total = rwr_score + nmf_score
        if total == 0:
            return {'graph_contribution': 0.0, 'cf_contribution': 0.0}

        return {
            'graph_contribution': (rwr_score / total) * 100,
            'cf_contribution': (nmf_score / total) * 100,
            'synergy_bonus': 10.0 if (rwr_score > 0.3 and nmf_score > 0.3) else 0.0,
        }

    def _generate_category_insights(self, paths: list, competence_info: dict) -> dict:
        """カテゴリーに関する洞察を生成"""
        insights = {
            'category': competence_info.get('カテゴリー', '不明'),
            'category_depth': 0,
            'related_categories': [],
        }

        # カテゴリー階層情報を追加
        if self.category_hierarchy and competence_info.get('カテゴリー'):
            category = competence_info['カテゴリー']
            insights['category_depth'] = self.category_hierarchy.get_level(category)
            insights['is_leaf'] = self.category_hierarchy.is_leaf(category)

            # 兄弟カテゴリーを取得
            siblings = self.category_hierarchy.get_siblings(category)
            insights['related_categories'] = siblings[:3]  # 最大3つ

        return insights

    def _calculate_confidence(self, rwr_score: float, nmf_score: float) -> str:
        """信頼度レベルを計算"""
        avg_score = (rwr_score + nmf_score) / 2

        if avg_score > 0.8:
            return "非常に高い"
        elif avg_score > 0.6:
            return "高い"
        elif avg_score > 0.4:
            return "中程度"
        elif avg_score > 0.2:
            return "低い"
        else:
            return "非常に低い"


def format_explanation_for_display(explanation: dict) -> str:
    """
    説明を表示用にフォーマット

    Args:
        explanation: generate_detailed_explanationの返り値

    Returns:
        マークダウン形式の説明文
    """
    lines = []

    # サマリー
    lines.append(f"### 📝 推薦サマリー")
    lines.append(explanation['summary'])
    lines.append("")

    # 推薦理由
    lines.append(f"### 🎯 推薦理由")
    for reason in explanation['path_based_reasons']:
        lines.append(f"- {reason}")
    lines.append("")

    # スコア内訳
    breakdown = explanation['score_breakdown']
    lines.append(f"### 📊 スコア内訳")
    lines.append(f"- グラフベース貢献度: {breakdown['graph_contribution']:.1f}%")
    lines.append(f"- 協調フィルタリング貢献度: {breakdown['cf_contribution']:.1f}%")
    if breakdown['synergy_bonus'] > 0:
        lines.append(f"- シナジーボーナス: +{breakdown['synergy_bonus']:.1f}%")
    lines.append("")

    # カテゴリー情報
    category_info = explanation['category_insights']
    lines.append(f"### 🗂️ カテゴリー情報")
    lines.append(f"- カテゴリー: {category_info['category']}")
    if category_info.get('category_depth', 0) > 0:
        lines.append(f"- 階層レベル: {category_info['category_depth']}")
    if category_info.get('related_categories'):
        related = ', '.join(category_info['related_categories'])
        lines.append(f"- 関連カテゴリー: {related}")
    lines.append("")

    # 信頼度
    lines.append(f"### ✅ 信頼度: {explanation['confidence_level']}")

    return "\n".join(lines)
