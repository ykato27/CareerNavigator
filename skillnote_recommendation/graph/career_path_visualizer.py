"""
Career Path Roadmap Visualizer

キャリアパスをガントチャート風に可視化
"""

import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict
from .career_path import (
    CareerPathAnalysis,
    CompetenceGap,
    PHASE_BASIC,
    PHASE_INTERMEDIATE,
    PHASE_EXPERT,
)


# フェーズごとの色設定
PHASE_COLORS = {
    PHASE_BASIC: "#4ECDC4",  # 青緑（基礎）
    PHASE_INTERMEDIATE: "#FFD93D",  # 黄色（中級）
    PHASE_EXPERT: "#FF6B6B",  # 赤（上級）
}


class CareerPathVisualizer:
    """キャリアパスの可視化クラス"""

    def __init__(self):
        """初期化"""
        pass

    def create_roadmap(self, career_path: CareerPathAnalysis, target_member_name: str) -> go.Figure:
        """
        ガントチャート風のロードマップを作成

        Args:
            career_path: キャリアパス分析結果
            target_member_name: 目標メンバー名

        Returns:
            Plotly Figure
        """
        # データを準備
        tasks = []
        task_id = 0

        # Phase 1
        for gap in career_path.phase_1_competences:
            tasks.append(
                {
                    "Task": gap.competence_name,
                    "Start": task_id,
                    "Finish": task_id + 1,
                    "Phase": PHASE_BASIC,
                    "Priority": gap.priority_score,
                    "Ease": gap.ease_score,
                    "Importance": gap.importance_score,
                    "Category": gap.category,
                }
            )
            task_id += 1

        # Phase 2
        for gap in career_path.phase_2_competences:
            tasks.append(
                {
                    "Task": gap.competence_name,
                    "Start": task_id,
                    "Finish": task_id + 1,
                    "Phase": PHASE_INTERMEDIATE,
                    "Priority": gap.priority_score,
                    "Ease": gap.ease_score,
                    "Importance": gap.importance_score,
                    "Category": gap.category,
                }
            )
            task_id += 1

        # Phase 3
        for gap in career_path.phase_3_competences:
            tasks.append(
                {
                    "Task": gap.competence_name,
                    "Start": task_id,
                    "Finish": task_id + 1,
                    "Phase": PHASE_EXPERT,
                    "Priority": gap.priority_score,
                    "Ease": gap.ease_score,
                    "Importance": gap.importance_score,
                    "Category": gap.category,
                }
            )
            task_id += 1

        # 空の場合
        if not tasks:
            return self._create_empty_figure("学習すべき力量はありません")

        # Figureを作成
        fig = go.Figure()

        # フェーズごとにバーを追加
        for task in tasks:
            fig.add_trace(
                go.Bar(
                    name=task["Task"],
                    x=[task["Finish"] - task["Start"]],
                    y=[task["Task"]],
                    orientation="h",
                    marker=dict(
                        color=PHASE_COLORS[task["Phase"]], line=dict(color="white", width=1)
                    ),
                    hovertemplate=(
                        f"<b>{task['Task']}</b><br>"
                        + f"フェーズ: {task['Phase']}<br>"
                        + f"カテゴリー: {task['Category']}<br>"
                        + f"優先度: {task['Priority']:.2f}<br>"
                        + f"習得容易性: {task['Ease']:.2f}<br>"
                        + f"重要度: {task['Importance']:.2f}<br>"
                        + "<extra></extra>"
                    ),
                    base=task["Start"],
                    showlegend=False,
                )
            )

        # レイアウト設定
        fig.update_layout(
            title=dict(
                text=f"📚 キャリアロードマップ → {target_member_name}さん",
                x=0.5,
                xanchor="center",
                font=dict(size=20),
            ),
            xaxis=dict(
                title="学習順序",
                showgrid=True,
                gridcolor="lightgray",
            ),
            yaxis=dict(
                title="力量",
                autorange="reversed",  # 上から下に表示
            ),
            barmode="overlay",
            height=max(400, len(tasks) * 40),  # タスク数に応じて高さを調整
            plot_bgcolor="white",
            hovermode="closest",
        )

        # フェーズの境界線を追加
        phase_boundaries = self._calculate_phase_boundaries(
            career_path.phase_1_competences,
            career_path.phase_2_competences,
            career_path.phase_3_competences,
        )

        for boundary in phase_boundaries:
            fig.add_vline(
                x=boundary["x"],
                line_dash="dash",
                line_color="gray",
                annotation_text=boundary["label"],
                annotation_position="top",
            )

        return fig

    def create_progress_gauge(self, completion_rate: float) -> go.Figure:
        """
        到達度ゲージを作成

        Args:
            completion_rate: 完了率（0-1）

        Returns:
            Plotly Figure
        """
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=completion_rate * 100,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "目標到達度", "font": {"size": 24}},
                delta={"reference": 100, "increasing": {"color": "green"}},
                gauge={
                    "axis": {"range": [None, 100], "ticksuffix": "%"},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 30], "color": "#FFE5E5"},
                        {"range": [30, 70], "color": "#FFF9E5"},
                        {"range": [70, 100], "color": "#E5FFE5"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 90,
                    },
                },
            )
        )

        fig.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=20))

        return fig

    def create_category_breakdown(self, career_path: CareerPathAnalysis) -> go.Figure:
        """
        カテゴリー別の内訳を作成

        Args:
            career_path: キャリアパス分析結果

        Returns:
            Plotly Figure
        """
        # カテゴリーごとに集計
        category_counts = {}
        for gap in career_path.missing_competences:
            cat = gap.category
            if cat not in category_counts:
                category_counts[cat] = 0
            category_counts[cat] += 1

        # 円グラフを作成
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=list(category_counts.keys()),
                    values=list(category_counts.values()),
                    hole=0.3,
                    marker=dict(colors=px.colors.qualitative.Pastel),
                )
            ]
        )

        fig.update_layout(
            title=dict(text="カテゴリー別の不足力量", x=0.5, xanchor="center", font=dict(size=18)),
            height=400,
            showlegend=True,
        )

        return fig

    def _calculate_phase_boundaries(
        self,
        phase_1: List[CompetenceGap],
        phase_2: List[CompetenceGap],
        phase_3: List[CompetenceGap],
    ) -> List[Dict]:
        """
        フェーズの境界線を計算

        Returns:
            境界線の情報リスト
        """
        boundaries = []
        x = 0

        if phase_1:
            x += len(phase_1)
            boundaries.append({"x": x - 0.5, "label": f"{PHASE_BASIC} → {PHASE_INTERMEDIATE}"})

        if phase_2:
            x += len(phase_2)
            boundaries.append({"x": x - 0.5, "label": f"{PHASE_INTERMEDIATE} → {PHASE_EXPERT}"})

        return boundaries

    def _create_empty_figure(self, message: str) -> go.Figure:
        """
        空のFigureを作成

        Args:
            message: 表示メッセージ

        Returns:
            Plotly Figure
        """
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20, color="gray"),
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            height=400,
        )
        return fig


def format_career_path_summary(career_path: CareerPathAnalysis, target_member_name: str) -> str:
    """
    キャリアパス分析のサマリーをMarkdown形式でフォーマット

    Args:
        career_path: キャリアパス分析結果
        target_member_name: 目標メンバー名

    Returns:
        Markdown形式の文字列
    """
    summary = f"""
## 🎯 目標メンバー: {target_member_name}さん

### 📊 ギャップ分析

- **共通力量**: {len(career_path.common_competences)}個 ✅
- **不足力量**: {len(career_path.missing_competences)}個 📚
- **到達度**: {career_path.estimated_completion_rate * 100:.1f}%

---

### 📚 推奨学習パス

#### ▶ Phase 1: {PHASE_BASIC} ({len(career_path.phase_1_competences)}個)
"""

    for i, gap in enumerate(career_path.phase_1_competences, 1):
        stars = "⭐" * min(int(gap.priority_score * 5), 5)
        summary += f"{i}. **{gap.competence_name}** {stars}\n"
        summary += f"   - カテゴリー: {gap.category}\n"
        summary += f"   - 優先度: {gap.priority_score:.2f} | 習得容易性: {gap.ease_score:.2f}\n\n"

    summary += f"""
#### ▶ Phase 2: {PHASE_INTERMEDIATE} ({len(career_path.phase_2_competences)}個)
"""

    for i, gap in enumerate(career_path.phase_2_competences, 1):
        stars = "⭐" * min(int(gap.priority_score * 5), 5)
        summary += f"{i}. **{gap.competence_name}** {stars}\n"
        summary += f"   - カテゴリー: {gap.category}\n"
        summary += f"   - 優先度: {gap.priority_score:.2f} | 習得容易性: {gap.ease_score:.2f}\n\n"

    summary += f"""
#### ▶ Phase 3: {PHASE_EXPERT} ({len(career_path.phase_3_competences)}個)
"""

    for i, gap in enumerate(career_path.phase_3_competences, 1):
        stars = "⭐" * min(int(gap.priority_score * 5), 5)
        summary += f"{i}. **{gap.competence_name}** {stars}\n"
        summary += f"   - カテゴリー: {gap.category}\n"
        summary += f"   - 優先度: {gap.priority_score:.2f} | 習得容易性: {gap.ease_score:.2f}\n\n"

    return summary
