"""
データ品質モニタリングページ

データの完全性、一貫性、適時性、異常値を検出し、
品質の問題を可視化します。
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

from skillnote_recommendation.core.data_quality_monitor import (
    DataQualityMonitor,
    Severity
)
from skillnote_recommendation.utils.streamlit_helpers import (
    check_data_loaded,
    display_error_details
)
from skillnote_recommendation.utils.ui_components import (
    apply_rich_ui_styles,
    render_gradient_header
)


def create_severity_distribution_chart(report):
    """重大度分布のチャート作成"""
    severity_order = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
    severity_colors = {
        'CRITICAL': '#dc3545',  # 赤
        'HIGH': '#fd7e14',      # オレンジ
        'MEDIUM': '#ffc107',    # 黄色
        'LOW': '#28a745'        # 緑
    }

    severities = []
    counts = []
    colors = []

    for sev in severity_order:
        if sev in report.issues_by_severity:
            severities.append(sev)
            counts.append(report.issues_by_severity[sev])
            colors.append(severity_colors[sev])

    if not severities:
        return None

    fig = go.Figure(data=[
        go.Bar(
            x=severities,
            y=counts,
            marker_color=colors,
            text=counts,
            textposition='auto'
        )
    ])

    fig.update_layout(
        title='問題の重大度分布',
        xaxis_title='重大度',
        yaxis_title='問題数',
        height=400
    )

    return fig


def create_category_distribution_chart(issues):
    """カテゴリ別問題分布のチャート作成"""
    categories = {}
    for issue in issues:
        cat = issue.category
        if cat not in categories:
            categories[cat] = 0
        categories[cat] += 1

    if not categories:
        return None

    category_names = {
        'completeness': '完全性',
        'consistency': '一貫性',
        'timeliness': '適時性',
        'anomaly': '異常値'
    }

    labels = [category_names.get(cat, cat) for cat in categories.keys()]
    values = list(categories.values())

    fig = go.Figure(data=[
        go.Pie(
            labels=labels,
            values=values,
            hole=0.3
        )
    ])

    fig.update_layout(
        title='カテゴリ別問題分布',
        height=400
    )

    return fig


def display_issue(issue, index):
    """個別の問題を表示"""
    severity_colors = {
        Severity.CRITICAL: '#dc3545',
        Severity.HIGH: '#fd7e14',
        Severity.MEDIUM: '#ffc107',
        Severity.LOW: '#28a745'
    }

    severity_icons = {
        Severity.CRITICAL: '🔴',
        Severity.HIGH: '🟠',
        Severity.MEDIUM: '🟡',
        Severity.LOW: '🟢'
    }

    color = severity_colors.get(issue.severity, '#6c757d')
    icon = severity_icons.get(issue.severity, '⚪')

    with st.container():
        st.markdown(f"""
        <div style="
            border-left: 4px solid {color};
            padding: 15px;
            margin: 10px 0;
            background-color: rgba(0,0,0,0.05);
            border-radius: 5px;
        ">
            <h4 style="margin: 0 0 10px 0;">{icon} {issue.title}</h4>
            <p><strong>重大度:</strong> {issue.severity.value}</p>
            <p><strong>カテゴリ:</strong> {issue.category}</p>
            <p><strong>影響レコード数:</strong> {issue.affected_records:,}件</p>
            <p>{issue.message}</p>
        </div>
        """, unsafe_allow_html=True)

        # 詳細情報（オプション）
        if issue.details:
            with st.expander("詳細情報を表示"):
                st.json(issue.details)

        # 推奨対応（オプション）
        if issue.recommendations:
            with st.expander("推奨対応を表示"):
                for i, rec in enumerate(issue.recommendations, 1):
                    st.markdown(f"{i}. {rec}")


def main():
    st.set_page_config(
        page_title="データ品質モニタリング - CareerNavigator",
        page_icon="🔍",
        layout="wide"
    )

    # Apply rich UI styles
    apply_rich_ui_styles()

    # リッチなヘッダー
    render_gradient_header(
        title="データ品質モニタリング",
        icon="🔍",
        description="スキルノートデータの品質をチェックし、潜在的な問題を検出します"
    )

    st.markdown("""
    **チェック項目:**
    - ✅ **完全性（Completeness）**: 欠損値の検出
    - ✅ **一貫性（Consistency）**: 論理的整合性の検証
    - ✅ **適時性（Timeliness）**: データの鮮度確認
    - ✅ **異常値（Anomaly）**: 重複や異常パターンの検出
    """)

    st.markdown("---")

    # =========================================================
    # 前提条件チェック
    # =========================================================

    check_data_loaded()

    # =========================================================
    # データ準備
    # =========================================================

    td = st.session_state.transformed_data
    member_competence = td["member_competence"]
    competence_master = td["competence_master"]

    st.success(f"✅ データ読み込み完了: {len(member_competence):,}件のレコード")

    # サイドバー設定
    st.sidebar.header("⚙️ モニタリング設定")

    missing_threshold = st.sidebar.slider(
        "欠損率の閾値（%）",
        min_value=1,
        max_value=50,
        value=5,
        help="この値を超える欠損率で警告を出します"
    ) / 100

    staleness_days = st.sidebar.slider(
        "データ鮮度の閾値（日数）",
        min_value=30,
        max_value=365,
        value=180,
        help="この日数以上古いデータで警告を出します"
    )

    max_skills_per_week = st.sidebar.slider(
        "週あたりの最大スキル習得数",
        min_value=1,
        max_value=10,
        value=3,
        help="この数を超えると異常な高速習得として警告します"
    )

    # データ品質チェック実行
    st.markdown("---")
    st.header("📊 品質チェック実行")

    if st.button("🔍 データ品質チェックを実行", type="primary"):
        with st.spinner("データ品質をチェック中..."):
            # スキル依存関係を設定（オプション）
            # 必要に応じて実際の依存関係を設定
            skill_dependencies = {}

            monitor = DataQualityMonitor(
                missing_threshold=missing_threshold,
                staleness_days=staleness_days,
                max_skills_per_week=max_skills_per_week,
                skill_dependencies=skill_dependencies
            )

            # membersデータがあれば取得
            members = td.get("members", None)

            report = monitor.check_all(
                member_competence=member_competence,
                competence_master=competence_master,
                members=members
            )

            # セッションステートに保存
            st.session_state['quality_report'] = report

    # レポート表示
    if 'quality_report' in st.session_state:
        report = st.session_state['quality_report']

        st.markdown("---")
        st.header("📈 チェック結果")

        # サマリー表示
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="総レコード数",
                value=f"{report.total_records:,}"
            )

        with col2:
            st.metric(
                label="検出された問題",
                value=report.total_issues
            )

        with col3:
            critical_count = report.issues_by_severity.get('CRITICAL', 0)
            high_count = report.issues_by_severity.get('HIGH', 0)
            st.metric(
                label="高優先度の問題",
                value=critical_count + high_count,
                delta=f"CRITICAL: {critical_count}, HIGH: {high_count}",
                delta_color="inverse"
            )

        with col4:
            if report.total_issues == 0:
                health_score = 100
                health_status = "優良"
                health_color = "green"
            else:
                critical = report.issues_by_severity.get('CRITICAL', 0)
                high = report.issues_by_severity.get('HIGH', 0)
                medium = report.issues_by_severity.get('MEDIUM', 0)
                low = report.issues_by_severity.get('LOW', 0)

                # スコア計算（重み付け）
                penalty = (critical * 25) + (high * 10) + (medium * 3) + (low * 1)
                health_score = max(0, 100 - penalty)

                if health_score >= 90:
                    health_status = "優良"
                    health_color = "green"
                elif health_score >= 70:
                    health_status = "良好"
                    health_color = "blue"
                elif health_score >= 50:
                    health_status = "注意"
                    health_color = "yellow"
                else:
                    health_status = "警告"
                    health_color = "red"

            st.metric(
                label="データ品質スコア",
                value=f"{health_score}点",
                delta=health_status
            )

        # チャート表示
        if report.total_issues > 0:
            st.markdown("---")
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                severity_chart = create_severity_distribution_chart(report)
                if severity_chart:
                    st.plotly_chart(severity_chart, use_container_width=True)

            with chart_col2:
                category_chart = create_category_distribution_chart(report.issues)
                if category_chart:
                    st.plotly_chart(category_chart, use_container_width=True)

            # 問題一覧
            st.markdown("---")
            st.header("📋 検出された問題一覧")

            # フィルタリング
            filter_col1, filter_col2 = st.columns(2)

            with filter_col1:
                severity_filter = st.multiselect(
                    "重大度でフィルタ",
                    options=['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'],
                    default=['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
                )

            with filter_col2:
                category_filter = st.multiselect(
                    "カテゴリでフィルタ",
                    options=['completeness', 'consistency', 'timeliness', 'anomaly'],
                    default=['completeness', 'consistency', 'timeliness', 'anomaly']
                )

            # フィルタリング適用
            filtered_issues = [
                issue for issue in report.issues
                if issue.severity.value in severity_filter and issue.category in category_filter
            ]

            if filtered_issues:
                st.markdown(f"**{len(filtered_issues)}件の問題を表示中**")

                for idx, issue in enumerate(filtered_issues, 1):
                    display_issue(issue, idx)
            else:
                st.info("フィルタ条件に一致する問題はありません。")
        else:
            st.success("🎉 データ品質に問題は検出されませんでした！")

        # レポートのエクスポート
        st.markdown("---")
        st.header("💾 レポートのエクスポート")

        # CSVエクスポート
        if report.total_issues > 0:
            export_data = []
            for issue in report.issues:
                export_data.append({
                    'タイムスタンプ': report.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    '重大度': issue.severity.value,
                    'カテゴリ': issue.category,
                    '問題タイトル': issue.title,
                    'メッセージ': issue.message,
                    '影響レコード数': issue.affected_records
                })

            export_df = pd.DataFrame(export_data)
            csv = export_df.to_csv(index=False, encoding='utf-8-sig')

            st.download_button(
                label="📥 CSVとしてダウンロード",
                data=csv,
                file_name=f"data_quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

        # メタ情報
        with st.expander("📊 データサマリー情報"):
            st.json(report.summary)
            st.markdown(f"**チェック実行日時:** {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
