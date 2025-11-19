"""
UI Components for Streamlit pages

This module provides common UI components and styling utilities
with a Modern Enterprise Theme to ensure consistent look and feel across all pages.
"""

import streamlit as st


def apply_enterprise_styles():
    """
    Apply enterprise UI styles to the current Streamlit page.

    This function should be called at the beginning of each page
    to ensure consistent styling across the application.

    Modern Enterprise Theme: Clean, minimal design with corporate green color scheme.
    """
    st.markdown(
        """
<style>
    /* グローバルスタイル */
    .main {
        background: #ffffff;
    }

    /* サイドバースタイル */
    [data-testid="stSidebar"] {
        background: #f5f7fa;
    }

    [data-testid="stSidebar"] > div:first-child {
        background: #f5f7fa;
    }

    /* サイドバーテキストの色 */
    [data-testid="stSidebar"] label {
        color: #1a1a1a !important;
        font-weight: 500;
        font-size: 0.95rem;
    }

    [data-testid="stSidebar"] .stMarkdown {
        color: #1a1a1a;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #1a1a1a !important;
    }

    /* サイドバーの入力要素 */
    [data-testid="stSidebar"] .stSelectbox > div > div,
    [data-testid="stSidebar"] .stMultiSelect > div > div {
        background: #ffffff;
        border-radius: 4px;
        border: 1px solid #e0e0e0;
        transition: border-color 0.2s ease;
    }

    [data-testid="stSidebar"] .stSelectbox > div > div:hover,
    [data-testid="stSidebar"] .stMultiSelect > div > div:hover {
        border-color: #2E7D32;
    }

    [data-testid="stSidebar"] .stSelectbox > div > div:focus-within,
    [data-testid="stSidebar"] .stMultiSelect > div > div:focus-within {
        border-color: #2E7D32;
        box-shadow: 0 0 0 2px rgba(46, 125, 50, 0.1);
    }

    /* スライダー */
    [data-testid="stSidebar"] .stSlider {
        padding: 1rem 0;
    }

    [data-testid="stSidebar"] .stSlider > div > div > div {
        background: #e0e0e0;
    }

    [data-testid="stSidebar"] .stSlider > div > div > div > div {
        background: #2E7D32;
        border: 2px solid #2E7D32;
    }

    /* チェックボックス */
    [data-testid="stSidebar"] .stCheckbox {
        background: #ffffff;
        padding: 0.75rem;
        border-radius: 4px;
        margin: 0.5rem 0;
        transition: background-color 0.2s ease;
        border: 1px solid #e0e0e0;
    }

    [data-testid="stSidebar"] .stCheckbox:hover {
        background: #f5f7fa;
        border-color: #2E7D32;
    }

    [data-testid="stSidebar"] .stCheckbox label {
        font-size: 0.9rem;
        color: #1a1a1a;
    }

    /* サイドバー内のinfo/warning/error */
    [data-testid="stSidebar"] .stAlert {
        background: #ffffff;
        border-radius: 4px;
        border-left: 3px solid #2E7D32;
        margin: 1rem 0;
    }

    /* サイドバーセクション区切り */
    [data-testid="stSidebar"] hr {
        border: none;
        height: 1px;
        background: #e0e0e0;
        margin: 1.5rem 0;
    }

    /* サイドバー内のボタン */
    [data-testid="stSidebar"] .stButton > button {
        background: #2E7D32;
        color: white;
        border: none;
        border-radius: 4px;
        font-weight: 500;
        padding: 0.5rem 1rem;
        width: 100%;
        transition: background-color 0.2s ease;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        background: #1B5E20;
    }

    /* サイドバーキャプション */
    [data-testid="stSidebar"] .stCaption {
        color: #666666 !important;
        font-size: 0.85rem;
    }

    /* カードスタイル - シンプルな影 */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        margin: 1rem 0;
        border: 1px solid #f0f0f0;
    }

    /* シンプルなページヘッダー */
    .page-header {
        padding: 1.5rem 0;
        margin-bottom: 2rem;
        border-bottom: 2px solid #2E7D32;
    }

    .page-header h1 {
        color: #1a1a1a;
        margin: 0;
        font-weight: 600;
    }

    .page-header p {
        color: #666666;
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
    }

    /* メトリクスカード - ボーダーアクセント */
    .metric-card {
        background: white;
        padding: 1.25rem;
        border-radius: 4px;
        border-left: 4px solid #2E7D32;
        margin: 0.5rem 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
    }

    .metric-card h3 {
        color: #666666;
        font-size: 0.875rem;
        font-weight: 500;
        margin: 0 0 0.5rem 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .metric-card h1 {
        color: #1a1a1a;
        font-size: 2rem;
        font-weight: 600;
        margin: 0;
    }

    /* カラーバリエーション */
    .metric-card-green {
        border-left-color: #2E7D32;
    }

    .metric-card-blue {
        border-left-color: #1976D2;
    }

    .metric-card-orange {
        border-left-color: #F57C00;
    }

    .metric-card-purple {
        border-left-color: #7B1FA2;
    }

    /* バッジ - フラットデザイン */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 3px;
        font-size: 0.85rem;
        font-weight: 500;
        margin: 0.25rem;
    }

    .badge-success {
        background: #E8F5E9;
        color: #2E7D32;
    }

    .badge-info {
        background: #E3F2FD;
        color: #1976D2;
    }

    .badge-warning {
        background: #FFF3E0;
        color: #F57C00;
    }

    .badge-danger {
        background: #FFEBEE;
        color: #C62828;
    }

    /* タイトル装飾 */
    .title-icon {
        font-size: 2rem;
        margin-right: 0.75rem;
        vertical-align: middle;
    }

    /* ボタンスタイル */
    .stButton>button {
        border-radius: 4px;
        font-weight: 500;
        transition: all 0.2s ease;
        border: none;
    }

    .stButton>button[kind="primary"] {
        background: #2E7D32;
        color: white;
    }

    .stButton>button[kind="primary"]:hover {
        background: #1B5E20;
    }

    .stButton>button:hover {
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    /* タブスタイル */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 4px 4px 0 0;
        padding: 10px 20px;
        font-weight: 500;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #2E7D32;
        color: white;
    }

    /* セクション区切り */
    .section-divider {
        height: 2px;
        background: #2E7D32;
        margin: 2rem 0;
    }

    /* アラートスタイル */
    .stAlert {
        border-radius: 4px;
        border-left: 3px solid;
    }

    /* メインコンテンツのスペーシング */
    .element-container {
        margin-bottom: 1rem;
    }
</style>
""",
        unsafe_allow_html=True,
    )


def render_page_header(title: str, icon: str, description: str = ""):
    """
    Render a simple, clean page header with icon and description.

    Args:
        title: Page title
        icon: Icon emoji (e.g., "🎯")
        description: Optional description text
    """
    st.markdown(
        f"""
<div class="page-header">
    <h1><span class="title-icon">{icon}</span>{title}</h1>
    {f'<p>{description}</p>' if description else ''}
</div>
""",
        unsafe_allow_html=True,
    )


def render_section_divider():
    """Render a simple section divider."""
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


def render_card_header(title: str, description: str = "", icon: str = ""):
    """
    Render a card-style header section.

    Args:
        title: Card title
        description: Optional description
        icon: Optional icon emoji
    """
    icon_html = (
        f'<span style="font-size: 1.5rem; margin-right: 0.5rem;">{icon}</span>' if icon else ""
    )
    desc_html = f"<p style='color: #666666; margin-top: 0.5rem;'>{description}</p>" if description else ""

    st.markdown(
        f"""
<div class="card">
    <h2 style="margin: 0; color: #1a1a1a;">{icon_html}{title}</h2>
    {desc_html}
</div>
""",
        unsafe_allow_html=True,
    )


def render_metric_card(value: str, label: str, color: str = "green"):
    """
    Render a border-accent metric card.

    Args:
        value: The main value to display
        label: Label for the metric
        color: Color theme (green, blue, orange, purple)
    """
    color_class = f"metric-card-{color}" if color in ["green", "blue", "orange", "purple"] else "metric-card-green"

    return f"""
<div class="metric-card {color_class}">
    <h3>{label}</h3>
    <h1>{value}</h1>
</div>
"""


def render_success_message(title: str, message: str, additional_info: str = ""):
    """
    Render a success message card.

    Args:
        title: Success message title
        message: Main message
        additional_info: Optional additional information
    """
    additional_html = (
        f'<p style="font-size: 0.9rem; margin: 0.5rem 0 0 0; color: #666666;">{additional_info}</p>'
        if additional_info
        else ""
    )

    st.markdown(
        f"""
<div class="card" style="border-left: 4px solid #2E7D32;">
    <h2 style="margin: 0; color: #2E7D32;">🎉 {title}</h2>
    <p style="font-size: 1.1rem; margin: 0.75rem 0 0 0; color: #1a1a1a;">{message}</p>
    {additional_html}
</div>
""",
        unsafe_allow_html=True,
    )


# Legacy function aliases for backward compatibility
def apply_rich_ui_styles():
    """Legacy alias for apply_enterprise_styles(). Deprecated."""
    apply_enterprise_styles()


def render_gradient_header(title: str, icon: str, description: str = ""):
    """Legacy alias for render_page_header(). Deprecated."""
    render_page_header(title, icon, description)
