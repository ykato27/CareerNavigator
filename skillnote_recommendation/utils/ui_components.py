"""
UI Components for Streamlit pages

This module provides common UI components and styling utilities
to ensure consistent look and feel across all pages.
"""

import streamlit as st


def apply_rich_ui_styles():
    """
    Apply rich UI styles to the current Streamlit page.

    This function should be called at the beginning of each page
    to ensure consistent styling across the application.
    """
    st.markdown("""
<style>
    /* グローバルスタイル */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }

    /* サイドバースタイル */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }

    [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }

    /* サイドバーテキストの色 */
    [data-testid="stSidebar"] label {
        color: white !important;
        font-weight: 600;
        font-size: 0.95rem;
    }

    [data-testid="stSidebar"] .stMarkdown {
        color: white;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: white !important;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }

    /* サイドバーの入力要素 */
    [data-testid="stSidebar"] .stSelectbox > div > div,
    [data-testid="stSidebar"] .stMultiSelect > div > div {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
        border: 2px solid rgba(255, 255, 255, 0.3);
        transition: all 0.3s ease;
    }

    [data-testid="stSidebar"] .stSelectbox > div > div:hover,
    [data-testid="stSidebar"] .stMultiSelect > div > div:hover {
        border-color: rgba(255, 255, 255, 0.8);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }

    /* スライダー */
    [data-testid="stSidebar"] .stSlider {
        padding: 1rem 0;
    }

    [data-testid="stSidebar"] .stSlider > div > div > div {
        background: rgba(255, 255, 255, 0.3);
        border-radius: 10px;
    }

    [data-testid="stSidebar"] .stSlider > div > div > div > div {
        background: white;
        border: 2px solid rgba(255, 255, 255, 0.5);
    }

    /* チェックボックス */
    [data-testid="stSidebar"] .stCheckbox {
        background: rgba(255, 255, 255, 0.1);
        padding: 0.75rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }

    [data-testid="stSidebar"] .stCheckbox:hover {
        background: rgba(255, 255, 255, 0.2);
    }

    [data-testid="stSidebar"] .stCheckbox label {
        font-size: 0.9rem;
    }

    /* サイドバー内のinfo/warning/error */
    [data-testid="stSidebar"] .stAlert {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }

    /* サイドバーセクション区切り */
    [data-testid="stSidebar"] hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.5), transparent);
        margin: 1.5rem 0;
    }

    /* サイドバー内のボタン */
    [data-testid="stSidebar"] .stButton > button {
        background: white;
        color: #667eea;
        border: none;
        border-radius: 10px;
        font-weight: bold;
        padding: 0.5rem 1rem;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        background: #f0f0f0;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }

    /* サイドバーキャプション */
    [data-testid="stSidebar"] .stCaption {
        color: rgba(255, 255, 255, 0.8) !important;
        font-size: 0.85rem;
    }

    /* カードスタイル */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 12px rgba(0, 0, 0, 0.15);
    }

    /* グラデーションヘッダー */
    .gradient-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* メトリクスカード */
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .metric-card-blue {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }

    .metric-card-green {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
    }

    .metric-card-orange {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    }

    .metric-card-purple {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
    }

    /* バッジ */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: bold;
        margin: 0.25rem;
    }

    .badge-success {
        background: #28a745;
        color: white;
    }

    .badge-info {
        background: #17a2b8;
        color: white;
    }

    .badge-warning {
        background: #ffc107;
        color: black;
    }

    .badge-danger {
        background: #dc3545;
        color: white;
    }

    /* アニメーション */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .fade-in {
        animation: fadeIn 0.6s ease-out;
    }

    /* タイトル装飾 */
    .title-icon {
        font-size: 2.5rem;
        margin-right: 1rem;
        vertical-align: middle;
    }

    /* プログレスバー */
    .progress-bar {
        background: #e9ecef;
        border-radius: 10px;
        height: 20px;
        overflow: hidden;
    }

    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        transition: width 0.6s ease;
    }

    /* ボタンホバー効果 */
    .stButton>button {
        border-radius: 10px;
        font-weight: bold;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }

    /* タブスタイル */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        font-weight: bold;
    }

    /* セクション区切り */
    .section-divider {
        height: 3px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)


def render_gradient_header(title: str, icon: str, description: str = ""):
    """
    Render a gradient header with icon and description.

    Args:
        title: Page title
        icon: Icon emoji (e.g., "🎯")
        description: Optional description text
    """
    st.markdown(f"""
<div class="gradient-header fade-in">
    <h1><span class="title-icon">{icon}</span>{title}</h1>
    {f'<p style="font-size: 1.1rem; margin: 0;">{description}</p>' if description else ''}
</div>
""", unsafe_allow_html=True)


def render_section_divider():
    """Render a stylish section divider."""
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


def render_card_header(title: str, description: str = "", icon: str = ""):
    """
    Render a card-style header section.

    Args:
        title: Card title
        description: Optional description
        icon: Optional icon emoji
    """
    icon_html = f'<span style="font-size: 1.5rem; margin-right: 0.5rem;">{icon}</span>' if icon else ''
    desc_html = f'<p>{description}</p>' if description else ''

    st.markdown(f"""
<div class="card fade-in">
    <h2>{icon_html}{title}</h2>
    {desc_html}
</div>
""", unsafe_allow_html=True)


def render_metric_card(value: str, label: str, color: str = "blue"):
    """
    Render a colorful metric card.

    Args:
        value: The main value to display
        label: Label for the metric
        color: Color theme (blue, green, orange, purple)
    """
    color_class = f"metric-card-{color}" if color in ["blue", "green", "orange", "purple"] else ""

    return f"""
<div class="metric-card {color_class} fade-in">
    <h3 style="margin: 0;">{label}</h3>
    <h1 style="margin: 0.5rem 0;">{value}</h1>
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
    additional_html = f'<p style="font-size: 0.9rem; margin: 0; opacity: 0.9;">{additional_info}</p>' if additional_info else ''

    st.markdown(f"""
<div class="card metric-card-green fade-in" style="text-align: left;">
    <h2 style="margin: 0;">🎉 {title}</h2>
    <p style="font-size: 1.2rem; margin: 0.5rem 0;">{message}</p>
    {additional_html}
</div>
""", unsafe_allow_html=True)
