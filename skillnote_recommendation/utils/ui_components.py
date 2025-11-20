"""
UI Components for Streamlit pages

This module provides common UI components and styling utilities
with a Modern Enterprise Theme to ensure consistent look and feel across all pages.
"""

import streamlit as st


def apply_enterprise_styles():
    """
    Apply modern UI styles to the current Streamlit page.

    This function should be called at the beginning of each page
    to ensure consistent styling across the application.

    Modern Design System: Vibrant gradients, glassmorphism, and smooth animations.
    """
    st.markdown(
        """
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0%, 100% { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }
        50% { background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%); }
    }

    /* Sidebar Styles - Modern Glassmorphism */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.25);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
    }

    [data-testid="stSidebar"] .stMarkdown, 
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #1a202c !important;
        font-weight: 600;
    }

    /* Sidebar Input Elements - Modern Style */
    [data-testid="stSidebar"] .stSelectbox > div > div,
    [data-testid="stSidebar"] .stMultiSelect > div > div,
    [data-testid="stSidebar"] .stTextInput > div > div,
    [data-testid="stSidebar"] .stNumberInput > div > div {
        background: rgba(255, 255, 255, 0.9);
        border: 2px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        transition: all 0.3s ease;
    }
    
    [data-testid="stSidebar"] .stSelectbox > div > div:hover,
    [data-testid="stSidebar"] .stMultiSelect > div > div:hover,
    [data-testid="stSidebar"] .stTextInput > div > div:hover,
    [data-testid="stSidebar"] .stNumberInput > div > div:hover {
        border-color: rgba(102, 126, 234, 0.6);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
    }

    /* Buttons - Gradient with Animation */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Primary Button Variant */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        box-shadow: 0 4px 15px rgba(245, 87, 108, 0.4);
    }
    
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 20px rgba(245, 87, 108, 0.6);
    }

    /* Cards - Glassmorphism Effect */
    .stCard {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.5);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .stCard:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.2);
    }

    /* Metric Cards - Gradient Border */
    .stMetricCard {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border: 2px solid transparent;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    
    .stMetricCard::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        border-radius: 16px;
        padding: 2px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        -webkit-mask-composite: xor;
        mask-composite: exclude;
    }
    
    .stMetricCard.accent::before { 
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
    }
    
    .stMetricCard.warning::before { 
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
    }
    
    .stMetricCard.danger::before { 
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
    }
    
    .stMetricCard.info::before { 
        background: linear-gradient(135deg, #30cfd0 0%, #330867 100%); 
    }
    
    .stMetricCard:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.3);
    }

    .stMetricCard h3 {
        font-size: 0.85rem;
        color: #64748b;
        margin: 0;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stMetricCard h1 {
        font-size: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0.5rem 0 0 0;
        font-weight: 800;
    }

    /* Page Header - Gradient Background */
    .stPageHeader {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .stPageHeader::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
    }
    
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    .stPageHeader h1 {
        color: white;
        font-weight: 700;
        font-size: 2.5rem;
        margin: 0;
        display: flex;
        align-items: center;
        position: relative;
        z-index: 1;
        text-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .stPageHeader .icon {
        margin-right: 1rem;
        font-size: 3rem;
        filter: drop-shadow(0 2px 10px rgba(0,0,0,0.1));
    }
    
    .stPageHeader p {
        color: rgba(255, 255, 255, 0.95);
        font-size: 1.1rem;
        margin-top: 0.5rem;
        position: relative;
        z-index: 1;
        font-weight: 400;
    }

    /* Tabs - Modern Style */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.5);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 0.5rem;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border: none;
        color: #64748b;
        font-weight: 600;
        padding: 12px 24px;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(102, 126, 234, 0.1);
        color: #667eea;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }

    /* Expander - Modern Style */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border: 2px solid rgba(102, 126, 234, 0.2);
        border-radius: 12px;
        color: #1a202c;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: rgba(102, 126, 234, 0.5);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
    }
    
    /* Section Divider */
    .section-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
        border-radius: 2px;
    }

    /* Badge - Gradient Style */
    .badge {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .badge-success { 
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
    }
    .badge-warning { 
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
    }
    .badge-danger { 
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
    }
    .badge-info { 
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
    }
    
    /* Info/Warning/Success boxes - Modern Style */
    .stAlert {
        border-radius: 12px;
        border: none;
        backdrop-filter: blur(10px);
    }
    
    /* Dataframe - Modern Style */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    /* Metrics - Enhanced */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

</style>
""",
        unsafe_allow_html=True,
    )


def render_page_header(title: str, icon: str, description: str = ""):
    """
    Render a clean, professional page header.

    Args:
        title: Page title
        icon: Icon emoji
        description: Optional description text
    """
    st.markdown(
        f"""
<div class="stPageHeader">
    <h1><span class="icon">{icon}</span>{title}</h1>
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
    Render a card-style header section (Clean style).
    """
    icon_html = (
        f'<span style="font-size: 1.2rem; margin-right: 0.5rem;">{icon}</span>' if icon else ""
    )
    desc_html = f"<p style='color: #6C757D; margin-top: 0.5rem;'>{description}</p>" if description else ""

    st.markdown(
        f"""
<div class="stCard">
    <h2 style="margin: 0; font-size: 1.5rem; color: #212529;">{icon_html}{title}</h2>
    {desc_html}
</div>
""",
        unsafe_allow_html=True,
    )


def render_metric_card(value: str, label: str, color: str = "green"):
    """
    Render a clean metric card with border accent.

    Args:
        value: The main value to display
        label: Label for the metric
        color: Color theme (green, accent, warning, danger, info)
    """
    # Map old colors to new classes if needed, or use direct mapping
    color_map = {
        "blue": "info",
        "green": "", # Default
        "orange": "warning",
        "purple": "accent",
        "red": "danger"
    }
    
    color_class = color_map.get(color, color)
    
    # If color is not in map and not a valid class, default to green (empty string in CSS)
    if color_class not in ["accent", "warning", "danger", "info", ""]:
         color_class = ""

    return f"""
<div class="stMetricCard {color_class}">
    <h3>{label}</h3>
    <h1>{value}</h1>
</div>
"""


def render_success_message(title: str, message: str, additional_info: str = ""):
    """
    Render a success message card (Clean style).
    """
    additional_html = (
        f'<p style="font-size: 0.9rem; margin-top: 0.5rem; color: #6C757D;">{additional_info}</p>'
        if additional_info
        else ""
    )

    st.markdown(
        f"""
<div class="stMetricCard" style="border-left-color: #28a745;">
    <h2 style="margin: 0; font-size: 1.2rem; color: #28a745;">🎉 {title}</h2>
    <p style="font-size: 1.1rem; margin: 0.5rem 0; color: #212529;">{message}</p>
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
