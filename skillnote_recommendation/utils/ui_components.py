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
    /* Global Styles */
    .main {
        background-color: #FFFFFF;
    }

    /* Sidebar Styles */
    [data-testid="stSidebar"] {
        background-color: #F8F9FA;
        border-right: 1px solid #E9ECEF;
    }

    [data-testid="stSidebar"] .stMarkdown, 
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #495057 !important;
    }

    /* Sidebar Input Elements */
    [data-testid="stSidebar"] .stSelectbox > div > div,
    [data-testid="stSidebar"] .stMultiSelect > div > div,
    [data-testid="stSidebar"] .stTextInput > div > div,
    [data-testid="stSidebar"] .stNumberInput > div > div {
        background-color: #FFFFFF;
        border: 1px solid #CED4DA;
        border-radius: 4px;
    }

    /* Buttons */
    .stButton > button {
        background-color: #2E7D32; /* Corporate Green */
        color: white;
        border: none;
        border-radius: 4px;
        font-weight: 600;
        padding: 0.5rem 1rem;
        transition: background-color 0.2s ease;
    }

    .stButton > button:hover {
        background-color: #1B5E20; /* Darker Green */
        border-color: #1B5E20;
        color: white;
    }
    
    .stButton > button:focus {
        color: white;
    }

    /* Cards (Custom Container) */
    .stCard {
        background-color: #FFFFFF;
        border: 1px solid #E9ECEF;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }

    /* Metric Cards (Border Style) */
    .stMetricCard {
        background-color: #FFFFFF;
        border: 1px solid #E9ECEF;
        border-left: 4px solid #2E7D32; /* Default Green */
        border-radius: 6px;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
    
    .stMetricCard.accent { border-left-color: #82C91E; } /* Lime */
    .stMetricCard.warning { border-left-color: #FFC107; } /* Amber */
    .stMetricCard.danger { border-left-color: #DC3545; } /* Red */
    .stMetricCard.info { border-left-color: #17A2B8; } /* Cyan */

    .stMetricCard h3 {
        font-size: 0.9rem;
        color: #6C757D;
        margin: 0;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stMetricCard h1 {
        font-size: 1.8rem;
        color: #212529;
        margin: 0.5rem 0 0 0;
        font-weight: 700;
    }

    /* Page Header */
    .stPageHeader {
        padding-bottom: 1rem;
        border-bottom: 2px solid #F1F3F5;
        margin-bottom: 2rem;
    }
    
    .stPageHeader h1 {
        color: #212529;
        font-weight: 600;
        font-size: 2.2rem;
        margin: 0;
        display: flex;
        align-items: center;
    }
    
    .stPageHeader .icon {
        margin-right: 1rem;
        font-size: 2.5rem;
    }
    
    .stPageHeader p {
        color: #6C757D;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        border-bottom: 2px solid #E9ECEF;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border: none;
        color: #6C757D;
        font-weight: 600;
        padding: 10px 20px;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #2E7D32;
        border-bottom: 2px solid #2E7D32;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: #FFFFFF;
        border: 1px solid #E9ECEF;
        border-radius: 4px;
        color: #212529;
        font-weight: 600;
    }
    
    /* Section Divider */
    .section-divider {
        height: 1px;
        background-color: #E9ECEF;
        margin: 2rem 0;
    }

    /* Badge */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .badge-success { background-color: #E8F5E9; color: #2E7D32; }
    .badge-warning { background-color: #FFF3E0; color: #E65100; }
    .badge-danger { background-color: #FFEBEE; color: #C62828; }
    .badge-info { background-color: #E1F5FE; color: #0277BD; }

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
