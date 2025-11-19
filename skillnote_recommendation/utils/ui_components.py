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
        border-left: 3px solid;
    }

    /* メインコンテンツのスペーシング */
    .element-container {
        margin-bottom: 1rem;
    }

# Legacy function aliases for backward compatibility
def apply_rich_ui_styles():
    """Legacy alias for apply_enterprise_styles(). Deprecated."""
    apply_enterprise_styles()


def render_gradient_header(title: str, icon: str, description: str = ""):
    """Legacy alias for render_page_header(). Deprecated."""
    render_page_header(title, icon, description)
