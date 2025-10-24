"""
Streamlit helper functions for the Career Navigator application.

This module provides common Streamlit UI utilities and session state
management functions used across multiple pages.
"""

import streamlit as st
from typing import Dict, Any


# =========================================================
# Session State Management
# =========================================================

def init_session_state(defaults: Dict[str, Any] = None) -> None:
    """
    Initialize Streamlit session state with default values.

    Args:
        defaults: Dictionary of session state keys and their default values.
                 If None, uses the standard CareerNavigator defaults.

    Example:
        >>> init_session_state()
        >>> init_session_state({"custom_key": "custom_value"})
    """
    if defaults is None:
        defaults = {
            "data_loaded": False,
            "model_trained": False,
            "raw_data": None,
            "transformed_data": None,
            "ml_recommender": None,
            "temp_dir": None,
            "last_recommendations_df": None,
            "last_recommendations": None,
            "last_target_member_code": None,
        }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# =========================================================
# State Checking Functions
# =========================================================

def check_data_loaded(stop_if_not_loaded: bool = True) -> bool:
    """
    Check if data has been loaded into the session.

    Args:
        stop_if_not_loaded: If True, displays a warning and stops execution
                           when data is not loaded.

    Returns:
        True if data is loaded, False otherwise.

    Example:
        >>> if check_data_loaded():
        ...     # Proceed with data processing
        ...     pass
    """
    is_loaded = st.session_state.get("data_loaded", False)

    if not is_loaded and stop_if_not_loaded:
        st.warning("⚠️ まずデータを読み込んでください。")
        st.info(
            "👉 サイドバーから「データ読み込み」ページに戻って"
            "CSVファイルをアップロードしてください。"
        )
        st.stop()

    return is_loaded


def check_model_trained(stop_if_not_trained: bool = True) -> bool:
    """
    Check if ML model has been trained.

    Args:
        stop_if_not_trained: If True, displays a warning and stops execution
                            when model is not trained.

    Returns:
        True if model is trained, False otherwise.

    Example:
        >>> if check_model_trained():
        ...     # Proceed with inference
        ...     pass
    """
    is_trained = st.session_state.get("model_trained", False)

    if not is_trained and stop_if_not_trained:
        st.warning("⚠️ まずMLモデルを学習してください。")
        st.info(
            "👉 サイドバーから「モデル学習」ページに移動して、"
            "MLモデルを学習してください。"
        )
        st.stop()

    return is_trained


# =========================================================
# Display Helper Functions
# =========================================================

def display_error_details(error: Exception, context: str = "") -> None:
    """
    Display detailed error information in an expandable section.

    Args:
        error: The exception that occurred.
        context: Optional context string describing where the error occurred.

    Example:
        >>> try:
        ...     # Some code
        ...     pass
        ... except Exception as e:
        ...     display_error_details(e, "during model training")
    """
    import traceback

    st.error(f"❌ エラーが発生しました{context}: {type(error).__name__}: {error}")

    with st.expander("🔍 詳細なエラー情報を表示"):
        st.code(traceback.format_exc())

        st.markdown("### デバッグ情報")
        st.write("**エラー型:**", type(error).__name__)
        st.write("**エラーメッセージ:**", str(error))

        st.info("💡 このエラー情報をスクリーンショットして開発者に共有してください。")


def show_metric_cards(metrics: Dict[str, Any], columns: int = 3) -> None:
    """
    Display multiple metrics in a card-like layout using Streamlit columns.

    Args:
        metrics: Dictionary where keys are metric labels and values are metric values.
        columns: Number of columns to display metrics in.

    Example:
        >>> show_metric_cards({
        ...     "会員数": 100,
        ...     "力量数": 50,
        ...     "保有力量レコード数": 500
        ... })
    """
    cols = st.columns(columns)

    for idx, (label, value) in enumerate(metrics.items()):
        col_idx = idx % columns
        with cols[col_idx]:
            st.metric(label, value)
