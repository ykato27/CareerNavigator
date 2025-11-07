"""
Streamlit integration helpers for persistence layer.
"""

import streamlit as st
from typing import Optional, Dict, Any, List
import logging

from .database import DatabaseManager
from .repository import UserRepository, RecommendationHistoryRepository
from .session_manager import SessionManager
from .model_storage import ModelStorage
from .models import User, RecommendationHistory

logger = logging.getLogger(__name__)


class StreamlitPersistenceManager:
    """Manager for integrating persistence with Streamlit."""

    def __init__(self, db_path: str = "career_navigator.db"):
        """
        Initialize persistence manager.

        Args:
            db_path: Path to database file
        """
        self.db = DatabaseManager(db_path)
        self.db.initialize_schema()

        self.user_repo = UserRepository(self.db)
        self.history_repo = RecommendationHistoryRepository(self.db)
        self.session_manager = SessionManager(self.db)
        self.model_storage = ModelStorage(self.db)

    def initialize_session(self):
        """Initialize Streamlit session state with persistence support."""
        # Initialize persistence-related session state
        if "persistence_initialized" not in st.session_state:
            st.session_state.persistence_initialized = True
            st.session_state.current_user = None
            st.session_state.current_session = None
            st.session_state.saved_models = []
            st.session_state.history_loaded = False

    def login_or_create_user(self, username: str, email: Optional[str] = None) -> User:
        """
        Login existing user or create new one.

        Args:
            username: Username
            email: Optional email

        Returns:
            User object
        """
        # Try to get existing user
        user = self.user_repo.get_user_by_username(username)

        if user is None:
            # Create new user
            user = self.user_repo.create_user(username, email)
            logger.info(f"Created new user: {username}")
        else:
            logger.info(f"Logged in existing user: {username}")

        # Update session state
        st.session_state.current_user = user

        # Create session
        session = self.session_manager.create_session(user.user_id)
        st.session_state.current_session = session

        return user

    def get_current_user(self) -> Optional[User]:
        """
        Get current logged-in user.

        Returns:
            User object or None
        """
        return st.session_state.get("current_user")

    def update_session_activity(self):
        """Update session activity timestamp."""
        if st.session_state.get("current_session"):
            session = st.session_state.current_session
            self.session_manager.update_session_activity(session.session_id)

    def save_recommendation_history(
        self,
        member_code: str,
        member_name: str,
        method: str,
        recommendations: List[Dict[str, Any]],
        reference_persons: Optional[List[Dict[str, Any]]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        execution_time: Optional[float] = None,
    ) -> Optional[RecommendationHistory]:
        """
        Save recommendation history.

        Args:
            member_code: Member code
            member_name: Member name
            method: Recommendation method
            recommendations: List of recommendations
            reference_persons: Optional reference persons
            parameters: Optional parameters used
            execution_time: Optional execution time

        Returns:
            RecommendationHistory object or None
        """
        user = self.get_current_user()
        if not user:
            logger.warning("No user logged in, cannot save history")
            return None

        history = RecommendationHistory(
            user_id=user.user_id,
            member_code=member_code,
            member_name=member_name,
            method=method,
            recommendations=recommendations,
            reference_persons=reference_persons or [],
            parameters=parameters or {},
            execution_time=execution_time,
        )

        return self.history_repo.create_history(history)

    def load_user_history(
        self, limit: Optional[int] = 20, member_code: Optional[str] = None
    ) -> List[RecommendationHistory]:
        """
        Load recommendation history for current user.

        Args:
            limit: Maximum number of records
            member_code: Optional filter by member code

        Returns:
            List of RecommendationHistory objects
        """
        user = self.get_current_user()
        if not user:
            return []

        if member_code:
            return self.history_repo.get_member_history(user.user_id, member_code, limit)
        else:
            return self.history_repo.get_user_history(user.user_id, limit=limit)

    def save_trained_model(
        self,
        model: Any,
        model_type: str,
        parameters: Dict[str, Any],
        metrics: Dict[str, float],
        training_data: Optional[Any] = None,
        description: Optional[str] = None,
    ) -> Optional[str]:
        """
        Save trained model.

        Args:
            model: Trained model object
            model_type: Model type
            parameters: Model parameters
            metrics: Model metrics
            training_data: Optional training data
            description: Optional description

        Returns:
            Model ID or None
        """
        user = self.get_current_user()
        if not user:
            logger.warning("No user logged in, cannot save model")
            return None

        try:
            metadata = self.model_storage.save_model(
                model=model,
                user_id=user.user_id,
                model_type=model_type,
                parameters=parameters,
                metrics=metrics,
                training_data=training_data,
                description=description,
            )

            # Update session
            if st.session_state.get("current_session"):
                self.session_manager.update_session_state(
                    st.session_state.current_session.session_id,
                    model_trained=True,
                    current_model_id=metadata.model_id,
                )

            return metadata.model_id
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return None

    def load_saved_model(
        self, model_type: str, training_data: Optional[Any] = None
    ) -> tuple[Optional[Any], Optional[Dict[str, Any]]]:
        """
        Load saved model for current user.

        Args:
            model_type: Model type
            training_data: Optional training data for hash matching

        Returns:
            Tuple of (model, metadata_dict) or (None, None)
        """
        user = self.get_current_user()
        if not user:
            return None, None

        model, metadata = self.model_storage.load_latest_model(
            user_id=user.user_id, model_type=model_type, training_data=training_data
        )

        if model and metadata:
            # Update session
            if st.session_state.get("current_session"):
                self.session_manager.update_session_state(
                    st.session_state.current_session.session_id,
                    model_trained=True,
                    current_model_id=metadata.model_id,
                )

            return model, metadata.to_dict()

        return None, None

    def list_saved_models(self, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List saved models for current user.

        Args:
            model_type: Optional filter by model type

        Returns:
            List of model metadata dictionaries
        """
        user = self.get_current_user()
        if not user:
            return []

        models = self.model_storage.list_user_models(user_id=user.user_id, model_type=model_type)

        return [m.to_dict() for m in models]

    def render_user_login(self):
        """Render user login UI."""
        st.sidebar.header("🔐 ユーザー管理")

        current_user = self.get_current_user()

        if current_user:
            st.sidebar.success(f"ログイン中: {current_user.username}")
            if st.sidebar.button("ログアウト"):
                st.session_state.current_user = None
                st.session_state.current_session = None
                st.rerun()
        else:
            with st.sidebar.form("login_form"):
                st.write("ユーザー名でログインまたは新規作成")
                username = st.text_input("ユーザー名", key="username_input")
                email = st.text_input("メールアドレス (任意)", key="email_input")
                submit = st.form_submit_button("ログイン")

                if submit and username:
                    try:
                        self.login_or_create_user(username=username, email=email if email else None)
                        st.rerun()
                    except Exception as e:
                        st.error(f"ログインエラー: {e}")

    def render_history_viewer(self):
        """Render recommendation history viewer."""
        user = self.get_current_user()
        if not user:
            st.info("履歴を表示するにはログインしてください")
            return

        st.subheader("📜 推薦履歴")

        # Load history
        history = self.load_user_history(limit=50)

        if not history:
            st.info("まだ推薦履歴がありません")
            return

        # Display history
        for record in history:
            with st.expander(
                f"{record.timestamp.strftime('%Y-%m-%d %H:%M')} - "
                f"{record.member_name} ({record.method})"
            ):
                st.write(f"**メンバーコード**: {record.member_code}")
                st.write(f"**推薦方法**: {record.method}")
                st.write(f"**実行時間**: {record.execution_time:.3f}秒")

                if record.parameters:
                    st.write("**パラメータ**:")
                    st.json(record.parameters)

                st.write(f"**推薦数**: {len(record.recommendations)}")

                if st.button(f"詳細表示", key=f"detail_{record.history_id}"):
                    st.json(record.recommendations)

    def render_saved_models_viewer(self):
        """Render saved models viewer."""
        user = self.get_current_user()
        if not user:
            st.info("保存済みモデルを表示するにはログインしてください")
            return

        st.subheader("💾 保存済みモデル")

        models = self.list_saved_models()

        if not models:
            st.info("保存済みモデルがありません")
            return

        # Display models
        for model in models:
            with st.expander(f"{model['model_type']} - " f"{model['created_at'][:10]}"):
                st.write(f"**モデルID**: {model['model_id']}")
                st.write(f"**作成日時**: {model['created_at']}")

                if model.get("description"):
                    st.write(f"**説明**: {model['description']}")

                if model.get("parameters"):
                    st.write("**パラメータ**:")
                    st.json(model["parameters"])

                if model.get("metrics"):
                    st.write("**メトリクス**:")
                    st.json(model["metrics"])
