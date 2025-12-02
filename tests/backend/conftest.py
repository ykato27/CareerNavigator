"""
Shared fixtures for backend API tests.
"""
import pytest
import tempfile
import shutil
from pathlib import Path
from fastapi.testclient import TestClient
import pandas as pd

from backend.main import app
from backend.utils.session_manager import SessionManager


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def temp_upload_dir(tmp_path):
    """Create a temporary upload directory."""
    upload_dir = tmp_path / "temp_uploads"
    upload_dir.mkdir()
    return upload_dir


@pytest.fixture
def sample_csv_files(tmp_path):
    """Create sample CSV files for testing."""
    # Members CSV
    members_data = pd.DataFrame({
        "メンバーコード": ["M001", "M002", "M003"],
        "メンバー名": ["山田太郎", "佐藤花子", "鈴木一郎"],
        "役職": ["部長", "課長", "主任"],
        "職種": ["開発", "開発", "営業"],
        "職能・等級": ["等級5", "等級4", "等級3"]
    })
    
    # Skills CSV
    skills_data = pd.DataFrame({
        "力量コード": ["S001", "S002", "S003"],
        "力量名": ["Python", "Java", "SQL"],
        "力量カテゴリ": ["プログラミング", "プログラミング", "データベース"],
        "力量カテゴリーコード": ["CAT001", "CAT001", "CAT002"],
        "カテゴリー": ["プログラミング", "プログラミング", "データベース"],
        "力量タイプ": ["技術", "技術", "技術"]
    })
    
    # Education CSV
    education_data = pd.DataFrame({
        "力量コード": ["S001", "S002"],
        "力量名": ["Python", "Java"],
        "力量カテゴリ": ["プログラミング", "プログラミング"]
    })
    
    # License CSV
    license_data = pd.DataFrame({
        "力量コード": ["S003"],
        "力量名": ["SQL"],
        "力量カテゴリ": ["データベース"]
    })
    
    # Categories CSV
    categories_data = pd.DataFrame({
        "力量カテゴリ": ["プログラミング", "データベース"],
        "大カテゴリ": ["技術", "技術"]
    })
    
    # Acquired CSV (Member competence)
    acquired_data = pd.DataFrame({
        "メンバーコード": ["M001", "M001", "M002", "M002", "M003"],
        "力量コード": ["S001", "S002", "S001", "S003", "S003"],
        "力量タイプ": ["SKILL", "SKILL", "SKILL", "SKILL", "SKILL"],
        "レベル": [3, 2, 4, 3, 2]
    })
    
    # Save to temporary directory
    csv_dir = tmp_path / "csv_files"
    csv_dir.mkdir()
    
    members_data.to_csv(csv_dir / "members.csv", index=False, encoding='utf-8-sig')
    skills_data.to_csv(csv_dir / "skills.csv", index=False, encoding='utf-8-sig')
    education_data.to_csv(csv_dir / "education.csv", index=False, encoding='utf-8-sig')
    license_data.to_csv(csv_dir / "license.csv", index=False, encoding='utf-8-sig')
    categories_data.to_csv(csv_dir / "categories.csv", index=False, encoding='utf-8-sig')
    acquired_data.to_csv(csv_dir / "acquired.csv", index=False, encoding='utf-8-sig')
    
    return csv_dir


@pytest.fixture
def session_manager_reset():
    """Reset session manager before each test."""
    manager = SessionManager()
    # Clear all data
    manager._sessions.clear()
    manager._models.clear()
    manager._cache.clear()
    yield manager
    # Clean up after test
    manager._sessions.clear()
    manager._models.clear()
    manager._cache.clear()


@pytest.fixture
def mock_session_data(tmp_path, sample_csv_files):
    """Create a mock session with uploaded files."""
    from backend.utils import get_upload_dir
    import shutil
    
    session_id = "test_session_001"
    upload_dir = get_upload_dir(session_id)
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy sample CSV files to upload directory
    for csv_file in sample_csv_files.glob("*.csv"):
        shutil.copy(csv_file, upload_dir / csv_file.name)
    
    yield session_id
    
    # Cleanup
    if upload_dir.exists():
        shutil.rmtree(upload_dir)


@pytest.fixture
def mock_trained_model(session_manager_reset, mock_session_data):
    """Create a mock trained model."""
    from skillnote_recommendation.ml.causal_graph_recommender import CausalGraphRecommender
    from backend.utils import load_and_transform_session_data
    
    # Load and transform data
    transformed_data = load_and_transform_session_data(mock_session_data)
    
    # Create and train model
    recommender = CausalGraphRecommender(
        member_competence=transformed_data["member_competence"],
        competence_master=transformed_data["competence_master"],
        learner_params={
            "correlation_threshold": 0.2,
            "min_cluster_size": 2
        },
        weights={"readiness": 0.6, "bayesian": 0.3, "utility": 0.1}
    )
    
    recommender.fit(min_members_per_skill=1)
    
    # Store model
    model_id = f"model_{mock_session_data}_test"
    session_manager_reset.add_model(model_id, recommender)
    
    return model_id, recommender
