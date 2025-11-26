"""
Common utilities and constants for the backend API.
"""
from pathlib import Path
from typing import Dict, Any
import pandas as pd
from fastapi import HTTPException

from skillnote_recommendation.core.data_transformer import DataTransformer


# Project root directory - centralized definition
PROJECT_ROOT = Path(__file__).parent.parent.parent


def get_upload_dir(session_id: str) -> Path:
    """
    Get the upload directory for a given session ID.

    Args:
        session_id: The session identifier

    Returns:
        Path object pointing to the session's upload directory
    """
    return PROJECT_ROOT / "backend" / "temp_uploads" / session_id


def load_csv_files(session_id: str) -> Dict[str, pd.DataFrame]:
    """
    Load all CSV files for a given session.

    Args:
        session_id: The session identifier

    Returns:
        Dictionary containing DataFrames for each CSV file

    Raises:
        HTTPException: If session directory or required files don't exist
    """
    session_dir = get_upload_dir(session_id)

    if not session_dir.exists():
        raise HTTPException(
            status_code=404,
            detail="Session data not found. Please upload data first."
        )

    csv_files = {
        'members': 'members.csv',
        'skills': 'skills.csv',
        'education': 'education.csv',
        'license': 'license.csv',
        'categories': 'categories.csv',
        'acquired': 'acquired.csv'
    }

    data = {}
    for key, filename in csv_files.items():
        filepath = session_dir / filename
        if not filepath.exists():
            raise HTTPException(
                status_code=404,
                detail=f"File not found: {filename}. Please upload all required files."
            )
        data[key] = pd.read_csv(filepath)

    return data


def transform_data(data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Transform raw CSV data into competence master and member competence.

    Args:
        data: Dictionary containing raw DataFrames

    Returns:
        Dictionary with transformed data including:
        - member_competence: Member competence DataFrame
        - competence_master: Competence master DataFrame
        - members_clean: Cleaned members DataFrame with valid members only
    """
    transformer = DataTransformer()

    competence_master = transformer.create_competence_master(data)
    member_competence, valid_members = transformer.create_member_competence(
        data, competence_master
    )

    # Create cleaned members DataFrame
    members_df = data['members'].copy()
    members_df = members_df[members_df['メンバーコード'].isin(valid_members)]

    return {
        "member_competence": member_competence,
        "competence_master": competence_master,
        "members_clean": members_df
    }


def load_and_transform_session_data(session_id: str) -> Dict[str, Any]:
    """
    Load CSV files and transform data for a given session.
    This is a convenience function combining load_csv_files and transform_data.

    Args:
        session_id: The session identifier

    Returns:
        Dictionary with transformed data

    Raises:
        HTTPException: If session or files don't exist
    """
    data = load_csv_files(session_id)
    return transform_data(data)


# Default model weights
DEFAULT_WEIGHTS = {
    'readiness': 0.6,
    'bayesian': 0.3,
    'utility': 0.1
}


def validate_weights(weights: Dict[str, float]) -> bool:
    """
    Validate that weights sum to approximately 1.0 and are all non-negative.

    Args:
        weights: Dictionary of weight values

    Returns:
        True if valid, raises HTTPException otherwise
    """
    total = sum(weights.values())
    if abs(total - 1.0) > 0.01:
        raise HTTPException(
            status_code=400,
            detail=f"Weights must sum to 1.0, got {total}"
        )

    for key, value in weights.items():
        if value < 0:
            raise HTTPException(
                status_code=400,
                detail=f"Weight '{key}' must be non-negative, got {value}"
            )

    return True
