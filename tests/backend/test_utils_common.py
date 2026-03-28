"""
Tests for backend/utils/common.py
"""
import pytest
import pandas as pd
from pathlib import Path
from fastapi import HTTPException

from backend.utils.common import (
    clean_column_name,
    clean_dataframe_columns,
    get_upload_dir,
    load_csv_files,
    transform_data,
    load_and_transform_session_data,
    validate_weights,
    DEFAULT_WEIGHTS,
    PROJECT_ROOT
)


class TestCleanColumnName:
    """Tests for clean_column_name function."""
    
    def test_clean_simple_column(self):
        """Test cleaning a column with metadata markers."""
        assert clean_column_name("メンバー名###[TEXT]###") == "メンバー名"
    
    def test_clean_column_without_markers(self):
        """Test column without markers remains unchanged."""
        assert clean_column_name("メンバー名") == "メンバー名"
    
    def test_clean_column_with_multiple_markers(self):
        """Test column with multiple markers."""
        result = clean_column_name("力量コード###[CODE]###力量名###[NAME]###")
        assert "###" not in result


class TestCleanDataFrameColumns:
    """Tests for clean_dataframe_columns function."""
    
    def test_clean_all_columns(self):
        """Test cleaning all columns in a DataFrame."""
        df = pd.DataFrame({
            "col1###[A]###": [1, 2],
            "col2###[B]###": [3, 4],
            "col3": [5, 6]
        })
        
        cleaned = clean_dataframe_columns(df)
        assert list(cleaned.columns) == ["col1", "col2", "col3"]
        assert cleaned["col1"].tolist() == [1, 2]
    
    def test_does_not_modify_original(self):
        """Test that original DataFrame is not modified."""
        df = pd.DataFrame({"col###[A]###": [1, 2]})
        original_cols = df.columns.tolist()
        
        cleaned = clean_dataframe_columns(df)
        
        assert df.columns.tolist() == original_cols
        assert cleaned.columns.tolist() != original_cols


class TestGetUploadDir:
    """Tests for get_upload_dir function."""
    
    def test_get_upload_dir_path(self):
        """Test upload directory path generation."""
        session_id = "test_session_123"
        result = get_upload_dir(session_id)
        
        assert isinstance(result, Path)
        assert str(result).endswith(f"temp_uploads{Path('/').as_posix() if Path('/').as_posix() != '/' else '/'}{session_id}".replace('/', Path('/').as_posix()))
        assert "temp_uploads" in str(result)
        assert session_id in str(result)


class TestLoadCsvFiles:
    """Tests for load_csv_files function."""
    
    def test_load_all_files_success(self, mock_session_data):
        """Test successfully loading all CSV files."""
        result = load_csv_files(mock_session_data)
        
        assert isinstance(result, dict)
        assert set(result.keys()) == {'members', 'skills', 'education', 'license', 'categories', 'acquired'}
        
        # Check DataFrames
        assert isinstance(result['members'], pd.DataFrame)
        assert len(result['members']) == 3
        assert len(result['skills']) == 3
    
    def test_load_nonexistent_session(self):
        """Test loading files for non-existent session."""
        with pytest.raises(HTTPException) as exc_info:
            load_csv_files("nonexistent_session")
        
        assert exc_info.value.status_code == 404
        assert "Session data not found" in exc_info.value.detail


class TestTransformData:
    """Tests for transform_data function."""
    
    def test_transform_data_success(self, mock_session_data):
        """Test successful data transformation."""
        raw_data = load_csv_files(mock_session_data)
        result = transform_data(raw_data)
        
        assert isinstance(result, dict)
        assert 'member_competence' in result
        assert 'competence_master' in result
        assert 'members_clean' in result
        
        # Check member_competence
        assert isinstance(result['member_competence'], pd.DataFrame)
        assert 'メンバーコード' in result['member_competence'].columns
        assert 'power量コード' in result['member_competence'].columns
        
        # Check competence_master
        assert isinstance(result['competence_master'], pd.DataFrame)
        
        # Check members_clean (should only include valid members)
        assert isinstance(result['members_clean'], pd.DataFrame)
        assert len(result['members_clean']) > 0


class TestLoadAndTransformSessionData:
    """Tests for load_and_transform_session_data function."""
    
    def test_load_and_transform_success(self, mock_session_data):
        """Test successful load and transform."""
        result = load_and_transform_session_data(mock_session_data)
        
        assert isinstance(result, dict)
        assert 'member_competence' in result
        assert 'competence_master' in result
        assert 'members_clean' in result
    
    def test_load_and_transform_nonexistent_session(self):
        """Test with non-existent session."""
        with pytest.raises(HTTPException) as exc_info:
            load_and_transform_session_data("nonexistent")
        
        assert exc_info.value.status_code == 404


class TestValidateWeights:
    """Tests for validate_weights function."""
    
    def test_valid_weights(self):
        """Test validation with valid weights."""
        weights = {'readiness': 0.5, 'bayesian': 0.3, 'utility': 0.2}
        assert validate_weights(weights) is True
    
    def test_weights_sum_to_one(self):
        """Test weights that sum exactly to 1.0."""
        weights = {'readiness': 0.6, 'bayesian': 0.3, 'utility': 0.1}
        assert validate_weights(weights) is True
    
    def test_weights_sum_not_one(self):
        """Test weights that don't sum to 1.0."""
        weights = {'readiness': 0.5, 'bayesian': 0.3, 'utility': 0.3}
        
        with pytest.raises(HTTPException) as exc_info:
            validate_weights(weights)
        
        assert exc_info.value.status_code == 400
        assert "must sum to 1.0" in exc_info.value.detail
    
    def test_negative_weight(self):
        """Test with negative weight value."""
        weights = {'readiness': 0.6, 'bayesian': 0.5, 'utility': -0.1}
        
        with pytest.raises(HTTPException) as exc_info:
            validate_weights(weights)
        
        assert exc_info.value.status_code == 400
        assert "must be non-negative" in exc_info.value.detail
    
    def test_default_weights_are_valid(self):
        """Test that DEFAULT_WEIGHTS are valid."""
        assert validate_weights(DEFAULT_WEIGHTS) is True
