"""
Data Validation Utilities

This module provides utility functions for validating data integrity
and checking for required columns, data types, and business rules.
"""

import pandas as pd
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for data validation errors."""

    pass


class DataValidator:
    """
    Data validation utility class.

    Provides static methods for validating DataFrames, checking required
    columns, and ensuring data quality.
    """

    @staticmethod
    def validate_required_columns(
        df: pd.DataFrame, required_columns: List[str], df_name: str = "DataFrame"
    ) -> None:
        """
        Validate that all required columns exist in the DataFrame.

        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            df_name: Name of the DataFrame for error messages

        Raises:
            ValidationError: If any required columns are missing

        Examples:
            >>> df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
            >>> DataValidator.validate_required_columns(
            ...     df, ['a', 'b', 'c'], 'test_df'
            ... )
            ValidationError: test_df is missing required columns: ['c']
        """
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            error_msg = (
                f"{df_name} is missing required columns: {missing_columns}\n"
                f"Available columns: {list(df.columns)}"
            )
            logger.error(error_msg)
            raise ValidationError(error_msg)

        logger.debug(f"{df_name}: All required columns present")

    @staticmethod
    def validate_non_empty(df: pd.DataFrame, df_name: str = "DataFrame") -> None:
        """
        Validate that DataFrame is not empty.

        Args:
            df: DataFrame to validate
            df_name: Name of the DataFrame for error messages

        Raises:
            ValidationError: If DataFrame is empty

        Examples:
            >>> df = pd.DataFrame()
            >>> DataValidator.validate_non_empty(df, 'test_df')
            ValidationError: test_df is empty
        """
        if df.empty:
            error_msg = f"{df_name} is empty (0 rows)"
            logger.error(error_msg)
            raise ValidationError(error_msg)

        logger.debug(f"{df_name}: Contains {len(df)} rows")

    @staticmethod
    def validate_column_data_type(
        df: pd.DataFrame, column: str, expected_type: type, df_name: str = "DataFrame"
    ) -> None:
        """
        Validate that a column has the expected data type.

        Args:
            df: DataFrame to validate
            column: Column name to check
            expected_type: Expected Python type (str, int, float, etc.)
            df_name: Name of the DataFrame for error messages

        Raises:
            ValidationError: If column type doesn't match

        Examples:
            >>> df = pd.DataFrame({'code': ['A001', 'B002']})
            >>> DataValidator.validate_column_data_type(
            ...     df, 'code', str, 'test_df'
            ... )
        """
        if column not in df.columns:
            error_msg = f"{df_name} doesn't have column '{column}'"
            logger.error(error_msg)
            raise ValidationError(error_msg)

        # Get actual dtype
        actual_dtype = df[column].dtype

        # Check compatibility
        if expected_type == str and not pd.api.types.is_string_dtype(actual_dtype):
            if not pd.api.types.is_object_dtype(actual_dtype):
                error_msg = f"{df_name}.{column}: Expected string type, " f"got {actual_dtype}"
                logger.error(error_msg)
                raise ValidationError(error_msg)

        logger.debug(f"{df_name}.{column}: Type {actual_dtype} is valid")

    @staticmethod
    def validate_no_duplicates(
        df: pd.DataFrame, columns: List[str], df_name: str = "DataFrame"
    ) -> None:
        """
        Validate that there are no duplicate rows based on specified columns.

        Args:
            df: DataFrame to validate
            columns: List of columns to check for duplicates
            df_name: Name of the DataFrame for error messages

        Raises:
            ValidationError: If duplicates are found

        Examples:
            >>> df = pd.DataFrame({'a': [1, 1, 2], 'b': [3, 3, 4]})
            >>> DataValidator.validate_no_duplicates(df, ['a', 'b'], 'test_df')
            ValidationError: test_df has 1 duplicate row(s) based on ['a', 'b']
        """
        duplicates = df[df.duplicated(subset=columns, keep=False)]

        if not duplicates.empty:
            error_msg = f"{df_name} has {len(duplicates)} duplicate row(s) " f"based on {columns}"
            logger.warning(error_msg)
            # Note: This is a warning, not an error, as duplicates might be handled
            # by the caller

    @staticmethod
    def validate_foreign_key(
        df: pd.DataFrame,
        column: str,
        reference_values: set,
        df_name: str = "DataFrame",
        reference_name: str = "reference",
    ) -> Dict[str, Any]:
        """
        Validate that all values in a column exist in a reference set.

        Args:
            df: DataFrame to validate
            column: Column name to check
            reference_values: Set of valid reference values
            df_name: Name of the DataFrame for error messages
            reference_name: Name of the reference data for error messages

        Returns:
            Dictionary with validation results:
            - valid_count: Number of valid values
            - invalid_count: Number of invalid values
            - invalid_values: Set of invalid values (sample if too many)

        Examples:
            >>> df = pd.DataFrame({'code': ['A', 'B', 'C']})
            >>> result = DataValidator.validate_foreign_key(
            ...     df, 'code', {'A', 'B'}, 'test_df', 'valid_codes'
            ... )
            >>> result['invalid_count']
            1
        """
        df_values = set(df[column].dropna().unique())
        invalid_values = df_values - reference_values
        valid_values = df_values & reference_values

        result = {
            "valid_count": len(valid_values),
            "invalid_count": len(invalid_values),
            "invalid_values": list(invalid_values)[:10],  # Sample first 10
            "total_reference_count": len(reference_values),
        }

        if invalid_values:
            logger.warning(
                f"{df_name}.{column}: {len(invalid_values)} value(s) not found "
                f"in {reference_name}. Sample: {list(invalid_values)[:5]}"
            )
        else:
            logger.debug(f"{df_name}.{column}: All {len(valid_values)} value(s) are valid")

        return result


# Convenient shortcut
validate_required_columns = DataValidator.validate_required_columns
validate_non_empty = DataValidator.validate_non_empty
