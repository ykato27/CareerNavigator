"""
Data Normalization Utilities

This module provides utility functions for normalizing and cleaning
various types of data used throughout the application.
"""

import unicodedata
import pandas as pd
from typing import Any, Optional


class DataNormalizer:
    """
    Data normalization utility class.

    Provides static methods for normalizing member codes, competence codes,
    and other data types to ensure consistency across the application.
    """

    @staticmethod
    def normalize_member_code(code: Any) -> str:
        """
        Normalize member code to standard format.

        Performs the following normalizations:
        1. Converts to string
        2. Strips leading/trailing whitespace
        3. Converts full-width to half-width characters (NFKC)

        Args:
            code: Member code in any format (str, int, float, etc.)

        Returns:
            Normalized member code as string. Returns empty string for NA values.

        Examples:
            >>> DataNormalizer.normalize_member_code("　A001　")
            'A001'
            >>> DataNormalizer.normalize_member_code("００１")
            '001'
            >>> DataNormalizer.normalize_member_code(123)
            '123'
        """
        if pd.isna(code):
            return ""

        # Convert to string
        code_str = str(code)

        # Strip whitespace
        code_str = code_str.strip()

        # Normalize full-width to half-width (NFKC normalization)
        code_str = unicodedata.normalize("NFKC", code_str)

        return code_str

    @staticmethod
    def normalize_competence_code(code: Any) -> str:
        """
        Normalize competence code to standard format.

        Uses the same normalization logic as member codes.

        Args:
            code: Competence code in any format

        Returns:
            Normalized competence code as string

        Examples:
            >>> DataNormalizer.normalize_competence_code("SKILL-001")
            'SKILL-001'
        """
        return DataNormalizer.normalize_member_code(code)

    @staticmethod
    def normalize_text(text: Any) -> str:
        """
        Normalize general text data.

        Args:
            text: Text to normalize

        Returns:
            Normalized text with:
            - Leading/trailing whitespace removed
            - Full-width characters converted to half-width
            - Empty string for NA values

        Examples:
            >>> DataNormalizer.normalize_text("　Hello　World　")
            'Hello　World'
        """
        if pd.isna(text):
            return ""

        text_str = str(text)
        text_str = text_str.strip()
        text_str = unicodedata.normalize("NFKC", text_str)

        return text_str

    @staticmethod
    def normalize_dataframe_column(
        df: pd.DataFrame, column: str, normalizer_func: Optional[callable] = None
    ) -> pd.DataFrame:
        """
        Apply normalization to an entire DataFrame column.

        Args:
            df: DataFrame to modify
            column: Column name to normalize
            normalizer_func: Normalization function to apply.
                           Defaults to normalize_text if not provided.

        Returns:
            DataFrame with normalized column (modifies in place)

        Examples:
            >>> df = pd.DataFrame({'code': ['　A001　', 'B002']})
            >>> DataNormalizer.normalize_dataframe_column(
            ...     df, 'code', DataNormalizer.normalize_member_code
            ... )
        """
        if normalizer_func is None:
            normalizer_func = DataNormalizer.normalize_text

        if column in df.columns:
            df[column] = df[column].apply(normalizer_func)

        return df


# Convenient shortcuts for common operations
normalize_member_code = DataNormalizer.normalize_member_code
normalize_competence_code = DataNormalizer.normalize_competence_code
normalize_text = DataNormalizer.normalize_text
