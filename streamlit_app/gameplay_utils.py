"""
Module for handling gameplay-based configuration functionality.
Provides utilities for detecting, validating, and processing gameplay configurations.
"""

import pandas as pd
from typing import Dict, List, Tuple, Any


def is_gameplay_config(config: dict) -> bool:
    """
    Check if a configuration is gameplay-based by detecting the 'gameplays' key.

    Args:
        config: Configuration dictionary

    Returns:
        True if config contains 'gameplays' key, False otherwise
    """
    return "gameplays" in config and isinstance(config["gameplays"], dict)


def get_available_gameplays(config: dict) -> List[str]:
    """
    Get list of available gameplay types from a gameplay configuration.

    Args:
        config: Gameplay configuration dictionary

    Returns:
        List of gameplay names, or empty list if not a gameplay config
    """
    if not is_gameplay_config(config):
        return []
    return list(config["gameplays"].keys())


def validate_gameplay_in_data(df: pd.DataFrame, gameplay_name: str) -> Tuple[bool, str]:
    """
    Validate that the specified gameplay exists in the dataframe.

    Args:
        df: DataFrame to validate
        gameplay_name: Name of the gameplay to check for

    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if gameplay exists, False otherwise
        - error_message: Empty string if valid, error description if invalid
    """
    if "gameplay_athena" not in df.columns:
        return (
            False,
            "Column 'gameplay_athena' not found in dataset. This column is required for gameplay mode.",
        )

    available = df["gameplay_athena"].dropna().unique().tolist()
    if gameplay_name not in available:
        return (
            False,
            f"Gameplay '{gameplay_name}' not found in dataset. Available gameplays: {', '.join(map(str, available))}",
        )

    return True, ""


def get_columns_for_gameplay(config: dict, gameplay_name: str) -> Dict[str, str]:
    """
    Get merged column descriptions for a specific gameplay.
    Combines common columns with gameplay-specific columns.

    Args:
        config: Gameplay configuration dictionary
        gameplay_name: Name of the gameplay

    Returns:
        Dictionary mapping column names to their descriptions
        Empty dict if not a gameplay config
    """
    if not is_gameplay_config(config):
        return {}

    # Common columns (all gameplays)
    common_columns = config.get("common_columns_to_all_gp", {})

    # Gameplay-specific columns
    gameplay = config["gameplays"].get(gameplay_name, {})
    gameplay_columns = gameplay.get("columns", {})

    # Merge (gameplay-specific can override common if same key)
    return {**common_columns, **gameplay_columns}


def validate_columns_exist(
    df: pd.DataFrame, required_columns: List[str]
) -> Tuple[bool, List[str]]:
    """
    Check if all required columns exist in the dataframe.

    Args:
        df: DataFrame to validate
        required_columns: List of column names that should exist

    Returns:
        Tuple of (all_exist, missing_columns)
        - all_exist: True if all columns exist, False otherwise
        - missing_columns: List of column names that are missing
    """
    missing = [col for col in required_columns if col not in df.columns]
    return len(missing) == 0, missing


def get_gameplay_config(config: dict, gameplay_name: str) -> Dict[str, Any]:
    """
    Get the complete merged configuration for a specific gameplay.
    Combines common configuration with gameplay-specific overrides.

    Args:
        config: Gameplay configuration dictionary
        gameplay_name: Name of the gameplay

    Returns:
        Dictionary containing merged configuration with keys:
        - column_descriptions: Merged column descriptions
        - selected_columns: List of all columns (common + gameplay-specific)
        - codebook: Gameplay-specific codebook or common codebook
        - examples: Gameplay-specific examples or common examples
        - selected_fields: Fields to extract from LLM
        - label_column: Label column name
        - label_type: Label data type
        - annotation_columns: List of annotation column names
        - text_columns: List of text columns to normalize
        Empty dict if not a gameplay config
    """
    if not is_gameplay_config(config):
        return {}

    common = config.get("common_config", {})
    gameplay = config["gameplays"].get(gameplay_name, {})

    # Merge column descriptions
    all_column_descriptions = get_columns_for_gameplay(config, gameplay_name)

    # Get text columns (common + gameplay-specific)
    common_text_cols = common.get("text_columns", [])
    gameplay_text_cols = gameplay.get("text_columns", [])
    all_text_columns = list(set(common_text_cols + gameplay_text_cols))

    # Use gameplay-specific codebook if present, otherwise use common
    codebook = gameplay.get("codebook", common.get("codebook", ""))

    # Use gameplay-specific examples if present, otherwise use common
    examples = gameplay.get("examples", common.get("examples", ""))

    return {
        "column_descriptions": all_column_descriptions,
        "selected_columns": list(all_column_descriptions.keys()),
        "codebook": codebook,
        "examples": examples,
        "selected_fields": common.get("selected_fields", []),
        "label_column": common.get("label_column"),
        "label_type": common.get("label_type"),
        "annotation_columns": common.get("annotation_columns", []),
        "text_columns": all_text_columns,
    }


def filter_df_by_gameplay(df: pd.DataFrame, gameplay_name: str) -> pd.DataFrame:
    """
    Filter DataFrame to only include rows matching the specified gameplay.

    Args:
        df: DataFrame to filter
        gameplay_name: Name of the gameplay to filter for

    Returns:
        Filtered DataFrame containing only rows where gameplay_athena == gameplay_name

    Raises:
        ValueError: If 'gameplay_athena' column not found in dataset
    """
    if "gameplay_athena" not in df.columns:
        raise ValueError(
            "Column 'gameplay_athena' not found in dataset. This column is required for gameplay mode."
        )

    return df[df["gameplay_athena"] == gameplay_name].copy()


def get_gameplay_description(config: dict, gameplay_name: str) -> str:
    """
    Get the description for a specific gameplay.

    Args:
        config: Gameplay configuration dictionary
        gameplay_name: Name of the gameplay

    Returns:
        Description string, or empty string if not found
    """
    if not is_gameplay_config(config):
        return ""

    gameplay = config["gameplays"].get(gameplay_name, {})
    return gameplay.get("description", "")
