"""
parsing.py

This module provides functions for parsing and extracting structured information from language model responses
and classification results. It is designed to handle text outputs, JSON format, and extract 
key data for further analysis.

Dependencies:
    - json: For parsing JSON-formatted text.
    - re: For regular expressions used in pattern matching.
    - pandas: For handling and processing tabular data.

Functions:
    - parse_llm_response(evaluation_text, selected_fields): 
        Extracts specific fields from a JSON object within a language model response.

    - extract_code_from_response(response_text, prefix=None): 
        Extracts an integer code from a language model response, with optional prefix matching.

    - extract_global_validity(results_df, id_pattern=r"Id:\s*(\w+)", label_column="Label", 
                              verbatim_column="Verbatim", global_validity_column="Global_Validity", 
                              verbatim_output_column="Verbatim"): 
        Aggregates binary classification results to determine the overall validity of grouped entries.

    - parse_key_value_lines(text: str) -> Dict[str, str]:
        Parses 'Key: Value' formatted lines from a text into a dictionary, supporting multi-line values.
"""

import json
import re
import pandas as pd
from typing import Optional, Dict


def parse_llm_response(
    evaluation_text: str, selected_fields: Optional[list] = None
) -> dict:
    """
    Parses a language model's response to extract specified fields from a JSON object.

    This function searches for a JSON-formatted substring within the provided `evaluation_text`.
    If a JSON object is found, it parses the JSON and extracts the values corresponding to
    the `selected_fields`. If no JSON object is found or parsing fails, it returns a dictionary
    with `None` values for each selected field.

    This function is useful for handling responses from applications like Streamlit,
    where the language model is instructed to return responses in JSON format.

    Parameters:
    ----------
    evaluation_text : str
        The full text response from the language model.

    selected_fields : list
        A list of field names (strings) to extract from the JSON object.

    Returns:
    -------
    dict
        A dictionary where each key is a field from `selected_fields` and each value
        is the corresponding value from the JSON object or `None` if not found or parsing failed.

    Raises:
    ------
    ValueError
        If no JSON object is found in the `evaluation_text`.

    Examples:
    --------
    >>> evaluation_text = '''
    ... Here is the evaluation of your entry:
    ... {
    ...     "Evaluation": "Positive",
    ...     "Comments": "Well-written and insightful."
    ... }
    ... Thank you!
    ... '''
    >>> selected_fields = ['Evaluation', 'Comments']
    >>> parse_llm_response(evaluation_text, selected_fields)
    {'Evaluation': 'Positive', 'Comments': 'Well-written and insightful.'}

    >>> incomplete_text = '''
    ... Response without a valid JSON.
    ... '''
    >>> parse_llm_response(incomplete_text, selected_fields)
    Error parsing LLM response: No JSON object found in the LLM response.
    {'Evaluation': None, 'Comments': None}
    """
    try:
        # Step 1: Find JSON block (including markdown variants)
        json_match = re.search(
            r"(?:```(?:json)?\n)?({.*?})(?:\n```)?", evaluation_text, re.DOTALL
        )

        if not json_match:
            raise ValueError("No JSON found in response")

        # Step 2: Clean JSON string
        json_str = json_match.group(1)

        # Remove comments and trailing commas
        json_str = re.sub(
            r"/\*.*?\*/|//.*?$|#.*?$|,\s*}(?=\s*$)",
            "",
            json_str,
            flags=re.DOTALL | re.MULTILINE,
        )

        # Step 3: Parse JSON
        parsed = json.loads(json_str)

        # Step 4: Validate types
        if "Validity" in parsed:
            if isinstance(parsed["Validity"], str):
                parsed["Validity"] = int(parsed["Validity"])  # Try conversion
            if parsed["Validity"] not in (0, 1):
                raise ValueError(f"Invalid Validity: {parsed['Validity']}")

        # If selected_fields is None or empty, return all fields from the parsed JSON
        if not selected_fields:
            return parsed
        else:
            return {field: parsed.get(field) for field in selected_fields}

    except Exception as e:
        print(f"Parsing Error: {str(e)}")
        print(f"Cleaned JSON Attempt: {json_str}")
        if not selected_fields:
            return {}  # Return empty dict if no selected_fields
        else:
            return {field: None for field in selected_fields}


def extract_code_from_response(
    response_text: str, prefix: Optional[str] = None
) -> Optional[int]:
    """
    Extracts an integer code from a language model response, optionally requiring a prefix.

    If a `prefix` is specified (e.g., "Validity:"), the function searches for a line matching:
        "<prefix> <digits>"
    (case-insensitively) and returns the integer.
    Example: `"Validity: 1"` → `1`

    If no `prefix` is provided, the function captures the first integer found in the text.
    Example: `"I think the code is 2"` → `2`

    Parameters:
    ----------
    response_text : str
        The text response from the language model.

    prefix : str, optional
        A string that must precede the integer.
        Example: `"Validity:"` or `"Score:"`.

    Returns:
    -------
    int or None
        The extracted integer code if found, otherwise `None`.

    Examples:
    --------
    Extracting a code with a specified prefix:

    >>> response_text = "Validity: 1"
    >>> extract_code_from_response(response_text, prefix="Validity")
    1

    Handling different formats with colons and hyphens:

    >>> response_text = "Score - 3"
    >>> extract_code_from_response(response_text, prefix="Score")
    3

    Extracting the first integer without a prefix:

    >>> response_text = "The final decision is 2."
    >>> extract_code_from_response(response_text)
    2

    Handling negative numbers:

    >>> response_text = "Validity: -1"
    >>> extract_code_from_response(response_text, prefix="Validity")
    -1

    When no integer is present:

    >>> response_text = "No valid code here."
    >>> extract_code_from_response(response_text)

    Case-insensitive matching:

    >>> response_text = "validity: 4"
    >>> extract_code_from_response(response_text, prefix="Validity")
    4
    """
    if prefix:
        # Case-insensitive search for the prefix followed by an integer
        pattern = rf"(?i)\b{re.escape(prefix)}\s*[:\-]?\s*([+-]?\d+)\s*$"
        match = re.search(pattern, response_text, re.MULTILINE)
        if match:
            return int(match.group(1))
        else:
            return None
    else:
        # Search for the first standalone integer if no prefix is given
        number_search_result = re.search(r"[+-]?\d+", response_text)
        if number_search_result:
            return int(number_search_result.group())
        else:
            return None


def extract_global_validity(
    results_df: pd.DataFrame,
    id_column: str = "Id",
    label_column: str = "Label",
    global_validity_column: str = "Global_Validity",
) -> pd.DataFrame:
    """
    Extracts the overall validity of each cycle based on sequential binary classification results.

    This function groups classification results by a unique identifier (`Id`) and evaluates whether all
    associated binary labels are positive (`1`). It prints a warning if NaN values are detected.

    Parameters:
    ----------
    results_df : pd.DataFrame
        DataFrame containing subject IDs and classification labels.

    id_column : str, optional
        Column name containing the subject ID. Default is 'Id'.

    label_column : str, optional
        Column name containing binary classification labels. Default is 'Label'.

    global_validity_column : str, optional
        Column name for the global validity output. Default is 'Global_Validity'.

    Returns:
    -------
    pd.DataFrame
        Original DataFrame with an additional column for the global validity,
        indicating if all labels for a subject are `1` (valid) or `0` (invalid).

    Examples:
    ---------
    **Example 1: All labels are 1 → Global Validity = 1**
    >>> data1 = {
    ...     'Id': ['BC23', 'BC23', 'BC23'],
    ...     'Theme': ['Identify Step', 'Guess Step', 'Assess Step'],
    ...     'Label': [1, 1, 1]
    ... }
    >>> df1 = pd.DataFrame(data1)
    >>> extract_global_validity(df1)
         Id          Theme  Label  Global_Validity
    0  BC23  Identify Step      1                1
    1  BC23     Guess Step      1                1
    2  BC23    Assess Step      1                1

    **Example 2: Contains a NaN → Treated as Invalid (0) with a Warning**
    >>> data2 = {
    ...     'Id': ['BC24', 'BC24', 'BC24'],
    ...     'Theme': ['Identify Step', 'Guess Step', 'Assess Step'],
    ...     'Label': [1, None, 1]
    ... }
    >>> df2 = pd.DataFrame(data2)
    >>> extract_global_validity(df2)
    Warning: NaN value detected in the 'Label' column for subject 'BC24'. Treated as invalid (0).
         Id          Theme  Label  Global_Validity
    0  BC24  Identify Step      1                0
    1  BC24     Guess Step      0                0
    2  BC24    Assess Step      1                0

    **Example 3: Multiple Subjects with Mixed Validity and NaNs**
    >>> data3 = {
    ...     'Id': ['BC25', 'BC25', 'BC26', 'BC26'],
    ...     'Theme': ['Identify Step', 'Assess Step', 'Guess Step', 'Assess Step'],
    ...     'Label': [1, 1, None, 1]
    ... }
    >>> df3 = pd.DataFrame(data3)
    >>> extract_global_validity(df3)
    Warning: NaN value detected in the 'Label' column for subject 'BC26'. Treated as invalid (0).
         Id          Theme  Label  Global_Validity
    0  BC25  Identify Step      1                1
    1  BC25    Assess Step      1                1
    2  BC26     Guess Step      0                0
    3  BC26    Assess Step      1                0
    """
    # Identify subjects (`Id`) that have NaN values in the label column
    nan_subjects = results_df[results_df[label_column].isna()][id_column].unique()

    # Print a warning for each subject with NaN
    for subject_id in nan_subjects:
        print(
            f"Warning: NaN value detected in the '{label_column}' column for subject '{subject_id}'. Treated as invalid (0)."
        )

    # Replace NaNs with 0 and ensure labels are integers
    results_df[label_column] = results_df[label_column].fillna(0).astype(int)

    # Compute global validity as integer (1 or 0)
    validity = (
        results_df.groupby(id_column)[label_column]
        .apply(lambda x: 1 if (x == 1).all() else 0)
        .reset_index()
    )
    validity = validity.rename(columns={label_column: global_validity_column})

    # Merge global validity back into the original DataFrame
    results_with_validity = results_df.merge(validity, on=id_column, how="left")

    return results_with_validity


def parse_key_value_lines(text: str) -> Dict[str, str]:
    """
    Parses 'Key: Value' formatted lines from a text into a dictionary.

    Allows multi-line values if they follow the same key.

    Parameters:
    ----------
    text : str
        The text containing 'Key: Value' lines to parse.

    Returns:
    -------
    Dict[str, str]
        A dictionary with keys and their corresponding multi-line or single-line values.

    Examples:
    --------
    Parsing a simple multi-line verbatim:

    >>> text = '''Id: 197
    ... Texte: Les enormes yeux des insectes font quasiment le tour de leurs tetes.
    ... Ils voient donc en meme temps vers lavant, vers larriere, a droite, a gauche, vers le haut et vers le bas.
    ... Encore mieux quun casque de vision en 3D!
    ... Question: Comment les point sont ensuite assembles'''
    >>> parse_key_value_lines(text)
    {'Id': '197', 'Texte': 'Les enormes yeux des insectes font quasiment le tour de leurs tetes.\\nIls voient donc en meme temps vers lavant, vers larriere, a droite, a gauche, vers le haut et vers le bas.\\nEncore mieux quun casque de vision en 3D!', 'Question': 'Comment les point sont ensuite assembles'}

    Handling sections without a colon:

    >>> text = '''Id: 198
    ... Texte: Sample text without question
    ... Another line without key'''
    >>> parse_key_value_lines(text)
    {'Id': '198', 'Texte': 'Sample text without question\\nAnother line without key'}

    Handling empty lines and multiple keys:

    >>> text = '''Id: 199
    ...
    ... Texte: Sample text for multiple keys
    ... Question: What is the purpose?
    ...
    ... Score: 5'''
    >>> parse_key_value_lines(text)
    {'Id': '199', 'Texte': 'Sample text for multiple keys', 'Question': 'What is the purpose?', 'Score': '5'}

    When no key-value pairs are present:

    >>> text = '''Just some random text without keys.'''
    >>> parse_key_value_lines(text)
    {'Section_1': 'Just some random text without keys.'}
    """
    lines = text.splitlines()
    result: Dict[str, str] = {}

    # Regular expression to match lines starting with 'Key: Value'
    key_pattern = re.compile(r"^(\S+)\s*:\s*(.*)$")
    # Explanation:
    #   ^(\S+)\s*:\s* matches "Id:  " or "Texte: "
    #   (.*)$ captures the rest of the line as the initial value

    current_key = None
    current_value_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            # Empty line
            if current_key:
                current_value_lines.append("")  # Preserve empty lines if desired
            continue

        match = key_pattern.match(line)
        if match:
            # Found a new 'Key: value' line
            # 1) Store the previous key-value if it exists
            if current_key is not None:
                result[current_key] = "\n".join(current_value_lines).strip()

            # 2) Start a new key-value
            current_key = match.group(1)
            initial_value = match.group(2)
            current_value_lines = [initial_value]
        else:
            # Continuation of the current key-value (multi-line value)
            if current_key is not None:
                current_value_lines.append(line)
            else:
                # No current key -> assign to a generic section
                generic_key = f"Section_{len(result) + 1}"
                current_key = generic_key
                current_value_lines = [line]

    # End of loop: store the last key-value pair if any
    if current_key is not None:
        result[current_key] = "\n".join(current_value_lines).strip()

    return result
