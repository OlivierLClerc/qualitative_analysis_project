"""
logging.py

This module provides a function to calculate and log various metrics dynamically
from a DataFrame, including the cumulative cost. All columns for the best model
(row with the highest accuracy) are logged dynamically.

Dependencies:
    - pandas as pd
    - datetime for timestamps

Functions:
    - calculate_and_log(df: pd.DataFrame, filename: str = "history.txt") -> None:
        Dynamically logs relevant metrics from a DataFrame into a text file.
"""

import pandas as pd
import datetime


def calculate_and_log(df: pd.DataFrame, filename: str = "history.txt") -> None:
    """
    Dynamically log all columns for the row with the best `accuracy_val`
    from the DataFrame, along with cumulative cost, into a text file.

    Parameters:
    ----------
    df : pd.DataFrame
        The DataFrame containing the data. Must include:
            - 'total_cost' (float): Cost associated with each row.
            - 'accuracy_val' (float): Accuracy value for each prompt.
    filename : str, optional
        The name of the text file to write the results to (default is "history.txt").

    Returns:
    -------
    None

    Raises:
    ------
    ValueError:
        If required columns ('total_cost', 'accuracy_val') are missing from the DataFrame.
    """
    required_columns = {"cost", "accuracy_val"}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"DataFrame is missing required columns: {', '.join(missing)}")

    # Calculate total cost and cumulative cost
    total_cost_sum = df["cost"].sum()
    cumulative_cost = 0.0

    # Read the cumulative cost from the file if it exists
    try:
        with open(filename, "r") as file:
            lines = file.readlines()
            if lines:
                last_line = lines[-1]
                # Extract the cumulative cost by parsing the last line
                if "Cumulative Cost all runs:" in last_line:
                    cumulative_cost = float(
                        last_line.split("Cumulative Cost all runs:")[-1].strip()
                    )
    except FileNotFoundError:
        pass  # File does not exist; start with a cumulative cost of 0

    # Update cumulative cost
    cumulative_cost += total_cost_sum

    # Find the row with the best accuracy value
    # Check if all values in accuracy_val are NaN
    if df["accuracy_val"].isna().all():
        # If all values are NaN, use the best accuracy_train as fallback
        if "accuracy_train" in df.columns and not df["accuracy_train"].isna().all():
            best_row = df.loc[df["accuracy_train"].idxmax()]
        else:
            # If no accuracy_train or all are NaN, use the first row
            best_row = df.iloc[0]
    else:
        # Otherwise, find the row with the maximum accuracy_val
        best_row = df.loc[df["accuracy_val"].idxmax()]

    # Get the current timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Prepare the log entry
    column_values = ", ".join([f"{col}: {best_row[col]}" for col in df.columns])
    log_entry = (
        f"{timestamp} - {column_values}, "
        f"Total Cost run: {total_cost_sum:.6f}, "
        f"Cumulative Cost all runs: {cumulative_cost:.6f}\n"
    )

    # Write the log entry to the file
    with open(filename, "a") as file:
        file.write(log_entry)
