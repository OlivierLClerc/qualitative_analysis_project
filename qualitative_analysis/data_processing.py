# data_processing.py
import pandas as pd
import unicodedata
import chardet
import csv


def load_data(file, file_type="csv", delimiter=",", **kwargs):
    """
    Loads data from a CSV or Excel file into a pandas DataFrame.

    This function attempts to read the file using 'utf-8' encoding and the specified delimiter.
    If it encounters a UnicodeDecodeError, it tries to detect the file's encoding using chardet.
    If detection fails or the encoding doesn't work, it falls back to 'ISO-8859-1'.

    Parameters:
        file (str or file-like object): The file path or file-like object to read.
        file_type (str, optional): The type of file to read. Options are 'csv' or 'xlsx'. Default is 'csv'.
        delimiter (str, optional): The delimiter used in the CSV file. Default is ','.
        **kwargs: Additional keyword arguments to pass to `pd.read_csv` or `pd.read_excel`.

    Returns:
        pandas.DataFrame: The loaded data as a DataFrame.

    Raises:
        ValueError: If an unsupported file type is specified.
        UnicodeDecodeError: If the file cannot be decoded with any of the attempted encodings.
        pd.errors.EmptyDataError: If the file is empty.
        pd.errors.ParserError: If there is a parsing error in the file.
    """
    if file_type == "csv":
        # Try reading with default 'utf-8' encoding first
        try:
            return pd.read_csv(file, delimiter=delimiter, **kwargs)
        except UnicodeDecodeError:
            pass  # Proceed to encoding detection
        except Exception as e:
            raise e  # Re-raise any other exceptions

        # Reset file pointer to the beginning if possible
        if hasattr(file, "seek"):
            file.seek(0)
        # Detect encoding
        encoding = detect_file_encoding(file)
        # Reset file pointer again if possible
        if hasattr(file, "seek"):
            file.seek(0)
        # Try reading again with detected encoding
        try:
            return pd.read_csv(file, encoding=encoding, delimiter=delimiter, **kwargs)
        except UnicodeDecodeError:
            pass  # Proceed to try 'ISO-8859-1'
        except Exception as e:
            raise e  # Re-raise any other exceptions

        # Reset file pointer again if possible
        if hasattr(file, "seek"):
            file.seek(0)
        # Try with 'ISO-8859-1' encoding
        try:
            return pd.read_csv(
                file, encoding="ISO-8859-1", delimiter=delimiter, **kwargs
            )
        except Exception:
            raise UnicodeDecodeError(
                "Failed to read the file with utf-8, detected encoding, or ISO-8859-1."
            )
    elif file_type == "xlsx":
        return pd.read_excel(file, **kwargs)
    else:
        raise ValueError("Unsupported file type. Please use 'csv' or 'xlsx'.")


def detect_file_encoding(file):
    """
    Detects the encoding of a file using the chardet library.

    Parameters:
        file (str or file-like object): The file path or file-like object to detect encoding.

    Returns:
        str: The detected encoding of the file.
    """
    # Read a portion of the file for encoding detection
    if hasattr(file, "read"):
        # If file is a file-like object
        rawdata = file.read(100000)
    else:
        # If file is a file path
        with open(file, "rb") as f:
            rawdata = f.read(100000)
    # Use chardet to detect encoding
    result = chardet.detect(rawdata)
    encoding = result["encoding"]
    if encoding is None:
        encoding = "utf-8"  # Default to 'utf-8' if detection fails
    return encoding


def clean_and_normalize(series):
    """
    Cleans and normalizes a pandas Series of text data.

    This function performs the following operations:
        - Converts all entries to strings.
        - Strips leading and trailing whitespace.
        - Normalizes Unicode characters using NFKD normalization.

    Parameters:
        series (pandas.Series): The Series containing text data to clean and normalize.

    Returns:
        pandas.Series: The cleaned and normalized Series.

    Example:
        # Clean and normalize a 'Comments' column
        data['Comments'] = clean_and_normalize(data['Comments'])
    """
    return (
        series.astype(str).str.strip().apply(lambda x: unicodedata.normalize("NFKD", x))
    )


def sanitize_dataframe(df):
    """
    Sanitizes a pandas DataFrame by replacing line breaks in string columns.

    This function replaces newline (`\n`) and carriage return (`\r`) characters with spaces in all string entries of the DataFrame.
    This is useful for preparing data for display or export, ensuring that line breaks do not disrupt formatting.

    Parameters:
        df (pandas.DataFrame): The DataFrame to sanitize.

    Returns:
        pandas.DataFrame: The sanitized DataFrame.

    Example:
        # Sanitize the entire DataFrame before exporting
        data = sanitize_dataframe(data)
    """
    return df.applymap(
        lambda x: x.replace("\n", " ").replace("\r", " ") if isinstance(x, str) else x
    )


def select_and_rename_columns(data, selected_columns, column_renames):
    """
    Selects specified columns from a DataFrame and renames them.

    Parameters:
        data (pandas.DataFrame): The original DataFrame.
        selected_columns (list): A list of column names to select from the DataFrame.
        column_renames (dict): A dictionary mapping original column names to new names.

    Returns:
        pandas.DataFrame: A new DataFrame containing only the selected and renamed columns.

    Raises:
        KeyError: If any of the `selected_columns` are not present in `data`.

    Example:
        # Select and rename columns
        selected_columns = ['col1', 'col2']
        column_renames = {'col1': 'ID', 'col2': 'Text'}
        processed_data = select_and_rename_columns(data, selected_columns, column_renames)
    """
    return data[selected_columns].rename(columns=column_renames)


def save_results_to_csv(coding, save_path, fieldnames=None, verbatims=None):
    """
    Saves coding results and verbatims to a CSV file.

    This function writes the coding results and associated verbatims to a CSV file.
    If `fieldnames` are not provided, they are inferred from the keys of the first row.
    If `verbatims` are provided, they are included in the output under the 'Verbatim' column.

    Parameters:
        coding (list): A list of coding results. Each element can be a dictionary of codes or a single value.
                       If dictionaries, keys represent field names and values represent the corresponding data.
        save_path (str): The file path where the CSV will be saved.
        fieldnames (list, optional): A list of field names (column headers) to include in the CSV.
                                     If not provided, field names are inferred from the coding data.
        verbatims (list, optional): A list of verbatim texts corresponding to each coding result.
                                    If provided, they are included in the 'Verbatim' column.

    Returns:
        None

    Raises:
        ValueError: If the lengths of `coding` and `verbatims` do not match.

    Example:
        coding = [{'Code': 1, 'Comments': 'Positive feedback'}, {'Code': 2, 'Comments': 'Negative feedback'}]
        verbatims = ['Great product!', 'Not satisfied with the service.']
        save_results_to_csv(coding, 'results.csv', verbatims=verbatims)
    """
    if verbatims and len(coding) != len(verbatims):
        raise ValueError("The length of 'coding' and 'verbatims' must be the same.")

    rows = []
    for i, code in enumerate(coding):
        if verbatims:
            verbatim = verbatims[i]
        else:
            verbatim = None
        if isinstance(code, dict):
            row = {"Verbatim": verbatim, **code}
        else:
            row = {"Verbatim": verbatim, "Code": code}
        rows.append(row)

    # Determine fieldnames if not provided
    if not fieldnames:
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = ["Verbatim"] + fieldnames

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results saved to: {save_path}")


def load_results_from_csv(load_path):
    """
    Loads coding results and verbatims from a CSV file.

    This function reads a CSV file containing coding results and verbatims.
    It returns the verbatims and coding results as separate lists.
    If the 'Verbatim' column is not present, only the coding results are returned.

    Parameters:
        load_path (str): The file path from which the CSV will be read.

    Returns:
        tuple or list:
            - If 'Verbatim' column is present:
                Returns a tuple (verbatims, coding), where:
                - verbatims (list): A list of verbatim texts.
                - coding (list): A list of coding results corresponding to each verbatim.
            - If 'Verbatim' column is not present:
                Returns coding (list): A list of coding results.

    Example:
        verbatims, coding = load_results_from_csv('results.csv')
    """
    verbatims = []
    coding = []
    with open(load_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            if "Verbatim" in fieldnames:
                verbatims.append(row["Verbatim"])
                code = {k: row[k] for k in fieldnames if k != "Verbatim"}
            else:
                code = row
            coding.append(code)
    print(f"Results loaded from: {load_path}")

    if verbatims:
        return verbatims, coding
    else:
        return coding


def extract_global_validity(
    results_df,
    id_pattern=r"Id:\s*(\w+)",
    label_column="Label",
    verbatim_column="Verbatim",
    global_validity_column="Global_Validity",
    verbatim_output_column="Verbatim",
):
    """
    Extracts the overall validity of each cycle based on sequential binary classification results.

    Parameters:
    ----------
    results_df : pd.DataFrame
        The DataFrame containing the classification results.
    id_pattern : str, optional
        Regular expression pattern to extract the 'Id' from the `verbatim_column`.
        Default is `r'Id:\s*(\w+)'`.
    label_column : str, optional
        Name of the column containing the binary classification labels. Default is `'Label'`.
    verbatim_column : str, optional
        Name of the column containing the text data. Default is `'Verbatim'`.
    global_validity_column : str, optional
        Name of the output column that stores the overall validity result. Default is `'Global_Validity'`.
    verbatim_output_column : str, optional
        Name of the output column that stores the combined verbatim text. Default is `'Verbatim'`.

    Returns:
    -------
    pd.DataFrame
        A DataFrame with the extracted ID, combined verbatim text, and overall validity for each cycle.
    """

    # Extract 'Id' from the specified 'verbatim_column'
    results_df["Id"] = results_df[verbatim_column].str.extract(id_pattern)

    # Ensure the 'label_column' is of integer type
    results_df[label_column] = results_df[label_column].astype(int)

    # Group by 'Id' and compute overall validity dynamically
    global_validity = (
        results_df.groupby("Id")
        .apply(
            lambda x: pd.Series(
                {
                    verbatim_output_column: " ".join(
                        x[verbatim_column].unique()
                    ),  # Dynamic verbatim output
                    global_validity_column: (
                        1 if (x[label_column] == 1).all() else 0
                    ),  # Dynamic global validity output
                }
            )
        )
        .reset_index()
    )

    return global_validity
