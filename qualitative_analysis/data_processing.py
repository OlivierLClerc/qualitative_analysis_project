# data_processing.py
import pandas as pd
import unicodedata
import chardet

# data_processing.py

import pandas as pd
import chardet

def load_data(file, file_type='csv', delimiter=',', **kwargs):
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
    if file_type == 'csv':
        # Try reading with default 'utf-8' encoding first
        try:
            return pd.read_csv(file, delimiter=delimiter, **kwargs)
        except UnicodeDecodeError:
            pass  # Proceed to encoding detection
        except Exception as e:
            raise e  # Re-raise any other exceptions

        # Reset file pointer to the beginning if possible
        if hasattr(file, 'seek'):
            file.seek(0)
        # Detect encoding
        encoding = detect_file_encoding(file)
        # Reset file pointer again if possible
        if hasattr(file, 'seek'):
            file.seek(0)
        # Try reading again with detected encoding
        try:
            return pd.read_csv(file, encoding=encoding, delimiter=delimiter, **kwargs)
        except UnicodeDecodeError:
            pass  # Proceed to try 'ISO-8859-1'
        except Exception as e:
            raise e  # Re-raise any other exceptions

        # Reset file pointer again if possible
        if hasattr(file, 'seek'):
            file.seek(0)
        # Try with 'ISO-8859-1' encoding
        try:
            return pd.read_csv(file, encoding='ISO-8859-1', delimiter=delimiter, **kwargs)
        except Exception as e:
            raise UnicodeDecodeError("Failed to read the file with utf-8, detected encoding, or ISO-8859-1.")
    elif file_type == 'xlsx':
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
    if hasattr(file, 'read'):
        # If file is a file-like object
        rawdata = file.read(100000)
    else:
        # If file is a file path
        with open(file, 'rb') as f:
            rawdata = f.read(100000)
    # Use chardet to detect encoding
    result = chardet.detect(rawdata)
    encoding = result['encoding']
    if encoding is None:
        encoding = 'utf-8'  # Default to 'utf-8' if detection fails
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
        series.astype(str)
        .str.strip()
        .apply(lambda x: unicodedata.normalize('NFKD', x))
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
        lambda x: x.replace('\n', ' ').replace('\r', ' ') if isinstance(x, str) else x
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