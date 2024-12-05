# data_processing.py
import pandas as pd
import unicodedata

def load_data(file, file_type='csv', **kwargs):
    """
    Loads data from a CSV or Excel file into a pandas DataFrame.
    
    This function can handle both file paths and file-like objects (e.g., those returned by `st.file_uploader` in Streamlit).
    It supports CSV and Excel files, and additional keyword arguments are passed to the underlying pandas read functions.
    
    Parameters:
        file (str or file-like object): The file path or file-like object to read.
        file_type (str, optional): The type of file to read. Options are 'csv' or 'xlsx'. Default is 'csv'.
        **kwargs: Additional keyword arguments to pass to `pd.read_csv` or `pd.read_excel`.
    
    Returns:
        pandas.DataFrame: The loaded data as a DataFrame.
    
    Raises:
        ValueError: If an unsupported file type is specified.
        FileNotFoundError: If the file path does not exist (when `file` is a string).
        pd.errors.EmptyDataError: If the file is empty.
        pd.errors.ParserError: If there is a parsing error in the file.
    
    Example:
        # Load a CSV file with a custom delimiter
        data = load_data('data.csv', file_type='csv', delimiter=';', encoding='utf-8')
        
        # Load an Excel file from a file-like object
        data = load_data(uploaded_file, file_type='xlsx')
    """
    if file_type == 'csv':
        return pd.read_csv(file, **kwargs)
    elif file_type == 'xlsx':
        return pd.read_excel(file, **kwargs)
    else:
        raise ValueError("Unsupported file type. Please use 'csv' or 'xlsx'.")

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