# utils.py
import csv

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
            row = {'Verbatim': verbatim, **code}
        else:
            row = {'Verbatim': verbatim, 'Code': code}
        rows.append(row)

    # Determine fieldnames if not provided
    if not fieldnames:
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = ['Verbatim'] + fieldnames

    with open(save_path, 'w', newline='', encoding='utf-8') as f:
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
    with open(load_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            if 'Verbatim' in fieldnames:
                verbatims.append(row['Verbatim'])
                code = {k: row[k] for k in fieldnames if k != 'Verbatim'}
            else:
                code = row
            coding.append(code)
    print(f"Results loaded from: {load_path}")

    if verbatims:
        return verbatims, coding
    else:
        return coding
