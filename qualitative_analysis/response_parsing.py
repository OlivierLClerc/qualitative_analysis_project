# response_parsing.py
import json
import re

def parse_llm_response(evaluation_text, selected_fields):
    """
    Parses a language model's response to extract specified fields from a JSON object.

    This function searches for a JSON-formatted substring within the provided `evaluation_text`.
    If a JSON object is found, it parses the JSON and extracts the values corresponding to
    the `selected_fields`. If no JSON object is found or parsing fails, it returns a dictionary
    with `None` values for each selected field.

    This function is suitable for handling responses from applications like Streamlit
    where the language model is instructed to return responses in JSON format.

    Parameters:
        evaluation_text (str): The full text response from the language model.
        selected_fields (list): A list of field names (strings) to extract from the JSON object.

    Returns:
        dict: A dictionary where each key is a field from `selected_fields` and each value
              is the corresponding value from the JSON object or `None` if not found or parsing failed.

    Raises:
        ValueError: If no JSON object is found in the `evaluation_text`.

    Example:
        evaluation_text = '''
        Here is the evaluation of your entry:
        {
            "Evaluation": "Positive",
            "Comments": "Well-written and insightful."
        }
        Thank you!
        '''
        selected_fields = ['Evaluation', 'Comments']
        result = parse_llm_response(evaluation_text, selected_fields)
        # result => {'Evaluation': 'Positive', 'Comments': 'Well-written and insightful.'}
    """
    try:
        # Search for a JSON object within the evaluation_text
        json_text_match = re.search(r'\{.*\}', evaluation_text, re.DOTALL)
        if json_text_match:
            json_text = json_text_match.group(0)
            # Parse the JSON text
            evaluation_json = json.loads(json_text)
            # Extract the specified fields
            return {field: evaluation_json.get(field, None) for field in selected_fields}
        else:
            raise ValueError("No JSON object found in the LLM response.")
    except json.JSONDecodeError as e:
        # Handle JSON decoding errors
        print(f"JSON decoding error: {e}")
        return {field: None for field in selected_fields}
    except Exception as e:
        # Handle other exceptions
        print(f"Error parsing LLM response: {e}")
        return {field: None for field in selected_fields}


def extract_code_from_response(response_text):
    """
    Extracts the first integer code from a language model's text response.

    This function searches for the first occurrence of one or more digits within the
    provided `response_text`. If a number is found, it converts and returns it as an integer.
    If no number is found, it returns `None`.

    Parameters:
        response_text (str): The text response from the language model.

    Returns:
        int or None: The extracted integer code if found; otherwise, `None`.

    Example:
        response_text = "Based on your input, the classification code is 2."
        code = extract_code_from_response(response_text)
        # code => 2
    """
    # Search for the first occurrence of one or more digits
    number_search_result = re.search(r'\d+', response_text)
    if number_search_result:
        return int(number_search_result.group())
    else:
        return None