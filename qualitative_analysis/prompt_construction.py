# prompt_construction.py
import json

def build_data_format_description(column_descriptions):
    """
    Builds a textual description of the data format based on column descriptions.

    This function takes a dictionary of column descriptions and constructs a formatted string
    that describes each column, which can be included in prompts for language models.

    Parameters:
        column_descriptions (dict): A dictionary where keys are column names and values are
                                    descriptions of those columns.

    Returns:
        str: A formatted string describing the data columns.

    Example:
        column_descriptions = {
            'ID': 'Unique identifier for each entry',
            'Text': 'The main text content to analyze',
            'Category': 'The category assigned to the text'
        }
        data_format_description = build_data_format_description(column_descriptions)
    """
    description = 'The data has the following columns:\n'
    for col, desc in column_descriptions.items():
        description += f'- "{col}": {desc}\n'
    return description

def construct_prompt(
    data_format_description,
    entry_text,
    codebook,
    examples,
    instructions,
    selected_fields=None,
    output_format_example=None,
    output_format_instructions=None,
    json_output=True
):
    """
    Constructs a prompt for a language model based on provided components.

    This function assembles various elements such as data format descriptions, entry text,
    codebooks, examples, and instructions into a single prompt string that can be used
    with a language model to generate responses.

    Parameters:
        data_format_description (str): A description of the data format, typically generated
                                       by `build_data_format_description`.
        entry_text (str): The text of the data entry to be evaluated by the language model.
        codebook (str): Instructions or guidelines (codebook) that the language model should
                        follow when evaluating the entry.
        examples (str): Examples of how entries should be evaluated or formatted.
        instructions (str): General instructions for the language model.
        selected_fields (list, optional): A list of fields that the language model should include
                                          in its response. Default is None.
        output_format_example (dict, optional): An example of the expected output format, provided
                                                as a dictionary. Default is None.
        output_format_instructions (str, optional): Specific instructions regarding the output format.
                                                   If not provided, a default instruction is generated
                                                   based on `selected_fields` and `output_format_example`.

    Returns:
        str: The constructed prompt string to be used with a language model.

    Example:
        data_format_description = build_data_format_description(column_descriptions)
        prompt = construct_prompt(
            data_format_description=data_format_description,
            entry_text=entry_text,
            codebook=codebook,
            examples=examples,
            instructions="You are an assistant that evaluates text entries.",
            selected_fields=['Evaluation', 'Comments'],
            output_format_example={'Evaluation': 'Positive', 'Comments': 'Well-written entry.'}
        )
    """
    # If selected_fields is not provided, default to an empty list
    if selected_fields is None:
        selected_fields = []

    # If user doesn't want JSON output
    if json_output:
        if output_format_instructions is None:
            output_format_instructions = f"""
- Your response should include the following fields: {', '.join(selected_fields)}.
- **Your response must be in JSON format only. Do not include any explanations, greetings, or additional text.**

**Example response format:**

{json.dumps(output_format_example, ensure_ascii=False, indent=2)}
"""

    prompt = f"""
{instructions}

You are provided with data entries in the following format:

{data_format_description}

Here is an entry to evaluate:

{entry_text}

{codebook}

{examples}

**Instructions:**

- Evaluate the entry according to the codebook and examples.
- Provide your evaluation in the specified format.
{output_format_instructions}
"""
    return prompt
