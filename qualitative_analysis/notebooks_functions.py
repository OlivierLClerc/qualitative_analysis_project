"""
notebooks_functions.py

This module provides utility functions for generating and processing classification results 
using large language models (LLMs) for both multiclass and binary classification tasks. 
It supports reasoning-based classification workflows and calculates token usage costs.

Dependencies:
    - pandas
    - qualitative_analysis.parsing (for extracting classification codes)
    - qualitative_analysis.cost_estimation (for calculating API costs)

Functions:
    - generate_answer(llm_client, model_name, base_prompt, multiclass_query, reasoning_query, ...):  
      Generates multiclass classification results with optional reasoning steps.

    - process_verbatims(verbatims_subset, codebooks, llm_client, model_name, prompt_template, ...):  
      Processes a set of verbatims (text samples) for multiclass classification and tracks API usage costs.

    - generate_binary_classification_answer(llm_client, model_name, final_prompt, reasoning_query, binary_query, ...):  
      Generates binary classification results (`0` or `1`) with optional reasoning.

    - process_verbatims_for_binary_criteria(verbatims_subset, codebooks, llm_client, model_name, prompt_template, ...):  
      Processes verbatims for binary classification across multiple themes and calculates token usage costs.
"""

from qualitative_analysis.parsing import extract_code_from_response
from qualitative_analysis.cost_estimation import openai_api_calculate_cost
from qualitative_analysis.cost_estimation import UsageProtocol
from typing import List, Dict, Tuple


def generate_answer(
    llm_client,
    model_name: str,
    base_prompt: str,
    multiclass_query: str,
    reasoning_query: str,
    reasoning: bool = False,
    temperature: float = 0.0001,
    verbose: bool = False,
) -> Tuple[str, UsageProtocol]:
    """
    Generates a classification result for a multiclass task using an LLM.

    If `reasoning` is False:
        - A single API call is made with the `base_prompt` and `multiclass_query`.

    If `reasoning` is True:
        - Two API calls are made:
            1. First call with `reasoning_query` for explanation.
            2. Second call using the reasoning result and `multiclass_query` for classification.

    Parameters:
    ----------
    llm_client : object
        The LLM client used to send prompts and receive responses.

    model_name : str
        The name of the LLM model to use (e.g., `"gpt-4"`).

    base_prompt : str
        The initial prompt providing context.

    multiclass_query : str
        The classification query for multiclass tasks.

    reasoning_query : str
        The reasoning query to enable step-by-step thinking.

    reasoning : bool, optional (default=False)
        If True, uses reasoning in the classification process.

    temperature : float, optional (default=0.0001)
        Controls randomness in the LLM output.

    verbose : bool, optional (default=False)
        If True, prints reasoning and classification steps.

    Returns:
    -------
    Tuple[str, UsageProtocol]
        - The classification result as a string.
        - The usage statistics object (tokens used, etc.).
    """
    if reasoning:
        # First call: get reasoning
        first_prompt = f"{base_prompt}\n\n{reasoning_query}"

        response_text_1, usage_1 = llm_client.get_response(
            prompt=first_prompt,
            model=model_name,
            max_tokens=500,
            temperature=temperature,
            verbose=verbose,
        )

        if verbose:
            print("\n=== Reasoning ===")

        # Second call: use reasoning answer for classification
        second_prompt = f"{base_prompt}\n\nReasoning about the entry:\n{response_text_1}\n\n{multiclass_query}"

        response_text_2, usage_2 = llm_client.get_response(
            prompt=second_prompt,
            model=model_name,
            max_tokens=500,
            temperature=temperature,
            verbose=verbose,
        )

        if verbose:
            print("\n=== Classification ===")

        # Combine usage stats
        usage_1.prompt_tokens += usage_2.prompt_tokens
        usage_1.completion_tokens += usage_2.completion_tokens
        usage_1.total_tokens += usage_2.total_tokens

        return response_text_2, usage_1

    else:
        # Single-step classification:
        # base prompt + multiclass query
        single_prompt = base_prompt

        response_text, usage = llm_client.get_response(
            prompt=single_prompt,
            model=model_name,
            max_tokens=500,
            temperature=temperature,
            verbose=verbose,
        )

        if verbose:
            print("\n=== Single-step Classification ===")

        return response_text, usage


def process_verbatims(
    verbatims_subset: List[str],
    codebooks: Dict[str, str],
    llm_client,
    model_name: str,
    prompt_template: str,
    multiclass_query: str,
    reasoning_query: str,
    valid_scores: List[int],
    reasoning: bool = False,
    verbose: bool = False,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """
    Processes and classifies a list of verbatims using the provided LLM.

    For each verbatim and each theme in the codebooks:
        - Generates a prompt.
        - Sends the prompt to the LLM for classification.
        - Tracks API usage and cost.

    Parameters:
    ----------
    verbatims_subset : List[str]
        A list of verbatim texts to classify.

    codebooks : Dict[str, str]
        Dictionary of themes and their descriptions.

    llm_client : object
        The LLM client used for response generation.

    model_name : str
        The LLM model name to use (e.g., `"gpt-4"`).

    prompt_template : str
        Template for formatting the prompt.

    multiclass_query : str
        Query for multiclass classification.

    reasoning_query : str
        Query for reasoning before classification.

    valid_scores : List[int]
        Valid classification labels (e.g., `[0, 1, 2]`).

    reasoning : bool, optional (default=False)
        If True, enables reasoning in classification.

    verbose : bool, optional (default=False)
        If True, prints progress logs.

    Returns:
    -------
    Tuple[List[Dict[str, object]], List[Dict[str, object]]]
        - A list of classification results.
        - A list of usage and cost per verbatim.
    """
    results = []
    verbatim_costs = []
    total_tokens_used = 0
    total_cost = 0.0

    for idx, verbatim_text in enumerate(verbatims_subset):
        print(f"\n=== Processing Verbatim {idx + 1}/{len(verbatims_subset)} ===")

        verbatim_tokens_used = 0
        verbatim_cost = 0.0

        # For each theme in the codebook dictionary
        for theme_name, codebook in codebooks.items():

            if verbose:
                print(f"\n--- Evaluating Theme: {theme_name} ---")

            # Build the final prompt
            final_prompt = prompt_template.format(
                verbatim_text=verbatim_text, codebook=codebook
            )

            try:
                # Generate the response
                response_content, usage = generate_answer(
                    llm_client=llm_client,
                    model_name=model_name,
                    base_prompt=final_prompt,
                    multiclass_query=multiclass_query,
                    reasoning_query=reasoning_query,
                    reasoning=reasoning,
                    temperature=0.0001,
                    verbose=verbose,
                )

                # Track usage/cost if usage is returned
                if usage:
                    tokens_used = usage.total_tokens
                    cost = openai_api_calculate_cost(usage, model=model_name)
                    total_tokens_used += tokens_used
                    total_cost += cost

                    verbatim_tokens_used += tokens_used
                    verbatim_cost += cost

                # Parse the classification score
                label = extract_code_from_response(response_content)
                if label in valid_scores:
                    results.append(
                        {"Verbatim": verbatim_text, "Theme": theme_name, "Label": label}
                    )
                else:
                    results.append(
                        {"Verbatim": verbatim_text, "Theme": theme_name, "Label": None}
                    )

            except Exception as e:
                print(
                    f"Error processing Verbatim {idx + 1} / Theme '{theme_name}': {e}"
                )
                results.append(
                    {"Verbatim": verbatim_text, "Theme": theme_name, "Label": None}
                )

        # Store usage/cost for this verbatim
        verbatim_costs.append(
            {
                "Verbatim": verbatim_text,
                "Tokens Used": verbatim_tokens_used,
                "Cost": verbatim_cost,
            }
        )

    print("\n=== Processing Complete ===")
    print(f"Total Tokens Used: {total_tokens_used}")
    print(f"Total Cost for Processing: ${total_cost:.4f}")

    return results, verbatim_costs


def generate_binary_classification_answer(
    llm_client,
    model_name: str,
    final_prompt: str,
    reasoning_query: str,
    binary_query: str,
    reasoning: bool = False,
    temperature: float = 0.0001,
    verbose: bool = False,
) -> Tuple[str, UsageProtocol]:
    """
    Generates a binary classification response ('1' or '0') using a language model.

    - If `reasoning` is False, a single API call is made with `final_prompt` and `binary_query`.
    - If `reasoning` is True, two API calls are made:
        1. First, to generate reasoning using `reasoning_query`.
        2. Second, to classify using the reasoning and `binary_query`.

    Parameters:
    ----------
    llm_client : object
        The LLM client used for making API calls.

    model_name : str
        The name of the LLM model (e.g., `"gpt-4"`).

    final_prompt : str
        The base prompt for classification.

    reasoning_query : str
        Query used to generate reasoning before classification.

    binary_query : str
        Query asking the model for binary classification.

    reasoning : bool, optional (default=False)
        Whether to include reasoning in the classification process.

    temperature : float, optional (default=0.0001)
        Controls the randomness of the model output.

    verbose : bool, optional (default=False)
        If True, prints intermediate outputs.

    Returns:
    -------
    Tuple[str, UsageProtocol]
        - Classification result (`'1'` or `'0'`).
        - API usage statistics.
    """
    if reasoning:
        # Two-step approach
        first_prompt = f"{final_prompt}\n\n{reasoning_query}"
        response_text_1, usage_1 = llm_client.get_response(
            prompt=first_prompt,
            model=model_name,
            max_tokens=500,
            temperature=temperature,
            verbose=verbose,
        )

        second_prompt = (
            f"{final_prompt}\n\nReasoning:\n{response_text_1}\n\n{binary_query}"
        )
        response_text_2, usage_2 = llm_client.get_response(
            prompt=second_prompt,
            model=model_name,
            max_tokens=500,
            temperature=temperature,
            verbose=verbose,
        )

        usage_1.prompt_tokens += usage_2.prompt_tokens
        usage_1.completion_tokens += usage_2.completion_tokens
        usage_1.total_tokens += usage_2.total_tokens

        return response_text_2, usage_1

    else:
        # Single-step
        single_prompt = f"{final_prompt}\n\n{binary_query}"
        response_text, usage = llm_client.get_response(
            prompt=single_prompt,
            model=model_name,
            max_tokens=500,
            temperature=temperature,
            verbose=verbose,
        )
        return response_text, usage


def process_verbatims_for_binary_criteria(
    verbatims_subset: List[str],
    codebooks: Dict[str, str],
    llm_client,
    model_name: str,
    prompt_template: str,
    reasoning_query: str,
    binary_query: str,
    reasoning: bool = False,
    verbose: bool = False,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """
    Processes verbatims and classifies them into binary categories (`0` or `1`) using an LLM.

    For each verbatim and theme in the `codebooks`, the function generates prompts to classify
    the text into binary labels. Reasoning can optionally be included.

    Parameters:
    ----------
    verbatims_subset : List[str]
        List of verbatim texts to classify.

    codebooks : Dict[str, str]
        Dictionary of themes and their descriptions.

    llm_client : object
        The LLM client used for generating responses.

    model_name : str
        The model to use for classification (e.g., `"gpt-4"`).

    prompt_template : str
        Template used to build the prompt with verbatims and themes.
        Example: `"{verbatim_text}\n\nTheme: {codebook}"`

    reasoning_query : str
        Query used for generating reasoning.

    binary_query : str
        Query used for binary classification.

    reasoning : bool, optional (default=False)
        Whether to include reasoning in the classification process.

    verbose : bool, optional (default=False)
        If True, prints detailed processing logs.

    Returns:
    -------
    Tuple[List[Dict[str, object]], List[Dict[str, object]]]
        - A list of classification results (`'Verbatim'`, `'Theme'`, `'Label'`).
        - A list of token usage and cost per verbatim.
    """

    results = []
    verbatim_costs = []
    total_tokens_used = 0
    total_cost = 0.0

    for idx, verbatim_text in enumerate(verbatims_subset):
        print(f"\n=== Processing Verbatim {idx+1}/{len(verbatims_subset)} ===")

        verbatim_tokens_used = 0
        verbatim_cost = 0.0

        # For each item in the codebook dictionary
        for theme_name, codebook in codebooks.items():

            if verbose:
                print(f"\n--- Evaluating Theme: {theme_name} ---")

            # Build the final prompt
            final_prompt = prompt_template.format(
                verbatim_text=verbatim_text, codebook=codebook
            )

            try:
                response_text, usage = generate_binary_classification_answer(
                    llm_client=llm_client,
                    model_name=model_name,
                    final_prompt=final_prompt,
                    reasoning_query=reasoning_query,
                    binary_query=binary_query,
                    reasoning=reasoning,
                    temperature=0.0001,
                    verbose=verbose,
                )

                # Track usage/cost if usage is returned
                if usage:
                    tokens_used = usage.total_tokens
                    # use your cost function
                    cost = openai_api_calculate_cost(usage, model=model_name)
                    total_tokens_used += tokens_used
                    total_cost += cost

                    verbatim_tokens_used += tokens_used
                    verbatim_cost += cost

                # parse the numeric classification (0 or 1)
                label = extract_code_from_response(response_text)
                if label in [0, 1]:
                    results.append(
                        {"Verbatim": verbatim_text, "Theme": theme_name, "Label": label}
                    )
                else:
                    results.append(
                        {"Verbatim": verbatim_text, "Theme": theme_name, "Label": None}
                    )

            except Exception as e:
                print(f"Error processing Verbatim {idx+1} / Theme '{theme_name}': {e}")
                results.append(
                    {"Verbatim": verbatim_text, "Theme": theme_name, "Label": None}
                )

        # Store usage/cost for this verbatim
        verbatim_costs.append(
            {
                "Verbatim": verbatim_text,
                "Tokens Used": verbatim_tokens_used,
                "Cost": verbatim_cost,
            }
        )

    print("\n=== Processing Complete ===")
    print(f"Total Tokens Used: {total_tokens_used}")
    print(f"Total Cost for Processing: ${total_cost:.4f}")

    return results, verbatim_costs
