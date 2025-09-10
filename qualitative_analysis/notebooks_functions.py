"""
notebooks_functions.py

This module provides general-purpose utility functions for processing text data (verbatims) 
using large language models (LLMs). It supports flexible classification workflows with both 
traditional prefix-based parsing and JSON-structured responses, along with token usage cost tracking.

Dependencies:
    - pandas
    - collections (Counter for majority voting)
    - typing (for type hints)
    - qualitative_analysis.parsing (for extracting classification codes and parsing JSON responses)
    - qualitative_analysis.cost_estimation (for calculating API costs)

Functions:
    - process_general_verbatims(verbatims_subset, llm_client, model_name, prompt_template, ...):
      Processes a list of verbatims by querying an LLM for each one. Supports both classic 
      prefix-based code extraction and JSON-structured response parsing. Includes majority 
      voting across multiple completions and comprehensive cost tracking.

    - majority_vote(labels):
      Returns the most common label from a list of labels, used for consensus across 
      multiple LLM completions.
"""

from qualitative_analysis.parsing import (
    extract_code_from_response,
    parse_llm_response,
)
from qualitative_analysis.cost_estimation import openai_api_calculate_cost
import pandas as pd
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional


def process_general_verbatims(
    verbatims_subset: List[str],
    llm_client,
    model_name: str,
    prompt_template: str,
    label_field: Optional[str] = None,
    temperature: float = 0.0,
    json_output: bool = False,
    selected_fields: Optional[List[str]] = None,
    n_completions: int = 1,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], Dict[str, float]]:
    """
    Processes a list of verbatims by querying a language model for each verbatim,
    then extracts either a code (default) or specific JSON fields from the response.

    This function can operate in two modes:
      1) **Classic/Prefix Mode** (if `json_output=False`):
         Uses `extract_code_from_response` to parse out a single "Label" from the LLM response
         based on a specified `prefix`.
      2) **JSON Mode** (if `json_output=True`):
         Expects the LLM response to be valid JSON; then uses `parse_llm_response`
         to extract the fields listed in `selected_fields`.
         The "Label" is taken from `parsed.get("Validity")` by default (though you can alter that as needed).

    Parameters
    ----------
    verbatims_subset : List[str]
        A list of verbatim (text) entries to process.

    llm_client : object
        An LLM client (or wrapper) that implements a `.get_response(...)` method
        to communicate with the model.

    model_name : str
        Name (or identifier) of the language model to use for classification/inference.

    prompt_template : str
        A string template used to build the final prompt for each verbatim.
        Must contain `"{verbatim_text}"` as a placeholder. Example:
          "You are an assistant...\n\nInput:\n{verbatim_text}\n\nRespond with 0 or 1."

    label_field : Optional[str], default=None
        The field name to use as the source for the "ModelPrediction" column.
        If `json_output=True`, this is the field name in the JSON response.
        If `json_output=False`, this is the prefix used by `extract_code_from_response`
        to locate the classification code within the LLM response.

    temperature : float, default=0.0
        Model generation temperature. Higher values produce more creative outputs.

    json_output : bool, default=False
        If True, the function expects valid JSON in the LLM response.
        Parsing is done via `parse_llm_response`, and you must provide `selected_fields`.

    selected_fields : Optional[List[str]], default=None
        The keys to extract from the LLM's JSON response. If `json_output=True` but
        no `selected_fields` is provided, the function raises a ValueError.
        Typically includes ["Validity", "Reasoning"], etc.

    n_completions : int, default=1
        Number of completions to request from the LLM for each verbatim.
        The final label is determined by majority vote among these completions.

    verbose : bool, default=False
        If True, prints intermediate debugging logs and partial results.

    Returns
    -------
    results_df : pd.DataFrame
        A DataFrame with columns:
          - "Verbatim": The original verbatim text.
          - "Label": The extracted code or JSON field (e.g. "Validity").

    verbatim_costs : List[Dict[str, Any]]
        A list of dictionaries, each containing:
          - "Verbatim": The original text.
          - "Tokens Used": The model's token usage for this call (if available).
          - "Cost": The estimated cost for this single inference call.

    totals : Dict[str, float]
        Aggregated totals across all verbatims:
          - "total_tokens_used": Sum of tokens used.
          - "total_cost": Accumulated cost for all requests.

    Raises
    ------
    ValueError
        If `json_output=True` but `selected_fields` is None or empty.
        (User must specify which fields to parse from JSON.)
    """

    results = []
    verbatim_costs = []
    total_tokens_used = 0
    total_cost = 0.0

    # If user sets json_output=True but didn't provide fields => enforce at least one field.
    if json_output and (not selected_fields or len(selected_fields) == 0):
        raise ValueError(
            "You must provide at least one field name in `selected_fields` "
            "when `json_output=True`."
        )

    for idx, verbatim_text in enumerate(verbatims_subset, start=1):
        if verbose:
            print(f"\n=== Processing Verbatim {idx}/{len(verbatims_subset)} ===")

        # Build the final prompt for this verbatim
        final_prompt = prompt_template.format(verbatim_text=verbatim_text)

        # We'll store each completion's label for majority voting
        completion_labels = []

        # We'll track usage/cost for all completions for this verbatim
        tokens_used_for_this_verbatim = 0
        cost_for_this_verbatim = 0.0

        try:
            # Request n_completions from the LLM
            for _ in range(n_completions):
                response_text, usage = llm_client.get_response(
                    prompt=final_prompt,
                    model=model_name,
                    max_tokens=10000,
                    temperature=temperature,
                    verbose=verbose,
                )

                # JSON mode
                if json_output:
                    # Parse the JSON response to extract all fields
                    parsed_fields = parse_llm_response(
                        response_text,
                        selected_fields=None,  # Pass None to extract all fields
                    )

                    # Get the label for majority voting
                    # By default we assume "Classification" if label_field is not provided
                    key_for_label = (
                        label_field if label_field is not None else "Classification"
                    )
                    label = parsed_fields.get(key_for_label, None)

                    # Fallback: If JSON parsing failed for the primary label field, try prefix extraction
                    if label is None and label_field is not None:
                        if verbose:
                            print(
                                f"JSON parsing failed for '{key_for_label}', attempting prefix extraction with '{label_field}'"
                            )
                        label = extract_code_from_response(
                            response_text, prefix=label_field
                        )
                        if label is not None and verbose:
                            print(
                                f"Successfully extracted label '{label}' using prefix method"
                            )

                        # Ultimate fallback: If prefix extraction also failed, try to extract any number from the response
                        if label is None:
                            if verbose:
                                print(
                                    "Prefix extraction also failed, attempting to extract any number from response"
                                )
                            # Look for any number in the response (0 or 1 for binary classification)
                            import re

                            numbers = re.findall(r"\b[01]\b", response_text)
                            if numbers:
                                # Take the last occurrence (often the final classification)
                                label = int(numbers[-1])
                                if verbose:
                                    print(
                                        f"Ultimate fallback extracted label '{label}' from numbers: {numbers}"
                                    )
                            else:
                                if verbose:
                                    print(
                                        "No valid binary classification number (0 or 1) found in response"
                                    )
                else:
                    # Classic prefix-based mode
                    # extract_code_from_response is a helper you have elsewhere
                    label = extract_code_from_response(
                        response_text, prefix=label_field
                    )
                    parsed_fields = None

                # Store the label for majority voting
                completion_labels.append(label)

                # Accumulate usage cost
                if usage:
                    single_cost = openai_api_calculate_cost(usage, model_name)
                    tokens_used_for_this_verbatim += usage.total_tokens
                    cost_for_this_verbatim += single_cost

            # Apply majority-vote on the n_completions
            final_label = majority_vote(completion_labels)

            # Record the results
            if json_output and parsed_fields:
                # Include all parsed fields in the results
                result_row = {"Verbatim": verbatim_text, "Label": final_label}

                # Add all fields from the parsed JSON response
                for field, value in parsed_fields.items():
                    result_row[field] = value

                results.append(result_row)
            else:
                # Just include the label for non-JSON output
                results.append({"Verbatim": verbatim_text, "Label": final_label})

            # Add cost info for this verbatim
            verbatim_costs.append(
                {
                    "Verbatim": verbatim_text,
                    "Tokens Used": tokens_used_for_this_verbatim,
                    "Cost": cost_for_this_verbatim,
                }
            )

            # Update global totals
            total_tokens_used += tokens_used_for_this_verbatim
            total_cost += cost_for_this_verbatim

            if verbose:
                print(f"Labels from {n_completions} completions => {completion_labels}")
                print(f"Final (majority) label => {final_label}")
                print(f"Tokens used => {tokens_used_for_this_verbatim}")
                print(f"Cost => {cost_for_this_verbatim:.4f}")

        except Exception as e:
            # Handle any errors in retrieving/parsing
            print(f"Critical Error processing verbatim {idx}")
            print(f"Error Type: {type(e).__name__}")
            print(f"Error Message: {str(e)}")

            # Store error details so you don't lose track of which verbatim failed
            results.append({"Verbatim": verbatim_text, "Label": None, "Error": str(e)})
            verbatim_costs.append(
                {
                    "Verbatim": verbatim_text,
                    "Tokens Used": 0,
                    "Cost": 0.0,
                    "Error": str(e),
                }
            )

    # Summarize total usage
    totals = {"total_tokens_used": total_tokens_used, "total_cost": total_cost}

    # Create a DataFrame of all results
    results_df = pd.DataFrame(results)

    return results_df, verbatim_costs, totals


def majority_vote(labels: List[str]) -> Optional[str]:
    """
    Returns the most common label in the list.
    Breaks ties by returning the first label with the highest count.
    If the list is empty, returns None.
    """
    if not labels:
        return None
    counter = Counter(labels)
    # `most_common(1)` returns [(label, count)] for the top label
    return counter.most_common(1)[0][0]
