# notebooks_functions.py
from qualitative_analysis.response_parsing import extract_code_from_response
from qualitative_analysis.cost_estimation import openai_api_calculate_cost

def generate_answer(
    llm_client,
    model_name,
    base_prompt,
    multiclass_query,
    reasoning_query,
    reasoning=False,
    temperature=0.0001,
    verbose=False
):
    """
    Generates a classification result for multiclass classification.

    If reasoning is False, make one API call:
        - Base prompt + multiclass query directly.

    If reasoning is True, make two API calls:
        1) Base prompt + reasoning query (no classification yet).
        2) Base prompt + reasoning answer_from_first_call + multiclass query.
    """
    if reasoning:
        # First call: get reasoning
        first_prompt = f"{base_prompt}\n\n{reasoning_query}"

        response_text_1, usage_1 = llm_client.get_response(
            prompt=first_prompt,
            model=model_name,
            max_tokens=500,
            temperature=temperature,
            verbose=verbose  # Changed from verbose=True
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
            verbose=verbose  # Changed from verbose=True
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
            verbose=verbose  # Changed from verbose=True
        )

        if verbose:
            print("\n=== Single-step Classification ===")

        return response_text, usage

def process_verbatims(
    verbatims_subset,
    codebooks,
    llm_client,
    model_name,
    prompt_template,
    multiclass_query,
    reasoning_query,
    valid_scores,
    reasoning=False,
    verbose=False
):
    """
    Process a subset of verbatims and classify them based on the provided themes 
    and descriptions from the codebooks.

    Args:
        verbatims_subset (list): List of verbatim texts to process.
        codebooks (dict): Dictionary of themes and their descriptions.
        llm_client: LLM client for generating responses.
        model_name (str): Model to use for predictions.
        prompt_template (str): Template to format the prompt.
        multiclass_query (str): Query for multiclass classification.
        reasoning_query (str): Query for reasoning (optional).
        valid_scores (list): List of valid classification scores.
        reasoning (bool): Whether to include reasoning in the responses.
        verbose (bool): Whether to print detailed logs.

    Returns:
        tuple: A list of results and a list of costs per verbatim.
    """
    results = []
    verbatim_costs = []
    total_tokens_used = 0
    total_cost = 0

    for idx, verbatim_text in enumerate(verbatims_subset):
        print(f"\n=== Processing Verbatim {idx + 1}/{len(verbatims_subset)} ===")

        verbatim_tokens_used = 0
        verbatim_cost = 0

        # For each theme in the codebook dictionary
        for theme_name, codebook in codebooks.items():

            if verbose:
                print(f"\n--- Evaluating Theme: {theme_name} ---")

            # Build the final prompt
            final_prompt = prompt_template.format(
                verbatim_text=verbatim_text,
                codebook=codebook
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
                    verbose=verbose
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
                score = extract_code_from_response(response_content)
                if score in valid_scores:
                    results.append({
                        'Verbatim': verbatim_text,
                        'Theme': theme_name,
                        'Label': score
                    })
                else:
                    results.append({
                        'Verbatim': verbatim_text,
                        'Theme': theme_name,
                        'Label': None
                    })

            except Exception as e:
                print(f"Error processing Verbatim {idx + 1} / Theme '{theme_name}': {e}")
                results.append({
                    'Verbatim': verbatim_text,
                    'Theme': theme_name,
                    'Label': None
                })

        # Store usage/cost for this verbatim
        verbatim_costs.append({
            'Verbatim': verbatim_text,
            'Tokens Used': verbatim_tokens_used,
            'Cost': verbatim_cost
        })

    print("\n=== Processing Complete ===")
    print(f"Total Tokens Used: {total_tokens_used}")
    print(f"Total Cost for Processing: ${total_cost:.4f}")

    return results, verbatim_costs




def generate_binary_classification_answer(
    llm_client,
    model_name,
    final_prompt,
    reasoning_query,
    binary_query,
    reasoning=False,
    temperature=0.0001,
    verbose=False
):
    """
    Generates a binary classification ('1' or '0').
    ...
    """
    if reasoning:
        # Two-step approach
        first_prompt = f"{final_prompt}\n\n{reasoning_query}"
        response_text_1, usage_1 = llm_client.get_response(
            prompt=first_prompt,
            model=model_name,
            max_tokens=500,
            temperature=temperature,
            verbose=verbose
        )

        second_prompt = f"{final_prompt}\n\nReasoning:\n{response_text_1}\n\n{binary_query}"
        response_text_2, usage_2 = llm_client.get_response(
            prompt=second_prompt,
            model=model_name,
            max_tokens=500,
            temperature=temperature,
            verbose=verbose
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
            verbose=verbose
        )
        return response_text, usage


def process_verbatims_for_binary_criteria(
    verbatims_subset,
    codebooks,
    llm_client,
    model_name,
    prompt_template,
    reasoning_query,
    binary_query,
    reasoning=False,
    verbose=False
):
    """
    Loops over each verbatim and each (theme_name, theme_description) in codebooks.
    For each combination, build a prompt and do a binary classification.
    ...
    """

    results = []
    verbatim_costs = []
    total_tokens_used = 0
    total_cost = 0

    for idx, verbatim_text in enumerate(verbatims_subset):
        print(f"\n=== Processing Verbatim {idx+1}/{len(verbatims_subset)} ===")

        verbatim_tokens_used = 0
        verbatim_cost = 0

        # For each item in the codebook dictionary
        for theme_name, codebook in codebooks.items():

            if verbose:
                print(f"\n--- Evaluating Theme: {theme_name} ---")

            # Build the final prompt
            final_prompt = prompt_template.format(
                verbatim_text=verbatim_text,
                codebook=codebook
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
                    verbose=verbose
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
                score = extract_code_from_response(response_text)
                if score in [0, 1]:
                    results.append({
                        'Verbatim': verbatim_text,
                        'Theme': theme_name,
                        'Score': score
                    })
                else:
                    results.append({
                        'Verbatim': verbatim_text,
                        'Theme': theme_name,
                        'Score': None
                    })

            except Exception as e:
                print(f"Error processing Verbatim {idx+1} / Theme '{theme_name}': {e}")
                results.append({
                    'Verbatim': verbatim_text,
                    'Theme': theme_name,
                    'Score': None
                })

        # Store usage/cost for this verbatim
        verbatim_costs.append({
            'Verbatim': verbatim_text,
            'Tokens Used': verbatim_tokens_used,
            'Cost': verbatim_cost
        })

    print("\n=== Processing Complete ===")
    print(f"Total Tokens Used: {total_tokens_used}")
    print(f"Total Cost for Processing: ${total_cost:.4f}")

    return results, verbatim_costs