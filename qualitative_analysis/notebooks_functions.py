# notebooks_functions.py
from qualitative_analysis.response_parsing import extract_code_from_response
from qualitative_analysis.cost_estimation import openai_api_calculate_cost

def generate_with_reasoning(
    llm_client,
    model_name,
    base_prompt,
    multiclass_query,  # Add multiclass_query as an argument
    reasoning_query,   # Add reasoning_query as an argument
    reasoning=False,
    temperature=0.0001,
    verbose=False
):
    """
    Generates a classification result for multiclass classification.

    If reasoning is False, make one API call:
        Base prompt + multiclass query directly.

    If reasoning is True, make two API calls:
        1) Base prompt + reasoning query (no classification yet).
        2) Base prompt + reasoning answer_from_first_call + multiclass query.
    """
    if reasoning:
        # First call: get reasoning
        first_prompt = f"{base_prompt}\n\n{reasoning_query}"
        print("=== First LLM Prompt (Reasoning) ===")

        response_text_1, usage_1 = llm_client.get_response(
            prompt=first_prompt,
            model=model_name,
            max_tokens=500,
            temperature=temperature,
            verbose=True
        )

        print("\n=== LLM Response (Reasoning) ===")
        print(response_text_1)

        # Second call: use reasoning answer for classification
        second_prompt = f"{base_prompt}\n\nReasoning:\n{response_text_1}\n\n{multiclass_query}"

        print("\n=== Second LLM Prompt (Classification) ===")

        response_text_2, usage_2 = llm_client.get_response(
            prompt=second_prompt,
            model=model_name,
            max_tokens=500,
            temperature=temperature,
            verbose=True
        )

        if verbose:
            print("\n=== LLM Response (Classification) ===")
            print(response_text_2)

        # Combine usage stats
        usage_1.prompt_tokens += usage_2.prompt_tokens
        usage_1.completion_tokens += usage_2.completion_tokens
        usage_1.total_tokens += usage_2.total_tokens

        return response_text_2, usage_1

    else:
        # Single-step classification: base prompt + multiclass query
        single_prompt = base_prompt

        print("=== Single-step Classification Prompt ===")

        response_text, usage = llm_client.get_response(
            prompt=single_prompt,
            model=model_name,
            max_tokens=500,
            temperature=temperature,
            verbose=True
        )
        
        print("\n=== LLM Response (Single-step Classification) ===")
        print(response_text)

        return response_text, usage

def process_verbatims(
    verbatims_subset,
    llm_client,
    model_name,
    theme_name,
    theme_description,
    data_format_description,
    construct_prompt,  # Accept the prompt construction function
    multiclass_query,  # Add multiclass_query as an argument
    reasoning_query,   # Add reasoning_query as an argument
    valid_scores,      # Add valid_scores as an argument
    reasoning=False,
    verbose=False
):
    """
    Process a subset of verbatims and classify them based on the provided theme and description.
    """
    results = []
    verbatim_costs = []
    total_tokens_used = 0
    total_cost = 0

    for idx, verbatim in enumerate(verbatims_subset):
        print(f"\n=== Processing Verbatim {idx + 1}/{len(verbatims_subset)} ===")
        verbatim_tokens_used = 0
        verbatim_cost = 0

        print(f"\n--- Evaluating Theme: {theme_name} ---")

        # Build the base prompt using the provided construct_prompt function
        base_prompt = construct_prompt(
            verbatim=verbatim,
            theme=theme_name,
            theme_description=theme_description,
            data_format_description=data_format_description
        )

        try:
            response_content, usage = generate_with_reasoning(
                llm_client=llm_client,
                model_name=model_name,
                base_prompt=base_prompt,
                multiclass_query=multiclass_query,  # Pass multiclass_query
                reasoning_query=reasoning_query,    # Pass reasoning_query
                reasoning=reasoning,
                temperature=0.0001,
                verbose=verbose
            )

            # Track token usage
            if usage:
                prompt_tokens = usage.prompt_tokens
                completion_tokens = usage.completion_tokens
                total_tokens = usage.total_tokens

                # Calculate the cost for this request
                cost = openai_api_calculate_cost(usage, model=model_name)
                total_tokens_used += total_tokens
                total_cost += cost
                verbatim_tokens_used += total_tokens
                verbatim_cost += cost

                # Print detailed token usage and cost
                print(f"\nTokens Used: {prompt_tokens} (prompt) + {completion_tokens} (completion) = {total_tokens} total")
                print(f"Cost for '{theme_name}': ${cost:.4f}")

            # Parse response
            score = extract_code_from_response(response_content)
            if score in valid_scores:  # Use the provided valid_scores
                results.append({
                    'Verbatim': verbatim,
                    'Label': score
                })
                print(f"Extracted Label: {score}")
            else:
                print("Failed to parse a valid label.")
                results.append({
                    'Verbatim': verbatim,
                    'Label': None
                })

        except Exception as e:
            print(f"Error processing Verbatim {idx + 1} for '{theme_name}': {e}")
            results.append({
                'Verbatim': verbatim,
                'Label': None
            })

        # Store verbatim-level cost
        verbatim_costs.append({'Verbatim': verbatim, 'Tokens Used': verbatim_tokens_used, 'Cost': verbatim_cost})

    # Final Summary
    print("\n=== Processing Complete ===")
    print(f"Total Tokens Used: {total_tokens_used}")
    print(f"Total Cost for Processing: ${total_cost:.4f}")

    return results, verbatim_costs