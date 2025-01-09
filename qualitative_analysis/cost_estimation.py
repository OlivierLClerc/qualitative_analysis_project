# cost_estimation.py
from qualitative_analysis.config import MODEL_PRICES


def openai_api_calculate_cost(usage, model="gpt-4o"):
    """
    Calculate the cost of API usage based on token consumption.

    Parameters:
        usage (object): An object with token usage details (prompt_tokens, completion_tokens).
        model (str): The OpenAI model name.

    Returns:
        float: The total cost rounded to 6 decimals.
    """
    pricing = MODEL_PRICES.get(model)
    if not pricing:
        raise ValueError(f"Invalid model specified: {model}")

    prompt_cost = usage.prompt_tokens * pricing["prompt"] / 1000
    completion_cost = usage.completion_tokens * pricing["completion"] / 1000
    total_cost = round(prompt_cost + completion_cost, 6)

    return total_cost
