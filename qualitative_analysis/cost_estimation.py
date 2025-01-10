"""
cost_estimation.py

This module calculates the cost of OpenAI API usage based on token consumption
for different models. It uses predefined model pricing data to compute the
total cost of API calls.
"""

from qualitative_analysis.config import MODEL_PRICES
from typing import Protocol


class UsageProtocol(Protocol):
    """
    Protocol to define the structure of the 'usage' object.
    The object must have integer attributes for token usage details.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


def openai_api_calculate_cost(usage: UsageProtocol, model: str = "gpt-4o") -> float:
    """
    Calculate the cost of OpenAI API usage based on token consumption.

    Parameters:
        usage (UsageProtocol): An object with token usage details, specifically:
            - prompt_tokens (int): Number of tokens used in the prompt.
            - completion_tokens (int): Number of tokens used in the completion.
            - total_tokens (int): Total tokens used (prompt + completion).
        model (str): The OpenAI model name (default: "gpt-4o"). Must exist in `MODEL_PRICES`.

    Returns:
        float: The total API usage cost in USD, rounded to 6 decimal places.

    Raises:
        ValueError: If the specified model is not found in `MODEL_PRICES`.

    Example:
        >>> class MockUsage:
        ...     prompt_tokens = 1000
        ...     completion_tokens = 500
        ...     total_tokens = 1500
        ...
        >>> usage = MockUsage()
        >>> openai_api_calculate_cost(usage, model="gpt-4o")
        0.0075

    Notes:
        The function assumes that the `MODEL_PRICES` dictionary contains pricing
        details with keys "prompt" and "completion" for each model.
    """
    pricing = MODEL_PRICES.get(model)
    if not pricing:
        raise ValueError(f"Invalid model specified: {model}")

    prompt_cost: float = usage.prompt_tokens * pricing["prompt"] / 1000
    completion_cost: float = usage.completion_tokens * pricing["completion"] / 1000
    total_cost: float = round(prompt_cost + completion_cost, 6)

    return total_cost
