"""
config.py

This module manages configuration settings and pricing information for different
OpenAI models and service providers. It securely loads environment variables and
defines pricing data for API usage cost estimation.
"""

import os
from dotenv import load_dotenv
from typing import Dict, Optional

# Load environment variables from a .env file
# This file should contain sensitive keys like API keys and endpoints.
load_dotenv()

# Configuration dictionary for different model providers.
# 'azure' contains settings for accessing Azure's OpenAI services
MODEL_CONFIG: Dict[str, Dict[str, Optional[str]]] = {
    "azure": {
        "api_key": os.getenv("AZURE_API_KEY"),
        "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "api_version": os.getenv("AZURE_API_VERSION"),
    },
    "openai": {
        "api_key": os.getenv("OPENAI_API_KEY"),
        # For standard OpenAI usage, you typically just need the API key,
        # no special endpoint or api_version.
        # But you could add more keys if needed (org ID, etc.).
    },
}


# Pricing information for different models.
# The prices are in dollars per token, where 'prompt' is the cost of input tokens
# and 'completion' is the cost of output tokens.
MODEL_PRICES: Dict[str, Dict[str, float]] = {
    "gpt-4o": {"prompt": 0.0025, "completion": 0.01},  # Pricing for GPT-4o model.
    "gpt-4o-mini": {
        "prompt": 0.00015,
        "completion": 0.0006,
    },  # Pricing for GPT-4o-mini model.
}

# NOTE:
# Ensure the .env file is properly set up with your keys:
# .env file might contain:
#   AZURE_API_KEY=...
#   AZURE_OPENAI_ENDPOINT=...
#   AZURE_API_VERSION=...
#   OPENAI_API_KEY=...
