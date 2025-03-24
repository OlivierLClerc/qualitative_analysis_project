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
    "together": {
        "api_key": os.getenv("TOGETHER_API_KEY"),
    },
    "vllm": {
        # Default to a small model that works with minimal resources
        "model_path": os.getenv(
            "VLLM_MODEL_PATH", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        ),
        # Optional parameters for vLLM
        "dtype": os.getenv("VLLM_DTYPE", "float16"),  # Use float16 for efficiency
        "gpu_memory_utilization": os.getenv(
            "VLLM_GPU_MEMORY_UTILIZATION", "0.95"
        ),  # Default to 95% GPU memory usage for supercomputers
        "tensor_parallel_size": os.getenv(
            "VLLM_TENSOR_PARALLEL_SIZE", "4"
        ),  # Default to 4 GPUs for supercomputers
        # Additional parameters for supercomputer usage
        "max_model_len": os.getenv(
            "VLLM_MAX_MODEL_LEN", "2048"
        ),  # Maximum sequence length
        "enable_prefix_caching": os.getenv(
            "VLLM_ENABLE_PREFIX_CACHING", "true"
        ),  # Enable prefix caching for better performance
        # Removed worker_multiproc_method as it's not supported in some vLLM versions
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
