"""
config.py

This module manages configuration settings and pricing information for different
OpenAI models and service providers. It securely loads environment variables and
defines pricing data for API usage cost estimation.
"""

import os
from dotenv import load_dotenv
from typing import Dict, Any

# Load environment variables from a .env file
# This file should contain sensitive keys like API keys and endpoints.
load_dotenv()

# Configuration dictionary for different model providers.
# 'azure' contains settings for accessing Azure's OpenAI services
MODEL_CONFIG: Dict[str, Dict[str, Any]] = {
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
    "anthropic": {
        "api_key": os.getenv("ANTHROPIC_API_KEY"),
        # For Anthropic, we just need the API key
    },
    "together": {
        "api_key": os.getenv("TOGETHER_API_KEY"),
    },
    "vllm": {
        # Explicitly set device type to fix "Failed to infer device type" error
        "device": os.getenv("VLLM_DEVICE", "cuda"),
        # Support both local paths and HuggingFace model IDs
        # For Jean Zay: set VLLM_LOCAL_MODEL_PATH to the path where you downloaded the model
        "model_path": os.getenv(
            "VLLM_LOCAL_MODEL_PATH",  # First check for a local path
            os.getenv(
                "VLLM_MODEL_PATH", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            ),  # Fall back to HF ID
        ),
        # Use half precision (equivalent to float16)
        "dtype": os.getenv("VLLM_DTYPE", "half"),
        # Force eager execution mode (convert string to boolean)
        "enforce_eager": os.getenv("VLLM_ENFORCE_EAGER", "true").lower() == "true",
        # Disable async output processing (convert string to boolean)
        "disable_async_output_proc": os.getenv("VLLM_DISABLE_ASYNC", "true").lower()
        == "true",
        # Start with tensor parallel size 1, can increase if needed (convert string to int)
        "tensor_parallel_size": int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1")),
        # Enable prefix caching for better performance (convert string to boolean)
        "enable_prefix_caching": os.getenv("VLLM_ENABLE_PREFIX_CACHING", "true").lower()
        == "true",
        # Explicitly set the worker class to fix the "not enough values to unpack" error
        "worker_cls": "vllm.worker.worker.Worker",
        # Set distributed executor backend to None for local execution
        "distributed_executor_backend": None,
        # Prevent HuggingFace from trying to download anything when using local models
        "trust_remote_code": os.getenv("VLLM_TRUST_REMOTE_CODE", "false").lower()
        == "true",
        # Don't try to fetch a specific revision when using local models
        "revision": os.getenv("VLLM_REVISION", None),
    },
    "gemini": {
        "api_key": os.getenv("GEMINI_API_KEY"),
    },
}


# Pricing information for different models.
# The prices are in dollars per token, where 'prompt' is the cost of input tokens
# and 'completion' is the cost of output tokens.
MODEL_PRICES: Dict[str, Dict[str, float]] = {
    # OpenAI models
    "gpt-4o": {"prompt": 0.0025, "completion": 0.01},  # Pricing for GPT-4o model.
    "gpt-4o-mini": {
        "prompt": 0.00015,
        "completion": 0.0006,
    },  # Pricing for GPT-4o-mini model.
    # Anthropic models
    "claude-3-7-sonnet-20250219": {
        "prompt": 0.0030,
        "completion": 0.0150,
    },
    "claude-3-5-haiku-20241022": {
        "prompt": 0.0008,
        "completion": 0.0040,
    },
    # Gemini models (example pricing - check Google AI Studio for current prices)
    "gemini-2.0-flash-001": {
        "prompt": 0.00005,  # Example: $0.50 per million input tokens
        "completion": 0.0001,  # Example: $1.00 per million output tokens
    },
    "gemini-2.5-pro-preview-03-25": {
        "prompt": 0.00005,  # Example: $0.50 per million input tokens
        "completion": 0.0001,  # Example: $1.00 per million output tokens
    },
}

# NOTE:
# Ensure the .env file is properly set up with your keys:
# .env file might contain:
#   AZURE_API_KEY=...
#   AZURE_OPENAI_ENDPOINT=...
#   AZURE_API_VERSION=...
#   OPENAI_API_KEY=...
#   ANTHROPIC_API_KEY=...
