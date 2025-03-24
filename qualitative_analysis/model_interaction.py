"""
model_interaction.py

This module provides a unified interface for interacting with different large language model (LLM) providers,
including:

    - OpenAI
    - Azure OpenAI
    - Together AI
    - vLLM (for open-source models)

It abstracts API interactions to simplify sending prompts and retrieving responses across multiple providers.

Dependencies:
    - openai: For interacting with Azure OpenAI or standard OpenAI models.
    - together: For interacting with Together AI models.
    - vllm: For running inference with open-source models locally.
    - abc: For defining the abstract base class.
    - types: For using SimpleNamespace to standardize usage object representation.

Classes:
    - LLMClient: Abstract base class defining the interface for LLM clients.
    - OpenAILLMClient: Client for interacting with standard OpenAI language models.
    - AzureOpenAILLMClient: Client for interacting with Azure OpenAI language models.
    - TogetherLLMClient: Client for interacting with Together AI language models.
    - VLLMLLMClient: Client for interacting with open-source models using vLLM.

Functions:
    - get_llm_client(provider, config): Factory function to instantiate the appropriate LLM client 
      based on the specified provider string.
"""

from abc import ABC, abstractmethod
import openai
from together import Together
from types import SimpleNamespace
from typing import Optional

# Try to import vLLM, but handle the case when it's not available
# This could be due to import errors or platform compatibility issues
try:
    from vllm import LLM, SamplingParams

    VLLM_AVAILABLE = True
except (ImportError, OSError):
    VLLM_AVAILABLE = False
    print("Warning: vLLM is not available. VLLMLLMClient will not be usable.")


class LLMClient(ABC):
    """
    Abstract base class for language model clients.

    This class defines a common interface for interacting with different language model providers.
    Subclasses must implement the `get_response` method to handle API communication with specific models.

    Methods:
        - get_response(prompt, model, **kwargs):
            Sends a prompt to the language model and retrieves the response.
            Must be implemented by all subclasses.

    Usage:
        This class cannot be instantiated directly. Subclasses should implement the `get_response` method.

        Example:
            class CustomLLMClient(LLMClient):
                def get_response(self, prompt, model, **kwargs):
                    # Custom implementation here
                    return "Sample response"

            client = CustomLLMClient()
            response = client.get_response("Hello!", model="custom-model")
    """

    @abstractmethod
    def get_response(self, prompt: str, model: str, **kwargs) -> tuple[str, object]:
        """
        Sends a prompt to the language model and retrieves the response.

        Parameters:
            - prompt (str): The input text prompt to send to the language model.
            - model (str): The identifier of the language model to use.
            - **kwargs: Additional keyword arguments specific to the language model API.

        Returns:
            - tuple[str, object]: A tuple containing:
                - The language model's response to the prompt (str).
                - Usage information or metadata (object).

        Raises:
            - NotImplementedError: If the method is not implemented in the subclass.
        """
        pass


class OpenAILLMClient(LLMClient):
    """
    Client for interacting with OpenAI language models (non-Azure).

    This class manages communication with the standard OpenAI API, enabling
    prompt-based interactions with models like "gpt-3.5-turbo" or "gpt-4".
    It handles authentication via an OpenAI API key (api_key).

    Attributes
    ----------
    api_key : str
        The OpenAI API key used for authentication.

    Methods
    -------
    get_response(prompt, model, **kwargs) -> tuple[str, object]
        Sends a prompt to the specified OpenAI language model and returns the
        response text plus usage metadata.
    """

    def __init__(self, api_key: str):
        """
        Initializes the OpenAI LLM client.

        Parameters
        ----------
        api_key : str
            The OpenAI API key to use (from OPENAI_API_KEY).
        """
        openai.api_type = "openai"
        openai.api_key = api_key
        # You may optionally set `openai.organization` if needed:
        # openai.organization = os.getenv("OPENAI_ORGANIZATION")

    def get_response(self, prompt: str, model: str, **kwargs) -> tuple[str, dict]:
        """
        Sends a prompt to the standard OpenAI model and retrieves the response.

        Parameters
        ----------
        prompt : str
            The user prompt to send to the language model.

        model : str
            The OpenAI model name (e.g., "gpt-3.5-turbo", "gpt-4", etc.).

        **kwargs : dict, optional
            Additional arguments such as:
            - temperature (float): Controls randomness (default is 0.0).
            - max_tokens (int): Max tokens in the response (default 500).
            - verbose (bool): If True, prints debug info.

        Returns
        -------
        tuple[str, object]
            A tuple containing:
                - The generated response text (str).
                - The usage object detailing token usage (object), if any.

        Raises
        ------
        openai.error.OpenAIError
            If the API request fails.
        """
        temperature = kwargs.get("temperature", 0.0)
        max_tokens = kwargs.get("max_tokens", 500)
        verbose = kwargs.get("verbose", False)

        if verbose:
            print(f"Prompt:\n{prompt}\n")

        # Standard openai call:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if verbose:
            print("\n=== LLM Response ===")
            print(f"{response.choices[0].message.content}\n")

        content = response.choices[0].message.content

        # Convert usage object to a dictionary for consistency
        if response.usage:
            usage_dict = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        else:
            # Fallback if usage is not available
            usage_dict = {
                "prompt_tokens": len(prompt.split()),  # Very rough estimate
                "completion_tokens": (
                    len(content.split()) if content else 0
                ),  # Very rough estimate
                "total_tokens": len(prompt.split())
                + (len(content.split()) if content else 0),
            }

        return (content.strip() if content else ""), usage_dict


class AzureOpenAILLMClient(LLMClient):
    """
    Client for interacting with Azure OpenAI language models.

    This class manages communication with the Azure OpenAI API, enabling prompt-based
    interactions with models like GPT-3/GPT-4 deployed on Azure. It handles authentication,
    configuration (endpoint, api_version), and response retrieval.

    Attributes
    ----------
    api_key : str
        The Azure API key for authenticating with the Azure OpenAI service.
    endpoint : str
        The base URL endpoint for the Azure OpenAI service.
    api_version : str
        The API version to use when making requests, e.g. '2023-05-15'.

    Methods:
    -------
    - get_response(prompt, model, **kwargs):
        Sends a prompt to the Azure OpenAI language model and retrieves the generated response.
    """

    def __init__(self, api_key: str, endpoint: str, api_version: str):
        """
        Initializes the OpenAILLMClient with the required authentication and API configuration.

        Parameters:
        ----------
        - api_key (str):
            The API key for authenticating with the Azure OpenAI service.
        - endpoint (str):
            The endpoint URL for the Azure OpenAI service.
        - api_version (str):
            The API version to use when making API requests.
        """
        openai.api_type = "azure"
        openai.api_key = api_key
        openai.azure_endpoint = endpoint
        openai.api_version = api_version

    def get_response(
        self, prompt: str, model: str, **kwargs
    ) -> tuple[str, SimpleNamespace]:
        """
        Sends a prompt to the Azure OpenAI language model and retrieves the response.

        Parameters:
        ----------
        - prompt (str):
            The input text prompt to send to the language model.
        - model (str):
            The deployment name of the Azure OpenAI model to use.
        - **kwargs:
            Additional keyword arguments for the OpenAI API call, such as:
                - temperature (float): Controls the randomness of the output (default is 0).
                - max_tokens (int): The maximum number of tokens to generate (default is 500).
                - verbose (bool): If True, prints the prompt and response for debugging (default is False).

        Returns:
        -------
        - tuple[str, object]:
            A tuple containing:
                - The model's generated response (str).
                - The usage object detailing token usage (object).

        Raises:
        ------
        - openai.error.OpenAIError:
            If the API request fails.
        """
        # Extract parameters or set defaults
        temperature = kwargs.get("temperature", 0)
        max_tokens = kwargs.get("max_tokens", 500)
        verbose = kwargs.get("verbose", False)

        if verbose:
            print(f"Prompt:\n{prompt}\n")

        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if verbose:
            print("\n=== LLM Response ===")
            print(f"{response.choices[0].message.content}\n")

        content = response.choices[0].message.content

        if response.usage:
            usage_obj = SimpleNamespace(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )
        else:
            usage_obj = SimpleNamespace(
                prompt_tokens=len(prompt.split()),
                completion_tokens=(len(content.split()) if content else 0),
                total_tokens=len(prompt.split())
                + (len(content.split()) if content else 0),
            )

        return (content.strip() if content else ""), usage_obj


class TogetherLLMClient(LLMClient):
    """
    Client for interacting with Together AI language models.

    This class handles communication with the Together AI API, allowing you to send prompts
    and receive responses from various language models.

    Attributes:
    ----------
    - client (Together):
        An instance of the Together client initialized with the API key.

    Methods:
    -------
    - get_response(prompt, model, **kwargs):
        Sends a prompt to the Together AI language model and retrieves the response.
    """

    def __init__(self, api_key: str):
        """
        Initializes the TogetherLLMClient with the provided API key.

        Parameters:
        ----------
        - api_key (str):
            The API key for the Together AI service.

        Example:
        -------
        >>> client = TogetherLLMClient(api_key='your_api_key')
        """
        self.client = Together(api_key=api_key)

    def get_response(
        self, prompt: str, model: str, **kwargs
    ) -> tuple[str, SimpleNamespace]:
        """
        Sends a prompt to the Together AI language model and retrieves the response.

        Parameters:
        ----------
        - prompt (str):
            The input text prompt to send to the language model.
        - model (str):
            The identifier of the Together AI model to use.
        - **kwargs:
            Additional keyword arguments for the Together AI API call, such as:
                - temperature (float): Controls the randomness of the output (default is 0.7).
                - max_tokens (int): The maximum number of tokens to generate (default is 500).
                - verbose (bool): If True, prints the prompt and response for debugging (default is False).

        Returns:
        -------
        - str:
            The language model's response to the prompt.

        Raises:
        ------
        - Exception:
            If the API request fails.
        """
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 500)
        verbose = kwargs.get("verbose", False)

        if verbose:
            print(f"Prompt:\n{prompt}\n")

        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if verbose:
            print(f"Generation:\n{response.choices[0].message.content}\n")

        content = response.choices[0].message.content.strip()

        # Create a simple usage object (Together AI doesn't always provide detailed token usage)
        usage_obj = SimpleNamespace(
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(content.split()),
            total_tokens=len(prompt.split()) + len(content.split()),
        )

        return content, usage_obj


class VLLMLLMClient(LLMClient):
    """
    Client for interacting with open-source language models using vLLM.

    This class manages local inference with open-source models using vLLM,
    which provides efficient inference for large language models. It handles
    model loading, inference, and response formatting.

    Attributes:
    ----------
    - llm (vllm.LLM):
        The vLLM LLM instance used for inference.
    - model_path (str):
        The path or HuggingFace model ID of the loaded model.

    Methods:
    -------
    - get_response(prompt, model, **kwargs):
        Sends a prompt to the local language model and retrieves the response.
    """

    def __init__(self, model_path: str, **kwargs):
        """
        Initializes the VLLMLLMClient with the specified model.

        Parameters:
        ----------
        - model_path (str):
            The path to the model or HuggingFace model ID.
        - **kwargs:
            Additional keyword arguments for vLLM initialization, such as:
                - dtype (str): Data type for model weights (e.g., 'float16', 'bfloat16').
                - gpu_memory_utilization (float): Target GPU memory utilization (0.0 to 1.0).
                - max_model_len (int): Maximum sequence length.

        Raises:
        ------
        - ImportError:
            If vLLM is not installed.
        - RuntimeError:
            If model loading fails.

        Example:
        -------
        >>> client = VLLMLLMClient(model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0", dtype="float16")
        """
        if not VLLM_AVAILABLE:
            raise ImportError(
                "vLLM is not installed. Please install it with 'pip install vllm'."
            )

        self.model_path = model_path
        try:
            self.llm = LLM(model=model_path, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to load model with vLLM: {e}")

    def get_response(
        self, prompt: str, model: str, **kwargs
    ) -> tuple[str, SimpleNamespace]:
        """
        Sends a prompt to the vLLM model and retrieves the response.

        Parameters:
        ----------
        - prompt (str):
            The input text prompt to send to the language model.
        - model (str):
            Ignored for vLLM as the model is already loaded during initialization.
        - **kwargs:
            Additional keyword arguments for the vLLM sampling, such as:
                - temperature (float): Controls the randomness of the output (default is 0.7).
                - max_tokens (int): The maximum number of tokens to generate (default is 500).
                - verbose (bool): If True, prints the prompt and response for debugging (default is False).

        Returns:
        -------
        - tuple[str, object]:
            A tuple containing:
                - The model's generated response (str).
                - A simple usage object with estimated token counts (dict).

        Example:
        -------
        >>> response, usage = client.get_response(
        ...     prompt="What is the capital of France?",
        ...     temperature=0.7,
        ...     max_tokens=100
        ... )
        >>> print(response)
        "The capital of France is Paris."
        """
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 500)
        verbose = kwargs.get("verbose", False)

        if verbose:
            print(f"Prompt:\n{prompt}\n")

        # Create sampling parameters for vLLM
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Generate response with vLLM
        outputs = self.llm.generate(prompt, sampling_params)

        # Extract the generated text
        generated_text = outputs[0].outputs[0].text.strip()

        if verbose:
            print(f"Generation:\n{generated_text}\n")

        # Create a simple usage object (vLLM doesn't provide detailed token usage)
        # This is a rough estimate for compatibility with other clients

        usage_obj = SimpleNamespace(
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(generated_text.split()),
            total_tokens=len(prompt.split()) + len(generated_text.split()),
        )

        return generated_text, usage_obj


def get_llm_client(
    provider: str, config: dict, model: Optional[str] = None
) -> LLMClient:
    """
    Factory function to instantiate an LLM client based on the specified provider.

    Parameters
    ----------
    provider : str
        The name of the language model provider. Supported values are:
            - 'azure': For Azure OpenAI models.
            - 'openai': For standard OpenAI models.
            - 'together': For Together AI models.
            - 'vllm': For open-source models using vLLM.

    config : dict
        A dictionary containing configuration parameters required by the selected provider.

        For **Azure OpenAI**:
            - 'api_key':       Azure API key.
            - 'endpoint':      Azure OpenAI endpoint URL.
            - 'api_version':   API version (e.g., '2023-05-15').

        For **OpenAI**:
            - 'api_key':       OpenAI API key (e.g., from OPENAI_API_KEY environment variable).

        For **Together AI**:
            - 'api_key':       Together AI API key.

        For **vLLM**:
            - 'model_path':    Path to the model or HuggingFace model ID.
            - 'dtype':         (optional) Data type for model weights (e.g., 'float16').
            - 'gpu_memory_utilization': (optional) Target GPU memory utilization (0.0 to 1.0).
            - 'max_model_len': (optional) Maximum sequence length.

    model : str, optional
        The model name to use. For vLLM, this can be used instead of config["model_path"].
        This allows using the model_name from the scenario directly.

    Returns
    -------
    LLMClient
        An instance of one of the following, depending on provider:
            - AzureOpenAILLMClient
            - OpenAILLMClient (standard OpenAI)
            - TogetherLLMClient
            - VLLMLLMClient

    Raises
    ------
    ValueError
        If an unknown provider is specified.
    ImportError
        If vLLM is requested but not installed.
    """
    if provider.lower() == "azure":
        return AzureOpenAILLMClient(
            api_key=config["api_key"],
            endpoint=config["endpoint"],
            api_version=config["api_version"],
        )
    elif provider == "openai":
        return OpenAILLMClient(api_key=config["api_key"])
    elif provider == "together":
        return TogetherLLMClient(api_key=config["api_key"])
    elif provider == "vllm":
        if not VLLM_AVAILABLE:
            raise ImportError(
                "vLLM is not available on this system. This could be due to installation issues or platform compatibility.\n"
                "vLLM may not work on Windows without WSL. Please consider using a different provider like 'azure', 'openai', or 'together'."
            )
        # Extract required parameters
        # Use model_name from scenario if provided in the function call
        # This allows users to specify the model in the scenario like other providers
        model_path = model if model else config["model_path"]

        # Extract optional parameters
        kwargs = {}
        # Define supported parameters based on successful Jean Zay configuration
        supported_params = [
            "device",
            "dtype",
            "enforce_eager",
            "disable_async_output_proc",
            "tensor_parallel_size",
            "enable_prefix_caching",
        ]

        for key in supported_params:
            if key in config:
                # Convert string 'true'/'false' to boolean for enable_prefix_caching
                if key == "enable_prefix_caching" and config[key] in ["true", "false"]:
                    kwargs[key] = config[key].lower() == "true"
                else:
                    kwargs[key] = config[key]

        return VLLMLLMClient(model_path=model_path, **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")
