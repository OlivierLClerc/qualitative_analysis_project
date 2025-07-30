"""
model_interaction.py

This module provides a unified interface for interacting with different large language model (LLM) providers,
including:

    - OpenAI
    - Azure OpenAI
    - Anthropic
    - Together AI
    - vLLM (for open-source models)

It abstracts API interactions to simplify sending prompts and retrieving responses across multiple providers.

Dependencies:
    - openai: For interacting with Azure OpenAI or standard OpenAI models.
    - anthropic: For interacting with Anthropic Claude models.
    - together: For interacting with Together AI models.
    - vllm: For running inference with open-source models locally.
    - abc: For defining the abstract base class.
    - types: For using SimpleNamespace to standardize usage object representation.

Classes:
    - LLMClient: Abstract base class defining the interface for LLM clients.
    - OpenAILLMClient: Client for interacting with standard OpenAI language models.
    - AzureOpenAILLMClient: Client for interacting with Azure OpenAI language models.
    - AnthropicLLMClient: Client for interacting with Anthropic Claude models.
    - TogetherLLMClient: Client for interacting with Together AI language models.
    - VLLMLLMClient: Client for interacting with open-source models using vLLM.

Functions:
    - get_llm_client(provider, config): Factory function to instantiate the appropriate LLM client 
      based on the specified provider string.
"""

from abc import ABC, abstractmethod
import openai
from anthropic import Anthropic
from together import Together
from types import SimpleNamespace
from typing import Optional
import google.generativeai as genai

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

    def get_response(
        self, prompt: str, model: str, **kwargs
    ) -> tuple[str, SimpleNamespace]:
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
        tuple[str, SimpleNamespace]
            A tuple containing:
                - The generated response text (str).
                - The usage object detailing token usage (SimpleNamespace).

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

        # Convert usage object to a SimpleNamespace for consistency with other clients
        if response.usage:
            usage_obj = SimpleNamespace(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )
        else:
            # Fallback if usage is not available
            usage_obj = SimpleNamespace(
                prompt_tokens=len(prompt.split()),  # Very rough estimate
                completion_tokens=(
                    len(content.split()) if content else 0
                ),  # Very rough estimate
                total_tokens=len(prompt.split())
                + (len(content.split()) if content else 0),
            )

        return (content.strip() if content else ""), usage_obj


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


class AnthropicLLMClient(LLMClient):
    """
    Client for interacting with Anthropic Claude language models.

    This class manages communication with the Anthropic API, enabling prompt-based
    interactions with Claude models. It handles authentication via an Anthropic API key.

    Attributes:
    ----------
    - client (Anthropic):
        An instance of the Anthropic client initialized with the API key.

    Methods:
    -------
    - get_response(prompt, model, **kwargs):
        Sends a prompt to the Anthropic language model and retrieves the response.
    """

    def __init__(self, api_key: str):
        """
        Initializes the AnthropicLLMClient with the provided API key.

        Parameters:
        ----------
        - api_key (str):
            The API key for the Anthropic service.

        Example:
        -------
        >>> client = AnthropicLLMClient(api_key='your_api_key')
        """
        self.client = Anthropic(api_key=api_key)

    def get_response(
        self, prompt: str, model: str, **kwargs
    ) -> tuple[str, SimpleNamespace]:
        """
        Sends a prompt to the Anthropic Claude model and retrieves the response.

        Parameters:
        ----------
        - prompt (str):
            The input text prompt to send to the language model.
        - model (str):
            The identifier of the Anthropic model to use (e.g., "claude-3-7-sonnet-20250219").
        - **kwargs:
            Additional keyword arguments for the Anthropic API call, such as:
                - temperature (float): Controls the randomness of the output (default is 0.0).
                - max_tokens (int): The maximum number of tokens to generate (default is 500).
                - verbose (bool): If True, prints the prompt and response for debugging (default is False).

        Returns:
        -------
        - tuple[str, SimpleNamespace]:
            A tuple containing:
                - The model's generated response (str).
                - A usage object with token counts (SimpleNamespace).

        Raises:
        ------
        - Exception:
            If the API request fails.
        """
        temperature = kwargs.get("temperature", 0.0)
        max_tokens = kwargs.get("max_tokens", 500)
        verbose = kwargs.get("verbose", False)

        if verbose:
            print(f"Prompt:\n{prompt}\n")

        # Create a message with Anthropic's API
        response = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system="You are a helpful assistant.",
            messages=[{"role": "user", "content": prompt}],
        )

        # Extract the content from the response
        # The content is a list of ContentBlock objects
        content_text = ""
        for content_block in response.content:
            # Check if the content block has a 'type' attribute and it's 'text'
            if hasattr(content_block, "type") and content_block.type == "text":
                content_text += content_block.text

        if verbose:
            print(f"Generation:\n{content_text}\n")

        # Create a usage object with token counts
        usage_obj = SimpleNamespace(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )

        return content_text.strip(), usage_obj


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
        - tuple[str, SimpleNamespace]:
            A tuple containing:
                - The language model's response to the prompt (str).
                - A usage object with token counts (SimpleNamespace).

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


class GeminiLLMClient(LLMClient):
    """
    Client for interacting with Google Gemini language models.

    This class manages communication with the Google Gemini API, enabling prompt-based
    interactions with Gemini models. It handles authentication via a Google API key.

    Attributes:
    ----------
    - api_key (str):
        The Google API key used for authentication.

    Methods:
    -------
    - get_response(prompt, model, **kwargs):
        Sends a prompt to the Gemini language model and retrieves the response.
    """

    def __init__(self, api_key: str):
        """
        Initializes the GeminiLLMClient with the provided API key.

        Parameters:
        ----------
        - api_key (str):
            The API key for the Google Gemini service.

        Example:
        -------
        >>> client = GeminiLLMClient(api_key='your_api_key')
        """
        genai.configure(api_key=api_key)

    def get_response(
        self, prompt: str, model: str, **kwargs
    ) -> tuple[str, SimpleNamespace]:
        """
        Sends a prompt to the Google Gemini model and retrieves the response.

        Parameters:
        ----------
        - prompt (str):
            The input text prompt to send to the language model.
        - model (str):
            The identifier of the Gemini model to use (e.g., "gemini-2.0-flash-001").
        - **kwargs:
            Additional keyword arguments for the Gemini API call, such as:
                - temperature (float): Controls the randomness of the output (default is 0.0).
                - max_tokens (int): The maximum number of tokens to generate (default is 500).
                - verbose (bool): If True, prints the prompt and response for debugging (default is False).

        Returns:
        -------
        - tuple[str, SimpleNamespace]:
            A tuple containing:
                - The model's generated response (str).
                - A usage object with estimated token counts (SimpleNamespace).

        Raises:
        ------
        - Exception:
            If the API request fails.
        """
        temperature = kwargs.get("temperature", 0.0)
        max_tokens = kwargs.get("max_tokens", 500)
        verbose = kwargs.get("verbose", False)

        if verbose:
            print(f"Prompt:\n{prompt}\n")

        # Create a GenerativeModel instance
        gemini_model = genai.GenerativeModel(model)

        # Generate content
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )

        # Extract the content from the response
        content_text = response.text.strip()

        if verbose:
            print(f"Generation:\n{content_text}\n")

        # Create a simple usage object (Gemini API doesn't provide token counts directly)
        # This is a rough estimate for compatibility with other clients
        usage_obj = SimpleNamespace(
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(content_text.split()),
            total_tokens=len(prompt.split()) + len(content_text.split()),
        )

        return content_text, usage_obj


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
        - tuple[str, SimpleNamespace]:
            A tuple containing:
                - The model's generated response (str).
                - A simple usage object with estimated token counts (SimpleNamespace).

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
            - 'anthropic': For Anthropic Claude models.
            - 'gemini': For Google Gemini models.
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

        For **Anthropic**:
            - 'api_key':       Anthropic API key (e.g., from ANTHROPIC_API_KEY environment variable).

        For **Gemini**:
            - 'api_key':       Google Gemini API key (e.g., from GEMINI_API_KEY environment variable).

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
            - AnthropicLLMClient
            - GeminiLLMClient
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
    elif provider == "anthropic":
        return AnthropicLLMClient(api_key=config["api_key"])
    elif provider == "gemini":
        return GeminiLLMClient(api_key=config["api_key"])
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
            "worker_cls",
            "distributed_executor_backend",
            "trust_remote_code",
            "revision",
        ]

        for key in supported_params:
            if key in config:
                # Handle type conversions for different parameter types
                if key in [
                    "enforce_eager",
                    "disable_async_output_proc",
                    "enable_prefix_caching",
                ]:
                    # These are boolean parameters
                    if isinstance(config[key], bool):
                        kwargs[key] = config[key]
                    elif isinstance(config[key], str) and config[key].lower() in [
                        "true",
                        "false",
                    ]:
                        kwargs[key] = config[key].lower() == "true"
                    else:
                        # Default to False for invalid values
                        kwargs[key] = False
                elif key == "tensor_parallel_size":
                    # This is an integer parameter
                    if isinstance(config[key], int):
                        kwargs[key] = config[key]
                    elif isinstance(config[key], str) and config[key].isdigit():
                        kwargs[key] = int(config[key])
                    else:
                        # Default to 1 for invalid values
                        kwargs[key] = 1
                else:
                    # Pass other parameters as-is
                    kwargs[key] = config[key]

        return VLLMLLMClient(model_path=model_path, **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")
