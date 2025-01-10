"""
model_interaction.py

This module provides a unified interface for interacting with different large language model (LLM) providers, 
such as Azure OpenAI and Together AI. It abstracts API interactions to simplify sending prompts and 
retrieving responses across multiple providers.

Dependencies:
    - openai: For interacting with Azure OpenAI models.
    - together: For interacting with Together AI models.
    - abc: For defining the abstract base class.

Classes:
    - LLMClient: Abstract base class defining the interface for LLM clients.
    - OpenAILLMClient: Client for interacting with Azure OpenAI language models.
    - TogetherLLMClient: Client for interacting with Together AI language models.

Functions:
    - get_llm_client(provider, config): Factory function to instantiate the appropriate LLM client 
      based on the provider.
"""

from abc import ABC, abstractmethod
import openai
from together import Together


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
    def get_response(self, prompt, model, **kwargs):
        """
        Sends a prompt to the language model and retrieves the response.

        Parameters:
            - prompt (str): The input text prompt to send to the language model.
            - model (str): The identifier of the language model to use.
            - **kwargs: Additional keyword arguments specific to the language model API.

        Returns:
            - str: The language model's response to the prompt.

        Raises:
            - NotImplementedError: If the method is not implemented in the subclass.
        """
        pass


class OpenAILLMClient(LLMClient):
    """
    Client for interacting with Azure OpenAI language models.

    This class manages communication with the Azure OpenAI API, enabling prompt-based interactions
    with models like **GPT-3** and **GPT-4**. It handles authentication, configuration, and
    response retrieval.

    Attributes:
    ----------
    - api_key (str):
        The API key for authenticating with the Azure OpenAI service.
    - endpoint (str):
        The base URL endpoint for the Azure OpenAI service.
    - api_version (str):
        The API version to use when making requests (e.g., '2023-05-15').

    Methods:
    -------
    - get_response(prompt, model, **kwargs):
        Sends a prompt to the Azure OpenAI language model and retrieves the generated response.

    Example (Mocked):
    -----------------
    This example demonstrates how to use the client with a mocked API call for testing purposes.

    >>> from unittest.mock import patch
    >>> client = OpenAILLMClient(
    ...     api_key="fake_api_key",
    ...     endpoint="https://fake-endpoint.openai.azure.com/",
    ...     api_version="2023-05-15"
    ... )
    >>> with patch.object(client, 'get_response', return_value=("Mocked response", None)):
    ...     response, _ = client.get_response(
    ...         prompt="Tell me a joke!",
    ...         model="gpt-4",
    ...         temperature=0.5,
    ...         max_tokens=50
    ...     )
    >>> print(response)
    Mocked response
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

        Example:
        -------
        >>> client = OpenAILLMClient(
        ...     api_key="your_api_key",
        ...     endpoint="https://your-endpoint.openai.azure.com/",
        ...     api_version="2023-05-15"
        ... )
        """
        openai.api_type = "azure"
        openai.api_key = api_key
        openai.azure_endpoint = endpoint
        openai.api_version = api_version

    def get_response(self, prompt: str, model: str, **kwargs) -> tuple[str, object]:
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

        Example (Mocked):
        -----------------
        This example uses a mocked response to safely test the function.

        >>> from unittest.mock import patch
        >>> client = OpenAILLMClient(
        ...     api_key="fake_api_key",
        ...     endpoint="https://fake-endpoint.openai.azure.com/",
        ...     api_version="2023-05-15"
        ... )
        >>> with patch.object(client, 'get_response', return_value=("Mocked response", None)):
        ...     response, _ = client.get_response(
        ...         prompt="Tell me a joke!",
        ...         model="gpt-4",
        ...         temperature=0.5,
        ...         max_tokens=50
        ...     )
        >>> print(response)
        Mocked response
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
        return (content.strip() if content else ""), response.usage

    # def get_embedding(self, input_text, model, **kwargs):
    #     """
    #     Retrieves the embedding for the given input text using Azure OpenAI's embeddings endpoint.

    #     Parameters:
    #         input_text (str): The text to embed.
    #         model (str): The deployment name of the Azure OpenAI embedding model to use.
    #         **kwargs: Additional keyword arguments for the OpenAI API call.

    #     Returns:
    #         list: The embedding vector as a list of floats.

    #     Raises:
    #         openai.error.OpenAIError: If the API request fails.

    #     Example:
    #         embedding = client.get_embedding(
    #             input_text="Sample text to embed.",
    #             model="text-embedding-ada-002"
    #         )
    #     """
    #     response = openai.embeddings.create(
    #         input=input_text,
    #         model=model,
    #         **kwargs
    #     )
    #     embedding = response['data'][0]['embedding']
    #     return embedding


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

    Example (Mocked):
    -----------------
    Demonstrates how to use the Together AI client with a mocked API call for safe testing.

    >>> from unittest.mock import patch
    >>> client = TogetherLLMClient(api_key="fake_api_key")
    >>> with patch.object(client, 'get_response', return_value="Mocked Together AI response"):
    ...     response = client.get_response(
    ...         prompt="Tell me a joke.",
    ...         model="together/gpt-neoxt-chat-20B",
    ...         temperature=0.9,
    ...         max_tokens=50
    ...     )
    >>> print(response)
    Mocked Together AI response
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

    def get_response(self, prompt: str, model: str, **kwargs) -> str:
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

        Example (Mocked):
        -----------------
        Demonstrates how to safely test the method without making an actual API call.

        >>> from unittest.mock import patch
        >>> client = TogetherLLMClient(api_key="fake_api_key")
        >>> with patch.object(client, 'get_response', return_value="Mocked Together AI response"):
        ...     response = client.get_response(
        ...         prompt="Tell me a joke.",
        ...         model="together/gpt-neoxt-chat-20B",
        ...         temperature=0.9,
        ...         max_tokens=50
        ...     )
        >>> print(response)
        Mocked Together AI response
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

        return response.choices[0].message.content.strip()


def get_llm_client(provider: str, config: dict) -> LLMClient:
    """
    Factory function to instantiate an LLM client based on the specified provider.

    Parameters:
    ----------
    - provider (str):
        The name of the language model provider. Supported values are:
            - `'azure'`: For Azure OpenAI models.
            - `'together'`: For Together AI models.

    - config (dict):
        A dictionary containing configuration parameters required by the provider.

        - For **Azure OpenAI**, required keys are:
            - `'api_key'`: Azure API key.
            - `'endpoint'`: Azure OpenAI endpoint URL.
            - `'api_version'`: API version (e.g., `'2023-05-15'`).

        - For **Together AI**, required key:
            - `'api_key'`: Together AI API key.

    Returns:
    -------
    - LLMClient:
        An instance of `OpenAILLMClient` or `TogetherLLMClient` based on the provider.

    Raises:
    ------
    - ValueError:
        If an unknown provider is specified.

    Examples (Mocked):
    ------------------
    Demonstrates how to instantiate clients with mocked API calls.

    >>> from unittest.mock import patch
    >>> azure_config = {
    ...     'api_key': 'fake_azure_key',
    ...     'endpoint': 'https://fake-endpoint.azure.com/',
    ...     'api_version': '2023-05-15'
    ... }
    >>> with patch.object(OpenAILLMClient, '__init__', return_value=None) as MockAzureClient:
    ...     client = get_llm_client(provider='azure', config=azure_config)
    ...     MockAzureClient.assert_called_with(
    ...         api_key='fake_azure_key',
    ...         endpoint='https://fake-endpoint.azure.com/',
    ...         api_version='2023-05-15'
    ... )

    >>> together_config = {'api_key': 'fake_together_key'}
    >>> with patch.object(TogetherLLMClient, '__init__', return_value=None) as MockTogetherClient:
    ...     client = get_llm_client(provider='together', config=together_config)
    ...     MockTogetherClient.assert_called_with(api_key='fake_together_key')

    Example (Error Case):
    ---------------------
    >>> invalid_config = {'api_key': 'fake_key'}
    >>> get_llm_client(provider='invalid', config=invalid_config)
    Traceback (most recent call last):
        ...
    ValueError: Unknown provider: invalid
    """
    if provider.lower() == "azure":
        return OpenAILLMClient(
            api_key=config["api_key"],
            endpoint=config["endpoint"],
            api_version=config["api_version"],
        )
    elif provider.lower() == "together":
        return TogetherLLMClient(api_key=config["api_key"])
    else:
        raise ValueError(f"Unknown provider: {provider}")
