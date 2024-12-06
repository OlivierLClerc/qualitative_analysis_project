# model_interaction.py
from abc import ABC, abstractmethod
import openai
from together import Together

class LLMClient(ABC):
    """
    Abstract base class for language model clients.

    This class defines the interface for interacting with different language model providers.
    Subclasses must implement the `get_response` method to communicate with the specific API.

    Methods:
        get_response(prompt, model, **kwargs):
            Sends a prompt to the language model and retrieves the response.
    """

    @abstractmethod
    def get_response(self, prompt, model, **kwargs):
        """
        Sends a prompt to the language model and retrieves the response.

        Parameters:
            prompt (str): The input text prompt to send to the language model.
            model (str): The identifier of the language model to use.
            **kwargs: Additional keyword arguments specific to the language model API.

        Returns:
            str: The language model's response to the prompt.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        pass

class OpenAILLMClient(LLMClient):
    """
    Client for interacting with Azure OpenAI language models.

    This class handles communication with the Azure OpenAI API, allowing you to send prompts
    and receive responses from models like GPT-3 and GPT-4.

    Attributes:
        api_key (str): The API key for authenticating with the Azure OpenAI service.
        endpoint (str): The base URL endpoint for the Azure OpenAI service.
        api_version (str): The API version to use when making requests.

    Methods:
        get_response(prompt, model, **kwargs):
            Sends a prompt to the Azure OpenAI language model and retrieves the response.
    """

    def __init__(self, api_key, endpoint, api_version):
        """
        Initializes the AzureLLMClient with authentication and configuration details.

        Parameters:
            api_key (str): The API key for the Azure OpenAI service.
            endpoint (str): The endpoint URL for the Azure OpenAI service.
            api_version (str): The API version to use (e.g., '2023-05-15').

        Example:
            client = AzureLLMClient(
                api_key='your_api_key',
                endpoint='https://your-endpoint.openai.azure.com/',
                api_version='2023-05-15'
            )
        """
        openai.api_type = "azure"
        openai.api_key = api_key
        openai.api_base = endpoint
        openai.api_version = api_version

    def get_response(self, prompt, model, **kwargs):
        """
        Sends a prompt to the Azure OpenAI language model and retrieves the response.

        Parameters:
            prompt (str): The input text prompt to send to the language model.
            model (str): The deployment name of the Azure OpenAI model to use.
            **kwargs: Additional keyword arguments for the OpenAI API call, such as:
                - temperature (float): Controls the randomness of the output (default is 0).
                - max_tokens (int): The maximum number of tokens to generate (default is 500).
                - verbose (bool): If True, prints the prompt and response for debugging (default is False).

        Returns:
            str: The language model's response to the prompt.

        Raises:
            openai.error.OpenAIError: If the API request fails.

        Example:
            response = client.get_response(
                prompt="Hello, how are you?",
                model="gpt-4",
                temperature=0.5,
                max_tokens=100
            )
        """
        # Extract parameters or set defaults
        temperature = kwargs.get('temperature', 0)
        max_tokens = kwargs.get('max_tokens', 500)
        verbose = kwargs.get('verbose', False)

        if verbose:
            print(f"Prompt:\n{prompt}\n")

        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )

        if verbose:
            print(f"Generation:\n{response.choices[0].message.content}\n")

        return response.choices[0].message.content.strip()

class TogetherLLMClient(LLMClient):
    """
    Client for interacting with Together AI language models.

    This class handles communication with the Together AI API, allowing you to send prompts
    and receive responses from various language models.

    Attributes:
        client (Together): An instance of the Together client initialized with the API key.

    Methods:
        get_response(prompt, model, **kwargs):
            Sends a prompt to the Together AI language model and retrieves the response.
    """

    def __init__(self, api_key):
        """
        Initializes the TogetherLLMClient with the provided API key.

        Parameters:
            api_key (str): The API key for the Together AI service.

        Example:
            client = TogetherLLMClient(api_key='your_api_key')
        """
        self.client = Together(api_key=api_key)

    def get_response(self, prompt, model, **kwargs):
        """
        Sends a prompt to the Together AI language model and retrieves the response.

        Parameters:
            prompt (str): The input text prompt to send to the language model.
            model (str): The identifier of the Together AI model to use.
            **kwargs: Additional keyword arguments for the Together AI API call, such as:
                - temperature (float): Controls the randomness of the output (default is 0.7).
                - max_tokens (int): The maximum number of tokens to generate (default is 500).
                - verbose (bool): If True, prints the prompt and response for debugging (default is False).

        Returns:
            str: The language model's response to the prompt.

        Raises:
            Exception: If the API request fails.

        Example:
            response = client.get_response(
                prompt="Tell me a joke.",
                model="together/gpt-neoxt-chat-20B",
                temperature=0.9,
                max_tokens=50
            )
        """
        temperature = kwargs.get('temperature', 0.7)
        max_tokens = kwargs.get('max_tokens', 500)
        verbose = kwargs.get('verbose', False)

        if verbose:
            print(f"Prompt:\n{prompt}\n")

        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )

        if verbose:
            print(f"Generation:\n{response.choices[0].message.content}\n")

        return response.choices[0].message.content.strip()

def get_llm_client(provider, config):
    """
    Factory function to get an instance of an LLM client based on the provider.

    Parameters:
        provider (str): The name of the language model provider ('azure' or 'together').
        config (dict): A dictionary containing configuration parameters required by the provider.

    Returns:
        LLMClient: An instance of a subclass of LLMClient corresponding to the provider.

    Raises:
        ValueError: If an unknown provider is specified.

    Example:
        # For Azure OpenAI
        config = {
            'api_key': 'your_api_key',
            'endpoint': 'https://your-endpoint.openai.azure.com/',
            'api_version': '2023-05-15'
        }
        client = get_llm_client(provider='azure', config=config)

        # For Together AI
        config = {
            'api_key': 'your_api_key'
        }
        client = get_llm_client(provider='together', config=config)
    """
    if provider == 'azure':
        return OpenAILLMClient(
            api_key=config['api_key'],
            endpoint=config['endpoint'],
            api_version=config['api_version']
        )
    elif provider == 'Together':
        return TogetherLLMClient(
            api_key=config['api_key']
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")