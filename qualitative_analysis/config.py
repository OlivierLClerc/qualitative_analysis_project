# config.py
import os
from dotenv import load_dotenv

load_dotenv()

MODEL_CONFIG = {
    'azure': {
        'api_key': os.getenv('AZURE_API_KEY'),
        'endpoint': os.getenv('AZURE_OPENAI_ENDPOINT'),
        'api_version': os.getenv('AZURE_API_VERSION'),
    }
}

MODEL_PRICES = {
    'gpt-4o': {'prompt': 0.0025, 'completion': 0.01},  # $2.50 per 1M input tokens, $10.00 per 1M output tokens
    'gpt-4o-mini': {'prompt': 0.00015, 'completion': 0.0006},  # $0.15 per 1M input tokens, $0.60 per 1M output tokens
}