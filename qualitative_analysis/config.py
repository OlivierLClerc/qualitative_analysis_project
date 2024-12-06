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
