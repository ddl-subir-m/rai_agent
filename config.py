import os
from dotenv import load_dotenv

load_dotenv()

EMPTY_MESSAGE_LIMIT = 1
LLM_CONFIG = {"config_list": [{"model": "gpt-4o", "api_key": os.getenv("OPENAI_API_KEY")}]}