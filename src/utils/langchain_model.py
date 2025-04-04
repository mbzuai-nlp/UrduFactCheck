import os
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from .logging import get_logger

logger = get_logger()

{
    "openai": [
        "gpt-4o",
        "gpt-4o-mini",
        "o1",
        "o1-mini",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-3.5-turbo",
        "chatgpt-4o-latest",
    ],
    "anthropic": [
        "claude-3-7-sonnet-latest",
        "claude-3-5-sonnet-latest",
        "claude-3-5-haiku-latest",
        "claude-3-opus-latest",
    ],
}


class ChatModel:
    def __new__(cls, provider: str = "", model_name: str = "", timeout: int = 20):
        # Get the model from the environment. If it doesn't exist - apply openai with gpt-4o
        provider = os.getenv("MODEL_PROVIDER", provider.lower())
        model_name = os.getenv("MODEL_NAME", model_name.lower())
        timeout = os.getenv("TIMEOUT", timeout)

        if not provider:
            logger.error("Provider not specified")
            raise ValueError("Provider not specified")
        if not model_name:
            logger.error("Model name not specified")
            raise ValueError("Model name not specified")
        if not timeout:
            logger.error("Timeout not specified")
            raise ValueError("Timeout not specified")

        logger.info(f"Using model: {provider} {model_name}")

        if provider == "openai":
            return ChatOpenAI(model=model_name, timeout=timeout)
        elif provider == "anthropic":
            return ChatAnthropic(model_name=model_name, timeout=timeout)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
