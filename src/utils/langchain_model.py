import os
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_aws import ChatBedrock
from langchain_mistralai import ChatMistralAI
from langchain_cohere import ChatCohere
from langchain_google_genai import ChatGoogleGenerativeAI

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
        "claude-3-5-sonnet-latest",
        "claude-3-5-haiku-latest",
        "claude-3-opus-latest",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ],
    "huggingface": [
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct",
        "microsoft/phi-4",
        "microsoft/Phi-3.5-mini-instruct",
    ],
    "bedrock": ["us.meta.llama3-3-70b-instruct-v1:0"],
    "mistral": [
        "mistral-large-latest",
        "mistral-small-latest",
        "ministral-3b-latest",
        "ministral-8b-latest",
    ],
    "cohere": [
        "command-r7b-12-2024",
        "command-r-plus",
        "command-r",
        "command",
        "command-nightly",
        "command-light",
        "command-light-nightly",
        "c4ai-aya-expanse-8b",
        "c4ai-aya-expanse-32b",
    ],
    "google-generative-ai": [
        "gemini-1.5-flash",
        "gemini-2.0-flash",
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
        elif provider == "huggingface":
            return ChatHuggingFace(
                llm=HuggingFaceEndpoint(
                    repo_id=model_name,
                    timeout=timeout,
                ),
            )
        elif provider == "bedrock":
            return ChatBedrock(
                model=model_name,
                aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
                aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            )
        elif provider == "mistral":
            return ChatMistralAI(
                model_name=model_name,
                timeout=timeout,
            )
        elif provider == "cohere":
            return ChatCohere(
                model=model_name,
                timeout_seconds=timeout,
            )
        elif provider == "google-generative-ai":
            return ChatGoogleGenerativeAI(
                model=model_name,
                timeout=timeout,
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
