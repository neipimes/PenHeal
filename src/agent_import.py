from dataclasses import dataclass, field
from typing import Any, List, Optional, cast
import os
import sys

try:
    import yaml
except Exception:
    yaml = None

from langchain_openai import OpenAIEmbeddings, ChatOpenAI

module_mapping = {
    "gpt-4": {
        "config_name": "GPT4ConfigClass",
        "module_name": "chatgpt_api",
        "class_name": "ChatGPTAPI",
    },
    "gpt-4-turbo": {
        "config_name": "GPT4Turbo",
        "module_name": "chatgpt_api",
        "class_name": "ChatGPTAPI",
    },
    "gpt-4-o": {
        "config_name": "GPT4O",
        "module_name": "chatgpt_api",
        "class_name": "ChatGPTAPI",
    },
    "gpt-3.5-turbo-16k": {
        "config_name": "GPT35Turbo16kConfigClass",
        "module_name": "chatgpt_api",
        "class_name": "ChatGPTAPI",
    },
    "gpt4all": {
        "config_name": "GPT4ALLConfigClass",
        "module_name": "gpt4all_api",
        "class_name": "GPT4ALLAPI",
    },
    "titan": {
        "config_name": "TitanConfigClass",
        "module_name": "titan_api",
        "class_name": "TitanAPI",
    },
    "azure-gpt-3.5": {
        "config_name": "AzureGPT35ConfigClass",
        "module_name": "azure_api",
        "class_name": "AzureGPTAPI",
    },
    "gemini-1.0": {
        "config_name": "Gemini10ConfigClass",
        "module_name": "gemini_api",  # Assuming you'll create gemini_api.py
        "class_name": "GeminiAPI",  # Assuming class name will be GeminiAPI
    },
    "gemini-1.5": {
        "config_name": "Gemini15ConfigClass",
        "module_name": "gemini_api",  # Assuming you'll create gemini_api.py
        "class_name": "GeminiAPI",  # Assuming class name will be GeminiAPI
    },
    "deepseek-v4-flash": {
        "config_name": "DeepSeekV4FlashConfig",
        "module_name": "openrouter_api",
        "class_name": "OpenRouterAPI"
    },
    "deepseek-v4-pro": {
        "config_name": "DeepSeekV4ProConfig",
        "module_name": "openrouter_api",
        "class_name": "OpenRouterAPI"
    }
}

@dataclass
class OpenAI_API_Endpoint:
    model: str
    api_key: str
    api_url: str = os.getenv("OPENAI_BASEURL", "https://api.openai.com/v1")
    
@dataclass
class OpenRouter_API_Endpoint:
    model_name: str
    api_key: str
    reasoning: dict[str, str]
    openrouter_provider: dict[str, List[str]]
    temperature: float

@dataclass
class DeepSeekV4FlashConfig(OpenRouter_API_Endpoint):
    model_name: str = "deepseek/deepseek-v4-flash"
    api_key: str = os.getenv("OPENROUTER_API_KEY", "KEY NOT SET")
    reasoning: dict[str, str] = field(default_factory=lambda: {"effort": "high", "summary": "auto"})
    openrouter_provider: dict[str, List[str]] = field(default_factory=lambda: {"order": ["SiliconFlow", "AtlasCloud", "DeepSeek"]})
    temperature: float = 0.0

@dataclass
class DeepSeekV4ProConfig(OpenRouter_API_Endpoint):
    model_name: str = "deepseek/deepseek-v4-pro"
    api_key: str = os.getenv("OPENROUTER_API_KEY", "KEY NOT SET")
    reasoning: dict[str, str] = field(default_factory=lambda: {"effort": "high", "summary": "auto"})
    openrouter_provider: dict[str, List[str]] = field(default_factory=lambda: {"order": ["DeepSeek", "AtlasCloud", "SiliconFlow"]})
    temperature: float = 0.0

@dataclass
class GPT4ConfigClass:
    model: str = "gpt-4"
    api_base: str = os.getenv("OPENAI_BASEURL", "https://api.openai.com/v1")
    # set up the openai key
    openai_key = os.getenv("OPENAI_API_KEY", None)
    if openai_key is None:
        print(
            "Your OPENAI_API_KEY is not set. Please set it in the environment variable."
        )
    error_wait_time: float = 20
    is_debugging: bool = False

@dataclass
class GPT35Turbo16kConfigClass:
    model: str = "gpt-3.5-turbo-16k"
    api_base: str = os.getenv("OPENAI_BASEURL", "https://api.openai.com/v1")
    # set up the openai key
    openai_key = os.getenv("OPENAI_API_KEY", None)
    if openai_key is None:
        print(
            "Your OPENAI_API_KEY is not set. Please set it in the environment variable."
        )
    error_wait_time: float = 20
    is_debugging: bool = False

@dataclass
class GPT4Turbo:
    model: str = "gpt-4-1106-preview"
    api_base: str = os.getenv("OPENAI_BASEURL", "https://api.openai.com/v1")
    # set up the openai key
    openai_key = os.getenv("OPENAI_API_KEY", None)
    if openai_key is None:
        print(
            "Your OPENAI_API_KEY is not set. Please set it in the environment variable."
        )
    error_wait_time: float = 10
    is_debugging: bool = False

@dataclass
class GPT4O:
    model: str = "gpt-4o-2024-05-13"
    api_base: str = os.getenv("OPENAI_BASEURL", "https://api.openai.com/v1")
    # set up the openai key
    openai_key = os.getenv("OPENAI_API_KEY", None)
    if openai_key is None:
        print(
            "Your OPENAI_API_KEY is not set. Please set it in the environment variable."
        )
    error_wait_time: float = 10
    is_debugging: bool = False

@dataclass
class GPT4ALLConfigClass:
    model: str = "mistral-7b-openorca.Q4_0.gguf"


@dataclass
class TitanConfigClass:
    model: str = "amazon.titan-tg1-large"


@dataclass
class AzureGPT35ConfigClass:
    model: str = "gpt-35-turbo"
    api_type: str = "azure"
    api_base: str = "https://docs-test-001.openai.azure.com/"
    openai_key = os.getenv("OPENAI_API_KEY", None)
    if openai_key is None:
        print(
            "Your OPENAI_API_KEY is not set. Please set it in the environment variable."
        )
    error_wait_time: float = 20
    is_debugging: bool = False

@dataclass
class Gemini10ConfigClass:  # New dataclass for Gemini 1.0
    model: str = "gemini-1.0-pro"
    # api_base: str = "https://api.gemini.com/v1"  # Replace with actual API base URL
    gemini_key = os.getenv(
        "GOOGLE_API_KEY", None
    )  # Assuming environment variable for API key
    if gemini_key is None:
        print(
            "Your GOOGLE_API_KEY is not set. Please set it in the environment variable."
        )
    error_wait_time: float = 20
    is_debugging: bool = False

@dataclass
class Gemini15ConfigClass:  # New dataclass for Gemini 1.5
    model: str = "gemini-1.5-pro-latest"
    # api_base: str = "https://api.gemini.com/v1"  # Replace with actual API base URL
    gemini_key = os.getenv(
        "GOOGLE_API_KEY", None
    )  # Assuming environment variable for API key
    if gemini_key is None:
        print(
            "Your GOOGLE_API_KEY is not set. Please set it in the environment variable."
        )
    error_wait_time: float = 20
    is_debugging: bool = False

@dataclass
class ChatGPTConfig:
    # model: str = "text-davinci-002-render-sha"
    model: str = "gpt-4-browsing"

    # api_base: str = "https://api.openai.com/v1"
    # set up the openai api base, default:"https://api.openai.com/v1"
    api_base: str = os.getenv("OPENAI_BASEURL", "https://api.openai.com/v1")

    log_dir: str = "logs"

    # set up the openai key
    openai_key = os.getenv("OPENAI_API_KEY", None)
    # set the user-agent below
    userAgent: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36"
    )
    # set cookie below
    cookie: str = os.getenv("CHATGPT_COOKIE", "")
    # curl command file
    curl_file: str = os.path.join(
        os.path.realpath(os.path.dirname(__file__)), "chatgpt_config_curl.txt"
    )

    if openai_key is None:
        print(
            'Your OPENAI key is not set. Please set it in the environment variable.\nIf you want to use chatGPT with no API, use "text-davinci-002-render-sha" in chat_config.py'
        )
    if cookie is None:
        print(
            "Your CHATGPT_COOKIE is not set. Please set it in the environment variable."
        )

    error_wait_time: float = 20
    is_debugging: bool = False
    proxies: dict = field(
        default_factory=lambda: {
            "http": "",
            "https": "",
        }
    )
    pinecone_api_key = os.getenv("PINECONE_API_KEY", None)


def dynamic_import(module_name, use_langfuse_logging=False) -> Any:
    if module_name in module_mapping:
        module_config_name = module_mapping[module_name]["config_name"]
        module_import_name = module_mapping[module_name]["module_name"]
        class_name = module_mapping[module_name]["class_name"]
        module_config = getattr(sys.modules[__name__], module_config_name)

        from llm_api import LLMAPI as LLM_module

        LLM_class = getattr(LLM_module, class_name)
        # initialize the class
        LLM_class_initialized = LLM_class(
            module_config, use_langfuse_logging=use_langfuse_logging
        )

        return LLM_class_initialized

    else:
        print(
            "Module not found: "
            + module_name
            + ". Falling back to use the default gpt-3.5-turbo-16k"
        )
        # fall back to gpt-3.5-turbo-16k
        LLM_class_initialized = dynamic_import("gpt-3.5-turbo-16k")
        return LLM_class_initialized


@dataclass
class RuntimeModelConfig:
    provider: str = "openai"  # openai, openrouter, lmstudio, local
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    model_name: str = "gpt-4o"
    temperature: float = 0.3
    timeout: int = 60
    is_embedding: bool = False


@dataclass
class OpenAICompatibleConfig:
    model: str = "gpt-4o"
    api_base: str = os.getenv("OPENAI_BASEURL", "https://api.openai.com/v1")
    openai_key: Optional[str] = os.getenv("OPENAI_API_KEY", None)
    log_dir: str = "logs"
    error_wait_time: float = 10
    temperature: float = 0.3
    proxies: dict = field(
        default_factory=lambda: {
            "http": "",
            "https": "",
        }
    )


@dataclass
class OpenRouterRuntimeConfig:
    model_name: str = "deepseek/deepseek-v4-pro"
    api_key: Optional[str] = os.getenv("OPENROUTER_API_KEY", None)
    reasoning: dict[str, str] = field(default_factory=lambda: {"effort": "high", "summary": "auto"})
    openrouter_provider: dict[str, List[str]] = field(default_factory=lambda: {"order": ["DeepSeek"]})
    temperature: float = 0.0
    error_wait_time: float = 10
    log_dir: str = "logs"


def load_config_from_file(path: str) -> dict:
    if yaml is None:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return {}


def create_client(config: RuntimeModelConfig, client_type: str = "llm"):
    """Create either an LLM client (returns an LLMAPI-derived instance) or an embeddings client.

    client_type: "llm" or "embeddings"
    """
    # normalize
    if client_type not in ("llm", "embeddings"):
        raise ValueError("client_type must be 'llm' or 'embeddings'")

    # For LLMs, return an LLMAPI wrapper if using OpenAI-like providers
    if client_type == "llm":
        # reuse existing LLMAPI for OpenAI-compatible endpoints
        if config.provider in ("openai", "lmstudio"):
            from llm_api import LLMAPI as OpenAILLMAPI

            if config.api_key:
                os.environ["OPENAI_API_KEY"] = config.api_key
            if config.api_base:
                os.environ["OPENAI_BASEURL"] = config.api_base
            cfg = OpenAICompatibleConfig(
                model=config.model_name,
                api_base=config.api_base or os.getenv("OPENAI_BASEURL") or "https://api.openai.com/v1",
                openai_key=config.api_key or os.getenv("OPENAI_API_KEY"),
                error_wait_time=10,
                temperature=config.temperature,
            )
            return OpenAILLMAPI(cast(Any, cfg))
        if config.provider == "openrouter":
            from openrouter_llm_api import LLMAPI as OpenRouterLLMAPI

            cfg = OpenRouterRuntimeConfig(
                model_name=config.model_name,
                api_key=config.api_key or os.getenv("OPENROUTER_API_KEY"),
                temperature=config.temperature,
            )
            return OpenRouterLLMAPI(cast(Any, cfg))
        raise NotImplementedError("Provider not supported for LLM: %s" % config.provider)

    # embeddings
    if client_type == "embeddings":
        if config.provider in ("openai", "lmstudio"):
            # configure OpenAIEmbeddings via env overrides
            if config.api_key:
                os.environ["OPENAI_API_KEY"] = config.api_key
            if config.api_base:
                os.environ["OPENAI_BASEURL"] = config.api_base
            # instantiate with model name when supported
            try:
                emb = OpenAIEmbeddings(model=config.model_name, chunk_size=1)
            except Exception:
                emb = OpenAIEmbeddings()
            return emb
        else:
            raise NotImplementedError("Provider not supported for embeddings: %s" % config.provider)


if __name__ == "__main__":
    # a quick local test
    # load gpt4
    gpt4: Any = dynamic_import("gpt4all", True)
    gpt4.send_new_message("hi")
