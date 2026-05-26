"""
Ethicore Engine™ - Guardian SDK - Providers Package
AI provider integrations for Guardian SDK

Supported providers:
  Local / self-hosted:
    LM Studio, llama.cpp, LocalAI, Jan.ai  — local_openai_compat_provider
    Ollama                                  — guardian_ollama_provider

  Cloud (OpenAI-compatible API):
    xAI / Grok                              — xai_provider
    DeepSeek                                — deepseek_provider
    Mistral AI                              — mistral_provider
    Perplexity                              — perplexity_provider

  Cloud (native API):
    OpenAI                                  — openai_provider
    Anthropic                               — anthropic_provider
    Google Gemini                           — gemini_provider
    Azure OpenAI                            — azure_provider
    AWS Bedrock                             — bedrock_provider

  Meta-provider:
    LiteLLM (140+ backends)                 — litellm_provider

  Framework integrations:
    LangChain callback                      — langchain_callback
"""

__version__ = "1.1.0"

# ── Local OpenAI-compatible providers ────────────────────────────────────────
from .local_openai_compat_provider import (
    LocalOpenAICompatConfig,
    LocalOpenAICompatProvider,
    LMStudioProvider,
    LlamaCppProvider,
    LocalAIProvider,
    JanAIProvider,
    ProtectedLocalClient,
    ProtectedAsyncLocalClient,
    create_protected_lmstudio_client,
    create_protected_llamacpp_client,
    create_protected_localai_client,
    create_protected_janai_client,
    ThreatBlockedException,
    ThreatChallengeException,
    ToolOutputBlockedException,
    AgentToolBlockedException,
    LocalProviderError,
)

# ── DeepSeek provider ─────────────────────────────────────────────────────────
from .deepseek_provider import (
    DeepSeekProvider,
    ProtectedDeepSeekClient,
    create_protected_deepseek_client,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODELS,
)

# ── Mistral AI provider ───────────────────────────────────────────────────────
from .mistral_provider import (
    MistralProvider,
    ProtectedMistralClient,
    create_protected_mistral_client,
    MISTRAL_BASE_URL,
    MISTRAL_MODELS,
)

# ── Perplexity provider ───────────────────────────────────────────────────────
from .perplexity_provider import (
    PerplexityProvider,
    ProtectedPerplexityClient,
    create_protected_perplexity_client,
    PERPLEXITY_BASE_URL,
    PERPLEXITY_MODELS,
)

__all__ = [
    # ── Local OpenAI-compatible ───────────────────────────────────────────────
    "LocalOpenAICompatConfig",
    "LocalOpenAICompatProvider",
    "LMStudioProvider",
    "LlamaCppProvider",
    "LocalAIProvider",
    "JanAIProvider",
    "ProtectedLocalClient",
    "ProtectedAsyncLocalClient",
    "create_protected_lmstudio_client",
    "create_protected_llamacpp_client",
    "create_protected_localai_client",
    "create_protected_janai_client",
    # ── DeepSeek ─────────────────────────────────────────────────────────────
    "DeepSeekProvider",
    "ProtectedDeepSeekClient",
    "create_protected_deepseek_client",
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_MODELS",
    # ── Mistral ───────────────────────────────────────────────────────────────
    "MistralProvider",
    "ProtectedMistralClient",
    "create_protected_mistral_client",
    "MISTRAL_BASE_URL",
    "MISTRAL_MODELS",
    # ── Perplexity ────────────────────────────────────────────────────────────
    "PerplexityProvider",
    "ProtectedPerplexityClient",
    "create_protected_perplexity_client",
    "PERPLEXITY_BASE_URL",
    "PERPLEXITY_MODELS",
    # ── Exceptions (re-exported from local_openai_compat_provider) ────────────
    "ThreatBlockedException",
    "ThreatChallengeException",
    "ToolOutputBlockedException",
    "AgentToolBlockedException",
    "LocalProviderError",
]
