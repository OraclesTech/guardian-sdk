"""
Ethicore Engine™ - Guardian SDK - Providers Package
AI provider integrations for Guardian SDK
"""

# This makes the providers directory a proper Python package.
# Providers are importable either directly from their module or from here.

__version__ = "1.0.0"

# ── Local OpenAI-compatible providers ────────────────────────────────────────
# LM Studio, llama.cpp server, LocalAI, Jan.ai — all share the same
# OpenAI-compatible REST API, covered by a single implementation.
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

__all__ = [
    # Config
    "LocalOpenAICompatConfig",
    # Base provider
    "LocalOpenAICompatProvider",
    # Named providers
    "LMStudioProvider",
    "LlamaCppProvider",
    "LocalAIProvider",
    "JanAIProvider",
    # Wrapped clients
    "ProtectedLocalClient",
    "ProtectedAsyncLocalClient",
    # Convenience factories
    "create_protected_lmstudio_client",
    "create_protected_llamacpp_client",
    "create_protected_localai_client",
    "create_protected_janai_client",
    # Exceptions
    "ThreatBlockedException",
    "ThreatChallengeException",
    "ToolOutputBlockedException",
    "AgentToolBlockedException",
    "LocalProviderError",
]
