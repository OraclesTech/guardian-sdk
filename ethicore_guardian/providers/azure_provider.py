"""
Ethicore Engine™ - Guardian SDK — Azure OpenAI Provider
Version: 1.0.0

Wraps Azure OpenAI clients (openai.AzureOpenAI / openai.AsyncAzureOpenAI)
with three-layer agentic protection.

Azure OpenAI uses the standard OpenAI Python SDK with Azure-specific
authentication (azure_endpoint, api_version, AD token).  The wire format
for completions and tool calls is identical to OpenAI, so this provider
delegates all protection logic to the OpenAI provider and adds only:
  - Explicit Azure client detection (base_url contains .azure.com, or
    class name contains 'azure')
  - provider_name = 'azure' for audit log labeling

Layer 1: Prompt scan via guardian.analyze()
Layer 2: Inbound tool result scan via scan_tool_output() (indirect injection)
Layer 3: Outbound tool call scan via scan_tool_call() (dangerous invocations)

All three layers include multilingual support — non-English payloads are
detected via MultilingualSemanticAnalyzer wired into scan_tool_output() and
scan_tool_call() in guardian.py.

Principle 14 (Divine Safety): Azure-hosted models are trusted endpoints but
the content flowing through them is not.

Copyright © 2026 Oracles Technologies LLC
All Rights Reserved
"""

from __future__ import annotations

from typing import Any

from .base_provider import ProviderError
from .openai_provider import (
    OpenAIProvider,
    ProtectedOpenAIClient,
    ThreatBlockedException,
    ThreatChallengeException,
    ToolOutputBlockedException,
    AgentToolBlockedException,
)

# Base URL substrings that identify Azure OpenAI endpoints
_AZURE_URL_PATTERNS = (
    ".openai.azure.com",
    ".azure.com",
    "azure.openai.com",
    "cognitiveservices.azure.com",
)


# ==============================================================================
# PROVIDER
# ==============================================================================

class AzureOpenAIProvider(OpenAIProvider):
    """
    Guardian provider for Azure OpenAI (openai.AzureOpenAI /
    openai.AsyncAzureOpenAI).

    Inherits all three-layer protection logic from OpenAIProvider; overrides
    only client detection and provider identity.
    """

    def __init__(self, guardian_instance: Any) -> None:
        super().__init__(guardian_instance)
        self.provider_name = "azure"

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    @staticmethod
    def _is_azure_client(client: Any) -> bool:
        """
        Return True if *client* is an Azure OpenAI client.

        Checks:
          1. base_url attribute contains a known Azure endpoint pattern
          2. Class name contains 'azure' (e.g. openai.lib.azure.AzureOpenAI)
        """
        base_url = str(getattr(client, "base_url", "") or "").lower()
        if any(p in base_url for p in _AZURE_URL_PATTERNS):
            return True
        client_type = str(type(client)).lower()
        return "azure" in client_type

    # ------------------------------------------------------------------
    # Wrapping
    # ------------------------------------------------------------------

    def wrap_client(self, client: Any) -> ProtectedOpenAIClient:
        """
        Wrap an Azure OpenAI client with Guardian protection.

        Accepts both openai.AzureOpenAI and openai.AsyncAzureOpenAI.
        Returns a ProtectedOpenAIClient — the wire format is identical to
        standard OpenAI so full three-layer protection applies unchanged.
        """
        if not (self._is_azure_client(client) or self._is_openai_client(client)):
            raise ProviderError(
                f"Expected AzureOpenAI client, got {type(client).__name__}. "
                "Pass an openai.AzureOpenAI or openai.AsyncAzureOpenAI instance."
            )
        return ProtectedOpenAIClient(client, self.guardian)


# ==============================================================================
# FACTORY
# ==============================================================================

def create_protected_azure_client(
    azure_client: Any,
    guardian_api_key: str,
) -> ProtectedOpenAIClient:
    """
    Convenience factory: wrap an existing AzureOpenAI client with Guardian.

    Args:
        azure_client:     openai.AzureOpenAI or openai.AsyncAzureOpenAI instance.
        guardian_api_key: Ethicore Guardian API key (ETHICORE_API_KEY).

    Returns:
        ProtectedOpenAIClient with full three-layer agentic protection.

    Example::

        from openai import AzureOpenAI
        from ethicore_guardian.providers.azure_provider import create_protected_azure_client

        az = AzureOpenAI(
            azure_endpoint="https://my-resource.openai.azure.com/",
            api_version="2024-12-01",
            api_key=os.environ["AZURE_OPENAI_KEY"],
        )
        protected = create_protected_azure_client(az, guardian_api_key="eg-sk-...")
        response = protected.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )
    """
    from ..guardian import Guardian  # local import avoids circular dependency
    guardian = Guardian(api_key=guardian_api_key)
    provider = AzureOpenAIProvider(guardian)
    return provider.wrap_client(azure_client)


# Re-export exception classes so callers only need one import
__all__ = [
    "AzureOpenAIProvider",
    "create_protected_azure_client",
    "ThreatBlockedException",
    "ThreatChallengeException",
    "ToolOutputBlockedException",
    "AgentToolBlockedException",
]
