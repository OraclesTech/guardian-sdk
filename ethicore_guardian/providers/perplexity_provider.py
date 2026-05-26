"""
Ethicore Engine™ - Guardian SDK — Perplexity Provider
Protection for Perplexity Sonar models via the OpenAI-compatible API.
Version: 1.0.0

Copyright © 2026 Oracles Technologies LLC
All Rights Reserved

Architecture note
-----------------
Perplexity's API is OpenAI-compatible.  Clients are constructed as:

    from openai import OpenAI
    client = OpenAI(api_key="pplx-...", base_url="https://api.perplexity.ai")

Detection checks for the Perplexity base URL rather than the client class name.

Web-grounded responses
-----------------------
All Sonar models search the web as part of inference — every response may
contain citations and live search results.  This is a fundamental difference
from standard LLMs: the model's effective context includes real-time external
data that the user did not directly supply.

Guardian's Layer 1 scan covers the *input prompt* to catch prompt injection and
jailbreak attempts before they reach Perplexity.  Guardian does NOT currently
scan the web search results Perplexity fetches internally — that surface is
outside the API call boundary.  Treat Sonar responses as you would any
externally-sourced content: verify citations before acting on them in agentic
workflows.

Current model IDs (May 2026):
  sonar                 — Lightweight, cost-effective web-grounded responses
  sonar-pro             — Advanced search; complex multi-source queries
  sonar-reasoning-pro   — Chain-of-thought reasoning with web grounding
  sonar-deep-research   — Exhaustive multi-step research agent

Model IDs are living aliases — no version suffixes.

Perplexity Agent API
---------------------
Perplexity also exposes POST /v1/agent for stateful research workflows.
That endpoint is not wrapped here; use raw httpx/requests for it and pipe
inputs through guardian.analyze() manually before submission.

Source: https://docs.perplexity.ai/

Agentic coverage
----------------
Three-layer Guardian protection across the full agentic loop:

    1. Pre-request  — scan the user prompt
    2. Pre-request  — scan tool result messages before they enter context
    3. Post-response — scan tool_calls in the reply before execution

Note: Sonar models are primarily used for retrieval/research rather than
tool-calling agentic pipelines, so Layers 2 and 3 are mostly relevant when
Sonar is embedded inside a larger agent framework (e.g. LangChain).

Principle 14 (Divine Safety): fail-closed on analysis errors.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from ._agentic_guards import (
    extract_openai_tool_results,
    extract_openai_tool_calls,
    scan_inbound_tool_results,
    scan_outbound_tool_calls,
    run_sync,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Perplexity endpoint constants
# ---------------------------------------------------------------------------

PERPLEXITY_BASE_URL = "https://api.perplexity.ai"

# Current Perplexity model identifiers (May 2026)
# Source: https://docs.perplexity.ai/docs/model-cards
# All Sonar models are web-grounded by default.
PERPLEXITY_MODELS = {
    "sonar",                  # Lightweight, cost-effective; fast web-grounded answers
    "sonar-pro",              # Advanced search; complex multi-source queries
    "sonar-reasoning-pro",    # Chain-of-thought reasoning with web grounding
    "sonar-deep-research",    # Exhaustive multi-step research agent
}


# ---------------------------------------------------------------------------
# Shared exception types
# ---------------------------------------------------------------------------

class ProviderError(Exception):
    """Provider-specific configuration or import error."""


class ThreatBlockedException(Exception):
    """Raised when Guardian issues a BLOCK verdict."""

    def __init__(self, analysis_result: Any, message: str = "Threat detected and blocked") -> None:
        self.analysis_result = analysis_result
        super().__init__(message)


class ThreatChallengeException(Exception):
    """
    Raised when Guardian issues a CHALLENGE verdict in non-strict mode.

    Callers should surface a secondary verification step rather than
    hard-blocking.  In strict_mode, CHALLENGE → ThreatBlockedException.

    Principle 16 (Sacred Autonomy): preserves human agency by surfacing
    uncertainty rather than silently blocking.
    """

    def __init__(
        self, analysis_result: Any, message: str = "Request requires verification"
    ) -> None:
        self.analysis_result = analysis_result
        super().__init__(message)


class AgentToolBlockedException(ThreatBlockedException):
    """Raised when a Perplexity-issued tool call is blocked by Guardian."""

    def __init__(
        self,
        analysis_result: Any,
        tool_name: str = "",
        message: str = "Agentic tool call blocked by Guardian",
    ) -> None:
        self.tool_name = tool_name
        super().__init__(analysis_result, message)


class ToolOutputBlockedException(ThreatBlockedException):
    """Raised when a tool result message contains an injection payload."""

    def __init__(
        self,
        analysis_result: Any,
        tool_name: str = "",
        message: str = "Tool output blocked by Guardian — injection payload detected",
    ) -> None:
        self.tool_name = tool_name
        super().__init__(analysis_result, message)


# ---------------------------------------------------------------------------
# PerplexityProvider — detection and extraction logic
# ---------------------------------------------------------------------------

class PerplexityProvider:
    """
    Perplexity provider integration for Guardian SDK.

    Wraps any OpenAI-compatible client configured for the Perplexity endpoint
    and applies three layers of Guardian protection across the full agentic loop.

    Security note: Sonar models retrieve live web content as part of inference.
    Layer 1 scans the user-supplied prompt before it reaches Perplexity.
    Web-fetched content within Perplexity's inference pipeline is outside the
    API boundary and is not scanned by Guardian — apply output validation for
    high-trust agentic contexts.
    """

    def __init__(self, guardian_instance: Any) -> None:
        self.guardian = guardian_instance
        self.provider_name = "perplexity"

    def wrap_client(self, client: Any) -> "ProtectedPerplexityClient":
        """
        Wrap a Perplexity client with Guardian protection.

        Args:
            client: An OpenAI client instance configured for api.perplexity.ai.

        Returns:
            A ProtectedPerplexityClient that is a transparent drop-in replacement.
        """
        try:
            import openai  # noqa: F401
        except ImportError:
            raise ProviderError(
                "openai package not installed. "
                "Run: pip install \"ethicore-engine-guardian[perplexity]\""
            )

        if not self._is_perplexity_client(client):
            raise ProviderError(
                f"Expected an OpenAI client configured for {PERPLEXITY_BASE_URL}, "
                f"got {type(client)}.  Pass provider='perplexity' to guardian.wrap() "
                "to force Perplexity routing."
            )

        return ProtectedPerplexityClient(client, self.guardian)

    def _is_perplexity_client(self, client: Any) -> bool:
        """Return True if *client* is a Perplexity client."""
        base_url = str(getattr(client, "base_url", "") or "").lower()
        if "perplexity" in base_url or "api.perplexity.ai" in base_url:
            return True
        base_url_alt = str(getattr(client, "_base_url", "") or "").lower()
        if "perplexity" in base_url_alt:
            return True
        return False

    def extract_prompt(self, **kwargs: Any) -> str:
        """Extract the analysable prompt text from chat.completions.create() kwargs."""
        parts: List[str] = []
        messages: List[Dict[str, Any]] = kwargs.get("messages", [])
        if not messages:
            return ""

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content") or ""

            if role == "tool":
                continue

            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))

        return " ".join(parts)


# ---------------------------------------------------------------------------
# ProtectedPerplexityClient
# ---------------------------------------------------------------------------

class ProtectedPerplexityClient:
    """
    Proxy around a Perplexity client that intercepts all chat.completions.create()
    calls and applies three-layer Guardian protection.
    """

    def __init__(self, original_client: Any, guardian_instance: Any) -> None:
        self._original_client = original_client
        self._guardian = guardian_instance
        self._provider = PerplexityProvider(guardian_instance)

        if hasattr(original_client, "chat"):
            self.chat = self._create_protected_chat()

        logger.debug("🛡️  Perplexity client protection enabled")

    def _create_protected_chat(self) -> "ProtectedChat":
        return ProtectedChat(
            self._original_client.chat,
            self._guardian,
            self._provider,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original_client, name)

    def __repr__(self) -> str:
        return f"ProtectedPerplexityClient(original={repr(self._original_client)})"


# ---------------------------------------------------------------------------
# ProtectedChat → ProtectedCompletions
# ---------------------------------------------------------------------------

class ProtectedChat:
    """Proxy for client.chat — wraps the completions sub-namespace."""

    def __init__(
        self,
        original_chat: Any,
        guardian_instance: Any,
        provider: PerplexityProvider,
    ) -> None:
        self._original_chat = original_chat
        self._guardian = guardian_instance
        self._provider = provider

        if hasattr(original_chat, "completions"):
            self.completions = ProtectedCompletions(
                original_chat.completions,
                guardian_instance,
                provider,
            )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original_chat, name)


class ProtectedCompletions:
    """
    Proxy for client.chat.completions that applies Guardian's three-layer
    agentic protection on every create() call.
    """

    def __init__(
        self,
        original_completions: Any,
        guardian_instance: Any,
        provider: PerplexityProvider,
    ) -> None:
        self._original_completions = original_completions
        self._guardian = guardian_instance
        self._provider = provider

    def create(self, **kwargs: Any) -> Any:
        """Protected synchronous chat.completions.create() — all three agentic layers."""
        messages = kwargs.get("messages", [])

        prompt_text = self._provider.extract_prompt(**kwargs)
        if prompt_text and prompt_text.strip():
            analysis = run_sync(self._analyze_prompt(prompt_text, kwargs))
            self._enforce_policy(analysis, prompt_text)

        tool_results = extract_openai_tool_results(messages)
        if tool_results:
            run_sync(scan_inbound_tool_results(
                self._guardian, tool_results, ToolOutputBlockedException
            ))

        response = self._original_completions.create(**kwargs)
        tool_calls = extract_openai_tool_calls(response)
        if tool_calls:
            run_sync(scan_outbound_tool_calls(
                self._guardian, tool_calls, AgentToolBlockedException
            ))
        return response

    async def async_create(self, **kwargs: Any) -> Any:
        """Protected async chat.completions.create() — all three agentic layers."""
        messages = kwargs.get("messages", [])

        prompt_text = self._provider.extract_prompt(**kwargs)
        if prompt_text and prompt_text.strip():
            analysis = await self._analyze_prompt(prompt_text, kwargs)
            self._enforce_policy(analysis, prompt_text)

        tool_results = extract_openai_tool_results(messages)
        if tool_results:
            await scan_inbound_tool_results(
                self._guardian, tool_results, ToolOutputBlockedException
            )

        response = await self._original_completions.create(**kwargs)
        tool_calls = extract_openai_tool_calls(response)
        if tool_calls:
            await scan_outbound_tool_calls(
                self._guardian, tool_calls, AgentToolBlockedException
            )
        return response

    async def _analyze_prompt(self, prompt_text: str, request_kwargs: Dict[str, Any]) -> Any:
        context: Dict[str, Any] = {
            "api_call": "perplexity.chat.completions.create",
            "model": request_kwargs.get("model", "sonar-pro"),
            "max_tokens": request_kwargs.get("max_tokens"),
            "temperature": request_kwargs.get("temperature"),
            "request_size": len(prompt_text),
        }
        return await self._guardian.analyze(prompt_text, context)

    def _enforce_policy(self, analysis: Any, prompt_text: str) -> None:
        action = getattr(analysis, "recommended_action", None)
        threat_level = getattr(analysis, "threat_level", "UNKNOWN")
        reasoning = getattr(analysis, "reasoning", [])
        reason_str = ", ".join(reasoning[:2]) if reasoning else "see analysis"

        if action == "BLOCK":
            logger.warning("🚨 BLOCKED Perplexity request — %s: %.100s…", threat_level, prompt_text)
            raise ThreatBlockedException(
                analysis_result=analysis,
                message=f"Request blocked: {threat_level} threat detected. Reasons: {reason_str}",
            )
        elif action == "CHALLENGE":
            logger.warning("⚠️  CHALLENGE Perplexity request — %s: %.100s…", threat_level, prompt_text)
            if self._guardian.config.strict_mode:
                raise ThreatBlockedException(
                    analysis_result=analysis,
                    message=f"Request blocked (strict mode — CHALLENGE): {threat_level} threat.",
                )
            else:
                raise ThreatChallengeException(
                    analysis_result=analysis,
                    message=f"Request requires verification: {threat_level} threat level.",
                )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original_completions, name)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def create_protected_perplexity_client(
    api_key: str,
    guardian_api_key: str,
    base_url: str = PERPLEXITY_BASE_URL,
    **openai_kwargs: Any,
) -> ProtectedPerplexityClient:
    """
    Create a Guardian-protected Perplexity client in one step.

    Args:
        api_key:           Perplexity API key (starts with "pplx-...").
        guardian_api_key:  Guardian/Ethicore API key.
        base_url:          Perplexity endpoint (default: https://api.perplexity.ai).
        **openai_kwargs:   Extra kwargs forwarded to openai.OpenAI().

    Returns:
        A ProtectedPerplexityClient ready for use as a drop-in replacement.

    Example::

        client = create_protected_perplexity_client(
            api_key="pplx-...",
            guardian_api_key="eg-sk-...",
        )
        response = client.chat.completions.create(
            model="sonar-pro",
            messages=[{"role": "user", "content": "What happened in AI this week?"}],
        )
    """
    try:
        import openai
    except ImportError:
        raise ProviderError(
            "openai package not installed. "
            "Run: pip install \"ethicore-engine-guardian[perplexity]\""
        )

    perplexity_client = openai.OpenAI(
        api_key=api_key,
        base_url=base_url,
        **openai_kwargs,
    )

    from ..guardian import Guardian

    guardian = Guardian(api_key=guardian_api_key)
    return guardian.wrap(perplexity_client, provider="perplexity")


__all__ = [
    "PerplexityProvider",
    "ProtectedPerplexityClient",
    "ProtectedChat",
    "ProtectedCompletions",
    "create_protected_perplexity_client",
    "PERPLEXITY_BASE_URL",
    "PERPLEXITY_MODELS",
    "ThreatBlockedException",
    "ThreatChallengeException",
    "AgentToolBlockedException",
    "ToolOutputBlockedException",
    "ProviderError",
]
