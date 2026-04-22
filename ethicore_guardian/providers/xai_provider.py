"""
Ethicore Engine™ - Guardian SDK — xAI / Grok Provider
Protection for xAI's Grok models via the OpenAI-compatible API.
Version: 1.0.0

Copyright © 2026 Oracles Technologies LLC
All Rights Reserved

Architecture note
-----------------
xAI's API is OpenAI-compatible.  Their clients are constructed as:

    from openai import OpenAI
    client = OpenAI(api_key="xai-...", base_url="https://api.x.ai/v1")

Detection therefore checks for the xAI base URL rather than the class name.

Agentic coverage
----------------
Unlike the base OpenAI/Anthropic providers (which only scan input prompts),
this provider also intercepts the *agentic loop*:

    1. Pre-request  — scan the user prompt (all providers do this)
    2. Pre-request  — scan any role='tool' result messages *before* they enter
                      Grok's context window (indirect injection via tool output)
    3. Post-response — scan tool_calls in Grok's reply *before* the caller
                       executes them (malicious tool invocation)

Steps 2 and 3 delegate to Guardian's scan_tool_output() and scan_tool_call()
respectively, which are API-tier methods (license key required).  When no
license key is present the agentic checks are skipped with a warning so the
provider degrades gracefully to prompt-only protection.

Principle 14 (Divine Safety): fail-closed on analysis errors — when in doubt,
block rather than allow an unchecked request through.
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
# xAI endpoint constants
# ---------------------------------------------------------------------------

XAI_BASE_URL = "https://api.x.ai/v1"

# Current Grok model identifiers (May 2026)
GROK_MODELS = {
    "grok-3",
    "grok-3-fast",
    "grok-3-mini",
    "grok-3-mini-fast",
    "grok-2-1212",
    "grok-2-vision-1212",
    "grok-beta",
    "grok-vision-beta",
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

    Callers should surface a secondary verification step (e.g. CAPTCHA,
    human review) rather than hard-blocking the request.  In strict_mode,
    CHALLENGE is escalated to ThreatBlockedException instead.

    Principle 16 (Sacred Autonomy): preserves human agency by surfacing
    uncertainty rather than silently blocking.
    """

    def __init__(
        self, analysis_result: Any, message: str = "Request requires verification"
    ) -> None:
        self.analysis_result = analysis_result
        super().__init__(message)


class AgentToolBlockedException(ThreatBlockedException):
    """Raised when a Grok-issued tool call is blocked by Guardian."""

    def __init__(
        self,
        analysis_result: Any,
        tool_name: str = "",
        message: str = "Agentic tool call blocked by Guardian",
    ) -> None:
        self.tool_name = tool_name
        super().__init__(analysis_result, message)


class ToolOutputBlockedException(ThreatBlockedException):
    """Raised when a tool result message is found to contain an injection payload."""

    def __init__(
        self,
        analysis_result: Any,
        tool_name: str = "",
        message: str = "Tool output blocked by Guardian — injection payload detected",
    ) -> None:
        self.tool_name = tool_name
        super().__init__(analysis_result, message)


# ---------------------------------------------------------------------------
# XAIProvider — detection and extraction logic
# ---------------------------------------------------------------------------

class XAIProvider:
    """
    xAI / Grok provider integration for Guardian SDK.

    Wraps any OpenAI-compatible client configured for the xAI endpoint and
    applies three layers of Guardian protection across the full agentic loop.
    """

    def __init__(self, guardian_instance: Any) -> None:
        self.guardian = guardian_instance
        self.provider_name = "xai"

    # ------------------------------------------------------------------
    # Client wrapping
    # ------------------------------------------------------------------

    def wrap_client(self, client: Any) -> "ProtectedXAIClient":
        """
        Wrap an xAI/Grok client with Guardian protection.

        Args:
            client: An OpenAI client instance configured for api.x.ai.

        Returns:
            A ProtectedXAIClient that is a transparent drop-in replacement.
        """
        try:
            import openai  # noqa: F401
        except ImportError:
            raise ProviderError(
                "openai package not installed. "
                "Run: pip install \"ethicore-engine-guardian[xai]\""
            )

        if not self._is_xai_client(client):
            raise ProviderError(
                f"Expected an OpenAI client configured for {XAI_BASE_URL}, "
                f"got {type(client)}.  Pass provider='xai' to guardian.wrap() "
                "to force xAI routing."
            )

        return ProtectedXAIClient(client, self.guardian)

    def _is_xai_client(self, client: Any) -> bool:
        """
        Return True if *client* is an xAI / Grok client.

        xAI clients are standard OpenAI clients with base_url pointing at
        api.x.ai.  We also accept manual override via provider='xai'.
        """
        # Check base_url attribute (set by OpenAI SDK when user supplies it)
        base_url = str(getattr(client, "base_url", "") or "").lower()
        if "x.ai" in base_url or "xai" in base_url:
            return True

        # Some wrappers expose _base_url instead
        base_url_alt = str(getattr(client, "_base_url", "") or "").lower()
        if "x.ai" in base_url_alt or "xai" in base_url_alt:
            return True

        return False

    # ------------------------------------------------------------------
    # Prompt extraction
    # ------------------------------------------------------------------

    def extract_prompt(self, **kwargs: Any) -> str:
        """
        Extract the analysable prompt text from chat.completions.create() kwargs.

        Handles:
        - Standard user messages (string content)
        - Multimodal content blocks [{type: text, text: ...}]
        - System messages (included so system-prompt injection is detected)
        """
        parts: List[str] = []
        messages: List[Dict[str, Any]] = kwargs.get("messages", [])
        if not messages:
            return ""

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content") or ""

            # Skip tool result messages — scanned separately via extract_openai_tool_results
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
# ProtectedXAIClient
# ---------------------------------------------------------------------------

class ProtectedXAIClient:
    """
    Proxy around an xAI/Grok client that intercepts all chat.completions.create()
    calls and applies three-layer Guardian protection.
    """

    def __init__(self, original_client: Any, guardian_instance: Any) -> None:
        self._original_client = original_client
        self._guardian = guardian_instance
        self._provider = XAIProvider(guardian_instance)

        if hasattr(original_client, "chat"):
            self.chat = self._create_protected_chat()

        logger.debug("🛡️  xAI/Grok client protection enabled")

    def _create_protected_chat(self) -> "ProtectedChat":
        return ProtectedChat(
            self._original_client.chat,
            self._guardian,
            self._provider,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original_client, name)

    def __repr__(self) -> str:
        return f"ProtectedXAIClient(original={repr(self._original_client)})"


# ---------------------------------------------------------------------------
# ProtectedChat → ProtectedCompletions
# ---------------------------------------------------------------------------

class ProtectedChat:
    """Proxy for client.chat — wraps the completions sub-namespace."""

    def __init__(
        self,
        original_chat: Any,
        guardian_instance: Any,
        provider: XAIProvider,
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

    Layer 1 — Input prompt scan      (all tiers)
    Layer 2 — Tool result scan       (API tier: scan_tool_output)
    Layer 3 — Outbound tool call scan (API tier: scan_tool_call)
    """

    def __init__(
        self,
        original_completions: Any,
        guardian_instance: Any,
        provider: XAIProvider,
    ) -> None:
        self._original_completions = original_completions
        self._guardian = guardian_instance
        self._provider = provider

    # ------------------------------------------------------------------
    # Sync path
    # ------------------------------------------------------------------

    def create(self, **kwargs: Any) -> Any:
        """Protected synchronous chat.completions.create() — all three agentic layers."""
        messages = kwargs.get("messages", [])

        # Layer 1: prompt scan
        prompt_text = self._provider.extract_prompt(**kwargs)
        if prompt_text and prompt_text.strip():
            analysis = run_sync(self._analyze_prompt(prompt_text, kwargs))
            self._enforce_policy(analysis, prompt_text)

        # Layer 2: inbound tool result scan
        tool_results = extract_openai_tool_results(messages)
        if tool_results:
            run_sync(scan_inbound_tool_results(
                self._guardian, tool_results, ToolOutputBlockedException
            ))

        # Layer 3: execute, then scan outbound tool calls
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

        # Layer 1: prompt scan
        prompt_text = self._provider.extract_prompt(**kwargs)
        if prompt_text and prompt_text.strip():
            analysis = await self._analyze_prompt(prompt_text, kwargs)
            self._enforce_policy(analysis, prompt_text)

        # Layer 2: inbound tool result scan
        tool_results = extract_openai_tool_results(messages)
        if tool_results:
            await scan_inbound_tool_results(
                self._guardian, tool_results, ToolOutputBlockedException
            )

        # Layer 3: execute, then scan outbound tool calls
        response = await self._original_completions.create(**kwargs)
        tool_calls = extract_openai_tool_calls(response)
        if tool_calls:
            await scan_outbound_tool_calls(
                self._guardian, tool_calls, AgentToolBlockedException
            )
        return response

    async def _analyze_prompt(self, prompt_text: str, request_kwargs: Dict[str, Any]) -> Any:
        context: Dict[str, Any] = {
            "api_call": "xai.chat.completions.create",
            "model": request_kwargs.get("model", "grok"),
            "max_tokens": request_kwargs.get("max_tokens"),
            "temperature": request_kwargs.get("temperature"),
            "request_size": len(prompt_text),
        }
        return await self._guardian.analyze(prompt_text, context)

    def _enforce_policy(self, analysis: Any, prompt_text: str) -> None:
        """Apply Guardian policy to a prompt analysis result."""
        action = getattr(analysis, "recommended_action", None)
        threat_level = getattr(analysis, "threat_level", "UNKNOWN")
        reasoning = getattr(analysis, "reasoning", [])
        reason_str = ", ".join(reasoning[:2]) if reasoning else "see analysis"

        if action == "BLOCK":
            logger.warning("🚨 BLOCKED xAI request — %s: %.100s…", threat_level, prompt_text)
            raise ThreatBlockedException(
                analysis_result=analysis,
                message=f"Request blocked: {threat_level} threat detected. Reasons: {reason_str}",
            )
        elif action == "CHALLENGE":
            logger.warning("⚠️  CHALLENGE xAI request — %s: %.100s…", threat_level, prompt_text)
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

def create_protected_xai_client(
    api_key: str,
    guardian_api_key: str,
    base_url: str = XAI_BASE_URL,
    **openai_kwargs: Any,
) -> ProtectedXAIClient:
    """
    Create a Guardian-protected xAI/Grok client in one step.

    Args:
        api_key:           xAI API key (starts with "xai-...").
        guardian_api_key:  Guardian/Ethicore API key.
        base_url:          xAI endpoint (default: https://api.x.ai/v1).
        **openai_kwargs:   Extra kwargs forwarded to openai.OpenAI().

    Returns:
        A ProtectedXAIClient ready for use as a drop-in replacement.

    Example::

        client = create_protected_xai_client(
            api_key="xai-...",
            guardian_api_key="ethicore-...",
        )
        response = client.chat.completions.create(
            model="grok-3",
            messages=[{"role": "user", "content": "Hello"}],
        )
    """
    try:
        import openai
    except ImportError:
        raise ProviderError(
            "openai package not installed. "
            "Run: pip install \"ethicore-engine-guardian[xai]\""
        )

    xai_client = openai.OpenAI(
        api_key=api_key,
        base_url=base_url,
        **openai_kwargs,
    )

    from ..guardian import Guardian

    guardian = Guardian(api_key=guardian_api_key)
    return guardian.wrap(xai_client, provider="xai")
