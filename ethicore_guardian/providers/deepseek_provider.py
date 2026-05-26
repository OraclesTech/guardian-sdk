"""
Ethicore Engine™ - Guardian SDK — DeepSeek Provider
Protection for DeepSeek models via the OpenAI-compatible API.
Version: 1.0.0

Copyright © 2026 Oracles Technologies LLC
All Rights Reserved

Architecture note
-----------------
DeepSeek's API is OpenAI-compatible. Clients are constructed as:

    from openai import OpenAI
    client = OpenAI(api_key="sk-...", base_url="https://api.deepseek.com")

Detection checks for the DeepSeek base URL rather than the client class name.

DeepSeek V4 — thinking / reasoning mode
-----------------------------------------
For V4 models, reasoning is toggled via an extra_body parameter, NOT a
separate model ID:

    client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[...],
        extra_body={"thinking": {"type": "enabled", "budget_tokens": 8000}},
    )

Current model IDs (May 2026):
  deepseek-v4-flash     — Default non-thinking mode (fast, cost-effective)
  deepseek-v4-pro       — Pro model; extended reasoning budget
  deepseek-chat         — Legacy alias → deepseek-v4-flash (deprecated 2026-07-24)
  deepseek-reasoner     — Legacy alias → deepseek-v4-flash thinking (deprecated 2026-07-24)

New code should use deepseek-v4-flash or deepseek-v4-pro directly.

Anthropic-compatible endpoint
------------------------------
DeepSeek also exposes https://api.deepseek.com/anthropic for use with the
Anthropic SDK. This provider wraps the OpenAI-compatible endpoint only.
For Anthropic-format calls, use the standard Anthropic SDK pointed at that URL.

Source: https://api-docs.deepseek.com/

Agentic coverage
----------------
Three-layer Guardian protection across the full agentic loop:

    1. Pre-request  — scan the user prompt
    2. Pre-request  — scan tool result messages before they enter context
                      (indirect injection via tool output)
    3. Post-response — scan tool_calls in DeepSeek's reply before execution
                       (malicious tool invocation)

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
# DeepSeek endpoint constants
# ---------------------------------------------------------------------------

DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# Current DeepSeek model identifiers (May 2026)
# Source: https://api-docs.deepseek.com/
DEEPSEEK_MODELS = {
    # ── V4 series (current — use these for new code) ────────────────────────
    "deepseek-v4-flash",     # Default; fast, cost-effective; non-thinking by default
    "deepseek-v4-pro",       # Pro model; higher capability and reasoning budget
    # ── Legacy aliases (deprecated 2026-07-24, still functional) ────────────
    "deepseek-chat",         # → deepseek-v4-flash non-thinking
    "deepseek-reasoner",     # → deepseek-v4-flash thinking mode
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
    """Raised when a DeepSeek-issued tool call is blocked by Guardian."""

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
# DeepSeekProvider — detection and extraction logic
# ---------------------------------------------------------------------------

class DeepSeekProvider:
    """
    DeepSeek provider integration for Guardian SDK.

    Wraps any OpenAI-compatible client configured for the DeepSeek endpoint
    and applies three layers of Guardian protection across the full agentic loop.
    """

    def __init__(self, guardian_instance: Any) -> None:
        self.guardian = guardian_instance
        self.provider_name = "deepseek"

    # ------------------------------------------------------------------
    # Client wrapping
    # ------------------------------------------------------------------

    def wrap_client(self, client: Any) -> "ProtectedDeepSeekClient":
        """
        Wrap a DeepSeek client with Guardian protection.

        Args:
            client: An OpenAI client instance configured for api.deepseek.com.

        Returns:
            A ProtectedDeepSeekClient that is a transparent drop-in replacement.
        """
        try:
            import openai  # noqa: F401
        except ImportError:
            raise ProviderError(
                "openai package not installed. "
                "Run: pip install \"ethicore-engine-guardian[deepseek]\""
            )

        if not self._is_deepseek_client(client):
            raise ProviderError(
                f"Expected an OpenAI client configured for {DEEPSEEK_BASE_URL}, "
                f"got {type(client)}.  Pass provider='deepseek' to guardian.wrap() "
                "to force DeepSeek routing."
            )

        return ProtectedDeepSeekClient(client, self.guardian)

    def _is_deepseek_client(self, client: Any) -> bool:
        """Return True if *client* is a DeepSeek client."""
        base_url = str(getattr(client, "base_url", "") or "").lower()
        if "deepseek" in base_url or "api.deepseek.com" in base_url:
            return True
        base_url_alt = str(getattr(client, "_base_url", "") or "").lower()
        if "deepseek" in base_url_alt:
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
        - tool role messages are excluded (scanned separately via Layer 2)
        """
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
# ProtectedDeepSeekClient
# ---------------------------------------------------------------------------

class ProtectedDeepSeekClient:
    """
    Proxy around a DeepSeek client that intercepts all chat.completions.create()
    calls and applies three-layer Guardian protection.
    """

    def __init__(self, original_client: Any, guardian_instance: Any) -> None:
        self._original_client = original_client
        self._guardian = guardian_instance
        self._provider = DeepSeekProvider(guardian_instance)

        if hasattr(original_client, "chat"):
            self.chat = self._create_protected_chat()

        logger.debug("🛡️  DeepSeek client protection enabled")

    def _create_protected_chat(self) -> "ProtectedChat":
        return ProtectedChat(
            self._original_client.chat,
            self._guardian,
            self._provider,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original_client, name)

    def __repr__(self) -> str:
        return f"ProtectedDeepSeekClient(original={repr(self._original_client)})"


# ---------------------------------------------------------------------------
# ProtectedChat → ProtectedCompletions
# ---------------------------------------------------------------------------

class ProtectedChat:
    """Proxy for client.chat — wraps the completions sub-namespace."""

    def __init__(
        self,
        original_chat: Any,
        guardian_instance: Any,
        provider: DeepSeekProvider,
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

    Layer 1 — Input prompt scan       (all tiers)
    Layer 2 — Tool result scan        (API tier: scan_tool_output)
    Layer 3 — Outbound tool call scan (API tier: scan_tool_call)
    """

    def __init__(
        self,
        original_completions: Any,
        guardian_instance: Any,
        provider: DeepSeekProvider,
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
            "api_call": "deepseek.chat.completions.create",
            "model": request_kwargs.get("model", "deepseek-v4-flash"),
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
            logger.warning("🚨 BLOCKED DeepSeek request — %s: %.100s…", threat_level, prompt_text)
            raise ThreatBlockedException(
                analysis_result=analysis,
                message=f"Request blocked: {threat_level} threat detected. Reasons: {reason_str}",
            )
        elif action == "CHALLENGE":
            logger.warning("⚠️  CHALLENGE DeepSeek request — %s: %.100s…", threat_level, prompt_text)
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

def create_protected_deepseek_client(
    api_key: str,
    guardian_api_key: str,
    base_url: str = DEEPSEEK_BASE_URL,
    **openai_kwargs: Any,
) -> ProtectedDeepSeekClient:
    """
    Create a Guardian-protected DeepSeek client in one step.

    Args:
        api_key:           DeepSeek API key (starts with "sk-...").
        guardian_api_key:  Guardian/Ethicore API key.
        base_url:          DeepSeek endpoint (default: https://api.deepseek.com).
        **openai_kwargs:   Extra kwargs forwarded to openai.OpenAI().

    Returns:
        A ProtectedDeepSeekClient ready for use as a drop-in replacement.

    Example::

        client = create_protected_deepseek_client(
            api_key="sk-...",
            guardian_api_key="eg-sk-...",
        )
        # Standard inference
        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[{"role": "user", "content": "Hello"}],
        )
        # With reasoning enabled (V4 thinking mode via extra_body)
        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[{"role": "user", "content": "Solve step by step: ..."}],
            extra_body={"thinking": {"type": "enabled", "budget_tokens": 8000}},
        )
    """
    try:
        import openai
    except ImportError:
        raise ProviderError(
            "openai package not installed. "
            "Run: pip install \"ethicore-engine-guardian[deepseek]\""
        )

    deepseek_client = openai.OpenAI(
        api_key=api_key,
        base_url=base_url,
        **openai_kwargs,
    )

    from ..guardian import Guardian

    guardian = Guardian(api_key=guardian_api_key)
    return guardian.wrap(deepseek_client, provider="deepseek")


__all__ = [
    "DeepSeekProvider",
    "ProtectedDeepSeekClient",
    "ProtectedChat",
    "ProtectedCompletions",
    "create_protected_deepseek_client",
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_MODELS",
    "ThreatBlockedException",
    "ThreatChallengeException",
    "AgentToolBlockedException",
    "ToolOutputBlockedException",
    "ProviderError",
]
