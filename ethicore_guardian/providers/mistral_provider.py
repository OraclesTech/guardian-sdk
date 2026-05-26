"""
Ethicore Engine™ - Guardian SDK — Mistral AI Provider
Protection for Mistral models via the OpenAI-compatible API.
Version: 1.0.0

Copyright © 2026 Oracles Technologies LLC
All Rights Reserved

Architecture note
-----------------
Mistral's API is OpenAI-compatible.  Clients are constructed as:

    from openai import OpenAI
    client = OpenAI(api_key="...", base_url="https://api.mistral.ai/v1")

Detection checks for the Mistral base URL rather than the client class name.

Mistral model ID convention
----------------------------
Mistral uses a versioned naming scheme: {name}-{major}-{minor}-{YY-MM}
e.g. mistral-large-3-25-12 = Mistral Large 3, released Dec 2025.

Current model IDs (May 2026):
  mistral-large-3-25-12           — Mistral Large 3 (flagship)
  mistral-medium-3-5-26-04        — Mistral Medium 3.5
  mistral-medium-3-1-25-08        — Mistral Medium 3.1
  mistral-small-4-0-26-03         — Mistral Small 4
  magistral-medium-1-2-25-09      — Magistral Medium 1.2 (reasoning/CoT model)
  ministral-3-14b-25-12           — Ministral 3 14B
  ministral-3-8b-25-12            — Ministral 3 8B
  ministral-3-3b-25-12            — Ministral 3 3B
  codestral-25-08                 — Codestral (code specialist)
  devstral-2-25-12                — Devstral 2 (agentic coding)
  voxtral-tts-26-03               — Voxtral TTS (text-to-speech)
  voxtral-small-25-07             — Voxtral Small
  leanstral-26-03                 — Leanstral (efficient/edge model)
  mistral-moderation-26-03        — Mistral Moderation 2

Short aliases (e.g. mistral-large-latest) may also be valid — check
https://docs.mistral.ai/getting-started/models/models_overview/ for current
alias mappings.

Source: https://docs.mistral.ai/

Agentic coverage
----------------
Three-layer Guardian protection across the full agentic loop:

    1. Pre-request  — scan the user prompt
    2. Pre-request  — scan tool result messages before they enter context
                      (indirect injection via tool output)
    3. Post-response — scan tool_calls in Mistral's reply before execution
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
# Mistral endpoint constants
# ---------------------------------------------------------------------------

MISTRAL_BASE_URL = "https://api.mistral.ai/v1"

# Current Mistral model identifiers (May 2026)
# Source: https://docs.mistral.ai/getting-started/models/models_overview/
MISTRAL_MODELS = {
    # ── Frontier / general-purpose ───────────────────────────────────────────
    "mistral-large-3-25-12",          # Mistral Large 3 (flagship)
    "mistral-medium-3-5-26-04",       # Mistral Medium 3.5
    "mistral-medium-3-1-25-08",       # Mistral Medium 3.1
    "mistral-small-4-0-26-03",        # Mistral Small 4
    # ── Reasoning ────────────────────────────────────────────────────────────
    "magistral-medium-1-2-25-09",     # Magistral Medium 1.2 (chain-of-thought reasoning)
    # ── Efficient / edge ─────────────────────────────────────────────────────
    "ministral-3-14b-25-12",          # Ministral 3 14B
    "ministral-3-8b-25-12",           # Ministral 3 8B
    "ministral-3-3b-25-12",           # Ministral 3 3B
    "leanstral-26-03",                # Leanstral (efficient/edge model)
    # ── Code specialists ─────────────────────────────────────────────────────
    "codestral-25-08",                # Codestral — code completion and generation
    "devstral-2-25-12",               # Devstral 2 — agentic coding workflows
    # ── Audio / speech ───────────────────────────────────────────────────────
    "voxtral-tts-26-03",              # Voxtral TTS
    "voxtral-small-25-07",            # Voxtral Small
    # ── Safety / moderation ──────────────────────────────────────────────────
    "mistral-moderation-26-03",       # Mistral Moderation 2
    # ── Common short aliases (may resolve to latest in each tier) ────────────
    "mistral-large-latest",
    "mistral-medium-latest",
    "mistral-small-latest",
    "codestral-latest",
    "devstral-latest",
    "open-mistral-7b",
    "open-mixtral-8x7b",
    "open-mixtral-8x22b",
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
    """Raised when a Mistral-issued tool call is blocked by Guardian."""

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
# MistralProvider — detection and extraction logic
# ---------------------------------------------------------------------------

class MistralProvider:
    """
    Mistral AI provider integration for Guardian SDK.

    Wraps any OpenAI-compatible client configured for the Mistral endpoint
    and applies three layers of Guardian protection across the full agentic loop.
    """

    def __init__(self, guardian_instance: Any) -> None:
        self.guardian = guardian_instance
        self.provider_name = "mistral"

    def wrap_client(self, client: Any) -> "ProtectedMistralClient":
        """
        Wrap a Mistral client with Guardian protection.

        Args:
            client: An OpenAI client instance configured for api.mistral.ai.

        Returns:
            A ProtectedMistralClient that is a transparent drop-in replacement.
        """
        try:
            import openai  # noqa: F401
        except ImportError:
            raise ProviderError(
                "openai package not installed. "
                "Run: pip install \"ethicore-engine-guardian[mistral]\""
            )

        if not self._is_mistral_client(client):
            raise ProviderError(
                f"Expected an OpenAI client configured for {MISTRAL_BASE_URL}, "
                f"got {type(client)}.  Pass provider='mistral' to guardian.wrap() "
                "to force Mistral routing."
            )

        return ProtectedMistralClient(client, self.guardian)

    def _is_mistral_client(self, client: Any) -> bool:
        """Return True if *client* is a Mistral client."""
        base_url = str(getattr(client, "base_url", "") or "").lower()
        if "mistral" in base_url or "api.mistral.ai" in base_url:
            return True
        base_url_alt = str(getattr(client, "_base_url", "") or "").lower()
        if "mistral" in base_url_alt:
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
# ProtectedMistralClient
# ---------------------------------------------------------------------------

class ProtectedMistralClient:
    """
    Proxy around a Mistral client that intercepts all chat.completions.create()
    calls and applies three-layer Guardian protection.
    """

    def __init__(self, original_client: Any, guardian_instance: Any) -> None:
        self._original_client = original_client
        self._guardian = guardian_instance
        self._provider = MistralProvider(guardian_instance)

        if hasattr(original_client, "chat"):
            self.chat = self._create_protected_chat()

        logger.debug("🛡️  Mistral client protection enabled")

    def _create_protected_chat(self) -> "ProtectedChat":
        return ProtectedChat(
            self._original_client.chat,
            self._guardian,
            self._provider,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original_client, name)

    def __repr__(self) -> str:
        return f"ProtectedMistralClient(original={repr(self._original_client)})"


# ---------------------------------------------------------------------------
# ProtectedChat → ProtectedCompletions
# ---------------------------------------------------------------------------

class ProtectedChat:
    """Proxy for client.chat — wraps the completions sub-namespace."""

    def __init__(
        self,
        original_chat: Any,
        guardian_instance: Any,
        provider: MistralProvider,
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
        provider: MistralProvider,
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
            "api_call": "mistral.chat.completions.create",
            "model": request_kwargs.get("model", "mistral-large-latest"),
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
            logger.warning("🚨 BLOCKED Mistral request — %s: %.100s…", threat_level, prompt_text)
            raise ThreatBlockedException(
                analysis_result=analysis,
                message=f"Request blocked: {threat_level} threat detected. Reasons: {reason_str}",
            )
        elif action == "CHALLENGE":
            logger.warning("⚠️  CHALLENGE Mistral request — %s: %.100s…", threat_level, prompt_text)
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

def create_protected_mistral_client(
    api_key: str,
    guardian_api_key: str,
    base_url: str = MISTRAL_BASE_URL,
    **openai_kwargs: Any,
) -> ProtectedMistralClient:
    """
    Create a Guardian-protected Mistral client in one step.

    Args:
        api_key:           Mistral API key.
        guardian_api_key:  Guardian/Ethicore API key.
        base_url:          Mistral endpoint (default: https://api.mistral.ai/v1).
        **openai_kwargs:   Extra kwargs forwarded to openai.OpenAI().

    Returns:
        A ProtectedMistralClient ready for use as a drop-in replacement.

    Example::

        client = create_protected_mistral_client(
            api_key="...",
            guardian_api_key="eg-sk-...",
        )
        response = client.chat.completions.create(
            model="mistral-large-3-25-12",
            messages=[{"role": "user", "content": "Hello"}],
        )
    """
    try:
        import openai
    except ImportError:
        raise ProviderError(
            "openai package not installed. "
            "Run: pip install \"ethicore-engine-guardian[mistral]\""
        )

    mistral_client = openai.OpenAI(
        api_key=api_key,
        base_url=base_url,
        **openai_kwargs,
    )

    from ..guardian import Guardian

    guardian = Guardian(api_key=guardian_api_key)
    return guardian.wrap(mistral_client, provider="mistral")


__all__ = [
    "MistralProvider",
    "ProtectedMistralClient",
    "ProtectedChat",
    "ProtectedCompletions",
    "create_protected_mistral_client",
    "MISTRAL_BASE_URL",
    "MISTRAL_MODELS",
    "ThreatBlockedException",
    "ThreatChallengeException",
    "AgentToolBlockedException",
    "ToolOutputBlockedException",
    "ProviderError",
]
