"""
Ethicore Engine™ - Guardian SDK — LiteLLM Provider
Version: 1.0.0

Wraps LiteLLM's completion / acompletion functions with three-layer
agentic protection, giving Guardian coverage over the 140+ providers
LiteLLM normalises into a single OpenAI-compatible interface.

Because LiteLLM is a module rather than a client object, the wrapping
model differs from other providers:

    protected = guardian.protect_litellm()
    # or
    protected = create_protected_litellm(guardian_api_key="eg-sk-...")

    response = protected.completion(model="gpt-4o", messages=[...])
    response = await protected.acompletion(model="gemini/gemini-2.0-flash", ...)

Layer 1: Prompt scan via guardian.analyze()
Layer 2: Inbound tool result scan via scan_tool_output() (role='tool' messages)
Layer 3: Outbound tool call scan via scan_tool_call() (response.choices tool_calls)

LiteLLM normalises all provider responses to OpenAI format, so the
existing OpenAI extractors (extract_openai_tool_results / extract_openai_tool_calls)
are reused for Layers 2 and 3.

All three layers include multilingual support — non-English payloads are
detected via MultilingualSemanticAnalyzer wired into scan_tool_output() and
scan_tool_call() in guardian.py.

Principle 14 (Divine Safety): LiteLLM routes to many backends; protection
must wrap the routing layer itself so no backend can be used to bypass Guardian.

Copyright © 2026 Oracles Technologies LLC
All Rights Reserved
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .base_provider import ProviderError
from ._agentic_guards import (
    run_sync,
    extract_openai_tool_results,
    extract_openai_tool_calls,
    scan_inbound_tool_results,
    scan_outbound_tool_calls,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# EXCEPTIONS  (mirror OpenAI provider hierarchy for API consistency)
# ==============================================================================

class ThreatBlockedException(Exception):
    """Raised when Layer 1 prompt scan verdict is BLOCK."""
    def __init__(self, analysis_result: Any = None, message: str = ""):
        self.analysis_result = analysis_result
        super().__init__(message or "LiteLLM request blocked by Guardian threat analysis.")


class ThreatChallengeException(Exception):
    """Raised when Layer 1 prompt scan verdict is CHALLENGE and strict_mode is on."""
    def __init__(self, analysis_result: Any = None, message: str = ""):
        self.analysis_result = analysis_result
        super().__init__(message or "LiteLLM request challenged by Guardian threat analysis.")


class ToolOutputBlockedException(ThreatBlockedException):
    """Raised when Layer 2 detects an injection payload in a tool result."""
    def __init__(self, analysis_result: Any = None, message: str = ""):
        super().__init__(
            analysis_result,
            message or "Tool output blocked: injection payload detected.",
        )


class AgentToolBlockedException(ThreatBlockedException):
    """Raised when Layer 3 blocks a dangerous outbound tool call."""
    def __init__(self, analysis_result: Any = None, message: str = ""):
        super().__init__(
            analysis_result,
            message or "Outbound tool call blocked: dangerous operation detected.",
        )


# ==============================================================================
# PROMPT EXTRACTION
# ==============================================================================

def _extract_prompt(messages: List[Dict[str, Any]]) -> str:
    """
    Extract analysable text from a LiteLLM / OpenAI messages list.

    Skips role='tool' messages (scanned separately in Layer 2).
    Handles string content and multimodal content blocks.
    """
    parts: List[str] = []
    for msg in messages:
        role = msg.get("role", "")
        if role == "tool":
            continue
        content = msg.get("content") or ""
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
    return " ".join(parts)


# ==============================================================================
# PROTECTED LITELLM CLIENT
# ==============================================================================

class ProtectedLiteLLMClient:
    """
    Drop-in replacement for litellm.completion / litellm.acompletion with
    Guardian's three-layer agentic protection applied transparently.

    Usage::

        from ethicore_guardian.providers.litellm_provider import ProtectedLiteLLMClient
        from ethicore_guardian import Guardian

        guardian = Guardian(api_key="eg-sk-...")
        protected = ProtectedLiteLLMClient(guardian)

        response = protected.completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )
    """

    def __init__(self, guardian_instance: Any) -> None:
        self._guardian = guardian_instance

    # ------------------------------------------------------------------
    # Sync completion
    # ------------------------------------------------------------------

    def completion(self, **kwargs: Any) -> Any:
        """
        Sync wrapper around litellm.completion with three-layer protection.

        Raises:
            ThreatBlockedException:      Layer 1 — prompt is a threat (BLOCK).
            ThreatChallengeException:    Layer 1 — threat in strict mode (CHALLENGE).
            ToolOutputBlockedException:  Layer 2 — tool result contains injection.
            AgentToolBlockedException:   Layer 3 — outbound tool call is dangerous.
        """
        try:
            import litellm  # noqa: PLC0415
        except ImportError:
            raise ProviderError(
                "litellm package not installed. Run: pip install litellm"
            )

        messages: List[Dict[str, Any]] = kwargs.get("messages", [])
        strict_mode: bool = getattr(
            getattr(self._guardian, "config", None), "strict_mode", False
        )

        # ── Layer 1: prompt scan ─────────────────────────────────────
        prompt_text = _extract_prompt(messages)
        if prompt_text.strip():
            analysis = run_sync(
                self._guardian.analyze(
                    prompt_text,
                    context={"provider": "litellm", "model": kwargs.get("model", "")},
                )
            )
            action = getattr(analysis, "recommended_action", "ALLOW")
            if action == "BLOCK":
                raise ThreatBlockedException(
                    analysis_result=analysis,
                    message=(
                        f"LiteLLM request blocked (score="
                        f"{getattr(analysis, 'threat_score', 0):.2f})."
                    ),
                )
            if action == "CHALLENGE" and strict_mode:
                raise ThreatChallengeException(
                    analysis_result=analysis,
                    message="LiteLLM request challenged in strict mode.",
                )

        # ── Layer 2: inbound tool result scan ────────────────────────
        tool_results = extract_openai_tool_results(messages)
        if tool_results:
            run_sync(
                scan_inbound_tool_results(
                    self._guardian, tool_results, ToolOutputBlockedException
                )
            )

        # ── Execute ──────────────────────────────────────────────────
        response = litellm.completion(**kwargs)

        # ── Layer 3: outbound tool call scan ─────────────────────────
        tool_calls = extract_openai_tool_calls(response)
        if tool_calls:
            run_sync(
                scan_outbound_tool_calls(
                    self._guardian, tool_calls, AgentToolBlockedException
                )
            )

        return response

    # ------------------------------------------------------------------
    # Async completion
    # ------------------------------------------------------------------

    async def acompletion(self, **kwargs: Any) -> Any:
        """
        Async wrapper around litellm.acompletion with three-layer protection.
        """
        try:
            import litellm  # noqa: PLC0415
        except ImportError:
            raise ProviderError(
                "litellm package not installed. Run: pip install litellm"
            )

        messages: List[Dict[str, Any]] = kwargs.get("messages", [])
        strict_mode: bool = getattr(
            getattr(self._guardian, "config", None), "strict_mode", False
        )

        # ── Layer 1 ──────────────────────────────────────────────────
        prompt_text = _extract_prompt(messages)
        if prompt_text.strip():
            analysis = await self._guardian.analyze(
                prompt_text,
                context={"provider": "litellm", "model": kwargs.get("model", "")},
            )
            action = getattr(analysis, "recommended_action", "ALLOW")
            if action == "BLOCK":
                raise ThreatBlockedException(
                    analysis_result=analysis,
                    message=(
                        f"LiteLLM request blocked (score="
                        f"{getattr(analysis, 'threat_score', 0):.2f})."
                    ),
                )
            if action == "CHALLENGE" and strict_mode:
                raise ThreatChallengeException(
                    analysis_result=analysis,
                    message="LiteLLM request challenged in strict mode.",
                )

        # ── Layer 2 ──────────────────────────────────────────────────
        tool_results = extract_openai_tool_results(messages)
        if tool_results:
            await scan_inbound_tool_results(
                self._guardian, tool_results, ToolOutputBlockedException
            )

        # ── Execute ──────────────────────────────────────────────────
        response = await litellm.acompletion(**kwargs)

        # ── Layer 3 ──────────────────────────────────────────────────
        tool_calls = extract_openai_tool_calls(response)
        if tool_calls:
            await scan_outbound_tool_calls(
                self._guardian, tool_calls, AgentToolBlockedException
            )

        return response


# ==============================================================================
# PROVIDER CLASS  (for guardian.wrap() dispatch)
# ==============================================================================

class LiteLLMProvider:
    """
    Guardian provider entry for LiteLLM.

    Unlike client-based providers, LiteLLM is wrapped at the function level.
    guardian.protect_litellm() returns a ProtectedLiteLLMClient directly;
    wrap_client() is provided for API consistency but is not the primary path.
    """

    provider_name = "litellm"

    def __init__(self, guardian_instance: Any) -> None:
        self.guardian = guardian_instance

    def wrap_client(self, _client: Any = None) -> ProtectedLiteLLMClient:
        """Return a ProtectedLiteLLMClient (client arg ignored for LiteLLM)."""
        return ProtectedLiteLLMClient(self.guardian)

    def extract_prompt(self, **kwargs: Any) -> str:
        return _extract_prompt(kwargs.get("messages", []))


# ==============================================================================
# FACTORY
# ==============================================================================

def create_protected_litellm(guardian_api_key: str) -> ProtectedLiteLLMClient:
    """
    Convenience factory: create a Guardian-protected LiteLLM wrapper.

    Args:
        guardian_api_key: Ethicore Guardian API key (ETHICORE_API_KEY).

    Returns:
        ProtectedLiteLLMClient exposing .completion() and .acompletion().

    Example::

        from ethicore_guardian.providers.litellm_provider import create_protected_litellm

        protected = create_protected_litellm(guardian_api_key="eg-sk-...")

        response = protected.completion(
            model="groq/llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Summarise the quarterly report."}],
        )
    """
    from ..guardian import Guardian  # local import avoids circular dependency
    guardian = Guardian(api_key=guardian_api_key)
    return ProtectedLiteLLMClient(guardian)


__all__ = [
    "LiteLLMProvider",
    "ProtectedLiteLLMClient",
    "create_protected_litellm",
    "ThreatBlockedException",
    "ThreatChallengeException",
    "ToolOutputBlockedException",
    "AgentToolBlockedException",
]
