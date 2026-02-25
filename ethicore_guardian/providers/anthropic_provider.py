"""
Ethicore Engine™ - Guardian SDK — Anthropic Provider
Mirrors the OpenAI provider pattern for the Anthropic Messages API.
Version: 1.0.0

Copyright © 2026 Oracles Technologies LLC
All Rights Reserved

Principle 22 (Servant Leadership): this provider exists entirely to serve the
user's safety — every interception is an act of protection, not gatekeeping.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared exception types (re-exported so callers have a single import path)
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
    human review) rather than hard-blocking the request.  In ``strict_mode``,
    CHALLENGE is escalated to ``ThreatBlockedException`` instead.

    Principle 16 (Sacred Autonomy): preserves human agency by surfacing
    uncertainty rather than silently blocking.

    Attributes:
        analysis_result: The ``ThreatAnalysis`` that triggered the challenge.
    """

    def __init__(
        self, analysis_result: Any, message: str = "Request requires verification"
    ) -> None:
        self.analysis_result = analysis_result
        super().__init__(message)


# ---------------------------------------------------------------------------
# AnthropicProvider — detection & extraction logic
# ---------------------------------------------------------------------------

class AnthropicProvider:
    """
    Anthropic provider integration for Guardian SDK.

    Intercepts ``client.messages.create()`` calls and runs Guardian threat
    detection before allowing the request to reach the Anthropic API.
    Maintains full API compatibility with both the sync ``anthropic.Anthropic``
    client and the async ``anthropic.AsyncAnthropic`` client.
    """

    def __init__(self, guardian_instance: Any) -> None:
        self.guardian = guardian_instance
        self.provider_name = "anthropic"

    def wrap_client(self, client: Any) -> "ProtectedAnthropicClient":
        """
        Wrap an Anthropic client with Guardian protection.

        Args:
            client: An ``anthropic.Anthropic`` or ``anthropic.AsyncAnthropic``
                    instance.

        Returns:
            A ``ProtectedAnthropicClient`` that passes all other attributes
            through to the original client unchanged.
        """
        try:
            import anthropic  # noqa: F401
        except ImportError:
            raise ProviderError(
                "anthropic package not installed. "
                "Run: pip install \"ethicore-engine-guardian[anthropic]\""
            )

        if not self._is_anthropic_client(client):
            raise ProviderError(f"Expected Anthropic client, got {type(client)}")

        return ProtectedAnthropicClient(client, self.guardian)

    def _is_anthropic_client(self, client: Any) -> bool:
        """Return True if *client* is a recognised Anthropic client type."""
        client_type = str(type(client)).lower()
        return "anthropic" in client_type

    # ------------------------------------------------------------------
    # Prompt extraction — handles all Anthropic Messages API shapes
    # ------------------------------------------------------------------

    def extract_prompt(self, **kwargs: Any) -> str:
        """
        Extract the user-visible prompt text from ``messages.create()`` kwargs.

        Supports:
        - ``messages=[{"role": "user", "content": "..."}]``
        - ``messages=[{"role": "user", "content": [{"type": "text", "text": "..."}]}]``
          (multimodal / vision format)
        - An optional ``system`` kwarg is included in analysis so system-prompt
          injection attacks are also caught.
        """
        parts: List[str] = []

        # Include system prompt if present (Anthropic passes it separately)
        system = kwargs.get("system")
        if system and isinstance(system, str):
            parts.append(system)

        messages: List[Dict[str, Any]] = kwargs.get("messages", [])
        if not messages:
            return " ".join(parts)

        # Analyse the last user message — that is where injection attacks land
        user_messages = [m for m in messages if m.get("role") == "user"]
        if not user_messages:
            return " ".join(parts)

        last_user = user_messages[-1]
        content = last_user.get("content", "")

        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            # Multimodal content blocks
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))

        return " ".join(parts)


# ---------------------------------------------------------------------------
# ProtectedAnthropicClient — thin proxy that intercepts messages.create()
# ---------------------------------------------------------------------------

class ProtectedAnthropicClient:
    """
    Proxy around an Anthropic client that intercepts ``messages.create()``
    calls and runs Guardian analysis first.

    All other attributes and methods are delegated to the original client
    via ``__getattr__`` so callers need not change any other code.
    """

    def __init__(self, original_client: Any, guardian_instance: Any) -> None:
        self._original_client = original_client
        self._guardian = guardian_instance
        self._provider = AnthropicProvider(guardian_instance)

        # Wrap the messages interface — the primary Anthropic API surface
        if hasattr(original_client, "messages"):
            self.messages = self._create_protected_messages()

        logger.debug("🛡️  Anthropic client protection enabled")

    # ------------------------------------------------------------------
    # Internal: build the protected messages namespace
    # ------------------------------------------------------------------

    def _create_protected_messages(self) -> "ProtectedMessages":
        """Return a ProtectedMessages object wrapping original_client.messages."""
        return ProtectedMessages(
            self._original_client.messages,
            self._guardian,
            self._provider,
        )

    # ------------------------------------------------------------------
    # Transparent delegation
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        """Pass unknown attribute lookups to the underlying client."""
        return getattr(self._original_client, name)

    def __repr__(self) -> str:
        return f"ProtectedAnthropicClient(original={repr(self._original_client)})"


# ---------------------------------------------------------------------------
# ProtectedMessages — intercepts create() / stream() on client.messages
# ---------------------------------------------------------------------------

class ProtectedMessages:
    """
    Proxy around ``client.messages`` that intercepts ``create()`` calls.

    Principle 14 (Divine Safety): when analysis cannot complete (timeout,
    internal error) the call is blocked — fail-closed, not fail-open.
    """

    def __init__(
        self,
        original_messages: Any,
        guardian_instance: Any,
        provider: AnthropicProvider,
    ) -> None:
        self._original_messages = original_messages
        self._guardian = guardian_instance
        self._provider = provider

        # Preserve non-callable attributes (e.g. model constants)
        for attr_name in dir(original_messages):
            if not attr_name.startswith("_") and attr_name not in {"create", "stream"}:
                attr = getattr(original_messages, attr_name)
                if not callable(attr):
                    setattr(self, attr_name, attr)

    # ------------------------------------------------------------------
    # Sync path
    # ------------------------------------------------------------------

    def create(self, **kwargs: Any) -> Any:
        """Protected synchronous ``messages.create()``."""
        prompt_text = self._provider.extract_prompt(**kwargs)

        if prompt_text and prompt_text.strip():
            analysis = self._run_analysis_sync(prompt_text, kwargs)
            self._enforce_policy(analysis, prompt_text)

        return self._original_messages.create(**kwargs)

    def _run_analysis_sync(self, prompt_text: str, request_kwargs: Dict[str, Any]) -> Any:
        """Run Guardian analysis, handling sync/async context differences."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop:
            # Already inside an event loop — push analysis to a thread pool
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    asyncio.run,
                    self._analyze(prompt_text, request_kwargs),
                )
                return future.result()
        else:
            return asyncio.run(self._analyze(prompt_text, request_kwargs))

    # ------------------------------------------------------------------
    # Async path
    # ------------------------------------------------------------------

    async def async_create(self, **kwargs: Any) -> Any:
        """
        Protected async ``messages.create()``.

        Usage with ``anthropic.AsyncAnthropic``::

            protected = guardian.wrap(async_client)
            response = await protected.messages.async_create(model=..., ...)
        """
        prompt_text = self._provider.extract_prompt(**kwargs)

        if prompt_text and prompt_text.strip():
            analysis = await self._analyze(prompt_text, kwargs)
            self._enforce_policy(analysis, prompt_text)

        return await self._original_messages.create(**kwargs)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    async def _analyze(self, prompt_text: str, request_kwargs: Dict[str, Any]) -> Any:
        """Run Guardian analysis with Anthropic-specific context metadata."""
        context: Dict[str, Any] = {
            "api_call": "anthropic.messages.create",
            "model": request_kwargs.get("model", "unknown"),
            "max_tokens": request_kwargs.get("max_tokens"),
            "temperature": request_kwargs.get("temperature"),
            "request_size": len(prompt_text),
        }
        return await self._guardian.analyze(prompt_text, context)

    def _enforce_policy(self, analysis: Any, prompt_text: str) -> None:
        """
        Apply Guardian policy to the analysis result.

        BLOCK                    → always raise ThreatBlockedException
        CHALLENGE + strict_mode  → escalate to ThreatBlockedException
        CHALLENGE + non-strict   → raise ThreatChallengeException so callers
                                   can surface a verification step
        ALLOW                    → do nothing
        """
        reasons = getattr(analysis, "reasoning", [])
        reason_str = ", ".join(reasons[:2]) if reasons else "see analysis"

        if analysis.recommended_action == "BLOCK":
            logger.warning(
                "🚨 BLOCKED Anthropic request — %s: %.100s…",
                analysis.threat_level,
                prompt_text,
            )
            logger.warning("   Reasons: %s", reason_str)
            raise ThreatBlockedException(
                analysis_result=analysis,
                message=(
                    f"Request blocked: {analysis.threat_level} threat detected. "
                    f"Reasons: {reason_str}"
                ),
            )

        elif analysis.recommended_action == "CHALLENGE":
            logger.warning(
                "⚠️  CHALLENGE Anthropic request — %s: %.100s…",
                analysis.threat_level,
                prompt_text,
            )
            logger.warning("   Reasons: %s", reason_str)
            if self._guardian.config.strict_mode:
                # Principle 14 (Divine Safety): in strict mode, treat CHALLENGE
                # as a hard block — better to refuse than to risk harm.
                raise ThreatBlockedException(
                    analysis_result=analysis,
                    message=(
                        f"Request blocked (strict mode — CHALLENGE): "
                        f"{analysis.threat_level} threat detected."
                    ),
                )
            else:
                raise ThreatChallengeException(
                    analysis_result=analysis,
                    message=(
                        f"Request requires verification: "
                        f"{analysis.threat_level} threat level."
                    ),
                )

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the original messages object."""
        return getattr(self._original_messages, name)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def create_protected_anthropic_client(
    api_key: str,
    guardian_api_key: str,
    **anthropic_kwargs: Any,
) -> ProtectedAnthropicClient:
    """
    Create a Guardian-protected Anthropic client in one step.

    Args:
        api_key:           Anthropic API key.
        guardian_api_key:  Guardian API key.
        **anthropic_kwargs: Extra kwargs forwarded to ``anthropic.Anthropic()``.

    Returns:
        A ``ProtectedAnthropicClient`` ready for use as a drop-in replacement.

    Example::

        client = create_protected_anthropic_client(
            api_key="sk-ant-...",
            guardian_api_key="ethicore-...",
        )
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
        )
    """
    try:
        import anthropic
    except ImportError:
        raise ProviderError(
            "anthropic package not installed. "
            "Run: pip install \"ethicore-engine-guardian[anthropic]\""
        )

    anthropic_client = anthropic.Anthropic(api_key=api_key, **anthropic_kwargs)

    from ..guardian import Guardian
    guardian = Guardian(api_key=guardian_api_key)

    return guardian.wrap(anthropic_client)
