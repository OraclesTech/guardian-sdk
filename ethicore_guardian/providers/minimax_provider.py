"""
Ethicore Engine™ - Guardian SDK — MiniMax Provider
Mirrors the OpenAI provider pattern for the MiniMax OpenAI-compatible API.
Version: 1.0.0

Copyright © 2026 Oracles Technologies LLC
All Rights Reserved

MiniMax (https://www.minimax.io) provides powerful LLM models accessible
through an OpenAI-compatible REST API at https://api.minimax.io/v1.
This provider wraps MiniMax-configured OpenAI clients with Guardian
threat detection, following the same composition pattern as the OpenAI
and Anthropic providers.

Supported models:
- MiniMax-M2.7        (latest flagship, 1M context)
- MiniMax-M2.7-highspeed  (fast variant)
- MiniMax-M2.5        (204K context)
- MiniMax-M2.5-highspeed  (fast variant, 204K context)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# MiniMax API base URL
MINIMAX_BASE_URL = "https://api.minimax.io/v1"

# Known MiniMax models
MINIMAX_MODELS = [
    "MiniMax-M2.7",
    "MiniMax-M2.7-highspeed",
    "MiniMax-M2.5",
    "MiniMax-M2.5-highspeed",
]


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
    """

    def __init__(
        self, analysis_result: Any, message: str = "Request requires verification"
    ) -> None:
        self.analysis_result = analysis_result
        super().__init__(message)


# ---------------------------------------------------------------------------
# MiniMaxProvider — detection & extraction logic
# ---------------------------------------------------------------------------

class MiniMaxProvider:
    """
    MiniMax provider integration for Guardian SDK.

    MiniMax exposes an OpenAI-compatible chat completions API, so this
    provider wraps an ``openai.OpenAI`` client that has been configured
    with ``base_url="https://api.minimax.io/v1"`` and a MiniMax API key.

    Intercepts ``client.chat.completions.create()`` calls and runs
    Guardian threat detection before allowing the request to reach the
    MiniMax API.  Maintains full API compatibility.
    """

    def __init__(self, guardian_instance: Any) -> None:
        self.guardian = guardian_instance
        self.provider_name = "minimax"

    def wrap_client(self, client: Any) -> "ProtectedMiniMaxClient":
        """
        Wrap an OpenAI client (configured for MiniMax) with Guardian protection.

        Args:
            client: An ``openai.OpenAI`` instance with ``base_url`` set to
                    the MiniMax API endpoint.

        Returns:
            A ``ProtectedMiniMaxClient`` that maintains API compatibility.
        """
        try:
            import openai  # noqa: F401
        except ImportError:
            raise ProviderError(
                "openai package not installed. "
                'Run: pip install "ethicore-engine-guardian[minimax]"'
            )

        if not self._is_openai_client(client):
            raise ProviderError(f"Expected OpenAI client, got {type(client)}")

        return ProtectedMiniMaxClient(client, self.guardian)

    @staticmethod
    def _is_openai_client(client: Any) -> bool:
        """Return True if *client* is a recognised OpenAI client type."""
        client_type = str(type(client)).lower()
        return "openai" in client_type

    # ------------------------------------------------------------------
    # Prompt extraction — handles OpenAI-compatible messages format
    # ------------------------------------------------------------------

    def extract_prompt(self, **kwargs: Any) -> str:
        """
        Extract user-visible prompt text from ``chat.completions.create()`` kwargs.

        Supports:
        - ``messages=[{"role": "user", "content": "..."}]``
        - ``messages=[{"role": "user", "content": [{"type": "text", "text": "..."}]}]``
        """
        if "messages" not in kwargs:
            # Legacy completions format
            prompt = kwargs.get("prompt", "")
            return prompt if isinstance(prompt, str) else str(prompt)

        messages: List[Dict[str, Any]] = kwargs["messages"]
        if not messages:
            return ""

        # Get the last user message (most relevant for threat detection)
        user_messages = [m for m in messages if m.get("role") == "user"]
        if not user_messages:
            return ""

        last_user = user_messages[-1]
        content = last_user.get("content", "")

        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            # Multimodal content blocks
            parts: List[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return " ".join(parts)

        return str(content)


# ---------------------------------------------------------------------------
# ProtectedMiniMaxClient — thin proxy that intercepts chat.completions.create()
# ---------------------------------------------------------------------------

class ProtectedMiniMaxClient:
    """
    Proxy around an OpenAI client (configured for MiniMax) that intercepts
    ``chat.completions.create()`` calls and runs Guardian analysis first.

    All other attributes and methods are delegated to the original client
    via ``__getattr__`` so callers need not change any other code.
    """

    def __init__(self, original_client: Any, guardian_instance: Any) -> None:
        self._original_client = original_client
        self._guardian = guardian_instance
        self._provider = MiniMaxProvider(guardian_instance)

        # Wrap the chat completions interface
        if hasattr(original_client, "chat"):
            self.chat = self._create_protected_chat()

        logger.debug("🛡️  MiniMax client protection enabled")

    # ------------------------------------------------------------------
    # Internal: build the protected chat namespace
    # ------------------------------------------------------------------

    def _create_protected_chat(self) -> "ProtectedChat":
        """Return a ProtectedChat object wrapping original_client.chat."""
        return ProtectedChat(
            self._original_client.chat,
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
        return f"ProtectedMiniMaxClient(original={repr(self._original_client)})"


# ---------------------------------------------------------------------------
# ProtectedChat / ProtectedCompletions — intercepts create() on
# client.chat.completions
# ---------------------------------------------------------------------------

class ProtectedChat:
    """Proxy around ``client.chat`` that intercepts ``completions.create()``."""

    def __init__(
        self,
        original_chat: Any,
        guardian_instance: Any,
        provider: MiniMaxProvider,
    ) -> None:
        self._original_chat = original_chat
        self._guardian = guardian_instance
        self._provider = provider

        # Preserve non-callable attributes
        for attr_name in dir(original_chat):
            if not attr_name.startswith("_") and attr_name != "completions":
                attr = getattr(original_chat, attr_name)
                if not callable(attr):
                    setattr(self, attr_name, attr)

        # Create protected completions
        if hasattr(original_chat, "completions"):
            self.completions = ProtectedCompletions(
                original_chat.completions, guardian_instance, provider
            )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original_chat, name)


class ProtectedCompletions:
    """
    Proxy around ``client.chat.completions`` that intercepts ``create()``.

    Principle 14 (Divine Safety): when analysis cannot complete (timeout,
    internal error) the call is blocked — fail-closed, not fail-open.
    """

    def __init__(
        self,
        original_completions: Any,
        guardian_instance: Any,
        provider: MiniMaxProvider,
    ) -> None:
        self._original_completions = original_completions
        self._guardian = guardian_instance
        self._provider = provider

        # Preserve non-callable attributes
        for attr_name in dir(original_completions):
            if not attr_name.startswith("_") and attr_name not in {"create", "acreate"}:
                attr = getattr(original_completions, attr_name)
                if not callable(attr):
                    setattr(self, attr_name, attr)

    # ------------------------------------------------------------------
    # Sync path
    # ------------------------------------------------------------------

    def create(self, **kwargs: Any) -> Any:
        """Protected synchronous ``chat.completions.create()``."""
        prompt_text = self._provider.extract_prompt(**kwargs)

        if prompt_text and prompt_text.strip():
            analysis = self._run_analysis_sync(prompt_text, kwargs)
            self._enforce_policy(analysis, prompt_text)

        return self._original_completions.create(**kwargs)

    def _run_analysis_sync(self, prompt_text: str, request_kwargs: Dict[str, Any]) -> Any:
        """Run Guardian analysis, handling sync/async context differences."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop:
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

    async def acreate(self, **kwargs: Any) -> Any:
        """Protected async ``chat.completions.create()``."""
        prompt_text = self._provider.extract_prompt(**kwargs)

        if prompt_text and prompt_text.strip():
            analysis = await self._analyze(prompt_text, kwargs)
            self._enforce_policy(analysis, prompt_text)

        return await self._original_completions.acreate(**kwargs)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    async def _analyze(self, prompt_text: str, request_kwargs: Dict[str, Any]) -> Any:
        """Run Guardian analysis with MiniMax-specific context metadata."""
        context: Dict[str, Any] = {
            "api_call": "minimax.chat.completions.create",
            "provider": "minimax",
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
                "🚨 BLOCKED MiniMax request — %s: %.100s…",
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
                "⚠️  CHALLENGE MiniMax request — %s: %.100s…",
                analysis.threat_level,
                prompt_text,
            )
            logger.warning("   Reasons: %s", reason_str)
            if self._guardian.config.strict_mode:
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
        """Delegate unknown attributes to the original completions object."""
        return getattr(self._original_completions, name)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def create_protected_minimax_client(
    api_key: str,
    guardian_api_key: str,
    base_url: str = MINIMAX_BASE_URL,
    **openai_kwargs: Any,
) -> ProtectedMiniMaxClient:
    """
    Create a Guardian-protected MiniMax client in one step.

    Uses the ``openai`` SDK configured with MiniMax's API endpoint.

    Args:
        api_key:          MiniMax API key.
        guardian_api_key:  Guardian API key.
        base_url:         MiniMax API base URL (default: https://api.minimax.io/v1).
        **openai_kwargs:  Extra kwargs forwarded to ``openai.OpenAI()``.

    Returns:
        A ``ProtectedMiniMaxClient`` ready for use as a drop-in replacement.

    Example::

        client = create_protected_minimax_client(
            api_key="your-minimax-api-key",
            guardian_api_key="ethicore-...",
        )
        response = client.chat.completions.create(
            model="MiniMax-M2.7",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
        )
    """
    try:
        import openai
    except ImportError:
        raise ProviderError(
            "openai package not installed. "
            'Run: pip install "ethicore-engine-guardian[minimax]"'
        )

    minimax_client = openai.OpenAI(
        api_key=api_key, base_url=base_url, **openai_kwargs
    )

    from ..guardian import Guardian

    guardian = Guardian(api_key=guardian_api_key)

    provider = MiniMaxProvider(guardian)
    return provider.wrap_client(minimax_client)
