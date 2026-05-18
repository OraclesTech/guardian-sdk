"""
Ethicore Engine™ - Guardian SDK — Local OpenAI-Compatible Providers
Version: 1.0.0

Covers any inference server that exposes an OpenAI-compatible REST API
at a local address.  All four layers of Guardian protection apply:

    Layer 1 — Prompt scan (pre-flight, before any tokens are generated)
    Layer 2 — Inbound tool result scan (injection in tool/function results)
    Layer 3 — Outbound tool call scan (dangerous calls issued by the model)

Named convenience classes with their default ports:

    LMStudioProvider   → http://localhost:1234/v1   (LM Studio)
    LlamaCppProvider   → http://localhost:8080/v1   (llama.cpp server)
    LocalAIProvider    → http://localhost:8080/v1   (LocalAI)
    JanAIProvider      → http://localhost:1337/v1   (Jan.ai)

Any of these can be pointed at a non-default URL via the ``base_url``
argument to ``LocalOpenAICompatConfig``.

Usage — LM Studio example:
    from ethicore_guardian import Guardian
    from ethicore_guardian.providers import LMStudioProvider

    guardian = Guardian(api_key="eg-sk-...")
    provider = LMStudioProvider(guardian)
    client   = provider.wrap_client()   # ProtectedLocalClient

    response = client.chat.completions.create(
        model="lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF",
        messages=[{"role": "user", "content": user_input}],
    )

Copyright © 2026 Oracles Technologies LLC
All Rights Reserved
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
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
# Exceptions (mirror the OpenAI provider — same interface contract)
# ---------------------------------------------------------------------------

class ThreatBlockedException(Exception):
    """Raised when Guardian issues a BLOCK verdict."""
    def __init__(self, analysis_result: Any, message: str = "Threat detected and blocked") -> None:
        self.analysis_result = analysis_result
        super().__init__(message)


class ThreatChallengeException(Exception):
    """
    Raised when Guardian issues a CHALLENGE verdict in non-strict mode.

    Callers should surface a secondary verification step (e.g. human review)
    rather than hard-blocking.  In strict_mode, CHALLENGE is escalated to
    ThreatBlockedException instead.
    """
    def __init__(self, analysis_result: Any, message: str = "Request requires verification") -> None:
        self.analysis_result = analysis_result
        super().__init__(message)


class ToolOutputBlockedException(ThreatBlockedException):
    """Raised when a tool result contains an injection payload."""
    def __init__(self, analysis_result: Any, message: str = "Tool output blocked — injection detected") -> None:
        super().__init__(analysis_result, message)


class AgentToolBlockedException(ThreatBlockedException):
    """Raised when the model issues a dangerous tool invocation."""
    def __init__(self, analysis_result: Any, message: str = "Agentic tool call blocked") -> None:
        super().__init__(analysis_result, message)


class LocalProviderError(Exception):
    """Configuration or connectivity error for a local provider."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LocalOpenAICompatConfig:
    """
    Connection settings for any local OpenAI-compatible inference server.

    Args:
        base_url:   Full URL including /v1, e.g. "http://localhost:1234/v1".
        api_key:    Most local servers ignore this, but the OpenAI SDK
                    requires a non-empty string.  Defaults to "local".
        timeout:    HTTP request timeout in seconds.
        provider_name: Human-readable name surfaced in logs and threat context.
        extra_headers: Optional headers forwarded on every request (e.g. for
                    LocalAI auth or custom proxies).
    """
    base_url: str = "http://localhost:1234/v1"
    api_key: str = "local"
    timeout: int = 60
    provider_name: str = "local"
    extra_headers: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Base provider
# ---------------------------------------------------------------------------

class LocalOpenAICompatProvider:
    """
    Guardian-protected wrapper for any OpenAI-compatible local inference server.

    Instantiate this directly with a custom ``LocalOpenAICompatConfig``, or use
    one of the named convenience subclasses (LMStudioProvider, LlamaCppProvider,
    LocalAIProvider, JanAIProvider) which pre-fill sensible defaults.
    """

    def __init__(self,
                 guardian_instance: Any,
                 config: Optional[LocalOpenAICompatConfig] = None) -> None:
        self.guardian = guardian_instance
        self.config = config or LocalOpenAICompatConfig()
        logger.info(
            "🖥️  %s provider initialised: %s",
            self.config.provider_name,
            self.config.base_url,
        )

    def wrap_client(self) -> "ProtectedLocalClient":
        """
        Create and return a ProtectedLocalClient.

        The returned client exposes ``client.chat.completions.create(...)``
        and ``client.chat.completions.acreate(...)`` with Guardian protection
        wired in.  All other openai.OpenAI attributes are delegated unchanged.
        """
        try:
            import openai
        except ImportError:
            raise LocalProviderError(
                "openai package is required for local OpenAI-compatible providers. "
                "Run: pip install openai"
            )

        raw_client = openai.OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            timeout=self.config.timeout,
            default_headers=self.config.extra_headers or {},
        )
        return ProtectedLocalClient(raw_client, self.guardian, self.config)

    def async_wrap_client(self) -> "ProtectedAsyncLocalClient":
        """
        Create and return a ProtectedAsyncLocalClient (AsyncOpenAI under the hood).

        Use this inside async contexts (FastAPI, async agents, etc.).
        """
        try:
            import openai
        except ImportError:
            raise LocalProviderError(
                "openai package is required. Run: pip install openai"
            )

        raw_client = openai.AsyncOpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            timeout=self.config.timeout,
            default_headers=self.config.extra_headers or {},
        )
        return ProtectedAsyncLocalClient(raw_client, self.guardian, self.config)

    def get_available_models(self) -> List[str]:
        """
        Query /v1/models and return model IDs.

        Returns an empty list if the server is unreachable or returns no models.
        """
        try:
            import openai
            raw = openai.OpenAI(
                base_url=self.config.base_url,
                api_key=self.config.api_key,
                timeout=10,
            )
            models = raw.models.list()
            return [m.id for m in models.data]
        except Exception as exc:
            logger.warning(
                "⚠️  Could not list models from %s: %s",
                self.config.base_url, exc,
            )
            return []


# ---------------------------------------------------------------------------
# Protected sync client
# ---------------------------------------------------------------------------

class ProtectedLocalClient:
    """
    Sync client with Guardian protection — drop-in replacement for openai.OpenAI
    pointed at a local server.
    """

    def __init__(self,
                 raw_client: Any,
                 guardian_instance: Any,
                 config: LocalOpenAICompatConfig) -> None:
        self._raw = raw_client
        self._guardian = guardian_instance
        self._config = config
        self.chat = _ProtectedChatNamespace(raw_client.chat, guardian_instance, config)
        logger.debug("🛡️  Protected %s client created (sync)", config.provider_name)

    def models(self):
        """Delegate model listing to the raw client."""
        return self._raw.models

    def __getattr__(self, name: str) -> Any:
        return getattr(self._raw, name)


class _ProtectedChatNamespace:
    """Mirrors openai.OpenAI().chat — exposes .completions."""

    def __init__(self, raw_chat: Any, guardian: Any, config: LocalOpenAICompatConfig) -> None:
        self.completions = _ProtectedCompletions(raw_chat.completions, guardian, config)


class _ProtectedCompletions:
    """Mirrors openai.OpenAI().chat.completions — exposes .create()."""

    def __init__(self, raw_completions: Any, guardian: Any, config: LocalOpenAICompatConfig) -> None:
        self._raw = raw_completions
        self._guardian = guardian
        self._config = config

    # ── sync create ────────────────────────────────────────────────────────

    def create(self, **kwargs: Any) -> Any:
        """
        Protected sync create — all three agentic layers.

        Layer 1: prompt scan before sending to local model.
        Layer 2: tool result scan (inbound injection protection).
        Layer 3: tool call scan after model responds (outbound escalation).
        """
        messages: List[Dict[str, Any]] = kwargs.get("messages", [])

        # Layer 1 — prompt
        prompt = _extract_user_text(messages)
        if prompt:
            analysis = run_sync(self._analyze(prompt, kwargs))
            _enforce_policy(analysis, prompt, self._guardian)

        # Layer 2 — inbound tool results
        tool_results = extract_openai_tool_results(messages)
        if tool_results:
            run_sync(scan_inbound_tool_results(
                self._guardian, tool_results, ToolOutputBlockedException
            ))

        # Layer 3 — execute then scan outbound tool calls
        response = self._raw.create(**kwargs)
        tool_calls = extract_openai_tool_calls(response)
        if tool_calls:
            run_sync(scan_outbound_tool_calls(
                self._guardian, tool_calls, AgentToolBlockedException
            ))
        return response

    async def acreate(self, **kwargs: Any) -> Any:
        """Async variant (delegates to AsyncOpenAI — not usable from sync client)."""
        raise LocalProviderError(
            "acreate() is only available on ProtectedAsyncLocalClient. "
            "Use async_wrap_client() on the provider."
        )

    async def _analyze(self, prompt: str, kwargs: Dict[str, Any]) -> Any:
        context = {
            "provider": self._config.provider_name,
            "base_url": self._config.base_url,
            "model": kwargs.get("model", "unknown"),
            "local_llm": True,
            "request_size": len(prompt),
        }
        return await self._guardian.analyze(prompt, context)


# ---------------------------------------------------------------------------
# Protected async client
# ---------------------------------------------------------------------------

class ProtectedAsyncLocalClient:
    """
    Async client with Guardian protection — drop-in for openai.AsyncOpenAI
    pointed at a local server.
    """

    def __init__(self,
                 raw_client: Any,
                 guardian_instance: Any,
                 config: LocalOpenAICompatConfig) -> None:
        self._raw = raw_client
        self._guardian = guardian_instance
        self._config = config
        self.chat = _ProtectedAsyncChatNamespace(raw_client.chat, guardian_instance, config)
        logger.debug("🛡️  Protected %s client created (async)", config.provider_name)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._raw, name)


class _ProtectedAsyncChatNamespace:
    def __init__(self, raw_chat: Any, guardian: Any, config: LocalOpenAICompatConfig) -> None:
        self.completions = _ProtectedAsyncCompletions(raw_chat.completions, guardian, config)


class _ProtectedAsyncCompletions:
    def __init__(self, raw_completions: Any, guardian: Any, config: LocalOpenAICompatConfig) -> None:
        self._raw = raw_completions
        self._guardian = guardian
        self._config = config

    async def create(self, **kwargs: Any) -> Any:
        """
        Protected async create — all three agentic layers.
        """
        messages: List[Dict[str, Any]] = kwargs.get("messages", [])

        # Layer 1 — prompt
        prompt = _extract_user_text(messages)
        if prompt:
            analysis = await self._analyze(prompt, kwargs)
            _enforce_policy(analysis, prompt, self._guardian)

        # Layer 2 — inbound tool results
        tool_results = extract_openai_tool_results(messages)
        if tool_results:
            await scan_inbound_tool_results(
                self._guardian, tool_results, ToolOutputBlockedException
            )

        # Layer 3 — execute then scan outbound tool calls
        response = await self._raw.create(**kwargs)
        tool_calls = extract_openai_tool_calls(response)
        if tool_calls:
            await scan_outbound_tool_calls(
                self._guardian, tool_calls, AgentToolBlockedException
            )
        return response

    async def _analyze(self, prompt: str, kwargs: Dict[str, Any]) -> Any:
        context = {
            "provider": self._config.provider_name,
            "base_url": self._config.base_url,
            "model": kwargs.get("model", "unknown"),
            "local_llm": True,
            "request_size": len(prompt),
        }
        return await self._guardian.analyze(prompt, context)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_user_text(messages: List[Dict[str, Any]]) -> str:
    """Return the last user message text from an OpenAI-format message list."""
    user_msgs = [m for m in messages if m.get("role") == "user"]
    if not user_msgs:
        return ""
    content = user_msgs[-1].get("content", "")
    if isinstance(content, list):
        # Multi-part content (vision / tool use)
        return " ".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(content)


def _enforce_policy(analysis: Any, prompt: str, guardian: Any) -> None:
    """
    Apply Guardian's recommended_action to the analysis result.

    BLOCK                  → always raise ThreatBlockedException
    CHALLENGE + strict     → escalate to ThreatBlockedException
    CHALLENGE + non-strict → raise ThreatChallengeException
    ALLOW                  → pass through silently
    """
    action = getattr(analysis, "recommended_action", None)
    level  = getattr(analysis, "threat_level", "UNKNOWN")
    reasons: List[str] = getattr(analysis, "reasoning", [])

    if action == "BLOCK":
        logger.warning("🚨 BLOCKED local LLM request — %s: %.100s…", level, prompt)
        logger.warning("   Reasons: %s", ", ".join(reasons[:2]))
        raise ThreatBlockedException(
            analysis_result=analysis,
            message=(
                f"Local LLM request blocked: {level} threat detected. "
                f"Reasons: {', '.join(reasons[:2])}"
            ),
        )

    if action == "CHALLENGE":
        logger.warning("⚠️  CHALLENGE local LLM request — %s: %.100s…", level, prompt)
        logger.warning("   Reasons: %s", ", ".join(reasons[:2]))
        if getattr(getattr(guardian, "config", None), "strict_mode", False):
            raise ThreatBlockedException(
                analysis_result=analysis,
                message=(
                    f"Local LLM request blocked (strict mode — CHALLENGE): "
                    f"{level} threat detected."
                ),
            )
        raise ThreatChallengeException(
            analysis_result=analysis,
            message=f"Local LLM request requires verification: {level} threat level.",
        )


# ---------------------------------------------------------------------------
# Named convenience providers
# ---------------------------------------------------------------------------

class LMStudioProvider(LocalOpenAICompatProvider):
    """
    Guardian-protected provider for LM Studio (default port 1234).

    LM Studio exposes an OpenAI-compatible server at http://localhost:1234/v1.
    Models are referenced by their full GGUF identifier as shown in the LM
    Studio UI (e.g. "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF").

    Example:
        provider = LMStudioProvider(guardian)
        client   = provider.wrap_client()
        response = client.chat.completions.create(
            model="lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF",
            messages=[{"role": "user", "content": prompt}],
        )
    """
    DEFAULT_BASE_URL = "http://localhost:1234/v1"

    def __init__(self,
                 guardian_instance: Any,
                 base_url: Optional[str] = None,
                 timeout: int = 60) -> None:
        config = LocalOpenAICompatConfig(
            base_url=base_url or self.DEFAULT_BASE_URL,
            api_key="lmstudio",
            timeout=timeout,
            provider_name="lmstudio",
        )
        super().__init__(guardian_instance, config)


class LlamaCppProvider(LocalOpenAICompatProvider):
    """
    Guardian-protected provider for llama.cpp server (default port 8080).

    llama.cpp's ``--server`` mode exposes an OpenAI-compatible API at
    http://localhost:8080.  Start the server with:
        ./llama-server -m model.gguf --host 0.0.0.0 --port 8080

    The model name reported by /v1/models is the filename.  Pass it verbatim
    or use "local-model" which llama.cpp also accepts.

    Example:
        provider = LlamaCppProvider(guardian)
        client   = provider.wrap_client()
        response = client.chat.completions.create(
            model="local-model",
            messages=[{"role": "user", "content": prompt}],
        )
    """
    DEFAULT_BASE_URL = "http://localhost:8080/v1"

    def __init__(self,
                 guardian_instance: Any,
                 base_url: Optional[str] = None,
                 timeout: int = 60) -> None:
        config = LocalOpenAICompatConfig(
            base_url=base_url or self.DEFAULT_BASE_URL,
            api_key="llamacpp",
            timeout=timeout,
            provider_name="llamacpp",
        )
        super().__init__(guardian_instance, config)


class LocalAIProvider(LocalOpenAICompatProvider):
    """
    Guardian-protected provider for LocalAI (default port 8080).

    LocalAI (https://localai.io) is a drop-in OpenAI-compatible server that
    supports GGUF, GPTQ, AWQ, and many other formats.  Start with:
        local-ai --address :8080

    Example:
        provider = LocalAIProvider(guardian, base_url="http://localhost:8080/v1")
        client   = provider.wrap_client()
        response = client.chat.completions.create(
            model="mistral-7b-instruct",
            messages=[{"role": "user", "content": prompt}],
        )
    """
    DEFAULT_BASE_URL = "http://localhost:8080/v1"

    def __init__(self,
                 guardian_instance: Any,
                 base_url: Optional[str] = None,
                 timeout: int = 60) -> None:
        config = LocalOpenAICompatConfig(
            base_url=base_url or self.DEFAULT_BASE_URL,
            api_key="localai",
            timeout=timeout,
            provider_name="localai",
        )
        super().__init__(guardian_instance, config)


class JanAIProvider(LocalOpenAICompatProvider):
    """
    Guardian-protected provider for Jan.ai (default port 1337).

    Jan.ai (https://jan.ai) exposes an OpenAI-compatible API at
    http://localhost:1337/v1.  Enable it under Jan → Settings → Local API Server.

    Example:
        provider = JanAIProvider(guardian)
        client   = provider.wrap_client()
        response = client.chat.completions.create(
            model="mistral-ins-7b-q4",
            messages=[{"role": "user", "content": prompt}],
        )
    """
    DEFAULT_BASE_URL = "http://localhost:1337/v1"

    def __init__(self,
                 guardian_instance: Any,
                 base_url: Optional[str] = None,
                 timeout: int = 60) -> None:
        config = LocalOpenAICompatConfig(
            base_url=base_url or self.DEFAULT_BASE_URL,
            api_key="janai",
            timeout=timeout,
            provider_name="janai",
        )
        super().__init__(guardian_instance, config)


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------

def create_protected_lmstudio_client(
    guardian_api_key: str,
    base_url: str = "http://localhost:1234/v1",
    strict_mode: bool = True,
) -> ProtectedLocalClient:
    """
    One-step factory: Guardian + LM Studio client.

    Args:
        guardian_api_key: Your Ethicore Engine API key (eg-sk-…).
        base_url:         LM Studio server URL (override if on a non-default port).
        strict_mode:      When True, CHALLENGE verdicts escalate to BLOCK.

    Returns:
        ProtectedLocalClient ready for use.

    Example:
        client = create_protected_lmstudio_client("eg-sk-...")
        response = client.chat.completions.create(
            model="lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF",
            messages=[{"role": "user", "content": user_input}],
        )
    """
    from ethicore_guardian import Guardian
    guardian = Guardian(api_key=guardian_api_key, strict_mode=strict_mode)
    return LMStudioProvider(guardian, base_url=base_url).wrap_client()


def create_protected_llamacpp_client(
    guardian_api_key: str,
    base_url: str = "http://localhost:8080/v1",
    strict_mode: bool = True,
) -> ProtectedLocalClient:
    """
    One-step factory: Guardian + llama.cpp server client.

    Args:
        guardian_api_key: Your Ethicore Engine API key (eg-sk-…).
        base_url:         llama.cpp server URL.
        strict_mode:      When True, CHALLENGE verdicts escalate to BLOCK.

    Returns:
        ProtectedLocalClient ready for use.
    """
    from ethicore_guardian import Guardian
    guardian = Guardian(api_key=guardian_api_key, strict_mode=strict_mode)
    return LlamaCppProvider(guardian, base_url=base_url).wrap_client()


def create_protected_localai_client(
    guardian_api_key: str,
    base_url: str = "http://localhost:8080/v1",
    strict_mode: bool = True,
) -> ProtectedLocalClient:
    """
    One-step factory: Guardian + LocalAI client.

    Args:
        guardian_api_key: Your Ethicore Engine API key (eg-sk-…).
        base_url:         LocalAI server URL.
        strict_mode:      When True, CHALLENGE verdicts escalate to BLOCK.

    Returns:
        ProtectedLocalClient ready for use.
    """
    from ethicore_guardian import Guardian
    guardian = Guardian(api_key=guardian_api_key, strict_mode=strict_mode)
    return LocalAIProvider(guardian, base_url=base_url).wrap_client()


def create_protected_janai_client(
    guardian_api_key: str,
    base_url: str = "http://localhost:1337/v1",
    strict_mode: bool = True,
) -> ProtectedLocalClient:
    """
    One-step factory: Guardian + Jan.ai client.

    Args:
        guardian_api_key: Your Ethicore Engine API key (eg-sk-…).
        base_url:         Jan.ai server URL.
        strict_mode:      When True, CHALLENGE verdicts escalate to BLOCK.

    Returns:
        ProtectedLocalClient ready for use.
    """
    from ethicore_guardian import Guardian
    guardian = Guardian(api_key=guardian_api_key, strict_mode=strict_mode)
    return JanAIProvider(guardian, base_url=base_url).wrap_client()
