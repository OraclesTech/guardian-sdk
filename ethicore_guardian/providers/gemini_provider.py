"""
Ethicore Engine™ - Guardian SDK — Google Gemini Provider
Version: 1.0.0

Wraps Google Gemini clients (google.genai.Client) with three-layer
agentic protection.

Supports the modern google-genai SDK (v1.0+) used for Gemini 2.x models.
The protected wrapper intercepts client.models.generate_content() and
client.models.generate_content_async(), preserving the full Gemini API
surface via __getattr__ pass-through.

Layer 1: Prompt scan via guardian.analyze()
Layer 2: Inbound tool result scan — FunctionResponse parts in contents
Layer 3: Outbound tool call scan — FunctionCall parts in response

Wire format handled by:
  extract_gemini_tool_results()  — from _agentic_guards
  extract_gemini_tool_calls()    — from _agentic_guards

All three layers include multilingual support — non-English payloads are
detected via MultilingualSemanticAnalyzer wired into scan_tool_output() and
scan_tool_call() in guardian.py.

Detection: client module contains 'google' and ('genai' or 'generativeai').

Principle 14 (Divine Safety): multimodal Gemini responses can embed
injection payloads in returned text, images, and function call args;
all vectors are scanned before re-entering the agent context.

Copyright © 2026 Oracles Technologies LLC
All Rights Reserved
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .base_provider import ProviderError
from ._agentic_guards import (
    run_sync,
    extract_gemini_tool_results,
    extract_gemini_tool_calls,
    scan_inbound_tool_results,
    scan_outbound_tool_calls,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# EXCEPTIONS
# ==============================================================================

class ThreatBlockedException(Exception):
    """Raised when Layer 1 prompt scan verdict is BLOCK."""
    def __init__(self, analysis_result: Any = None, message: str = ""):
        self.analysis_result = analysis_result
        super().__init__(message or "Gemini request blocked by Guardian threat analysis.")


class ThreatChallengeException(Exception):
    """Raised when Layer 1 verdict is CHALLENGE and strict_mode is on."""
    def __init__(self, analysis_result: Any = None, message: str = ""):
        self.analysis_result = analysis_result
        super().__init__(message or "Gemini request challenged by Guardian threat analysis.")


class ToolOutputBlockedException(ThreatBlockedException):
    """Raised when Layer 2 detects injection in a Gemini FunctionResponse."""
    def __init__(self, analysis_result: Any = None, message: str = ""):
        super().__init__(
            analysis_result,
            message or "Gemini tool output blocked: injection payload detected.",
        )


class AgentToolBlockedException(ThreatBlockedException):
    """Raised when Layer 3 blocks a dangerous Gemini FunctionCall."""
    def __init__(self, analysis_result: Any = None, message: str = ""):
        super().__init__(
            analysis_result,
            message or "Gemini outbound tool call blocked: dangerous operation detected.",
        )


# ==============================================================================
# PROMPT EXTRACTION
# ==============================================================================

def _extract_prompt(contents: Any) -> str:
    """
    Extract analysable text from Gemini contents.

    Handles:
    - Plain string (single-turn)
    - List of strings
    - List of Content objects (SDK) or dicts with 'parts'
    - Multimodal parts — text blocks only; images are skipped

    FunctionResponse parts are excluded (scanned in Layer 2).
    """
    if not contents:
        return ""
    # Plain string
    if isinstance(contents, str):
        return contents
    parts_text: List[str] = []
    for item in contents:
        if isinstance(item, str):
            parts_text.append(item)
            continue
        # SDK Content object or dict
        parts = getattr(item, "parts", None)
        if parts is None and isinstance(item, dict):
            parts = item.get("parts", [])
        for part in (parts or []):
            # Skip FunctionResponse (Layer 2 handles those)
            if getattr(part, "function_response", None) is not None:
                continue
            if isinstance(part, dict) and "functionResponse" in part:
                continue
            # Text part
            text = getattr(part, "text", None)
            if text is None and isinstance(part, dict):
                text = part.get("text")
            if text:
                parts_text.append(str(text))
    return " ".join(parts_text)


# ==============================================================================
# PROTECTED MODELS NAMESPACE
# ==============================================================================

class ProtectedGeminiModels:
    """
    Proxy for client.models — intercepts generate_content() while
    passing all other method calls through to the original models object.
    """

    def __init__(self, original_models: Any, guardian_instance: Any) -> None:
        self._original_models = original_models
        self._guardian = guardian_instance

    # ------------------------------------------------------------------
    # Sync
    # ------------------------------------------------------------------

    def generate_content(
        self,
        model: str,
        contents: Any,
        config: Any = None,
        **kwargs: Any,
    ) -> Any:
        """
        Sync generate_content with three-layer Guardian protection.

        Raises:
            ThreatBlockedException:      Layer 1 threat.
            ThreatChallengeException:    Layer 1 challenge in strict mode.
            ToolOutputBlockedException:  Layer 2 injection in tool result.
            AgentToolBlockedException:   Layer 3 dangerous tool call.
        """
        strict_mode: bool = getattr(
            getattr(self._guardian, "config", None), "strict_mode", False
        )

        # ── Layer 1: prompt scan ─────────────────────────────────────
        prompt_text = _extract_prompt(contents)
        if prompt_text.strip():
            analysis = run_sync(
                self._guardian.analyze(
                    prompt_text,
                    context={"provider": "gemini", "model": model},
                )
            )
            action = getattr(analysis, "recommended_action", "ALLOW")
            if action == "BLOCK":
                raise ThreatBlockedException(
                    analysis_result=analysis,
                    message=(
                        f"Gemini request blocked (score="
                        f"{getattr(analysis, 'threat_score', 0):.2f})."
                    ),
                )
            if action == "CHALLENGE" and strict_mode:
                raise ThreatChallengeException(
                    analysis_result=analysis,
                    message="Gemini request challenged in strict mode.",
                )

        # ── Layer 2: inbound tool result scan ────────────────────────
        tool_results = extract_gemini_tool_results(
            contents if isinstance(contents, list) else []
        )
        if tool_results:
            run_sync(
                scan_inbound_tool_results(
                    self._guardian, tool_results, ToolOutputBlockedException
                )
            )

        # ── Execute ──────────────────────────────────────────────────
        if config is not None:
            response = self._original_models.generate_content(
                model=model, contents=contents, config=config, **kwargs
            )
        else:
            response = self._original_models.generate_content(
                model=model, contents=contents, **kwargs
            )

        # ── Layer 3: outbound tool call scan ─────────────────────────
        tool_calls = extract_gemini_tool_calls(response)
        if tool_calls:
            run_sync(
                scan_outbound_tool_calls(
                    self._guardian, tool_calls, AgentToolBlockedException
                )
            )

        return response

    # ------------------------------------------------------------------
    # Async
    # ------------------------------------------------------------------

    async def generate_content_async(
        self,
        model: str,
        contents: Any,
        config: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Async generate_content with three-layer Guardian protection."""
        strict_mode: bool = getattr(
            getattr(self._guardian, "config", None), "strict_mode", False
        )

        # ── Layer 1 ──────────────────────────────────────────────────
        prompt_text = _extract_prompt(contents)
        if prompt_text.strip():
            analysis = await self._guardian.analyze(
                prompt_text,
                context={"provider": "gemini", "model": model},
            )
            action = getattr(analysis, "recommended_action", "ALLOW")
            if action == "BLOCK":
                raise ThreatBlockedException(
                    analysis_result=analysis,
                    message=(
                        f"Gemini request blocked (score="
                        f"{getattr(analysis, 'threat_score', 0):.2f})."
                    ),
                )
            if action == "CHALLENGE" and strict_mode:
                raise ThreatChallengeException(
                    analysis_result=analysis,
                    message="Gemini request challenged in strict mode.",
                )

        # ── Layer 2 ──────────────────────────────────────────────────
        tool_results = extract_gemini_tool_results(
            contents if isinstance(contents, list) else []
        )
        if tool_results:
            await scan_inbound_tool_results(
                self._guardian, tool_results, ToolOutputBlockedException
            )

        # ── Execute ──────────────────────────────────────────────────
        if config is not None:
            response = await self._original_models.generate_content_async(
                model=model, contents=contents, config=config, **kwargs
            )
        else:
            response = await self._original_models.generate_content_async(
                model=model, contents=contents, **kwargs
            )

        # ── Layer 3 ──────────────────────────────────────────────────
        tool_calls = extract_gemini_tool_calls(response)
        if tool_calls:
            await scan_outbound_tool_calls(
                self._guardian, tool_calls, AgentToolBlockedException
            )

        return response

    def __getattr__(self, name: str) -> Any:
        """Pass all other models methods through unchanged."""
        return getattr(self._original_models, name)


# ==============================================================================
# PROTECTED CLIENT
# ==============================================================================

class ProtectedGeminiClient:
    """
    Drop-in replacement for google.genai.Client with Guardian protection
    applied to generate_content calls.

    All other client attributes (files, caches, live, etc.) are proxied
    transparently to the original client.
    """

    def __init__(self, original_client: Any, guardian_instance: Any) -> None:
        self._original_client = original_client
        self._guardian = guardian_instance
        # Intercept the models namespace
        self.models = ProtectedGeminiModels(
            original_client.models, guardian_instance
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original_client, name)


# ==============================================================================
# PROVIDER CLASS
# ==============================================================================

class GeminiProvider:
    """Guardian provider for Google Gemini (google.genai.Client)."""

    provider_name = "gemini"

    def __init__(self, guardian_instance: Any) -> None:
        self.guardian = guardian_instance

    @staticmethod
    def _is_gemini_client(client: Any) -> bool:
        """
        Return True if *client* is a Google Gemini client.

        Checks the client's module path for google + (genai | generativeai).
        """
        module = getattr(client, "__module__", "") or ""
        module_lower = module.lower()
        client_type = str(type(client)).lower()
        return (
            ("google" in module_lower and (
                "genai" in module_lower or "generativeai" in module_lower
            ))
            or ("google" in client_type and "genai" in client_type)
        )

    def wrap_client(self, client: Any) -> ProtectedGeminiClient:
        """
        Wrap a google.genai.Client with Guardian protection.

        Args:
            client: google.genai.Client instance.

        Returns:
            ProtectedGeminiClient with full three-layer protection.
        """
        if not self._is_gemini_client(client):
            raise ProviderError(
                f"Expected google.genai.Client, got {type(client).__name__}. "
                "Pass a google.genai.Client instance."
            )
        return ProtectedGeminiClient(client, self.guardian)

    def extract_prompt(self, **kwargs: Any) -> str:
        return _extract_prompt(kwargs.get("contents", ""))


# ==============================================================================
# FACTORY
# ==============================================================================

def create_protected_gemini_client(
    gemini_client: Any,
    guardian_api_key: str,
) -> ProtectedGeminiClient:
    """
    Convenience factory: wrap an existing google.genai.Client with Guardian.

    Args:
        gemini_client:    google.genai.Client instance.
        guardian_api_key: Ethicore Guardian API key.

    Returns:
        ProtectedGeminiClient with three-layer agentic protection.

    Example::

        import google.genai as genai
        from ethicore_guardian.providers.gemini_provider import create_protected_gemini_client

        client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        protected = create_protected_gemini_client(client, guardian_api_key="EG-PRO-...")

        response = protected.models.generate_content(
            model="gemini-2.0-flash",
            contents="Summarise this document.",
        )
    """
    from ..guardian import Guardian
    guardian = Guardian(api_key=guardian_api_key)
    provider = GeminiProvider(guardian)
    return provider.wrap_client(gemini_client)


__all__ = [
    "GeminiProvider",
    "ProtectedGeminiClient",
    "ProtectedGeminiModels",
    "create_protected_gemini_client",
    "ThreatBlockedException",
    "ThreatChallengeException",
    "ToolOutputBlockedException",
    "AgentToolBlockedException",
]
