"""
Ethicore Engine™ - Guardian SDK — AWS Bedrock Provider
Version: 1.0.0

Wraps AWS Bedrock runtime clients (boto3.client('bedrock-runtime')) with
three-layer agentic protection via the Bedrock Converse API.

The Bedrock Converse API provides a unified, model-agnostic interface for
all Bedrock-hosted models (Claude, Llama, Titan, Mistral, etc.).  This
provider intercepts converse() and converse_stream() calls.

Layer 1: Prompt scan via guardian.analyze() — user/system messages
Layer 2: Inbound tool result scan — toolResult blocks in user messages
Layer 3: Outbound tool call scan — toolUse blocks in assistant response

Wire format handled by:
  extract_bedrock_tool_results()  — from _agentic_guards
  extract_bedrock_tool_calls()    — from _agentic_guards

All three layers include multilingual support — non-English payloads are
detected via MultilingualSemanticAnalyzer wired into scan_tool_output() and
scan_tool_call() in guardian.py.

Detection: boto3 client whose service model name contains 'bedrock'.

Principle 14 (Divine Safety): Bedrock hosts regulated-industry workloads
(healthcare, finance, government) where supply-chain attacks on tool
outputs carry the highest consequence; every result is scanned before
re-entering the model context.

Copyright © 2026 Oracles Technologies LLC
All Rights Reserved
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .base_provider import ProviderError
from ._agentic_guards import (
    run_sync,
    extract_bedrock_tool_results,
    extract_bedrock_tool_calls,
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
        super().__init__(message or "Bedrock request blocked by Guardian threat analysis.")


class ThreatChallengeException(Exception):
    """Raised when Layer 1 verdict is CHALLENGE and strict_mode is on."""
    def __init__(self, analysis_result: Any = None, message: str = ""):
        self.analysis_result = analysis_result
        super().__init__(message or "Bedrock request challenged by Guardian threat analysis.")


class ToolOutputBlockedException(ThreatBlockedException):
    """Raised when Layer 2 detects injection in a Bedrock toolResult."""
    def __init__(self, analysis_result: Any = None, message: str = ""):
        super().__init__(
            analysis_result,
            message or "Bedrock tool output blocked: injection payload detected.",
        )


class AgentToolBlockedException(ThreatBlockedException):
    """Raised when Layer 3 blocks a dangerous Bedrock toolUse call."""
    def __init__(self, analysis_result: Any = None, message: str = ""):
        super().__init__(
            analysis_result,
            message or "Bedrock outbound tool call blocked: dangerous operation detected.",
        )


# ==============================================================================
# PROMPT EXTRACTION
# ==============================================================================

def _extract_prompt(messages: List[Dict[str, Any]], system: Optional[List] = None) -> str:
    """
    Extract analysable text from Bedrock converse API messages.

    Handles:
    - Text content blocks: {"text": "..."}
    - System prompt blocks (passed separately in the converse API)
    - Skips toolResult blocks (scanned in Layer 2)
    - Skips toolUse blocks (Layer 3)
    """
    parts: List[str] = []

    # System prompt (list of {"text": "..."} dicts)
    for block in (system or []):
        if isinstance(block, dict) and "text" in block:
            parts.append(block["text"])

    for msg in (messages or []):
        for block in msg.get("content", []):
            if not isinstance(block, dict):
                continue
            # Skip tool-specific blocks
            if "toolResult" in block or "toolUse" in block:
                continue
            if "text" in block:
                parts.append(block["text"])

    return " ".join(parts)


# ==============================================================================
# PROTECTED CLIENT
# ==============================================================================

class ProtectedBedrockClient:
    """
    Drop-in replacement for boto3.client('bedrock-runtime') with Guardian
    protection applied to converse() and converse_stream() calls.

    All other boto3 methods (list_foundation_models, get_model, etc.) are
    proxied transparently via __getattr__.
    """

    def __init__(self, original_client: Any, guardian_instance: Any) -> None:
        self._original_client = original_client
        self._guardian = guardian_instance

    # ------------------------------------------------------------------
    # converse  (sync)
    # ------------------------------------------------------------------

    def converse(self, **kwargs: Any) -> Any:
        """
        Sync converse() with three-layer Guardian protection.

        Raises:
            ThreatBlockedException:      Layer 1 threat.
            ThreatChallengeException:    Layer 1 challenge in strict mode.
            ToolOutputBlockedException:  Layer 2 injection in toolResult.
            AgentToolBlockedException:   Layer 3 dangerous toolUse.
        """
        messages: List[Dict[str, Any]] = kwargs.get("messages", [])
        system: Optional[List] = kwargs.get("system")
        strict_mode: bool = getattr(
            getattr(self._guardian, "config", None), "strict_mode", False
        )

        # ── Layer 1: prompt scan ─────────────────────────────────────
        prompt_text = _extract_prompt(messages, system)
        if prompt_text.strip():
            analysis = run_sync(
                self._guardian.analyze(
                    prompt_text,
                    context={
                        "provider": "bedrock",
                        "model": kwargs.get("modelId", ""),
                    },
                )
            )
            action = getattr(analysis, "recommended_action", "ALLOW")
            if action == "BLOCK":
                raise ThreatBlockedException(
                    analysis_result=analysis,
                    message=(
                        f"Bedrock request blocked (score="
                        f"{getattr(analysis, 'threat_score', 0):.2f})."
                    ),
                )
            if action == "CHALLENGE" and strict_mode:
                raise ThreatChallengeException(
                    analysis_result=analysis,
                    message="Bedrock request challenged in strict mode.",
                )

        # ── Layer 2: inbound tool result scan ────────────────────────
        tool_results = extract_bedrock_tool_results(messages)
        if tool_results:
            run_sync(
                scan_inbound_tool_results(
                    self._guardian, tool_results, ToolOutputBlockedException
                )
            )

        # ── Execute ──────────────────────────────────────────────────
        response = self._original_client.converse(**kwargs)

        # ── Layer 3: outbound tool call scan ─────────────────────────
        tool_calls = extract_bedrock_tool_calls(response)
        if tool_calls:
            run_sync(
                scan_outbound_tool_calls(
                    self._guardian, tool_calls, AgentToolBlockedException
                )
            )

        return response

    # ------------------------------------------------------------------
    # converse_stream  (sync — returns streaming response)
    # ------------------------------------------------------------------

    def converse_stream(self, **kwargs: Any) -> Any:
        """
        converse_stream() with Layer 1 and Layer 2 protection applied
        before streaming begins.  Layer 3 is not applied to streaming
        responses since tool calls arrive incrementally; use converse()
        for agentic tool-call workflows.
        """
        messages: List[Dict[str, Any]] = kwargs.get("messages", [])
        system: Optional[List] = kwargs.get("system")
        strict_mode: bool = getattr(
            getattr(self._guardian, "config", None), "strict_mode", False
        )

        # Layer 1
        prompt_text = _extract_prompt(messages, system)
        if prompt_text.strip():
            analysis = run_sync(
                self._guardian.analyze(
                    prompt_text,
                    context={
                        "provider": "bedrock",
                        "model": kwargs.get("modelId", ""),
                    },
                )
            )
            action = getattr(analysis, "recommended_action", "ALLOW")
            if action == "BLOCK":
                raise ThreatBlockedException(
                    analysis_result=analysis,
                    message="Bedrock streaming request blocked by Guardian.",
                )
            if action == "CHALLENGE" and strict_mode:
                raise ThreatChallengeException(
                    analysis_result=analysis,
                    message="Bedrock streaming request challenged in strict mode.",
                )

        # Layer 2
        tool_results = extract_bedrock_tool_results(messages)
        if tool_results:
            run_sync(
                scan_inbound_tool_results(
                    self._guardian, tool_results, ToolOutputBlockedException
                )
            )

        return self._original_client.converse_stream(**kwargs)

    def __getattr__(self, name: str) -> Any:
        """Proxy all other boto3 client methods unchanged."""
        return getattr(self._original_client, name)


# ==============================================================================
# PROVIDER CLASS
# ==============================================================================

class BedrockProvider:
    """Guardian provider for AWS Bedrock (boto3 bedrock-runtime client)."""

    provider_name = "bedrock"

    def __init__(self, guardian_instance: Any) -> None:
        self.guardian = guardian_instance

    @staticmethod
    def _is_bedrock_client(client: Any) -> bool:
        """
        Return True if *client* is a boto3 bedrock-runtime client.

        Checks the boto3 service model name (most reliable) and falls back
        to class/module name matching.
        """
        # Primary check: boto3 service model
        try:
            svc = client.meta.service_model.service_name.lower()
            if "bedrock" in svc:
                return True
        except Exception:
            pass
        # Fallback: class/module string
        client_type = str(type(client)).lower()
        module = getattr(client, "__module__", "") or ""
        return "bedrock" in client_type or "bedrock" in module.lower()

    def wrap_client(self, client: Any) -> ProtectedBedrockClient:
        """
        Wrap a boto3 bedrock-runtime client with Guardian protection.

        Args:
            client: boto3.client('bedrock-runtime') instance.

        Returns:
            ProtectedBedrockClient with three-layer agentic protection.
        """
        if not self._is_bedrock_client(client):
            raise ProviderError(
                f"Expected boto3 bedrock-runtime client, got {type(client).__name__}. "
                "Pass boto3.client('bedrock-runtime')."
            )
        return ProtectedBedrockClient(client, self.guardian)

    def extract_prompt(self, **kwargs: Any) -> str:
        return _extract_prompt(
            kwargs.get("messages", []),
            kwargs.get("system"),
        )


# ==============================================================================
# FACTORY
# ==============================================================================

def create_protected_bedrock_client(
    bedrock_client: Any,
    guardian_api_key: str,
) -> ProtectedBedrockClient:
    """
    Convenience factory: wrap an existing boto3 bedrock-runtime client.

    Args:
        bedrock_client:   boto3.client('bedrock-runtime') instance.
        guardian_api_key: Ethicore Guardian API key.

    Returns:
        ProtectedBedrockClient with three-layer agentic protection.

    Example::

        import boto3
        from ethicore_guardian.providers.bedrock_provider import create_protected_bedrock_client

        client = boto3.client("bedrock-runtime", region_name="us-east-1")
        protected = create_protected_bedrock_client(client, guardian_api_key="eg-sk-...")

        response = protected.converse(
            modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
            messages=[{"role": "user", "content": [{"text": "Summarise this report."}]}],
        )
    """
    from ..guardian import Guardian
    guardian = Guardian(api_key=guardian_api_key)
    provider = BedrockProvider(guardian)
    return provider.wrap_client(bedrock_client)


__all__ = [
    "BedrockProvider",
    "ProtectedBedrockClient",
    "create_protected_bedrock_client",
    "ThreatBlockedException",
    "ThreatChallengeException",
    "ToolOutputBlockedException",
    "AgentToolBlockedException",
]
