"""
Ethicore Engine® - Guardian SDK — System One (decision-model) Provider
Runtime protection for "System One" models — TypeSafe AI's **Jev** and its kin.

Why this is a NET-NEW provider (not covered by the OpenAI/base provider)
------------------------------------------------------------------------
Every other provider here speaks the OpenAI chat-completions wire format:
``messages[]`` in, ``choices[].message.content`` / ``tool_calls`` out. Guardian's
extraction (extract_prompt / extract_openai_tool_results / extract_openai_tool_calls)
is built around that shape.

A **System One model** does not. Named for Kahneman's fast, intuitive "System 1",
it is non-autoregressive: you send it a *state* plus a set of *questions* and it
returns **typed, calibrated decisions** — never free text. TypeSafe's Jev exposes
one endpoint, ``POST https://api.typesafe.ai/v1/systemone``:

    request  = {"state": <text|json>, "model": "jev-latest", "questions": {...}}
    response = {"model": ..., "answers": {...}, "usage": {...}}

with three decision primitives — **Noul** (yes/no), **Choice** (pick one of N),
**Score** (ordered level). There are no ``messages`` and no assistant text, so
routing a Jev call through the OpenAI provider would extract *nothing* and pass
everything through — hollow protection. Hence a dedicated provider.

The threat model inverts
------------------------
Classic LLM risks (jailbreak-to-text, hallucination, unsafe content, system-prompt
extraction) mostly do not apply — there is no text output to weaponise. What *does*
apply, and is arguably higher-stakes because the output is a control-flow decision:

    1. Poisoned STATE  → a flipped decision. The state is assembled from RAG, tool
       outputs, documents, and upstream agents — all untrusted. Guardian scans it
       (holistic ``analyze`` + per-field ``scan_tool_output``) BEFORE the model decides.
    2. Decision → ACTION. A System One answer usually gates something (a tool call,
       a branch, a payout). Guardian scans the intended action (``scan_tool_call``)
       BEFORE the caller executes it.

Graceful degradation: the per-field / per-action scans use Guardian's API-tier
methods; without an API key they skip (PermissionError) and the provider degrades to
holistic state analysis only — same contract as every other provider here.

Copyright © 2026 Oracles Technologies LLC
All Rights Reserved
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ._agentic_guards import (
    run_sync,
    scan_inbound_tool_results,
    scan_outbound_tool_calls,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TypeSafe / Jev constants — convenience only; never used for gating or validation.
# Source: https://docs.typesafe.ai/api
# ---------------------------------------------------------------------------

JEV_SYSTEMONE_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
JEV_DEFAULT_MODEL = "jev-latest"
JEV_PRIMITIVES = frozenset({"noul", "choice", "score"})  # yes/no · pick-one · ordered level


# ---------------------------------------------------------------------------
# Exceptions (mirror the house provider pattern)
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

    Principle 16 (Sacred Autonomy): surface uncertainty for human review rather
    than silently blocking. In strict_mode, CHALLENGE escalates to a block.
    """

    def __init__(self, analysis_result: Any, message: str = "Decision requires verification") -> None:
        self.analysis_result = analysis_result
        super().__init__(message)


class PoisonedStateException(ThreatBlockedException):
    """Raised when the state fed to a System One model is malicious/injected."""

    def __init__(self, analysis_result: Any, message: str = "State blocked — poisoned input detected") -> None:
        super().__init__(analysis_result, message)


class DecisionActionBlockedException(ThreatBlockedException):
    """Raised when an action triggered by a System One decision is dangerous."""

    def __init__(
        self,
        analysis_result: Any,
        tool_name: str = "",
        message: str = "Decision-triggered action blocked by Guardian",
    ) -> None:
        self.tool_name = tool_name
        super().__init__(analysis_result, message)


# ---------------------------------------------------------------------------
# SystemOneProvider — the reusable, SDK-agnostic guard logic
# ---------------------------------------------------------------------------

# A decision "answer" is whatever the model returned for a named question — a bool,
# a string/choice, a number/score, or a dict wrapping one of those with a probability.
Answer = Any
# An intended action derived from a decision: {"name": <tool>, "arguments": <dict>}.
Call = Dict[str, Any]


class SystemOneProvider:
    """
    Guardian protection for decision-only ("System One") models, independent of any
    specific vendor SDK. ``JevProvider`` is the concrete TypeSafe/Jev subclass.

    Two guard points map to the inverted threat model:

        guard_state(...)            — Layer 1 (holistic) + Layer 2 (per untrusted field)
        guard_decision_actions(...) — Layer 3 on the actions a decision triggers

    Both have sync and async forms; the sync forms bridge via ``run_sync``.
    """

    def __init__(
        self,
        guardian_instance: Any,
        *,
        provider_name: str = "system_one",
        model: Optional[str] = None,
        source_type: str = "tool_output",
    ) -> None:
        self.guardian = guardian_instance
        self.provider_name = provider_name
        self.model = model
        # State is typically assembled from retrieval/tools, so treat it as
        # tool-output-sourced by default (source-type-aware weighting in the engine).
        self.source_type = source_type

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def extract_state(state: Any) -> str:
        """Coerce a state (text or JSON-serialisable object) to analysable text."""
        if state is None:
            return ""
        if isinstance(state, str):
            return state
        try:
            return json.dumps(state, default=str, ensure_ascii=False)
        except Exception:  # pragma: no cover - defensive
            return str(state)

    @staticmethod
    def _normalize_fields(
        untrusted_fields: Optional[Union[Dict[str, Any], List[Any]]]
    ) -> List[Dict[str, Any]]:
        """Normalise untrusted state fields to [{tool_name, content}] for Layer 2."""
        if not untrusted_fields:
            return []
        out: List[Dict[str, Any]] = []
        if isinstance(untrusted_fields, dict):
            items = untrusted_fields.items()
        else:
            items = []
            for f in untrusted_fields:
                if isinstance(f, dict):
                    items.append((f.get("tool_name") or f.get("name") or "state_field",
                                  f.get("content")))
                elif isinstance(f, (list, tuple)) and len(f) == 2:
                    items.append((f[0], f[1]))
        for name, content in items:
            out.append({"tool_name": str(name), "content": content})
        return out

    @staticmethod
    def _normalize_calls(intended_calls: Optional[List[Any]]) -> List[Call]:
        """Normalise intended actions to [{name, arguments}] for Layer 3."""
        if not intended_calls:
            return []
        out: List[Call] = []
        for c in intended_calls:
            if not isinstance(c, dict):
                continue
            name = c.get("name") or c.get("tool_name")
            if not name:
                continue
            out.append({"name": str(name), "arguments": c.get("arguments", c.get("args", {})) or {}})
        return out

    @staticmethod
    def _answer_value(answer: Answer) -> Any:
        """Extract the decision value from an answer that may wrap it in a dict."""
        if isinstance(answer, dict):
            for k in ("value", "answer", "decision", "choice", "label", "score"):
                if k in answer:
                    return answer[k]
        return answer

    @classmethod
    def decisions_to_calls(
        cls,
        answers: Dict[str, Answer],
        action_map: Dict[str, Dict[str, Any]],
    ) -> List[Call]:
        """
        Turn typed decisions into the tool calls they would trigger, per a
        declarative map. Each ``action_map[question]`` is::

            {"name": <tool>, "arguments": <dict>, "when": <trigger>}

        ``when`` fires the action if:
          - omitted / None  → the answer is truthy (Noul "yes"/True, non-zero Score)
          - a callable      → ``when(value)`` is truthy
          - a set/list      → the Choice value is in it
          - any other value → the value equals it

        Only questions present in BOTH ``answers`` and ``action_map`` are considered.
        """
        calls: List[Call] = []
        # NB: bind ``x or {}`` to locals before iterating — inlining
        # ``(action_map or {}).items()`` crashes Cython's AnalyseExpressionsTransform
        # in some 3.0.x releases (a compiler bug), so keep the expressions simple.
        amap = action_map if action_map else {}
        ans = answers if answers else {}
        for q, spec in amap.items():
            if q not in ans:
                continue
            value = cls._answer_value(ans[q])
            when = spec.get("when", None)
            fire: bool
            if when is None:
                v = value
                if isinstance(v, str):
                    fire = v.strip().lower() in ("yes", "true", "1", "y")
                else:
                    fire = bool(v)
            elif callable(when):
                fire = bool(when(value))
            elif isinstance(when, (set, list, tuple)):
                fire = value in when
            else:
                fire = value == when
            if fire:
                calls.append({"name": spec["name"], "arguments": spec.get("arguments", {}) or {}})
        return calls

    # ------------------------------------------------------------------ Layer 1 + 2

    async def aguard_state(
        self,
        state: Any,
        *,
        untrusted_fields: Optional[Union[Dict[str, Any], List[Any]]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Guard the state BEFORE it is sent to the model.

        Layer 1 — ``guardian.analyze`` on the whole state (BLOCK → PoisonedStateException;
                   CHALLENGE → ThreatChallengeException, or block in strict_mode).
        Layer 2 — ``scan_tool_output`` on each declared untrusted field (RAG chunk, tool
                   output) → PoisonedStateException on an injection payload. Skipped
                   automatically at community tier (no API key).

        Returns the Layer-1 analysis (or None if the state was empty).
        """
        text = self.extract_state(state)
        analysis = None
        if text.strip():
            ctx: Dict[str, Any] = {
                "provider": self.provider_name,
                "model": self.model or "unknown",
                "system_one": True,
                "decision": True,
                "source_type": self.source_type,
                "request_size": len(text),
            }
            if context:
                ctx.update(context)
            analysis = await self.guardian.analyze(text, ctx)
            self._enforce_state_policy(analysis, text)

        fields = self._normalize_fields(untrusted_fields)
        if fields:
            await scan_inbound_tool_results(self.guardian, fields, PoisonedStateException)
        return analysis

    def guard_state(self, state: Any, **kwargs: Any) -> Any:
        """Synchronous ``aguard_state``."""
        return run_sync(self.aguard_state(state, **kwargs))

    # ------------------------------------------------------------------ Layer 3

    async def aguard_decision_actions(self, intended_calls: List[Any]) -> None:
        """
        Guard the actions a decision would trigger, BEFORE the caller executes them.

        Layer 3 — ``scan_tool_call`` per intended action → DecisionActionBlockedException
        on a dangerous call (or a CHALLENGE escalated in strict_mode). Skipped
        automatically at community tier (no API key).
        """
        calls = self._normalize_calls(intended_calls)
        if calls:
            await scan_outbound_tool_calls(self.guardian, calls, DecisionActionBlockedException)

    def guard_decision_actions(self, intended_calls: List[Any]) -> None:
        """Synchronous ``aguard_decision_actions``."""
        return run_sync(self.aguard_decision_actions(intended_calls))

    # ------------------------------------------------------------------ policy

    def _enforce_state_policy(self, analysis: Any, state_text: str) -> None:
        action = getattr(analysis, "recommended_action", None)
        level = getattr(analysis, "threat_level", "UNKNOWN")
        reasoning = getattr(analysis, "reasoning", []) or []
        reason_str = ", ".join(reasoning[:2]) if reasoning else "see analysis"

        if action == "BLOCK":
            logger.warning("🚨 BLOCKED %s state — %s: %.100s…", self.provider_name, level, state_text)
            raise PoisonedStateException(
                analysis_result=analysis,
                message=f"State blocked: {level} threat detected. Reasons: {reason_str}",
            )
        if action == "CHALLENGE":
            logger.warning("⚠️  CHALLENGE %s state — %s: %.100s…", self.provider_name, level, state_text)
            if getattr(getattr(self.guardian, "config", None), "strict_mode", False):
                raise PoisonedStateException(
                    analysis_result=analysis,
                    message=f"State blocked (strict mode — CHALLENGE): {level} threat.",
                )
            raise ThreatChallengeException(
                analysis_result=analysis,
                message=f"State requires verification: {level} threat level.",
            )

    # ------------------------------------------------------------------ client wrap

    def wrap_client(self, client: Any, *, decision_method: str = "system_one") -> "ProtectedSystemOneClient":
        """
        Wrap a vendor SDK client so its decision call is state-guarded (Layer 1/2).

        ``decision_method`` is the client method that takes ``state`` / ``questions``
        (default ``"system_one"`` — the method name on TypeSafe's ``TypeSafeClient`` /
        ``AsyncTypeSafeClient``). Action gating (Layer 3) is post-decision and
        app-specific — call ``guard_decision_actions`` with your intended actions.
        """
        return ProtectedSystemOneClient(client, self, decision_method)


class JevProvider(SystemOneProvider):
    """
    Guardian protection for TypeSafe AI's **Jev** — the first System One model.

    Example (explicit guard API, SDK-agnostic)::

        from ethicore_guardian import Guardian
        from ethicore_guardian.providers import JevProvider

        guardian = Guardian(api_key="eg-sk-...")
        jev = JevProvider(guardian)

        # 1) Guard the state before deciding (raises on poisoned input)
        jev.guard_state(
            state=assembled_state,
            untrusted_fields={"retrieved_doc": rag_chunk, "tool_out": api_result},
        )

        # 2) Ask Jev for typed decisions (your typesafe_sdk call) ...
        resp = typesafe_client.system_one(state=assembled_state, questions=questions)
        answers = resp.answers

        # 3) Guard any action the decision triggers, before executing it
        calls = JevProvider.decisions_to_calls(
            answers, {"approve_wire": {"name": "wire_transfer", "arguments": {...}, "when": "yes"}}
        )
        jev.guard_decision_actions(calls)
    """

    def __init__(self, guardian_instance: Any, *, model: str = JEV_DEFAULT_MODEL) -> None:
        super().__init__(guardian_instance, provider_name="jev", model=model)
        self.endpoint = JEV_SYSTEMONE_ENDPOINT


# ---------------------------------------------------------------------------
# ProtectedSystemOneClient — transparent proxy that guards the decision call
# ---------------------------------------------------------------------------

class ProtectedSystemOneClient:
    """
    Proxy around a System One SDK client. Intercepts the configured decision method
    to run ``guard_state`` first (Layer 1/2), then delegates untouched. Every other
    attribute passes straight through to the wrapped client.
    """

    DEFAULT_DECISION_METHOD = "system_one"  # TypeSafe TypeSafeClient.system_one(state, questions, *, model)

    def __init__(self, client: Any, provider: SystemOneProvider, decision_method: str) -> None:
        self._client = client
        self._provider = provider
        self._decision_method = decision_method
        logger.debug("🛡️  %s client protection enabled (method=%s)",
                     provider.provider_name, decision_method)

    def __getattr__(self, name: str) -> Any:
        target = getattr(self._client, name)
        if name != self._decision_method or not callable(target):
            return target

        provider = self._provider

        def _guarded(*args: Any, **kwargs: Any) -> Any:
            state = kwargs.get("state", args[0] if args else None)
            # Optional Layer-2 hint: caller may pass which fields are untrusted.
            untrusted = kwargs.pop("guardian_untrusted_fields", None)
            provider.guard_state(state, untrusted_fields=untrusted)
            return target(*args, **kwargs)

        _guarded.__name__ = name
        return _guarded

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"ProtectedSystemOneClient(provider={self._provider.provider_name}, client={self._client!r})"


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def create_protected_jev_client(
    typesafe_client: Any,
    guardian_api_key: str,
    *,
    model: str = JEV_DEFAULT_MODEL,
    decision_method: str = "system_one",
    **guardian_kwargs: Any,
) -> ProtectedSystemOneClient:
    """
    Wrap an existing TypeSafe/Jev SDK client with Guardian state-guarding in one step.

    Args:
        typesafe_client:  A constructed ``typesafe_sdk`` client (its API key is its own).
        guardian_api_key: Your Ethicore Engine / Guardian API key (eg-sk-…).
        model:            Jev model alias (forwarded only as threat context).
        decision_method:  The client method to guard (default ``"system_one"``).
        **guardian_kwargs: Extra kwargs forwarded to ``Guardian(...)`` (e.g. strict_mode).

    Returns:
        A ProtectedSystemOneClient — a transparent drop-in whose decision call is guarded.
    """
    if typesafe_client is None:
        raise ProviderError(
            "create_protected_jev_client requires a constructed typesafe_sdk client. "
            "Build it first (pip install typesafe_sdk; reads TYPESAFE_API_KEY), then pass it here."
        )
    from ..guardian import Guardian

    guardian = Guardian(api_key=guardian_api_key, **guardian_kwargs)
    return JevProvider(guardian, model=model).wrap_client(
        typesafe_client, decision_method=decision_method
    )


__all__ = [
    "SystemOneProvider",
    "JevProvider",
    "ProtectedSystemOneClient",
    "create_protected_jev_client",
    "JEV_SYSTEMONE_ENDPOINT",
    "JEV_DEFAULT_MODEL",
    "JEV_PRIMITIVES",
    "ProviderError",
    "ThreatBlockedException",
    "ThreatChallengeException",
    "PoisonedStateException",
    "DecisionActionBlockedException",
]
