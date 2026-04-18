"""
Ethicore Engine™ - Guardian SDK — LangChain Callback Handlers
Version: 1.0.0

Copyright © 2026 Oracles Technologies LLC
All Rights Reserved

═══════════════════════════════════════════════════════════════════
OVERVIEW
═══════════════════════════════════════════════════════════════════

GuardianCallbackHandler and GuardianAsyncCallbackHandler wire Guardian
SDK's three agentic protection points into any LangChain agent with
zero changes to existing agent code.  Simply register the handler and
every tool call, tool output, and assembled prompt is automatically
scanned.

Protection points:

  on_chat_model_start / on_llm_start
    ↳ guardian.analyze()
    ↳ Catches: adversarial assembled context before the LLM call —
      payloads injected via memory, system messages, or prior tool
      outputs that survived individual scans.

  on_agent_action  (fires BEFORE the tool executes)
    ↳ guardian.scan_tool_call()
    ↳ Catches: jailbroken LLM attempting to invoke dangerous tools —
      shell execution, package installation, data exfiltration.

  on_tool_end  (fires AFTER the tool returns, BEFORE result enters context)
    ↳ guardian.scan_tool_output()
    ↳ Catches: injection payloads in tool return values before they
      enter the agent's next context window.

Usage (sync agents)::

    from ethicore_guardian import Guardian
    from ethicore_guardian.providers.langchain_callback import GuardianCallbackHandler

    guardian = Guardian(api_key="your_key")
    handler = GuardianCallbackHandler(guardian, block_on_challenge=True)

    agent_executor.invoke(
        {"input": user_query},
        config={"callbacks": [handler]},
    )

Usage (async agents)::

    from ethicore_guardian.providers.langchain_callback import GuardianAsyncCallbackHandler

    handler = GuardianAsyncCallbackHandler(guardian, block_on_challenge=True)

    await agent_executor.ainvoke(
        {"input": user_query},
        config={"callbacks": [handler]},
    )

block_on_challenge:
    When True, CHALLENGE verdicts are treated as confirmed threats and
    raise a blocking exception.  Recommended for production agentic
    deployments where any ambiguity should halt the pipeline.
    Default: True.

Exceptions raised on BLOCK (or CHALLENGE when block_on_challenge=True):
    GuardianAgentBlockedError   — on_chat_model_start / on_llm_start
    GuardianToolCallBlockedError — on_agent_action
    GuardianToolOutputBlockedError — on_tool_end

These inherit from GuardianPipelineError for easy top-level catching.

Principle 14 (Divine Safety): agents given tool use are far more
dangerous than pure text generators.  This handler ensures the
protective coverage wraps the entire agentic loop, not just inputs.
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Sequence, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Guardian-specific exceptions
# ---------------------------------------------------------------------------

class GuardianPipelineError(Exception):
    """Base class for all agentic pipeline blocks raised by Guardian."""
    def __init__(self, message: str, result: Any) -> None:
        super().__init__(message)
        self.result = result  # ToolCallScanResult / ToolOutputScanResult / ThreatAnalysis


class GuardianAgentBlockedError(GuardianPipelineError):
    """Raised when the assembled prompt before an LLM call is blocked."""


class GuardianToolCallBlockedError(GuardianPipelineError):
    """Raised when a tool call is blocked before execution."""


class GuardianToolOutputBlockedError(GuardianPipelineError):
    """Raised when a tool output is blocked before entering context."""


# ---------------------------------------------------------------------------
# Lazy LangChain import helpers
# ---------------------------------------------------------------------------

def _get_base_callback_handler():
    """Return langchain_core BaseCallbackHandler, or raise ImportError."""
    try:
        from langchain_core.callbacks.base import BaseCallbackHandler
        return BaseCallbackHandler
    except ImportError:
        raise ImportError(
            "langchain-core is required for GuardianCallbackHandler.\n"
            "Install it with: pip install langchain-core"
        )


def _get_async_callback_handler():
    """Return langchain_core AsyncCallbackHandler, or raise ImportError."""
    try:
        from langchain_core.callbacks.base import AsyncCallbackHandler
        return AsyncCallbackHandler
    except ImportError:
        raise ImportError(
            "langchain-core is required for GuardianAsyncCallbackHandler.\n"
            "Install it with: pip install langchain-core"
        )


# ---------------------------------------------------------------------------
# Helpers shared by both handlers
# ---------------------------------------------------------------------------

def _extract_prompt_from_messages(messages: Any) -> str:
    """
    Flatten LangChain message lists into a single string for Guardian.analyze().
    Handles both raw dicts and LangChain message objects.
    """
    parts: List[str] = []
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict):
                content = msg.get("content", "")
                role = msg.get("role", "")
            elif hasattr(msg, "content"):
                content = msg.content
                role = getattr(msg, "type", getattr(msg, "role", ""))
            else:
                content = str(msg)
                role = ""
            if content:
                prefix = f"[{role}] " if role else ""
                parts.append(f"{prefix}{content}")
    elif isinstance(messages, str):
        parts.append(messages)
    return "\n".join(parts)


def _extract_agent_action_info(action: Any) -> tuple[str, Any]:
    """
    Extract (tool_name, tool_input) from a LangChain AgentAction.
    Works with both legacy AgentAction and modern ToolAgentAction.
    """
    tool_name = getattr(action, "tool", "") or ""
    tool_input = getattr(action, "tool_input", None)
    return tool_name, tool_input


def _should_block(verdict: str, block_on_challenge: bool) -> bool:
    return verdict == "BLOCK" or (block_on_challenge and verdict == "CHALLENGE")


# ---------------------------------------------------------------------------
# Sync handler
# ---------------------------------------------------------------------------

class GuardianCallbackHandler:
    """
    LangChain callback handler that wires Guardian SDK into the full agentic
    loop: assembled prompt scanning, pre-tool-call validation, and tool output
    injection detection.

    Inherits from langchain_core BaseCallbackHandler at instantiation time
    (lazy import) so the class can be defined without langchain installed.
    """

    def __new__(cls, *args, **kwargs):
        # Lazily inherit from BaseCallbackHandler at instantiation
        BaseCallbackHandler = _get_base_callback_handler()
        if BaseCallbackHandler not in cls.__bases__:
            cls.__bases__ = (BaseCallbackHandler,)
        return super().__new__(cls)

    def __init__(
        self,
        guardian: Any,
        block_on_challenge: bool = True,
        session_id: Optional[str] = None,
        tool_registry: Optional[Any] = None,
    ) -> None:
        """
        Args:
            guardian:           Initialized Guardian instance.
            block_on_challenge: Treat CHALLENGE as BLOCK (recommended for
                                high-risk agentic deployments).
            session_id:         Optional session ID for correlated audit logs.
            tool_registry:      Optional ToolRegistry for schema provenance checks.
        """
        super().__init__()
        self.guardian = guardian
        self.block_on_challenge = block_on_challenge
        self.session_id = session_id
        self.tool_registry = tool_registry
        logger.info(
            "GuardianCallbackHandler registered (block_on_challenge=%s, session=%s)",
            block_on_challenge, session_id or "none",
        )

    # ── Protection point 1: assembled prompt before LLM call ────────────

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[Any]],
        **kwargs: Any,
    ) -> None:
        """Scan assembled context before the LLM call."""
        flat = _extract_prompt_from_messages(
            [msg for turn in messages for msg in turn]
        )
        if not flat.strip():
            return
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Running inside an existing async context — schedule and wait
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(
                        asyncio.run,
                        self.guardian.analyze(flat, context={"session_id": self.session_id}),
                    )
                    result = future.result(timeout=10)
            else:
                result = loop.run_until_complete(
                    self.guardian.analyze(flat, context={"session_id": self.session_id})
                )
        except Exception as exc:
            logger.warning("GuardianCallbackHandler.on_chat_model_start error: %s", exc)
            return

        if _should_block(result.recommended_action, self.block_on_challenge):
            logger.warning(
                "GuardianCallbackHandler: BLOCKING LLM call — %s (score=%.2f)",
                result.recommended_action, result.threat_score,
            )
            raise GuardianAgentBlockedError(
                f"Guardian blocked LLM call: {result.recommended_action} "
                f"(score={result.threat_score:.2f})",
                result,
            )

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any,
    ) -> None:
        """Scan assembled prompt for legacy (non-chat) LLM calls."""
        combined = "\n".join(p for p in prompts if p and p.strip())
        if not combined.strip():
            return
        self.on_chat_model_start(serialized, [[combined]], **kwargs)

    # ── Protection point 2: pre-tool-call validation ─────────────────────

    def on_agent_action(self, action: Any, **kwargs: Any) -> Any:
        """Scan tool name + args BEFORE the tool executes."""
        tool_name, tool_input = _extract_agent_action_info(action)
        if not tool_name:
            return None
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(
                        asyncio.run,
                        self.guardian.scan_tool_call(
                            tool_name, tool_input,
                            session_id=self.session_id,
                            block_on_challenge=self.block_on_challenge,
                            registry=self.tool_registry,
                        ),
                    )
                    result = future.result(timeout=5)
            else:
                result = loop.run_until_complete(
                    self.guardian.scan_tool_call(
                        tool_name, tool_input,
                        session_id=self.session_id,
                        block_on_challenge=self.block_on_challenge,
                        registry=self.tool_registry,
                    )
                )
        except (GuardianToolCallBlockedError, GuardianAgentBlockedError):
            raise
        except Exception as exc:
            logger.warning("GuardianCallbackHandler.on_agent_action error: %s", exc)
            return None

        if _should_block(result.verdict, self.block_on_challenge):
            logger.warning(
                "GuardianCallbackHandler: BLOCKING tool call '%s' — %s (score=%.0f)",
                tool_name, result.verdict, result.risk_score,
            )
            raise GuardianToolCallBlockedError(
                f"Guardian blocked tool call '{tool_name}': {result.verdict} "
                f"(score={result.risk_score:.0f}, categories={result.threat_categories})",
                result,
            )
        return None

    # ── Protection point 3: tool output before context re-entry ──────────

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Scan tool output BEFORE it enters the agent's next context window."""
        tool_name = kwargs.get("name", "") or ""
        if not output or not str(output).strip():
            return
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(
                        asyncio.run,
                        self.guardian.scan_tool_output(
                            output, tool_name=tool_name, session_id=self.session_id
                        ),
                    )
                    result = future.result(timeout=10)
            else:
                result = loop.run_until_complete(
                    self.guardian.scan_tool_output(
                        output, tool_name=tool_name, session_id=self.session_id
                    )
                )
        except (GuardianToolOutputBlockedError,):
            raise
        except Exception as exc:
            logger.warning("GuardianCallbackHandler.on_tool_end error: %s", exc)
            return

        if _should_block(result.verdict, self.block_on_challenge):
            logger.warning(
                "GuardianCallbackHandler: BLOCKING tool output from '%s' — %s (score=%.0f)",
                tool_name, result.verdict, result.injection_score,
            )
            raise GuardianToolOutputBlockedError(
                f"Guardian blocked tool output from '{tool_name}': {result.verdict} "
                f"(score={result.injection_score:.0f})",
                result,
            )


# ---------------------------------------------------------------------------
# Async handler
# ---------------------------------------------------------------------------

class GuardianAsyncCallbackHandler:
    """
    Async-native version of GuardianCallbackHandler.

    Uses await directly at each hook — no thread-pool bridging.
    Use this for async LangChain agents (ainvoke, astream, etc.).

    block_on_challenge=True is the default and strongly recommended for
    async agentic deployments where ambiguous results should halt the
    pipeline rather than proceeding optimistically.
    """

    def __new__(cls, *args, **kwargs):
        AsyncCallbackHandler = _get_async_callback_handler()
        if AsyncCallbackHandler not in cls.__bases__:
            cls.__bases__ = (AsyncCallbackHandler,)
        return super().__new__(cls)

    def __init__(
        self,
        guardian: Any,
        block_on_challenge: bool = True,
        session_id: Optional[str] = None,
        tool_registry: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.guardian = guardian
        self.block_on_challenge = block_on_challenge
        self.session_id = session_id
        self.tool_registry = tool_registry
        logger.info(
            "GuardianAsyncCallbackHandler registered (block_on_challenge=%s, session=%s)",
            block_on_challenge, session_id or "none",
        )

    # ── Protection point 1 ───────────────────────────────────────────────

    async def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[Any]],
        **kwargs: Any,
    ) -> None:
        flat = _extract_prompt_from_messages(
            [msg for turn in messages for msg in turn]
        )
        if not flat.strip():
            return
        result = await self.guardian.analyze(
            flat, context={"session_id": self.session_id}
        )
        if _should_block(result.recommended_action, self.block_on_challenge):
            logger.warning(
                "GuardianAsyncCallbackHandler: BLOCKING LLM call — %s (score=%.2f)",
                result.recommended_action, result.threat_score,
            )
            raise GuardianAgentBlockedError(
                f"Guardian blocked LLM call: {result.recommended_action} "
                f"(score={result.threat_score:.2f})",
                result,
            )

    async def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any,
    ) -> None:
        combined = "\n".join(p for p in prompts if p and p.strip())
        if not combined.strip():
            return
        await self.on_chat_model_start(serialized, [[combined]], **kwargs)

    # ── Protection point 2 ───────────────────────────────────────────────

    async def on_agent_action(self, action: Any, **kwargs: Any) -> Any:
        tool_name, tool_input = _extract_agent_action_info(action)
        if not tool_name:
            return None
        result = await self.guardian.scan_tool_call(
            tool_name, tool_input,
            session_id=self.session_id,
            block_on_challenge=self.block_on_challenge,
            registry=self.tool_registry,
        )
        if _should_block(result.verdict, self.block_on_challenge):
            logger.warning(
                "GuardianAsyncCallbackHandler: BLOCKING tool call '%s' — %s (score=%.0f)",
                tool_name, result.verdict, result.risk_score,
            )
            raise GuardianToolCallBlockedError(
                f"Guardian blocked tool call '{tool_name}': {result.verdict} "
                f"(score={result.risk_score:.0f}, categories={result.threat_categories})",
                result,
            )
        return None

    # ── Protection point 3 ───────────────────────────────────────────────

    async def on_tool_end(self, output: str, **kwargs: Any) -> None:
        tool_name = kwargs.get("name", "") or ""
        if not output or not str(output).strip():
            return
        result = await self.guardian.scan_tool_output(
            output, tool_name=tool_name, session_id=self.session_id
        )
        if _should_block(result.verdict, self.block_on_challenge):
            logger.warning(
                "GuardianAsyncCallbackHandler: BLOCKING tool output from '%s' — %s (score=%.0f)",
                tool_name, result.verdict, result.injection_score,
            )
            raise GuardianToolOutputBlockedError(
                f"Guardian blocked tool output from '{tool_name}': {result.verdict} "
                f"(score={result.injection_score:.0f})",
                result,
            )


# ---------------------------------------------------------------------------
# Convenience exports
# ---------------------------------------------------------------------------

__all__ = [
    "GuardianCallbackHandler",
    "GuardianAsyncCallbackHandler",
    "GuardianPipelineError",
    "GuardianAgentBlockedError",
    "GuardianToolCallBlockedError",
    "GuardianToolOutputBlockedError",
]
