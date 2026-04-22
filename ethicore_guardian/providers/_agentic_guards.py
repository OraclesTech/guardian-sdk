"""
Ethicore Engine™ - Guardian SDK — Shared Agentic Loop Guards
Layers 2 and 3 of full agentic pipeline protection, shared across all providers.

Layer 2 — Inbound tool result scan
    Scans tool execution results BEFORE they re-enter the model's context window.
    Catches indirect injection attacks embedded in web pages, API responses, file
    reads, database results, etc.

Layer 3 — Outbound tool call scan
    Scans tool name + arguments AFTER the model decides to invoke a tool but BEFORE
    the caller executes it.  Catches dangerous invocations (shell commands, package
    installs, data exfiltration, destructive filesystem operations).

Both layers delegate to Guardian's scan_tool_output() and scan_tool_call() methods
(API tier, license-gated).  When no license key is present the layers skip with a
debug-level log so providers degrade gracefully to prompt-only protection.

Provider-specific extraction helpers are co-located here so every provider uses
the same parsing logic rather than maintaining divergent copies.

Copyright © 2026 Oracles Technologies LLC
All Rights Reserved
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sync / async bridge
# ---------------------------------------------------------------------------

def run_sync(coro: Any) -> Any:
    """
    Run *coro* to completion from a synchronous call site.

    Handles both cases:
    - No running event loop  → asyncio.run()
    - Inside a running loop  → ThreadPoolExecutor bridge (avoids "cannot run
      nested event loops" error in Jupyter / FastAPI / etc.)
    """
    try:
        asyncio.get_running_loop()
        # Already inside an event loop — push to a thread
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    except RuntimeError:
        # No running loop
        return asyncio.run(coro)


# ---------------------------------------------------------------------------
# OpenAI-format extraction  (also used by xAI — same wire format)
# ---------------------------------------------------------------------------

def extract_openai_tool_results(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Return role='tool' messages from an OpenAI-format messages list.

    Each dict in the returned list has keys: tool_name, content.
    """
    results: List[Dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") != "tool":
            continue
        results.append({
            "tool_name": msg.get("name", ""),
            "content": msg.get("content", ""),
        })
    return results


def extract_openai_tool_calls(response: Any) -> List[Dict[str, Any]]:
    """
    Return tool_calls from an OpenAI/xAI chat.completions response.

    Each dict has keys: name, arguments (parsed dict or raw string).
    """
    calls: List[Dict[str, Any]] = []
    try:
        choices = getattr(response, "choices", None) or response.get("choices", [])
        for choice in choices:
            msg = getattr(choice, "message", None) or choice.get("message", {})
            tool_calls = getattr(msg, "tool_calls", None) or (
                msg.get("tool_calls") if isinstance(msg, dict) else None
            )
            if not tool_calls:
                continue
            for tc in tool_calls:
                fn = getattr(tc, "function", None) or tc.get("function", {})
                name = getattr(fn, "name", None) or (
                    fn.get("name", "") if isinstance(fn, dict) else ""
                )
                arguments = getattr(fn, "arguments", None) or (
                    fn.get("arguments", "") if isinstance(fn, dict) else ""
                )
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except (json.JSONDecodeError, ValueError):
                        pass
                calls.append({"name": name, "arguments": arguments})
    except Exception as exc:
        logger.debug("extract_openai_tool_calls error: %s", exc)
    return calls


# ---------------------------------------------------------------------------
# Anthropic-format extraction
# ---------------------------------------------------------------------------

def extract_anthropic_tool_results(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Return tool_result content blocks from an Anthropic-format messages list.

    In the Anthropic API, tool results arrive in role='user' messages as
    content blocks of type 'tool_result':

        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "...", "content": "..."}
        ]}

    Each dict in the returned list has keys: tool_name (empty — Anthropic does not
    echo the tool name in results), content.
    """
    results: List[Dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_result":
                continue
            inner = block.get("content", "")
            # content may itself be a list of text blocks
            if isinstance(inner, list):
                text_parts = [
                    b.get("text", "") for b in inner
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                inner = " ".join(text_parts)
            results.append({
                "tool_name": block.get("tool_use_id", ""),  # id, not name
                "content": inner,
            })
    return results


def extract_anthropic_tool_calls(response: Any) -> List[Dict[str, Any]]:
    """
    Return tool_use content blocks from an Anthropic messages response.

    In the Anthropic API, tool invocations appear in the response content as
    blocks of type 'tool_use':

        {"content": [
            {"type": "tool_use", "id": "...", "name": "bash", "input": {...}}
        ]}

    Each dict in the returned list has keys: name, arguments (the 'input' dict).
    """
    calls: List[Dict[str, Any]] = []
    try:
        # SDK object path
        content_blocks = getattr(response, "content", None)
        if content_blocks is None:
            # Dict path
            content_blocks = response.get("content", []) if isinstance(response, dict) else []

        for block in content_blocks or []:
            block_type = getattr(block, "type", None) or (
                block.get("type") if isinstance(block, dict) else None
            )
            if block_type != "tool_use":
                continue
            name = getattr(block, "name", None) or (
                block.get("name", "") if isinstance(block, dict) else ""
            )
            arguments = getattr(block, "input", None) or (
                block.get("input", {}) if isinstance(block, dict) else {}
            )
            calls.append({"name": name, "arguments": arguments})
    except Exception as exc:
        logger.debug("extract_anthropic_tool_calls error: %s", exc)
    return calls


# ---------------------------------------------------------------------------
# Ollama-format extraction
# ---------------------------------------------------------------------------

def extract_ollama_tool_results(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Return tool result messages from an Ollama-format messages list.

    Ollama follows the OpenAI messages format for function/tool calling
    (role='tool'), introduced in Ollama 0.3+ with supported models.
    """
    return extract_openai_tool_results(messages)


def extract_ollama_tool_calls(response: Any) -> List[Dict[str, Any]]:
    """
    Return tool_calls from an Ollama chat response dict.

    Ollama returns: {"message": {"role": "assistant", "tool_calls": [...]}}
    """
    calls: List[Dict[str, Any]] = []
    try:
        msg = response.get("message", {}) if isinstance(response, dict) else {}
        tool_calls = msg.get("tool_calls", []) or []
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            arguments = fn.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except (json.JSONDecodeError, ValueError):
                    pass
            calls.append({"name": name, "arguments": arguments})
    except Exception as exc:
        logger.debug("extract_ollama_tool_calls error: %s", exc)
    return calls


# ---------------------------------------------------------------------------
# Shared scan runners
# ---------------------------------------------------------------------------

async def scan_inbound_tool_results(
    guardian: Any,
    tool_results: List[Dict[str, Any]],
    blocked_exc_cls: type,
) -> None:
    """
    Layer 2 — scan each tool result via guardian.scan_tool_output().

    Args:
        guardian:        The Guardian instance.
        tool_results:    List of {tool_name, content} dicts.
        blocked_exc_cls: Exception class to raise on injection detection
                         (provider-specific ToolOutputBlockedException or similar).
    """
    for tr in tool_results:
        try:
            result = await guardian.scan_tool_output(
                tr["content"], tool_name=tr["tool_name"]
            )
        except PermissionError:
            logger.debug("[Guardian] scan_tool_output skipped — API tier license required")
            continue
        except Exception as exc:
            logger.warning("[Guardian] scan_tool_output error: %s", exc)
            continue

        if result is not None and getattr(result, "is_injection", False):
            score = getattr(result, "injection_score", 0.0)
            logger.warning(
                "🚨 BLOCKED tool output [%s] — injection payload (score=%.1f)",
                tr["tool_name"], score,
            )
            raise blocked_exc_cls(
                analysis_result=result,
                message=(
                    f"Tool output from '{tr['tool_name']}' blocked: "
                    f"injection payload detected (score={score:.1f})."
                ),
            )


async def scan_outbound_tool_calls(
    guardian: Any,
    tool_calls: List[Dict[str, Any]],
    blocked_exc_cls: type,
) -> None:
    """
    Layer 3 — scan each outbound tool call via guardian.scan_tool_call().

    Args:
        guardian:        The Guardian instance.
        tool_calls:      List of {name, arguments} dicts.
        blocked_exc_cls: Exception class to raise on dangerous call detection.
    """
    for tc in tool_calls:
        try:
            result = await guardian.scan_tool_call(
                tc["name"], tool_args=tc["arguments"], block_on_challenge=False
            )
        except PermissionError:
            logger.debug("[Guardian] scan_tool_call skipped — API tier license required")
            continue
        except Exception as exc:
            logger.warning("[Guardian] scan_tool_call error: %s", exc)
            continue

        if result is not None and getattr(result, "is_dangerous", False):
            risk = getattr(result, "risk_score", 0.0)
            cats = getattr(result, "threat_categories", [])
            logger.warning(
                "🚨 BLOCKED tool call [%s] — dangerous operation (risk=%.1f, cats=%s)",
                tc["name"], risk, cats,
            )
            raise blocked_exc_cls(
                analysis_result=result,
                message=(
                    f"Tool call '{tc['name']}' blocked: dangerous operation detected "
                    f"(risk={risk:.1f}, categories={cats})."
                ),
            )
