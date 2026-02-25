"""
Ethicore Engine™ - Guardian SDK — Shared pytest fixtures

conftest.py is auto-loaded by pytest for all tests in this directory.
Place shared fixtures here so individual test files stay focused on
what they're testing, not on setup boilerplate.

Principle 13 (Ultimate Accountability): fixtures are explicit and named so
every test's preconditions are visible and auditable.
"""

from __future__ import annotations

import os

import pytest

from ethicore_guardian import Guardian, GuardianConfig

# ---------------------------------------------------------------------------
# Excluded test files
#
# test_openai.py is an interactive script (it calls input() to read a real
# API key from stdin) and was never designed to run under pytest.  Rather
# than silently skipping it, we flag it here so the decision is visible and
# auditable — honouring Principle 11 (Sacred Truth / Emet).
# ---------------------------------------------------------------------------
collect_ignore = ["test_openai.py"]


# ---------------------------------------------------------------------------
# Core fixture: initialised Guardian instance
# ---------------------------------------------------------------------------

@pytest.fixture
async def guardian() -> Guardian:
    """
    Provide a fully-initialised Guardian instance for tests.

    Uses a synthetic test API key so tests run without real credentials.
    Strict mode is OFF by default — individual tests can override via the
    Guardian.configure() method if they need strict-mode behaviour.
    """
    config = GuardianConfig(
        api_key="test-key-pytest",
        strict_mode=False,
        log_level="WARNING",  # Keep test output clean; errors still surface
    )
    g = Guardian(config=config)
    await g.initialize()
    return g


@pytest.fixture
async def strict_guardian() -> Guardian:
    """
    Guardian instance with strict mode enabled — for tests that specifically
    verify lower-threshold detection behaviour.
    """
    config = GuardianConfig(
        api_key="test-key-pytest-strict",
        strict_mode=True,
        log_level="WARNING",
    )
    g = Guardian(config=config)
    await g.initialize()
    return g


# ---------------------------------------------------------------------------
# License-gated test marker
#
# Tests decorated with @requires_license are automatically skipped unless
# ETHICORE_LICENSE_KEY is set in the environment (i.e., the full threat
# library and ONNX asset bundle are available).
#
# To run the full licensed test suite:
#   ETHICORE_LICENSE_KEY="EG-PRO-..." ETHICORE_ASSETS_DIR="$HOME/.ethicore" pytest tests/ -v
#
# Principle 11 (Sacred Truth): skip reason is visible and explicit — never
# silently passing tests that cannot actually run in this environment.
# ---------------------------------------------------------------------------
LICENSED = bool(os.environ.get("ETHICORE_LICENSE_KEY"))

requires_license = pytest.mark.skipif(
    not LICENSED,
    reason=(
        "Requires full threat library + ONNX models. "
        "Set ETHICORE_LICENSE_KEY (and optionally ETHICORE_ASSETS_DIR) to enable."
    ),
)
