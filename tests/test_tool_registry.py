"""
Ethicore Engine™ - Guardian SDK — Tool Registry Tests
Version: 1.0.0

Covers ToolRegistry (tool_registry.py):
  - register() / deregister() / verify() happy paths
  - Schema-change detection → BLOCK
  - Unregistered tool → CHALLENGE
  - Hash stability and key-order invariance
  - Call count and timestamp tracking
  - Thread safety
  - JSON persistence (save/load via tmp_path)
  - Edge cases: None schema, empty schema, non-dict schema

Copyright © 2026 Oracles Technologies LLC
All Rights Reserved
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from ethicore_guardian.analyzers.tool_registry import (
    ToolRegistry,
    ToolRegistration,
    ToolProvenanceResult,
    _hash_schema,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def registry() -> ToolRegistry:
    """Fresh in-memory ToolRegistry for each test."""
    return ToolRegistry(challenge_on_unregistered=True, block_on_schema_change=True)


SAMPLE_SCHEMA = {"input": {"query": "string"}, "output": {"results": "list"}, "version": 1}
MUTATED_SCHEMA = {"input": {"query": "string"}, "output": {"results": "list"}, "version": 2}


# ---------------------------------------------------------------------------
# TestRegistration
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_register_returns_tool_registration(self, registry):
        reg = registry.register("web_search", SAMPLE_SCHEMA)
        assert isinstance(reg, ToolRegistration)

    def test_register_stores_sha256_hash(self, registry):
        reg = registry.register("web_search", SAMPLE_SCHEMA)
        expected_hash = _hash_schema(SAMPLE_SCHEMA)
        assert reg.schema_hash == expected_hash
        assert len(reg.schema_hash) == 64  # SHA-256 hex

    def test_register_sets_tool_name(self, registry):
        reg = registry.register("file_reader", SAMPLE_SCHEMA)
        assert reg.tool_name == "file_reader"

    def test_register_sets_registered_at_timestamp(self, registry):
        import time
        before = time.time()
        reg = registry.register("tool_a", SAMPLE_SCHEMA)
        after = time.time()
        assert before <= reg.registered_at <= after

    def test_register_initial_call_count_zero(self, registry):
        reg = registry.register("tool_a", SAMPLE_SCHEMA)
        assert reg.call_count == 0

    def test_register_initial_last_called_none(self, registry):
        reg = registry.register("tool_a", SAMPLE_SCHEMA)
        assert reg.last_called_at is None

    def test_register_trust_level_default(self, registry):
        reg = registry.register("tool_a", SAMPLE_SCHEMA)
        assert reg.trust_level == "verified"

    def test_register_trust_level_custom(self, registry):
        reg = registry.register("tool_a", SAMPLE_SCHEMA, trust_level="observed")
        assert reg.trust_level == "observed"

    def test_register_idempotent_same_schema(self, registry):
        """Re-registering with the same schema returns existing reg and preserves call_count."""
        reg1 = registry.register("tool_a", SAMPLE_SCHEMA)
        # Simulate a call
        registry.verify("tool_a", SAMPLE_SCHEMA)
        reg2 = registry.register("tool_a", SAMPLE_SCHEMA)
        assert reg1.schema_hash == reg2.schema_hash
        assert reg2.call_count == 1  # preserved from verify()

    def test_register_overwrite_different_schema(self, registry):
        registry.register("tool_a", SAMPLE_SCHEMA)
        reg2 = registry.register("tool_a", MUTATED_SCHEMA)
        assert reg2.schema_hash == _hash_schema(MUTATED_SCHEMA)

    def test_tool_count_increments(self, registry):
        assert registry.tool_count == 0
        registry.register("tool_a", SAMPLE_SCHEMA)
        assert registry.tool_count == 1
        registry.register("tool_b", SAMPLE_SCHEMA)
        assert registry.tool_count == 2

    def test_deregister_existing_returns_true(self, registry):
        registry.register("tool_a", SAMPLE_SCHEMA)
        assert registry.deregister("tool_a") is True
        assert registry.tool_count == 0

    def test_deregister_missing_returns_false(self, registry):
        assert registry.deregister("nonexistent") is False

    def test_list_registrations_empty(self, registry):
        assert registry.list_registrations() == []

    def test_list_registrations_returns_all(self, registry):
        registry.register("a", SAMPLE_SCHEMA)
        registry.register("b", MUTATED_SCHEMA)
        names = {r.tool_name for r in registry.list_registrations()}
        assert names == {"a", "b"}

    def test_get_registration_found(self, registry):
        registry.register("tool_a", SAMPLE_SCHEMA)
        reg = registry.get_registration("tool_a")
        assert reg is not None
        assert reg.tool_name == "tool_a"

    def test_get_registration_missing(self, registry):
        assert registry.get_registration("missing") is None


# ---------------------------------------------------------------------------
# TestVerification
# ---------------------------------------------------------------------------

class TestVerification:
    def test_verify_registered_matching_schema_returns_allow(self, registry):
        registry.register("web_search", SAMPLE_SCHEMA)
        result = registry.verify("web_search", SAMPLE_SCHEMA)
        assert result.verdict == "ALLOW"

    def test_verify_allow_is_not_suspicious(self, registry):
        registry.register("web_search", SAMPLE_SCHEMA)
        result = registry.verify("web_search", SAMPLE_SCHEMA)
        assert result.is_suspicious is False

    def test_verify_schema_matches_true_on_allow(self, registry):
        registry.register("web_search", SAMPLE_SCHEMA)
        result = registry.verify("web_search", SAMPLE_SCHEMA)
        assert result.schema_matches is True

    def test_verify_increments_call_count(self, registry):
        registry.register("tool_a", SAMPLE_SCHEMA)
        registry.verify("tool_a", SAMPLE_SCHEMA)
        registry.verify("tool_a", SAMPLE_SCHEMA)
        reg = registry.get_registration("tool_a")
        assert reg.call_count == 2

    def test_verify_updates_last_called_at(self, registry):
        import time
        registry.register("tool_a", SAMPLE_SCHEMA)
        before = time.time()
        registry.verify("tool_a", SAMPLE_SCHEMA)
        reg = registry.get_registration("tool_a")
        assert reg.last_called_at is not None
        assert reg.last_called_at >= before

    def test_verify_unregistered_returns_challenge(self, registry):
        result = registry.verify("unknown_tool", SAMPLE_SCHEMA)
        assert result.verdict == "CHALLENGE"
        assert result.is_suspicious is True
        assert result.is_registered is False

    def test_verify_unregistered_challenge_off_returns_allow(self):
        reg = ToolRegistry(challenge_on_unregistered=False)
        result = reg.verify("unknown_tool", SAMPLE_SCHEMA)
        assert result.verdict == "ALLOW"

    def test_verify_schema_changed_returns_block(self, registry):
        registry.register("tool_a", SAMPLE_SCHEMA)
        result = registry.verify("tool_a", MUTATED_SCHEMA)
        assert result.verdict == "BLOCK"
        assert result.is_suspicious is True
        assert result.schema_matches is False

    def test_verify_schema_changed_block_off_returns_challenge(self):
        reg = ToolRegistry(block_on_schema_change=False)
        reg.register("tool_a", SAMPLE_SCHEMA)
        result = reg.verify("tool_a", MUTATED_SCHEMA)
        assert result.verdict == "CHALLENGE"

    def test_verify_no_schema_provided_returns_allow(self, registry):
        registry.register("tool_a", SAMPLE_SCHEMA)
        result = registry.verify("tool_a")  # no schema
        assert result.verdict == "ALLOW"
        assert result.schema_matches is None

    def test_verify_returns_provenance_result_type(self, registry):
        result = registry.verify("any_tool", SAMPLE_SCHEMA)
        assert isinstance(result, ToolProvenanceResult)

    def test_verify_result_fields_present(self, registry):
        registry.register("tool_a", SAMPLE_SCHEMA)
        result = registry.verify("tool_a", SAMPLE_SCHEMA)
        assert hasattr(result, "verdict")
        assert hasattr(result, "is_suspicious")
        assert hasattr(result, "tool_name")
        assert hasattr(result, "is_registered")
        assert hasattr(result, "schema_matches")
        assert hasattr(result, "observed_hash")
        assert hasattr(result, "expected_hash")
        assert hasattr(result, "trust_level")
        assert hasattr(result, "reasoning")
        assert hasattr(result, "analysis_time_ms")
        assert result.analysis_time_ms >= 0.0

    def test_verify_reasoning_non_empty(self, registry):
        result = registry.verify("tool_a", SAMPLE_SCHEMA)
        assert isinstance(result.reasoning, list)
        assert len(result.reasoning) > 0

    def test_verify_observed_hash_on_allow(self, registry):
        registry.register("tool_a", SAMPLE_SCHEMA)
        result = registry.verify("tool_a", SAMPLE_SCHEMA)
        assert result.observed_hash == _hash_schema(SAMPLE_SCHEMA)

    def test_verify_expected_hash_on_allow(self, registry):
        registry.register("tool_a", SAMPLE_SCHEMA)
        result = registry.verify("tool_a", SAMPLE_SCHEMA)
        assert result.expected_hash == _hash_schema(SAMPLE_SCHEMA)


# ---------------------------------------------------------------------------
# TestHashStability
# ---------------------------------------------------------------------------

class TestHashStability:
    def test_hash_same_schema_stable(self):
        h1 = _hash_schema(SAMPLE_SCHEMA)
        h2 = _hash_schema(SAMPLE_SCHEMA)
        assert h1 == h2

    def test_hash_key_order_invariant(self):
        schema_a = {"b": 2, "a": 1}
        schema_b = {"a": 1, "b": 2}
        assert _hash_schema(schema_a) == _hash_schema(schema_b)

    def test_hash_different_schemas_differ(self):
        assert _hash_schema(SAMPLE_SCHEMA) != _hash_schema(MUTATED_SCHEMA)

    def test_hash_none_does_not_raise(self):
        h = _hash_schema(None)
        assert isinstance(h, str)
        assert len(h) == 64

    def test_hash_empty_dict_does_not_raise(self):
        h = _hash_schema({})
        assert isinstance(h, str)

    def test_hash_string_schema_does_not_raise(self):
        h = _hash_schema("just a string schema")
        assert isinstance(h, str)

    def test_public_hash_schema_alias(self):
        assert ToolRegistry.hash_schema(SAMPLE_SCHEMA) == _hash_schema(SAMPLE_SCHEMA)


# ---------------------------------------------------------------------------
# TestThreadSafety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_verify_no_corruption(self, registry):
        """10 threads verify the same tool concurrently — call_count must be accurate."""
        registry.register("shared_tool", SAMPLE_SCHEMA)
        results = []
        errors = []

        def worker():
            try:
                r = registry.verify("shared_tool", SAMPLE_SCHEMA)
                results.append(r.verdict)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
        assert all(v == "ALLOW" for v in results)
        reg = registry.get_registration("shared_tool")
        assert reg.call_count == 10

    def test_concurrent_register_no_corruption(self):
        """10 threads register different tools concurrently — all must be present."""
        reg = ToolRegistry()
        errors = []

        def worker(i):
            try:
                reg.register(f"tool_{i}", {"id": i})
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert reg.tool_count == 10


# ---------------------------------------------------------------------------
# TestPersistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_persist_and_reload(self, tmp_path):
        persist_file = str(tmp_path / "registry.json")
        reg1 = ToolRegistry(persistence_path=persist_file)
        reg1.register("tool_a", SAMPLE_SCHEMA)
        reg1.register("tool_b", MUTATED_SCHEMA)

        # Load into a fresh instance
        reg2 = ToolRegistry(persistence_path=persist_file)
        assert reg2.tool_count == 2
        assert reg2.get_registration("tool_a") is not None
        assert reg2.get_registration("tool_b") is not None

    def test_persist_preserves_schema_hash(self, tmp_path):
        persist_file = str(tmp_path / "registry.json")
        reg1 = ToolRegistry(persistence_path=persist_file)
        reg1.register("tool_a", SAMPLE_SCHEMA)
        expected_hash = _hash_schema(SAMPLE_SCHEMA)

        reg2 = ToolRegistry(persistence_path=persist_file)
        reg = reg2.get_registration("tool_a")
        assert reg.schema_hash == expected_hash

    def test_persist_preserves_trust_level(self, tmp_path):
        persist_file = str(tmp_path / "registry.json")
        reg1 = ToolRegistry(persistence_path=persist_file)
        reg1.register("tool_a", SAMPLE_SCHEMA, trust_level="observed")

        reg2 = ToolRegistry(persistence_path=persist_file)
        reg = reg2.get_registration("tool_a")
        assert reg.trust_level == "observed"

    def test_deregister_persists(self, tmp_path):
        persist_file = str(tmp_path / "registry.json")
        reg1 = ToolRegistry(persistence_path=persist_file)
        reg1.register("tool_a", SAMPLE_SCHEMA)
        reg1.deregister("tool_a")

        reg2 = ToolRegistry(persistence_path=persist_file)
        assert reg2.tool_count == 0

    def test_verify_after_reload_allows(self, tmp_path):
        persist_file = str(tmp_path / "registry.json")
        reg1 = ToolRegistry(persistence_path=persist_file)
        reg1.register("tool_a", SAMPLE_SCHEMA)

        reg2 = ToolRegistry(persistence_path=persist_file)
        result = reg2.verify("tool_a", SAMPLE_SCHEMA)
        assert result.verdict == "ALLOW"
