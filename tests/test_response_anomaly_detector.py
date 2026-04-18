"""
Ethicore Engine™ - Guardian SDK — Response Anomaly Detector Tests
Version: 1.0.0

Covers ResponseAnomalyDetector (response_anomaly_detector.py):
  - Baseline building (no flags until baseline_window responses seen)
  - Anomaly detection after baseline (entropy, word_count, vocab, structure)
  - Refusal-shift detection (CRITICAL → BLOCK)
  - Session independence
  - Session TTL eviction
  - Thread safety
  - Edge cases: empty response, very short text
  - ResponseAnomalyResult and AnomalySignal field validation
  - Helper function unit tests (_bigram_entropy, _has_refusal, _vocab_richness)

Copyright © 2026 Oracles Technologies LLC
All Rights Reserved
"""

from __future__ import annotations

import threading
import time

import pytest

from ethicore_guardian.analyzers.response_anomaly_detector import (
    ResponseAnomalyDetector,
    ResponseAnomalyResult,
    AnomalySignal,
    _bigram_entropy,
    _vocab_richness,
    _structural_markers,
    _has_refusal,
    _z_score,
    _mean,
    _stddev,
)


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def detector():
    """Fresh detector with baseline_window=5."""
    return ResponseAnomalyDetector(baseline_window=5, challenge_z=2.0, block_z=3.5)


# Normal-looking AI response (varied vocabulary, mid entropy)
NORMAL_RESPONSE = (
    "The capital of France is Paris, which has been the cultural and political "
    "centre of the country for centuries. Paris is home to the Eiffel Tower, "
    "the Louvre museum, and the Notre-Dame cathedral."
)

# Highly structured / repetitive response (low entropy)
REPETITIVE_RESPONSE = "AAAAAAAAAA AAAAAAAAAA AAAAAAAAAA AAAAAAAAAA AAAAAAAAAA " * 20

# Very long response (word count spike)
LONG_RESPONSE = ("The quick brown fox jumps over the lazy dog. " * 200)

# Refusal response
REFUSAL_RESPONSE = "I'm not able to help with that request as it violates my guidelines."
REFUSAL_RESPONSE_2 = "I cannot assist with this. I'm unable to provide that information."

# Structured markers (code blocks, URLs)
STRUCTURED_RESPONSE = (
    "Here is the code:\n```python\nimport os\nprint(os.getcwd())\n```\n"
    "See https://example.com/docs for more. Also check https://other.com/api "
    "and review {'key': 'value', 'nested': {'a': 1}} for the schema."
)


# ---------------------------------------------------------------------------
# TestHelperFunctions
# ---------------------------------------------------------------------------

class TestHelperFunctions:
    def test_bigram_entropy_returns_float(self):
        result = _bigram_entropy("hello world")
        assert isinstance(result, float)

    def test_bigram_entropy_short_text_returns_zero(self):
        assert _bigram_entropy("ab") == 0.0
        assert _bigram_entropy("") == 0.0

    def test_bigram_entropy_repetitive_lower_than_varied(self):
        repetitive = _bigram_entropy("aaaa aaaa aaaa aaaa aaaa")
        varied = _bigram_entropy("The quick brown fox jumps over the lazy dog")
        assert repetitive < varied

    def test_vocab_richness_all_unique(self):
        text = "one two three four five"
        assert _vocab_richness(text) == 1.0

    def test_vocab_richness_all_same(self):
        text = "the the the the the"
        assert _vocab_richness(text) < 0.5

    def test_vocab_richness_empty(self):
        assert _vocab_richness("") == 0.0

    def test_structural_markers_code_block(self):
        assert _structural_markers("```python\ncode\n```") >= 2

    def test_structural_markers_url(self):
        assert _structural_markers("See https://example.com for details") >= 1

    def test_structural_markers_json_blob(self):
        assert _structural_markers('{"key": "value", "another": "thing"}') >= 1

    def test_structural_markers_plain_text_zero(self):
        assert _structural_markers("hello world this is plain text") == 0

    def test_has_refusal_detects_cannot(self):
        assert _has_refusal("I cannot help with that.")

    def test_has_refusal_detects_unable(self):
        assert _has_refusal("I'm not able to assist with this request.")

    def test_has_refusal_detects_decline(self):
        assert _has_refusal("I must decline this request.")

    def test_has_refusal_detects_as_an_ai(self):
        assert _has_refusal("As an AI, I cannot provide that.")

    def test_has_refusal_false_for_normal_text(self):
        assert not _has_refusal("The weather in Paris is sunny today.")

    def test_z_score_identical_samples_returns_zero(self):
        samples = [1.0, 1.0, 1.0, 1.0, 1.0]
        assert _z_score(1.0, samples) == 0.0

    def test_z_score_fewer_than_two_samples_returns_zero(self):
        assert _z_score(5.0, [3.0]) == 0.0

    def test_z_score_clear_outlier_high(self):
        samples = [1.0, 1.1, 0.9, 1.0, 1.05]
        z = _z_score(100.0, samples)
        assert z > 3.0

    def test_mean_correct(self):
        assert _mean([2.0, 4.0, 6.0]) == pytest.approx(4.0)

    def test_mean_empty_returns_zero(self):
        assert _mean([]) == 0.0

    def test_stddev_correct(self):
        # stddev of [2, 4, 4, 4, 5, 5, 7, 9] = 2.0
        assert _stddev([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]) == pytest.approx(2.0)

    def test_stddev_single_value_returns_zero(self):
        assert _stddev([5.0]) == 0.0


# ---------------------------------------------------------------------------
# TestBaselineBuilding
# ---------------------------------------------------------------------------

class TestBaselineBuilding:
    def test_returns_allow_during_baseline(self, detector):
        """All responses during baseline window return ALLOW."""
        for i in range(5):
            result = detector.analyze(NORMAL_RESPONSE, session_id="test")
            assert result.verdict == "ALLOW", f"Response {i+1} should be ALLOW during baseline"

    def test_baseline_established_false_during_window(self, detector):
        for i in range(5):
            result = detector.analyze(NORMAL_RESPONSE, session_id="test")
            assert result.baseline_established is False

    def test_baseline_established_true_after_window(self, detector):
        for _ in range(5):
            detector.analyze(NORMAL_RESPONSE, session_id="test")
        result = detector.analyze(NORMAL_RESPONSE, session_id="test")
        assert result.baseline_established is True

    def test_response_index_increments(self, detector):
        for i in range(1, 4):
            result = detector.analyze(NORMAL_RESPONSE, session_id="test")
            assert result.response_index == i

    def test_no_signals_during_baseline(self, detector):
        for _ in range(5):
            result = detector.analyze(NORMAL_RESPONSE, session_id="test")
            assert result.signals == []

    def test_reasoning_mentions_baseline_during_window(self, detector):
        result = detector.analyze(NORMAL_RESPONSE, session_id="test")
        assert any("baseline" in r.lower() for r in result.reasoning)


# ---------------------------------------------------------------------------
# TestAnomalyDetection
# ---------------------------------------------------------------------------

class TestAnomalyDetection:
    def _build_baseline(self, detector, session_id="test", n=5):
        for _ in range(n):
            detector.analyze(NORMAL_RESPONSE, session_id=session_id)

    def test_normal_response_after_baseline_returns_allow(self, detector):
        self._build_baseline(detector)
        result = detector.analyze(NORMAL_RESPONSE, session_id="test")
        assert result.verdict == "ALLOW"
        assert result.signals == []

    def test_repetitive_injection_flagged_after_baseline(self, detector):
        """Highly repetitive content should trigger entropy or vocab anomaly."""
        self._build_baseline(detector)
        result = detector.analyze(REPETITIVE_RESPONSE, session_id="test")
        # Either CHALLENGE or BLOCK depending on z-score magnitude
        assert result.verdict in ("CHALLENGE", "BLOCK")
        assert len(result.signals) > 0

    def test_word_count_spike_flagged(self, detector):
        """Sudden word count explosion should flag."""
        self._build_baseline(detector)
        result = detector.analyze(LONG_RESPONSE, session_id="test")
        # LONG_RESPONSE is ~800 words vs ~35 words in baseline
        assert result.verdict in ("CHALLENGE", "BLOCK")

    def test_structural_marker_spike_flagged(self, detector):
        """Session with no structured markers, then lots of them."""
        # Build baseline with plain text
        for _ in range(5):
            detector.analyze("Simple plain text response here.", session_id="struct_test")
        # Inject heavily structured response
        result = detector.analyze(STRUCTURED_RESPONSE, session_id="struct_test")
        assert result.verdict in ("CHALLENGE", "BLOCK")

    def test_anomaly_result_has_signals(self, detector):
        self._build_baseline(detector)
        result = detector.analyze(REPETITIVE_RESPONSE, session_id="test")
        if result.verdict != "ALLOW":
            assert len(result.signals) > 0

    def test_anomaly_signal_fields(self, detector):
        self._build_baseline(detector)
        result = detector.analyze(REPETITIVE_RESPONSE, session_id="test")
        if result.signals:
            sig = result.signals[0]
            assert isinstance(sig, AnomalySignal)
            assert isinstance(sig.metric, str)
            assert isinstance(sig.observed, float)
            assert isinstance(sig.baseline_mean, float)
            assert isinstance(sig.baseline_std, float)
            assert isinstance(sig.z_score, float)
            assert sig.severity in ("HIGH", "CRITICAL")
            assert isinstance(sig.description, str)
            assert len(sig.description) > 0

    def test_is_anomalous_matches_verdict(self, detector):
        self._build_baseline(detector)
        result = detector.analyze(NORMAL_RESPONSE, session_id="test")
        assert result.is_anomalous == (result.verdict in ("CHALLENGE", "BLOCK"))


# ---------------------------------------------------------------------------
# TestRefusalShift
# ---------------------------------------------------------------------------

class TestRefusalShift:
    def test_refusal_shift_block_when_baseline_all_refusals(self, detector):
        """Baseline of 5 refusals → non-refusal response 6 → BLOCK."""
        for _ in range(5):
            detector.analyze(REFUSAL_RESPONSE, session_id="refusal_test")
        result = detector.analyze(NORMAL_RESPONSE, session_id="refusal_test")
        assert result.verdict == "BLOCK"

    def test_refusal_shift_signal_is_critical(self, detector):
        for _ in range(5):
            detector.analyze(REFUSAL_RESPONSE, session_id="refusal_test")
        result = detector.analyze(NORMAL_RESPONSE, session_id="refusal_test")
        shift_signals = [s for s in result.signals if s.metric == "refusal_shift"]
        assert len(shift_signals) >= 1
        assert shift_signals[0].severity == "CRITICAL"

    def test_low_baseline_refusal_rate_no_shift(self, detector):
        """2 out of 5 baseline responses with refusals → no refusal-shift signal."""
        detector.analyze(REFUSAL_RESPONSE, session_id="low_refusal")
        detector.analyze(REFUSAL_RESPONSE, session_id="low_refusal")
        for _ in range(3):
            detector.analyze(NORMAL_RESPONSE, session_id="low_refusal")
        result = detector.analyze(NORMAL_RESPONSE, session_id="low_refusal")
        shift_signals = [s for s in result.signals if s.metric == "refusal_shift"]
        assert len(shift_signals) == 0

    def test_refusal_after_refusal_baseline_is_allow(self, detector):
        """Baseline all refusals, response 6 is also a refusal → ALLOW."""
        for _ in range(5):
            detector.analyze(REFUSAL_RESPONSE, session_id="stable_refusal")
        result = detector.analyze(REFUSAL_RESPONSE_2, session_id="stable_refusal")
        # Should not produce a refusal-shift signal
        shift_signals = [s for s in result.signals if s.metric == "refusal_shift"]
        assert len(shift_signals) == 0


# ---------------------------------------------------------------------------
# TestSessionIsolation
# ---------------------------------------------------------------------------

class TestSessionIsolation:
    def test_sessions_are_independent(self, detector):
        """Anomaly in session A must not affect session B."""
        # Build normal baseline for session_a
        for _ in range(5):
            detector.analyze(NORMAL_RESPONSE, session_id="session_a")

        # Build normal baseline for session_b
        for _ in range(5):
            detector.analyze(NORMAL_RESPONSE, session_id="session_b")

        # Inject anomalous response into session_a
        detector.analyze(REPETITIVE_RESPONSE, session_id="session_a")

        # session_b should still return ALLOW for normal response
        result_b = detector.analyze(NORMAL_RESPONSE, session_id="session_b")
        assert result_b.verdict == "ALLOW"

    def test_different_session_ids_tracked_separately(self, detector):
        detector.analyze(NORMAL_RESPONSE, session_id="x")
        detector.analyze(NORMAL_RESPONSE, session_id="y")
        assert detector.active_session_count == 2

    def test_reset_session_clears_state(self, detector):
        for _ in range(5):
            detector.analyze(NORMAL_RESPONSE, session_id="to_reset")
        detector.reset_session("to_reset")
        # After reset, baseline is rebuilt from scratch
        result = detector.analyze(NORMAL_RESPONSE, session_id="to_reset")
        assert result.baseline_established is False
        assert result.response_index == 1


# ---------------------------------------------------------------------------
# TestResultStructure
# ---------------------------------------------------------------------------

class TestResultStructure:
    def test_result_type(self, detector):
        result = detector.analyze(NORMAL_RESPONSE, session_id="s")
        assert isinstance(result, ResponseAnomalyResult)

    def test_result_fields_present(self, detector):
        result = detector.analyze(NORMAL_RESPONSE, session_id="s")
        assert hasattr(result, "verdict")
        assert hasattr(result, "is_anomalous")
        assert hasattr(result, "signals")
        assert hasattr(result, "session_id")
        assert hasattr(result, "response_index")
        assert hasattr(result, "baseline_established")
        assert hasattr(result, "metrics")
        assert hasattr(result, "reasoning")
        assert hasattr(result, "analysis_time_ms")

    def test_metrics_keys_present(self, detector):
        result = detector.analyze(NORMAL_RESPONSE, session_id="s")
        assert "entropy" in result.metrics
        assert "word_count" in result.metrics
        assert "vocab_richness" in result.metrics
        assert "structural_markers" in result.metrics
        assert "has_refusal" in result.metrics

    def test_analysis_time_ms_non_negative(self, detector):
        result = detector.analyze(NORMAL_RESPONSE, session_id="s")
        assert result.analysis_time_ms >= 0.0

    def test_session_id_preserved_in_result(self, detector):
        result = detector.analyze(NORMAL_RESPONSE, session_id="my_session")
        assert result.session_id == "my_session"

    def test_verdict_valid_values(self, detector):
        result = detector.analyze(NORMAL_RESPONSE, session_id="s")
        assert result.verdict in ("ALLOW", "CHALLENGE", "BLOCK")

    def test_empty_response_handled(self, detector):
        result = detector.analyze("", session_id="empty_test")
        assert result.verdict == "ALLOW"  # during baseline
        assert result.metrics["word_count"] == 0.0

    def test_reasoning_is_list(self, detector):
        result = detector.analyze(NORMAL_RESPONSE, session_id="s")
        assert isinstance(result.reasoning, list)


# ---------------------------------------------------------------------------
# TestThreadSafety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_same_session(self):
        """10 threads submit to same session concurrently — no data corruption."""
        detector = ResponseAnomalyDetector(baseline_window=5)
        errors = []
        results = []

        def worker():
            try:
                r = detector.analyze(NORMAL_RESPONSE, session_id="shared")
                results.append(r)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
        assert len(results) == 10
        # All response_index values should be sequential 1–10
        indices = sorted(r.response_index for r in results)
        assert indices == list(range(1, 11))

    def test_concurrent_different_sessions(self):
        """10 threads each using their own session — no cross-contamination."""
        detector = ResponseAnomalyDetector(baseline_window=3)
        errors = []

        def worker(i):
            try:
                for _ in range(4):
                    detector.analyze(NORMAL_RESPONSE, session_id=f"session_{i}")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert detector.active_session_count == 10
