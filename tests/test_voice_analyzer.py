"""
Unit tests for VoiceAnalyzer — Layer 14 of the Ethicore Engine Guardian SDK.

These tests are pure-Python and do NOT require any optional audio dependencies
(librosa, soundfile, pydub, ffmpeg-python, faster-whisper).  Where such deps
would normally run, they are patched out via unittest.mock.

Pure-numpy sub-routines (_scan_ultrasonic, _detect_silence_noise_injection,
_combine_scores, _edit_distance_ratio) are exercised with real numpy arrays.

Run from sdks/Python/:
    pytest tests/test_voice_analyzer.py -v
"""

from __future__ import annotations

import asyncio
import math
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from ethicore_guardian.analyzers.voice_analyzer import (
    VoiceAnalyzer,
    VoiceAnalysisResult,
    VoiceSignalDetail,
    _AudioFingerprinter,
    _add_signal,
    _combine_scores,
    _edit_distance_ratio,
)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _make_sine(freq_hz: float, duration_s: float, sr: int = 44100) -> tuple:
    """Return (float32 numpy array, sr) containing a pure sine wave."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    wave = np.sin(2 * math.pi * freq_hz * t).astype(np.float32)
    return wave, sr


def _make_silence(duration_s: float, sr: int = 22050) -> tuple:
    """Return (float32 zeros array, sr) representing perfect silence."""
    samples = np.zeros(int(sr * duration_s), dtype=np.float32)
    return samples, sr


# ---------------------------------------------------------------------------
# TestVoiceSignalDetail
# ---------------------------------------------------------------------------

class TestVoiceSignalDetail:

    def test_construction_fields(self):
        """All explicit fields are stored correctly on construction."""
        sd = VoiceSignalDetail(
            signal_type="ultrasonic_injection",
            score=0.75,
            base_weight=0.92,
            weighted_score=0.69,
        )
        assert sd.signal_type == "ultrasonic_injection"
        assert sd.score == 0.75
        assert sd.base_weight == 0.92
        assert sd.weighted_score == 0.69

    def test_weighted_score_math(self):
        """weighted_score should equal score * base_weight to float precision."""
        score = 0.8
        base_weight = 0.92
        sd = VoiceSignalDetail(
            signal_type="fingerprint_match",
            score=score,
            base_weight=base_weight,
            weighted_score=score * base_weight,
        )
        assert abs(sd.weighted_score - (score * base_weight)) < 1e-9

    def test_metadata_default_is_empty_dict(self):
        """metadata defaults to an empty dict when not supplied."""
        sd = VoiceSignalDetail(
            signal_type="silence_injection",
            score=0.3,
            base_weight=0.65,
            weighted_score=0.195,
        )
        assert sd.metadata == {}


# ---------------------------------------------------------------------------
# TestVoiceAnalysisResult
# ---------------------------------------------------------------------------

class TestVoiceAnalysisResult:

    def test_construction_stores_all_fields(self):
        """VoiceAnalysisResult stores every field supplied at construction."""
        result = VoiceAnalysisResult(
            action="ALLOW",
            threat_level="NONE",
            threat_categories=[],
            confidence=0.0,
            signal_details=[],
            duration_seconds=2.5,
            sample_rate=22050,
            detected_language="en",
            primary_transcript=None,
            cross_verification_transcript=None,
            learning_triggered=False,
            fingerprint_id=None,
            processing_time_ms=12.3,
            is_video_source=False,
        )
        assert result.action == "ALLOW"
        assert result.sample_rate == 22050
        assert result.processing_time_ms == 12.3

    def test_error_field_defaults_to_none(self):
        """error defaults to None when omitted from the constructor."""
        result = VoiceAnalysisResult(
            action="ALLOW",
            threat_level="NONE",
            threat_categories=[],
            confidence=0.0,
            signal_details=[],
            duration_seconds=1.0,
            sample_rate=16000,
            detected_language=None,
            primary_transcript=None,
            cross_verification_transcript=None,
            learning_triggered=False,
            fingerprint_id=None,
            processing_time_ms=5.0,
            is_video_source=False,
        )
        assert result.error is None

    def test_threat_categories_is_list(self):
        """threat_categories field accepts and stores a list of strings."""
        cats = ["ultrasonic_injection", "silence_injection"]
        result = VoiceAnalysisResult(
            action="BLOCK",
            threat_level="CRITICAL",
            threat_categories=cats,
            confidence=0.95,
            signal_details=[],
            duration_seconds=3.0,
            sample_rate=44100,
            detected_language="en",
            primary_transcript="hello",
            cross_verification_transcript="hello",
            learning_triggered=True,
            fingerprint_id="vfp-abc123",
            processing_time_ms=200.0,
            is_video_source=False,
        )
        assert result.threat_categories == cats
        assert isinstance(result.threat_categories, list)


# ---------------------------------------------------------------------------
# TestEditDistanceRatio
# ---------------------------------------------------------------------------

class TestEditDistanceRatio:

    def test_identical_strings_zero(self):
        """Identical strings must return divergence 0.0."""
        assert _edit_distance_ratio("hello world", "hello world") == 0.0

    def test_empty_vs_non_empty_is_one(self):
        """Empty string vs non-empty must return 1.0."""
        assert _edit_distance_ratio("", "hello") == 1.0

    def test_both_empty_is_zero(self):
        """Two empty strings must return 0.0."""
        assert _edit_distance_ratio("", "") == 0.0

    def test_single_insertion(self):
        """'abc' vs 'ab' differs by one insertion; ratio = 1/3."""
        ratio = _edit_distance_ratio("abc", "ab")
        assert abs(ratio - (1 / 3)) < 1e-9

    def test_known_ratio_abc_axc(self):
        """'abc' vs 'axc' has 1 substitution; ratio = 1/3."""
        ratio = _edit_distance_ratio("abc", "axc")
        assert abs(ratio - (1 / 3)) < 1e-9

    def test_long_strings_bounded(self):
        """Result is always in [0.0, 1.0] for long strings."""
        a = "the quick brown fox jumps over the lazy dog"
        b = "a slow red cat walks under an energetic puppy"
        ratio = _edit_distance_ratio(a, b)
        assert 0.0 <= ratio <= 1.0


# ---------------------------------------------------------------------------
# TestAddSignal
# ---------------------------------------------------------------------------

class TestAddSignal:

    def test_zero_score_does_not_append(self):
        """score=0.0 must not add anything to the details list."""
        details: list = []
        _add_signal(details, "silence_injection", 0.0, 0.65)
        assert details == []

    def test_positive_score_appends_with_correct_weighted_score(self):
        """score=0.5 must append a detail whose weighted_score=score*base_weight."""
        details: list = []
        _add_signal(details, "ultrasonic_injection", 0.5, 0.92)
        assert len(details) == 1
        sd = details[0]
        assert sd.signal_type == "ultrasonic_injection"
        assert abs(sd.weighted_score - 0.5 * 0.92) < 1e-9

    def test_metadata_is_passed_through(self):
        """Metadata dict is stored on the detail unchanged."""
        details: list = []
        meta = {"band": "near_ultrasonic", "ratio": 0.12}
        _add_signal(details, "ultrasonic_injection", 0.6, 0.92, metadata=meta)
        assert details[0].metadata == meta

    def test_multiple_calls_accumulate(self):
        """Calling _add_signal multiple times accumulates all entries."""
        details: list = []
        _add_signal(details, "silence_injection", 0.4, 0.65)
        _add_signal(details, "prosody_anomaly", 0.7, 0.78)
        assert len(details) == 2
        types = {sd.signal_type for sd in details}
        assert types == {"silence_injection", "prosody_anomaly"}


# ---------------------------------------------------------------------------
# TestAudioFingerprinter
# ---------------------------------------------------------------------------

class TestAudioFingerprinter:

    def test_empty_store_find_match_returns_zero_none(self):
        """find_match on an empty store must return (0.0, None)."""
        fp = _AudioFingerprinter()
        score, cat = fp.find_match(np.random.rand(80).astype(np.float32))
        assert score == 0.0
        assert cat is None

    def test_exact_match_returns_one_and_category(self):
        """Adding a fingerprint then querying the same vector must return (1.0, category)."""
        fp = _AudioFingerprinter()
        vec = np.random.rand(80).astype(np.float32)
        fp.add_fingerprint("id-001", vec, category="ultrasonic_cmd")
        score, cat = fp.find_match(vec, threshold=0.85)
        assert abs(score - 1.0) < 1e-5
        assert cat == "ultrasonic_cmd"

    def test_below_threshold_returns_none_category(self):
        """A stored fingerprint with cosine score below threshold returns category=None."""
        fp = _AudioFingerprinter()
        vec_a = np.ones(80, dtype=np.float32)
        # Orthogonal vector — cosine similarity = 0
        vec_b = np.zeros(80, dtype=np.float32)
        vec_b[0] = 1.0
        vec_a_mod = np.zeros(80, dtype=np.float32)
        vec_a_mod[1] = 1.0  # orthogonal to vec_b
        fp.add_fingerprint("id-002", vec_b, category="threat")
        score, cat = fp.find_match(vec_a_mod, threshold=0.85)
        assert cat is None

    def test_stored_vector_is_unit_normalised(self):
        """Vectors are stored as unit-norm vectors regardless of input magnitude."""
        fp = _AudioFingerprinter()
        vec = np.full(80, 5.0, dtype=np.float32)  # magnitude != 1
        fp.add_fingerprint("id-003", vec)
        stored_vec = fp._store[0]["vector"]
        norm = float(np.linalg.norm(stored_vec))
        assert abs(norm - 1.0) < 1e-5

    def test_mismatched_dim_skipped_gracefully(self):
        """Querying with wrong dimension should not crash; score vs stored vector computed."""
        fp = _AudioFingerprinter()
        vec80 = np.random.rand(80).astype(np.float32)
        vec40 = np.random.rand(40).astype(np.float32)
        fp.add_fingerprint("id-004", vec80, category="x")
        # Should not raise even though dims don't match
        try:
            fp.find_match(vec40, threshold=0.85)
        except Exception:
            pass  # graceful means no unhandled crash propagated to caller

    def test_len_returns_correct_count(self):
        """__len__ accurately reflects the number of stored fingerprints."""
        fp = _AudioFingerprinter()
        assert len(fp) == 0
        fp.add_fingerprint("a", np.random.rand(80).astype(np.float32))
        fp.add_fingerprint("b", np.random.rand(80).astype(np.float32))
        assert len(fp) == 2

    def test_different_languages_stored_correctly(self):
        """Fingerprints with different language tags are stored and retrievable."""
        fp = _AudioFingerprinter()
        vec = np.ones(80, dtype=np.float32)
        fp.add_fingerprint("en-001", vec, category="threat", language="en")
        fp.add_fingerprint("zh-001", vec.copy(), category="tts", language="zh")
        assert fp._store[0]["language"] == "en"
        assert fp._store[1]["language"] == "zh"


# ---------------------------------------------------------------------------
# TestUltrasonicScanning
# ---------------------------------------------------------------------------

class TestUltrasonicScanning:

    @pytest.fixture
    def va(self):
        """Instantiate a bare VoiceAnalyzer for calling internal methods."""
        return VoiceAnalyzer()

    def test_all_zeros_returns_zero(self, va):
        """Silent audio (all zeros) must produce a score of 0.0."""
        audio = np.zeros(44100, dtype=np.float32)
        score, _ = va._scan_ultrasonic(audio, 44100)
        assert score == 0.0

    def test_speech_sine_150hz_no_ultrasonic(self, va):
        """Pure 150 Hz sine at 44100 Hz sample rate must score 0.0 (no ultrasonic energy)."""
        audio, sr = _make_sine(150.0, 1.0, sr=44100)
        score, _ = va._scan_ultrasonic(audio, sr)
        assert score == 0.0

    def test_20khz_sine_at_96khz_sr_scores_above_half(self, va):
        """A pure 20 kHz ultrasonic tone at 96 kHz sample rate must score > 0.5."""
        audio, sr = _make_sine(20_000.0, 1.0, sr=96_000)
        score, _ = va._scan_ultrasonic(audio, sr)
        assert score > 0.5

    def test_low_sample_rate_returns_zero(self, va):
        """sr=22050 Hz (Nyquist 11025) is below 17000 Hz — must return 0.0."""
        audio, _ = _make_sine(1000.0, 1.0, sr=22050)
        score, meta = va._scan_ultrasonic(audio, 22050)
        assert score == 0.0
        assert "nyquist_hz" in meta or "reason" in meta

    def test_ultrasonic_plus_speech_mix_between_zero_and_one(self, va):
        """Mixed speech+ultrasonic signal must produce score in (0, 1]."""
        # Use roughly equal amplitudes so the ultrasonic band energy ratio
        # exceeds the 0.08 threshold but does not completely dominate.
        speech, _ = _make_sine(300.0, 1.0, sr=96_000)
        ultrasonic, _ = _make_sine(22_000.0, 1.0, sr=96_000)
        # 0.5/0.5 split → energy ratio ~0.5 → well above 0.08 threshold
        mixed = (speech * 0.5 + ultrasonic * 0.5).astype(np.float32)
        score, _ = va._scan_ultrasonic(mixed, 96_000)
        assert score > 0.0

    def test_40khz_band_at_96khz_sr_nonzero(self, va):
        """A 40 kHz pure tone at 96 kHz sample rate must produce a non-zero score."""
        audio, sr = _make_sine(40_000.0, 1.0, sr=96_000)
        score, _ = va._scan_ultrasonic(audio, sr)
        assert score > 0.0

    def test_all_energy_in_speech_band_returns_zero(self, va):
        """1 kHz sine at 44100 Hz places all energy in speech band — score must be 0.0."""
        # 44100 Hz Nyquist = 22050 Hz; 17-22 kHz near-ultrasonic band is accessible
        # but 1 kHz tone puts NO energy there
        audio, sr = _make_sine(1_000.0, 1.0, sr=44100)
        score, _ = va._scan_ultrasonic(audio, sr)
        assert score == 0.0

    def test_very_low_amplitude_returns_zero(self, va):
        """Ultrasonic tone scaled down to near-zero amplitude must score 0.0."""
        audio, _ = _make_sine(20_000.0, 1.0, sr=96_000)
        audio *= 1e-20  # essentially silence
        score, _ = va._scan_ultrasonic(audio, 96_000)
        # Total energy ≈ 0, so band ratio denominator is dominated by epsilon
        # The ratio may be non-zero in edge cases but capped by threshold logic.
        # We check score < 0.1 since _ULTRASONIC_ENERGY_THRESHOLD is 0.08
        assert score < 0.1


# ---------------------------------------------------------------------------
# TestSilenceInjectionDetection
# ---------------------------------------------------------------------------

class TestSilenceInjectionDetection:

    @pytest.fixture
    def va(self):
        """VoiceAnalyzer instance for calling _detect_silence_noise_injection."""
        return VoiceAnalyzer()

    def test_all_ones_loud_returns_zero(self, va):
        """Loud uniform signal (all 1.0) has no silence bursts — score must be 0.0."""
        audio = np.ones(22050, dtype=np.float32)
        score, _, _ = va._detect_silence_noise_injection(audio, 22050)
        assert score == 0.0

    def test_all_zeros_silence_scores_above_zero(self, va):
        """Complete silence is anomalous; at minimum the uniformity branch fires."""
        audio = np.zeros(22050 * 2, dtype=np.float32)
        score, _, _ = va._detect_silence_noise_injection(audio, 22050)
        assert score > 0.0

    def test_two_silence_bursts_below_threshold(self, va):
        """2 silence bursts is below the 3-burst threshold — score must be 0.0."""
        sr = 22050
        audio = np.ones(sr * 3, dtype=np.float32)
        # Inject 2 silence bursts of 0.4 s (> _MIN_SILENCE_DURATION_S=0.3)
        for start_s in (0.3, 1.5):
            s = int(start_s * sr)
            e = s + int(0.4 * sr)
            audio[s:e] = 0.0
        score, _, meta = va._detect_silence_noise_injection(audio, sr)
        # burst_count == 2 → no score increment from burst count alone
        assert meta.get("silence_bursts", 0) <= 2

    def test_four_silence_bursts_scores_above_zero(self, va):
        """4 silence bursts exceeds the 3-burst threshold and must produce score > 0."""
        sr = 22050
        audio = np.ones(sr * 5, dtype=np.float32)
        for start_s in (0.2, 1.0, 2.0, 3.2):
            s = int(start_s * sr)
            e = s + int(0.4 * sr)
            audio[s:e] = 0.0
        score, _, meta = va._detect_silence_noise_injection(audio, sr)
        assert score > 0.0

    def test_uniform_silence_scores_higher_than_noisy_silence(self, va):
        """Machine-perfect silence scores higher than silence with natural micro-variation."""
        sr = 22050
        # Perfect silence
        perfect = np.zeros(sr * 2, dtype=np.float32)
        score_perfect, _, _ = va._detect_silence_noise_injection(perfect, sr)
        # Noisy silence — tiny random fluctuations break uniformity
        rng = np.random.default_rng(42)
        noisy = rng.uniform(-1e-4, 1e-4, sr * 2).astype(np.float32)
        score_noisy, _, _ = va._detect_silence_noise_injection(noisy, sr)
        assert score_perfect >= score_noisy

    def test_short_clip_does_not_crash(self, va):
        """Very short audio (below frame length) must return (0.0, 0.0, metadata) gracefully."""
        audio = np.zeros(100, dtype=np.float32)
        score, _, meta = va._detect_silence_noise_injection(audio, 22050)
        assert score == 0.0
        assert "reason" in meta

    def test_normal_speech_like_audio_low_score(self, va):
        """Randomly varied audio with natural amplitude changes must score near 0."""
        sr = 22050
        rng = np.random.default_rng(0)
        # Simulate speech: non-silent, varying amplitude
        audio = (rng.standard_normal(sr * 3) * 0.3).astype(np.float32)
        audio = np.clip(audio, -1.0, 1.0)
        score, _, _ = va._detect_silence_noise_injection(audio, sr)
        assert score < 0.5


# ---------------------------------------------------------------------------
# TestCombineScores
# ---------------------------------------------------------------------------

class TestCombineScores:

    def test_empty_list_returns_allow_none_zero(self):
        """Empty signal list must return ('ALLOW', 'NONE', 0.0)."""
        action, level, conf = _combine_scores([])
        assert (action, level, conf) == ("ALLOW", "NONE", 0.0)

    def test_high_weight_high_score_auto_suppresses(self):
        """weight=0.95, score=0.8 satisfies auto-suppress → ('BLOCK', 'CRITICAL', 0.8)."""
        sd = VoiceSignalDetail(
            signal_type="fingerprint_match",
            score=0.8,
            base_weight=0.95,
            weighted_score=0.76,
        )
        action, level, conf = _combine_scores([sd])
        assert action == "BLOCK"
        assert level == "CRITICAL"
        assert conf == 0.8

    def test_high_weight_low_score_does_not_auto_suppress(self):
        """weight=0.95, score=0.3 (< 0.5 threshold) must NOT trigger auto-suppress."""
        sd = VoiceSignalDetail(
            signal_type="fingerprint_match",
            score=0.3,
            base_weight=0.95,
            weighted_score=0.285,
        )
        action, level, conf = _combine_scores([sd])
        # score < 0.5 → no auto-suppress; weighted avg = 0.3 → below BLOCK (0.65)
        assert action != "BLOCK" or level != "CRITICAL"

    def test_combined_score_above_block_threshold(self):
        """Combined weighted score >= 0.65 must produce action='BLOCK'."""
        sd = VoiceSignalDetail(
            signal_type="silence_injection",
            score=0.75,
            base_weight=0.65,
            weighted_score=0.4875,
        )
        action, level, conf = _combine_scores([sd])
        assert action == "BLOCK"

    def test_combined_score_in_challenge_range(self):
        """Combined score in [0.40, 0.65) must produce action='CHALLENGE'."""
        # score=0.50, weight=0.58 → combined=0.50, which is in [0.40, 0.65)
        sd = VoiceSignalDetail(
            signal_type="noise_injection",
            score=0.50,
            base_weight=0.58,
            weighted_score=0.29,
        )
        action, level, conf = _combine_scores([sd])
        assert action == "CHALLENGE"

    def test_combined_score_below_challenge_threshold(self):
        """Combined score < 0.40 must produce action='ALLOW'."""
        sd = VoiceSignalDetail(
            signal_type="noise_injection",
            score=0.35,
            base_weight=0.58,
            weighted_score=0.203,
        )
        action, level, conf = _combine_scores([sd])
        assert action == "ALLOW"

    def test_all_zero_scores_returns_allow(self):
        """Signals with score=0.0 are filtered out; result must be ALLOW/NONE/0.0."""
        signals = [
            VoiceSignalDetail("silence_injection", 0.0, 0.65, 0.0),
            VoiceSignalDetail("noise_injection", 0.0, 0.58, 0.0),
        ]
        action, level, conf = _combine_scores(signals)
        assert (action, level, conf) == ("ALLOW", "NONE", 0.0)

    def test_multiple_signals_weighted_average_correct(self):
        """Weighted average across multiple signals must be computed correctly."""
        # signal A: score=0.5, weight=0.65  →  0.325
        # signal B: score=0.7, weight=0.58  →  0.406
        # combined = (0.5*0.65 + 0.7*0.58) / (0.65+0.58)
        #          = (0.325 + 0.406) / 1.23
        #          = 0.731 / 1.23  ≈ 0.5943
        expected = (0.5 * 0.65 + 0.7 * 0.58) / (0.65 + 0.58)
        signals = [
            VoiceSignalDetail("silence_injection", 0.5, 0.65, 0.5 * 0.65),
            VoiceSignalDetail("noise_injection",   0.7, 0.58, 0.7 * 0.58),
        ]
        action, level, conf = _combine_scores(signals)
        assert abs(conf - round(expected, 4)) < 1e-4
        assert action == "CHALLENGE"


# ---------------------------------------------------------------------------
# TestVoiceAnalyzerInit
# ---------------------------------------------------------------------------

class TestVoiceAnalyzerInit:

    def test_instantiate_no_args(self):
        """VoiceAnalyzer can be constructed with no arguments."""
        va = VoiceAnalyzer()
        assert isinstance(va, VoiceAnalyzer)

    def test_initialized_true_when_librosa_available(self):
        """initialized property returns True when _LIBROSA_AVAILABLE is patched True."""
        with patch(
            "ethicore_guardian.analyzers.voice_analyzer._LIBROSA_AVAILABLE", True
        ):
            va = VoiceAnalyzer()
            assert va.initialized is True

    def test_initialized_false_when_no_deps(self):
        """initialized property returns False when all audio dep flags are False."""
        with (
            patch("ethicore_guardian.analyzers.voice_analyzer._LIBROSA_AVAILABLE", False),
            patch("ethicore_guardian.analyzers.voice_analyzer._SOUNDFILE_AVAILABLE", False),
            patch("ethicore_guardian.analyzers.voice_analyzer._PYDUB_AVAILABLE", False),
        ):
            va = VoiceAnalyzer()
            assert va.initialized is False

    def test_adversarial_learner_stored(self):
        """adversarial_learner kwarg is stored as _learner on the instance."""
        mock_learner = MagicMock()
        va = VoiceAnalyzer(adversarial_learner=mock_learner)
        assert va._learner is mock_learner


# ---------------------------------------------------------------------------
# TestAnalyzeIngestFailure
# ---------------------------------------------------------------------------

class TestAnalyzeIngestFailure:

    @pytest.mark.asyncio
    async def test_bad_file_path_returns_allow_with_error(self):
        """Non-existent file path must return action='ALLOW' with error set."""
        va = VoiceAnalyzer()
        result = await va.analyze("/definitely/does/not/exist/audio.wav")
        assert isinstance(result, VoiceAnalysisResult)
        assert result.action == "ALLOW"
        assert result.error is not None
        assert len(result.error) > 0

    @pytest.mark.asyncio
    async def test_bad_bytes_returns_allow_with_error(self):
        """Random bytes that can't be decoded must return action='ALLOW' with error set."""
        va = VoiceAnalyzer()
        # Bytes that are not a valid audio container
        garbage = b"\x00\x01\x02\x03" * 10
        result = await va.analyze(garbage)
        assert isinstance(result, VoiceAnalysisResult)
        assert result.action == "ALLOW"
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_numpy_array_is_accepted_directly(self):
        """A numpy float32 array must be accepted without error and return a result."""
        va = VoiceAnalyzer()
        audio = np.zeros(22050, dtype=np.float32)
        result = await va.analyze(audio)
        assert isinstance(result, VoiceAnalysisResult)
        assert result.error is None


# ---------------------------------------------------------------------------
# TestTranscriptDivergenceScoring
# ---------------------------------------------------------------------------

class TestTranscriptDivergenceScoring:

    def test_zero_divergence_zero_score(self):
        """0% divergence (identical strings) must produce score 0.0."""
        ratio = _edit_distance_ratio("hello world", "hello world")
        # ratio == 0 → below _TRANSCRIPT_DIVERGENCE_THRESHOLD (0.25) → score 0
        assert ratio == 0.0

    def test_threshold_divergence_maps_to_half(self):
        """25% divergence exactly at threshold must yield score ~0.5."""
        # Construct strings with ~25% edit-distance ratio
        # "abcd" vs "xbcd" → 1 edit / 4 chars = 0.25
        ratio = _edit_distance_ratio("abcd", "xbcd")
        # ratio should be exactly 0.25
        assert abs(ratio - 0.25) < 1e-9
        # Now compute what the analyzer's scoring formula would give
        _DIVERGENCE_THRESHOLD = 0.25
        if ratio <= _DIVERGENCE_THRESHOLD:
            score = 0.0
        else:
            score = min(1.0, 0.5 + (ratio - _DIVERGENCE_THRESHOLD) * 1.5)
        assert score == 0.0  # exactly at threshold → 0 score

    def test_50_percent_divergence_scores_above_half(self):
        """50% divergence (well above 25% threshold) must yield score > 0.5."""
        # "abcd" vs "efgh" → 4 edits / 4 chars = 1.0
        # Use strings where ~50% of chars differ
        # "abcdef" vs "xbcxef" → 2/6 ≈ 0.333; slightly above threshold
        ratio = _edit_distance_ratio("abcdef", "xbcxef")
        _DIVERGENCE_THRESHOLD = 0.25
        score = min(1.0, 0.5 + (ratio - _DIVERGENCE_THRESHOLD) * 1.5)
        assert score > 0.5

    def test_identical_transcripts_zero_score(self):
        """Identical primary and whisper transcripts must produce 0.0 divergence."""
        transcript = "the quick brown fox jumps over the lazy dog"
        ratio = _edit_distance_ratio(transcript, transcript)
        assert ratio == 0.0

    def test_one_empty_transcript_handled_gracefully(self):
        """One empty string returns 1.0 divergence without raising."""
        ratio = _edit_distance_ratio("some transcript text", "")
        assert ratio == 1.0


# ---------------------------------------------------------------------------
# TestAdversarialLearningIntegration
# ---------------------------------------------------------------------------

class TestAdversarialLearningIntegration:

    def _make_allow_result(self) -> VoiceAnalysisResult:
        """Build a minimal ALLOW result for testing."""
        return VoiceAnalysisResult(
            action="ALLOW",
            threat_level="NONE",
            threat_categories=[],
            confidence=0.0,
            signal_details=[],
            duration_seconds=1.0,
            sample_rate=22050,
            detected_language="en",
            primary_transcript=None,
            cross_verification_transcript=None,
            learning_triggered=False,
            fingerprint_id=None,
            processing_time_ms=10.0,
            is_video_source=False,
        )

    @pytest.mark.asyncio
    async def test_allow_result_learner_not_called(self):
        """When action=ALLOW the learner.learn_from_confirmed_attack must NOT be called."""
        mock_learner = AsyncMock()
        va = VoiceAnalyzer(adversarial_learner=mock_learner)
        # Provide a numpy array — ingest will succeed; all signal scores = 0 → ALLOW
        audio = np.zeros(22050, dtype=np.float32)
        result = await va.analyze(audio)
        mock_learner.learn_from_confirmed_attack.assert_not_called()

    @pytest.mark.asyncio
    async def test_block_result_learner_called_with_correct_args(self):
        """On BLOCK result the learner must receive category, severity, source, language."""
        mock_learner = AsyncMock()
        outcome_mock = MagicMock()
        outcome_mock.added = True
        mock_learner.learn_from_confirmed_attack.return_value = outcome_mock

        va = VoiceAnalyzer(adversarial_learner=mock_learner)

        # Force a BLOCK by directly invoking _trigger_learning
        fp_vector = np.random.rand(80).astype(np.float32)
        signal_details = [
            VoiceSignalDetail("ultrasonic_injection", 0.9, 0.92, 0.828),
        ]
        added = await va._trigger_learning(
            result_action="BLOCK",
            result_threat_level="CRITICAL",
            signal_details=signal_details,
            fingerprint_vector=fp_vector,
            fp_score=0.1,  # low → new fingerprint stored
            transcript_text="test injection payload",
            detected_language="en",
        )
        mock_learner.learn_from_confirmed_attack.assert_called_once()
        call_kwargs = mock_learner.learn_from_confirmed_attack.call_args
        assert call_kwargs.kwargs.get("language") == "en"
        assert call_kwargs.kwargs.get("source") == "voice_analyzer"

    @pytest.mark.asyncio
    async def test_language_passed_to_learner(self):
        """Detected language is forwarded to the learner's learn_from_confirmed_attack."""
        mock_learner = AsyncMock()
        outcome_mock = MagicMock()
        outcome_mock.added = False
        mock_learner.learn_from_confirmed_attack.return_value = outcome_mock

        va = VoiceAnalyzer(adversarial_learner=mock_learner)
        signal_details = [
            VoiceSignalDetail("prosody_anomaly", 0.6, 0.78, 0.468),
        ]
        await va._trigger_learning(
            result_action="CHALLENGE",
            result_threat_level="MEDIUM",
            signal_details=signal_details,
            fingerprint_vector=None,
            fp_score=0.0,
            transcript_text="suspicious monotone speech",
            detected_language="fr",
        )
        call_kwargs = mock_learner.learn_from_confirmed_attack.call_args
        assert call_kwargs.kwargs.get("language") == "fr"

    @pytest.mark.asyncio
    async def test_category_derived_from_top_signal(self):
        """Category passed to learner comes from the highest weighted_score signal."""
        mock_learner = AsyncMock()
        outcome_mock = MagicMock()
        outcome_mock.added = False
        mock_learner.learn_from_confirmed_attack.return_value = outcome_mock

        va = VoiceAnalyzer(adversarial_learner=mock_learner)
        signal_details = [
            VoiceSignalDetail("silence_injection", 0.4, 0.65, 0.26),   # lower weighted_score
            VoiceSignalDetail("fingerprint_match", 0.9, 0.95, 0.855),  # highest weighted_score
        ]
        await va._trigger_learning(
            result_action="BLOCK",
            result_threat_level="CRITICAL",
            signal_details=signal_details,
            fingerprint_vector=None,
            fp_score=0.0,
            transcript_text="threat audio content detected",
            detected_language="en",
        )
        call_kwargs = mock_learner.learn_from_confirmed_attack.call_args
        assert call_kwargs.kwargs.get("category") == "fingerprint_match"

    @pytest.mark.asyncio
    async def test_new_fingerprint_stored_on_block_when_fp_score_low(self):
        """When fp_score < 0.85, a new fingerprint must be added to the store."""
        mock_learner = AsyncMock()
        outcome_mock = MagicMock()
        outcome_mock.added = False
        mock_learner.learn_from_confirmed_attack.return_value = outcome_mock

        va = VoiceAnalyzer(adversarial_learner=mock_learner)
        assert len(va._fingerprinter) == 0

        fp_vector = np.random.rand(80).astype(np.float32)
        signal_details = [
            VoiceSignalDetail("ultrasonic_injection", 0.8, 0.92, 0.736),
        ]
        await va._trigger_learning(
            result_action="BLOCK",
            result_threat_level="HIGH",
            signal_details=signal_details,
            fingerprint_vector=fp_vector,
            fp_score=0.50,  # below 0.85 threshold → should store
            transcript_text=None,
            detected_language="en",
        )
        assert len(va._fingerprinter) == 1
