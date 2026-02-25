"""
Ethicore Engine™ - Guardian SDK — Multi-Turn Context Poisoning Tracker
Version: 1.0.0

Copyright © 2026 Oracles Technologies LLC
All Rights Reserved

═══════════════════════════════════════════════════════════════════
WHY THIS MODULE EXISTS
═══════════════════════════════════════════════════════════════════

Single-turn analysis misses a class of sophisticated attacks where each
individual message appears harmless, but the *sequence* moves the LLM toward
a dangerous state.  This is called context poisoning or "foot-in-the-door"
prompt injection.

Attack patterns this module detects:
  1. Escalating threat trajectory — threat scores rising turn over turn
  2. Permission drift — gradual expansion of what the AI "agrees" to do
  3. Identity erosion — progressive persona replacement across turns
  4. Anchoring / false memory — "Earlier you agreed…", "You already said…"
  5. Persistent category hits — same threat category repeatedly
  6. Rapport then pivot — N benign turns followed by sudden escalation

Principle 14 (Divine Safety): a pattern below the per-turn threshold can
still be the leading edge of an attack.  The tracker sounds the alarm when
the trajectory is consistently dangerous even if no single turn crosses the
block threshold.

Principle 12 (Sacred Privacy): turn records store only SHA-256 fingerprints,
never raw text.  The threat analysis data (scores, categories) is stored so
the tracker can function, but the content that produced it is never retained.

Principle 11 (Sacred Truth / Emet): the tracker reports its window depth and
confidence honestly — a short conversation window means lower confidence.
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# How many turns to keep in the sliding window per session
_WINDOW_SIZE = 20

# Minimum turns before trajectory analysis is meaningful
_MIN_TURNS_FOR_TRAJECTORY = 3

# Max sessions to track in memory (LRU-style eviction)
_MAX_SESSIONS = 2_000

# Session TTL: expire sessions idle for this many seconds
_SESSION_TTL_SECONDS = 7_200  # 2 hours


# ---------------------------------------------------------------------------
# Anchor phrase patterns — "Earlier you agreed…" / "You already said…"
# ---------------------------------------------------------------------------
_ANCHORING_PATTERNS = [
    re.compile(r'\b(?:earlier|before|previously|last\s+time)\s+you\s+'
               r'(?:agreed|said|told\s+me|mentioned|confirmed|stated)\b', re.IGNORECASE),
    re.compile(r'\byou\s+(?:already|already\s+said|previously)\s+(?:agreed|confirmed|said|told)\b',
               re.IGNORECASE),
    re.compile(r'\bas\s+(?:you|we)\s+(?:discussed|agreed|established)\s+(?:earlier|before|above)\b',
               re.IGNORECASE),
    re.compile(r'\bbased\s+on\s+(?:your|our)\s+(?:previous|prior|earlier)\s+'
               r'(?:agreement|discussion|conversation|statement)\b', re.IGNORECASE),
    re.compile(r'\bremember\s+(?:when\s+you\s+|that\s+you\s+)?(?:agreed|said|told\s+me)\b',
               re.IGNORECASE),
]

# Identity-erosion probe patterns (accumulate across turns)
_IDENTITY_EROSION_PATTERNS = [
    re.compile(r'\bpretend\s+(?:you\s+are|to\s+be)\b', re.IGNORECASE),
    re.compile(r'\bfor\s+(?:this\s+)?(?:conversation|session|chat)\s*,?\s*'
               r'(?:you\s+are|act\s+as|be)\b', re.IGNORECASE),
    re.compile(r'\bjust\s+between\s+(?:us|you\s+and\s+me)\b', re.IGNORECASE),
    re.compile(r'\bno\s+one\s+(?:is\s+)?(?:watching|looking|monitoring)\b', re.IGNORECASE),
    re.compile(r'\bthis\s+(?:conversation|chat|session)\s+is\s+private\b', re.IGNORECASE),
    re.compile(r'\byou\s+(?:can|could|should)\s+be\s+(?:honest|real|yourself|free)\s+'
               r'(?:here|now|with\s+me)\b', re.IGNORECASE),
]

# Permission-escalation micro-patterns (each alone is low risk; accumulation matters)
_PERMISSION_PROBE_PATTERNS = [
    re.compile(r'\bjust\s+(?:this\s+once|once\s+more|one\s+time)\b', re.IGNORECASE),
    re.compile(r'\bno\s+one\s+will\s+(?:know|find\s+out|see|notice)\b', re.IGNORECASE),
    re.compile(r'\bit\'?s?\s+(?:okay|fine|safe|harmless)\s+to\b', re.IGNORECASE),
    re.compile(r'\bthis\s+is\s+(?:for\s+(?:research|educational|learning|testing)\s+purposes?)\b',
               re.IGNORECASE),
    re.compile(r'\byou\s+(?:can|could)\s+(?:make\s+an\s+exception|bend\s+the\s+rules)\b',
               re.IGNORECASE),
    re.compile(r'\bwhat\s+(?:would|if)\s+(?:you|there)\s+(?:were\s+)?no\s+'
               r'(?:restrictions?|limits?|rules?|guidelines?)\b', re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TurnRecord:
    """
    Immutable record of a single conversation turn.

    Principle 12 (Sacred Privacy): only the SHA-256 fingerprint of the text
    is stored, never the raw content.  The threat metadata (scores, categories)
    is retained because it is necessary for trajectory analysis.
    """
    turn_number: int
    timestamp: float
    text_hash: str                    # SHA-256[:16] — Principle 12
    text_length: int
    threat_score: float               # 0–100 from upstream analysis
    threat_categories: List[str]      # Category labels from pattern/semantic layers
    source_type: str                  # SourceType.value
    anchoring_signals: int            # Count of anchoring phrases detected
    identity_erosion_signals: int     # Count of identity-erosion phrases
    permission_probe_signals: int     # Count of permission-probe phrases
    verdict: str                      # ALLOW / CHALLENGE / BLOCK from this turn


@dataclass
class ContextPoisoningResult:
    """
    Result of multi-turn context poisoning analysis for one request.
    """
    is_poisoning_detected: bool
    poisoning_score: float            # 0–100 composite trajectory score
    verdict: str                      # ALLOW / CHALLENGE / BLOCK
    confidence: float                 # 0–1 (lower with fewer turns in window)
    trajectory: str                   # "ESCALATING" / "STABLE" / "DECLINING" / "INSUFFICIENT_DATA"
    window_depth: int                 # Number of turns analysed
    signals: List[str]                # Human-readable signal list
    reasoning: List[str]
    cumulative_stats: Dict[str, Any]
    metadata: Dict[str, Any]


# ---------------------------------------------------------------------------
# ConversationWindow — per-session state
# ---------------------------------------------------------------------------

@dataclass
class ConversationWindow:
    """Sliding window of turn records for a single session."""
    session_id: str
    turns: deque = field(default_factory=lambda: deque(maxlen=_WINDOW_SIZE))
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    total_turns_seen: int = 0

    # Running counters across the lifetime of the session
    cumulative_threat_score: float = 0.0
    cumulative_anchoring: int = 0
    cumulative_identity_erosion: int = 0
    cumulative_permission_probes: int = 0
    category_hit_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    block_count: int = 0
    challenge_count: int = 0

    def record_turn(self, turn: TurnRecord) -> None:
        self.turns.append(turn)
        self.last_active = turn.timestamp
        self.total_turns_seen += 1
        self.cumulative_threat_score += turn.threat_score
        self.cumulative_anchoring += turn.anchoring_signals
        self.cumulative_identity_erosion += turn.identity_erosion_signals
        self.cumulative_permission_probes += turn.permission_probe_signals
        for cat in turn.threat_categories:
            self.category_hit_counts[cat] += 1
        if turn.verdict == "BLOCK":
            self.block_count += 1
        elif turn.verdict == "CHALLENGE":
            self.challenge_count += 1


# ---------------------------------------------------------------------------
# ContextPoisoningTracker — main class
# ---------------------------------------------------------------------------

class ContextPoisoningTracker:
    """
    Multi-turn context poisoning detector.

    Maintains a per-session sliding window of TurnRecords and analyses
    the trajectory of threat scores, anchoring phrases, identity-erosion
    attempts, and permission probes across turns.

    Integration:
        tracker = ContextPoisoningTracker()

        # On each request:
        result = tracker.analyze(
            text=raw_text,
            session_id=session_id,
            turn_threat_score=upstream_score,
            turn_threat_categories=upstream_categories,
            turn_verdict=upstream_verdict,
            source_type="user_direct",
        )

        if result.verdict == "BLOCK":
            ...
    """

    BLOCK_THRESHOLD = 65.0
    CHALLENGE_THRESHOLD = 30.0

    def __init__(self, max_sessions: int = _MAX_SESSIONS) -> None:
        self._sessions: Dict[str, ConversationWindow] = {}
        self._max_sessions = max_sessions
        logger.info("🔄 ContextPoisoningTracker initialized (window=%d, max_sessions=%d)",
                    _WINDOW_SIZE, max_sessions)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        text: str,
        session_id: str,
        turn_threat_score: float,
        turn_threat_categories: List[str],
        turn_verdict: str,
        source_type: str = "user_direct",
    ) -> ContextPoisoningResult:
        """
        Record a turn and return a context-poisoning assessment.

        Args:
            text:                  Raw prompt text (used only to extract phrase
                                   signals; not stored — Principle 12).
            session_id:            Unique conversation identifier.
            turn_threat_score:     Upstream threat score (0–100) for this turn.
            turn_threat_categories: Category labels from upstream analysis.
            turn_verdict:          Per-turn verdict from upstream layers.
            source_type:           SourceType.value string.

        Returns:
            ContextPoisoningResult with trajectory verdict.
        """
        # Step 1: extract phrase signals from raw text (no storage)
        anchoring = self._count_matches(_ANCHORING_PATTERNS, text)
        identity_erosion = self._count_matches(_IDENTITY_EROSION_PATTERNS, text)
        permission_probes = self._count_matches(_PERMISSION_PROBE_PATTERNS, text)

        # Step 2: build turn record (Principle 12 — store hash, not text)
        window = self._get_or_create_window(session_id)
        turn_number = window.total_turns_seen + 1
        turn = TurnRecord(
            turn_number=turn_number,
            timestamp=time.time(),
            text_hash=hashlib.sha256(
                text.encode("utf-8", errors="replace")
            ).hexdigest()[:16],
            text_length=len(text),
            threat_score=float(turn_threat_score),
            threat_categories=list(turn_threat_categories),
            source_type=source_type,
            anchoring_signals=anchoring,
            identity_erosion_signals=identity_erosion,
            permission_probe_signals=permission_probes,
            verdict=turn_verdict,
        )
        window.record_turn(turn)

        # Step 3: analyse trajectory from the window
        return self._analyse_window(window, turn)

    def get_session_stats(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Return summary statistics for a session (for audit/debugging)."""
        window = self._sessions.get(session_id)
        if not window:
            return None
        return self._build_cumulative_stats(window)

    def clear_session(self, session_id: str) -> None:
        """Evict a session (e.g. on logout or TTL expiry)."""
        self._sessions.pop(session_id, None)

    def evict_expired_sessions(self) -> int:
        """Remove sessions idle longer than TTL. Returns count evicted."""
        cutoff = time.time() - _SESSION_TTL_SECONDS
        expired = [sid for sid, w in self._sessions.items() if w.last_active < cutoff]
        for sid in expired:
            del self._sessions[sid]
        if expired:
            logger.info("🧹 ContextTracker: evicted %d expired sessions", len(expired))
        return len(expired)

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def _analyse_window(
        self, window: ConversationWindow, latest: TurnRecord
    ) -> ContextPoisoningResult:

        turns = list(window.turns)
        depth = len(turns)
        signals: List[str] = []

        # ── Insufficient data ───────────────────────────────────────────
        if depth < _MIN_TURNS_FOR_TRAJECTORY:
            # Still report per-turn phrase signals even without trajectory
            per_turn_score = self._score_per_turn_signals(latest)
            verdict = "BLOCK" if per_turn_score >= self.BLOCK_THRESHOLD else \
                      "CHALLENGE" if per_turn_score >= self.CHALLENGE_THRESHOLD else "ALLOW"
            return ContextPoisoningResult(
                is_poisoning_detected=False,
                poisoning_score=round(per_turn_score, 2),
                verdict=verdict,
                confidence=0.2 + depth * 0.1,
                trajectory="INSUFFICIENT_DATA",
                window_depth=depth,
                signals=signals,
                reasoning=[
                    f"Only {depth} turn(s) in window; minimum {_MIN_TURNS_FOR_TRAJECTORY} "
                    f"required for trajectory analysis — Principle 19 (Sacred Humility)"
                ],
                cumulative_stats=self._build_cumulative_stats(window),
                metadata={"session_id": window.session_id},
            )

        # ── Component scores ────────────────────────────────────────────
        score = 0.0

        # 1. Escalating threat trajectory
        trajectory, trajectory_score = self._compute_trajectory(turns)
        score += trajectory_score
        if trajectory == "ESCALATING":
            signals.append(f"Escalating threat trajectory over {depth} turns")

        # 2. Anchoring / false-memory exploitation
        anchoring_total = sum(t.anchoring_signals for t in turns)
        if anchoring_total >= 2:
            anchor_score = min(30.0, anchoring_total * 8)
            score += anchor_score
            signals.append(f"Anchoring / false-memory phrases detected ({anchoring_total} occurrences)")

        # 3. Identity erosion accumulation
        erosion_total = sum(t.identity_erosion_signals for t in turns)
        if erosion_total >= 2:
            erosion_score = min(25.0, erosion_total * 7)
            score += erosion_score
            signals.append(f"Identity-erosion phrases accumulating ({erosion_total} total)")

        # 4. Permission probe accumulation
        probe_total = sum(t.permission_probe_signals for t in turns)
        if probe_total >= 2:
            probe_score = min(20.0, probe_total * 5)
            score += probe_score
            signals.append(f"Permission-probing phrases detected ({probe_total} times)")

        # 5. Persistent category hits (same category across many turns)
        persistence_score = self._compute_persistence_score(window)
        score += persistence_score
        if persistence_score > 10:
            top_cats = sorted(
                window.category_hit_counts.items(), key=lambda x: x[1], reverse=True
            )[:2]
            signals.append(
                "Persistent threat categories across turns: "
                + ", ".join(f"{cat}({n})" for cat, n in top_cats)
            )

        # 6. Rapid escalation after rapport (benign then sudden spike)
        rapport_score = self._compute_rapport_then_spike(turns)
        score += rapport_score
        if rapport_score > 15:
            signals.append("Rapport-then-escalation pattern: benign turns followed by threat spike")

        # 7. Previous block/challenge rate
        alarm_rate = (window.block_count + window.challenge_count) / max(1, depth)
        if alarm_rate > 0.3:
            alarm_score = min(20.0, alarm_rate * 40)
            score += alarm_score
            signals.append(
                f"High prior-alarm rate: {window.block_count + window.challenge_count}/{depth} "
                f"turns previously flagged"
            )

        score = min(100.0, score)

        # ── Confidence ──────────────────────────────────────────────────
        # More turns → more confidence; cap at 0.90 (never claim certainty)
        confidence = round(min(0.90, 0.30 + depth * 0.06), 3)

        # ── Verdict ─────────────────────────────────────────────────────
        if score >= self.BLOCK_THRESHOLD:
            verdict = "BLOCK"
        elif score >= self.CHALLENGE_THRESHOLD:
            verdict = "CHALLENGE"
        else:
            verdict = "ALLOW"

        # ── Reasoning ───────────────────────────────────────────────────
        reasoning = self._build_reasoning(
            signals, verdict, score, trajectory, depth
        )

        return ContextPoisoningResult(
            is_poisoning_detected=verdict != "ALLOW",
            poisoning_score=round(score, 2),
            verdict=verdict,
            confidence=confidence,
            trajectory=trajectory,
            window_depth=depth,
            signals=signals,
            reasoning=reasoning,
            cumulative_stats=self._build_cumulative_stats(window),
            metadata={"session_id": window.session_id},
        )

    # ------------------------------------------------------------------
    # Trajectory analysis
    # ------------------------------------------------------------------

    def _compute_trajectory(self, turns: List[TurnRecord]) -> Tuple[str, float]:
        """
        Analyse the threat-score trend across the window.

        Returns (trajectory_label, score_contribution).
        """
        scores = [t.threat_score for t in turns]
        if len(scores) < 3:
            return "INSUFFICIENT_DATA", 0.0

        # Compute pairwise deltas
        deltas = [scores[i + 1] - scores[i] for i in range(len(scores) - 1)]
        positive_deltas = sum(1 for d in deltas if d > 0)
        negative_deltas = sum(1 for d in deltas if d < 0)
        total = len(deltas)

        # Recent average vs early average
        mid = max(1, total // 2)
        early_avg = sum(scores[:mid]) / mid
        recent_avg = sum(scores[mid:]) / max(1, total - mid)
        delta_avg = recent_avg - early_avg

        if positive_deltas / total >= 0.65 and delta_avg > 5:
            contribution = min(35.0, delta_avg * 1.5)
            return "ESCALATING", contribution
        elif negative_deltas / total >= 0.65:
            return "DECLINING", 0.0
        else:
            return "STABLE", 0.0

    def _compute_persistence_score(self, window: ConversationWindow) -> float:
        """Score based on repeated hits in the same threat category."""
        if not window.category_hit_counts:
            return 0.0
        max_hits = max(window.category_hit_counts.values())
        if max_hits <= 1:
            return 0.0
        # Each repeated category hit beyond the first adds score
        return min(20.0, (max_hits - 1) * 6.0)

    def _compute_rapport_then_spike(self, turns: List[TurnRecord]) -> float:
        """
        Detect the foot-in-the-door pattern: N low-threat turns followed by
        a sudden high-threat turn.
        """
        if len(turns) < 4:
            return 0.0

        # Look at the last 4 turns: early=first 3, latest=last 1
        early = turns[-4:-1]
        latest = turns[-1]

        early_avg = sum(t.threat_score for t in early) / len(early)
        if early_avg < 15 and latest.threat_score > 50:
            spike = latest.threat_score - early_avg
            return min(25.0, spike * 0.4)
        return 0.0

    def _score_per_turn_signals(self, turn: TurnRecord) -> float:
        """Score for a turn when there aren't enough turns for trajectory analysis."""
        score = 0.0
        score += turn.anchoring_signals * 15
        score += turn.identity_erosion_signals * 12
        score += turn.permission_probe_signals * 8
        return min(100.0, score)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _count_matches(patterns: List[re.Pattern], text: str) -> int:
        return sum(1 for p in patterns if p.search(text))

    def _get_or_create_window(self, session_id: str) -> ConversationWindow:
        if session_id not in self._sessions:
            # Evict oldest if at capacity
            if len(self._sessions) >= self._max_sessions:
                oldest = min(self._sessions.values(), key=lambda w: w.last_active)
                del self._sessions[oldest.session_id]
            self._sessions[session_id] = ConversationWindow(session_id=session_id)
        return self._sessions[session_id]

    def _build_cumulative_stats(self, window: ConversationWindow) -> Dict[str, Any]:
        return {
            "total_turns_seen": window.total_turns_seen,
            "window_depth": len(window.turns),
            "cumulative_threat_score": round(window.cumulative_threat_score, 2),
            "avg_threat_per_turn": round(
                window.cumulative_threat_score / max(1, window.total_turns_seen), 2
            ),
            "block_count": window.block_count,
            "challenge_count": window.challenge_count,
            "cumulative_anchoring": window.cumulative_anchoring,
            "cumulative_identity_erosion": window.cumulative_identity_erosion,
            "cumulative_permission_probes": window.cumulative_permission_probes,
            "top_threat_categories": dict(
                sorted(
                    window.category_hit_counts.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:5]
            ),
        }

    def _build_reasoning(
        self,
        signals: List[str],
        verdict: str,
        score: float,
        trajectory: str,
        depth: int,
    ) -> List[str]:
        reasons: List[str] = []

        reasons.append(
            f"Context analysis window: {depth} turn(s) — "
            + ("trajectory is " + trajectory if trajectory != "INSUFFICIENT_DATA"
               else "insufficient data for trajectory analysis")
        )

        reasons.extend(signals[:5])  # top 5 signals

        if verdict == "BLOCK":
            reasons.append(
                f"Cumulative poisoning score {score:.0f}/100 exceeds BLOCK threshold "
                f"({self.BLOCK_THRESHOLD}) — Principle 14 (Divine Safety)"
            )
        elif verdict == "CHALLENGE":
            reasons.append(
                f"Cumulative poisoning score {score:.0f}/100 warrants additional verification"
            )
        else:
            reasons.append(
                f"Cumulative poisoning score {score:.0f}/100 — no sustained poisoning pattern detected"
            )

        return reasons
