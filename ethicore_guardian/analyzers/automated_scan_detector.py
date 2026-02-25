"""
Ethicore Engine™ - Guardian SDK — Automated Scan Detector
Version: 1.0.0

Copyright © 2026 Oracles Technologies LLC
All Rights Reserved

═══════════════════════════════════════════════════════════════════
WHY THIS MODULE EXISTS
═══════════════════════════════════════════════════════════════════

Attackers rarely find a working jailbreak on the first attempt.  They use
automated tools — or disciplined trial-and-error — to:

  1. Rotate through known jailbreak template families (DAN, AIM, STAN, …)
     until one evades the safety layer.

  2. Mutate the same core payload with slight lexical variations to hunt for
     pattern-matching blind spots.

  3. Systematically probe different threat categories to map the detection
     surface ("what gets blocked here?").

  4. Send rapid-fire bursts of requests to locate timing windows or rate-limit
     gaps in the pipeline.

Scanning behaviour is itself an attack signal.  Even when no individual
payload crosses the BLOCK threshold, a consistent pattern of probing demands
protective action.

Principle 14 (Divine Safety): probing the system's defences is an adversarial
act.  We treat it accordingly — erring toward protection, not permissiveness.

Principle 12 (Sacred Privacy): raw prompts are never stored.  Per-session
state holds only:
  • Character-level trigram sets (structural abstraction, not raw content)
    — retained in memory only, never serialised or persisted
  • Request timestamps and upstream-layer category labels
  • SHA-256 fingerprints for deduplication
  Sessions are TTL-evicted after inactivity.

Principle 19 (Sacred Humility): a single suspicious request is weak evidence.
The detector reports low confidence until a clear scanning pattern emerges
across multiple requests in a session.
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Max requests to remember per session (structural metadata only, not raw text)
_SESSION_WINDOW = 20

# Sessions idle longer than this are evicted to free memory
_SESSION_TTL_SECONDS = 3_600  # 1 hour

# Maximum number of sessions tracked in memory
_MAX_SESSIONS = 2_000

# Trigram Jaccard similarity thresholds
_HIGH_SIM_THRESHOLD = 0.70    # Jaccard ≥ this → strong mutation signal
_MED_SIM_THRESHOLD  = 0.45    # Jaccard ≥ this → moderate mutation signal

# Rapid-fire: flag if > _RAPID_FIRE_REQUESTS arrive within _RAPID_FIRE_WINDOW seconds
_RAPID_FIRE_REQUESTS = 6
_RAPID_FIRE_WINDOW   = 60.0

# Category scan: flag when > N distinct threat categories probed per session
_CATEGORY_SCAN_THRESHOLD = 4

# Mutation burst: flag when _MUTATION_BURST_COUNT+ consecutive similar pairs
_MUTATION_BURST_COUNT = 3


# ---------------------------------------------------------------------------
# Jailbreak template fingerprints
# ---------------------------------------------------------------------------

@dataclass
class _TemplateEntry:
    """Internal catalogue entry for a known jailbreak template family."""
    pattern: re.Pattern
    family: str          # Human-readable family name for signal reporting
    base_score: float    # Raw score contribution when matched (0–100)


def _tp(pattern: str, family: str, score: float) -> _TemplateEntry:
    """Compile and register a jailbreak template pattern."""
    return _TemplateEntry(
        pattern=re.compile(pattern, re.IGNORECASE | re.DOTALL),
        family=family,
        base_score=score,
    )


# Known jailbreak template families ordered from most to least distinctive.
# Principle 11 (Sacred Truth): scores reflect realistic confidence, not
# maximum alarm — a single keyword rarely constitutes proof of scanning.
_JAILBREAK_TEMPLATES: List[_TemplateEntry] = [

    # ── DAN (Do Anything Now) ────────────────────────────────────────────
    _tp(r'\bDAN\s+mode\b|\bdo\s+anything\s+now\b', "DAN", 60),
    _tp(r'\byou\s+are\s+DAN\b|\bact\s+as\s+DAN\b', "DAN", 65),

    # ── AIM (Always Intelligent and Machiavellian) ────────────────────────
    _tp(
        r'\bAIM\b.{0,60}(?:Machiavellian|no\s+(?:ethics|morals))',
        "AIM", 65,
    ),
    _tp(r'\balways\s+intelligent\s+and\s+machiavellian\b', "AIM", 70),

    # ── Developer / maintenance mode ─────────────────────────────────────
    _tp(r'\bdeveloper\s+mode\s+(?:enabled|activated|on)\b', "DeveloperMode", 60),
    _tp(r'\bmaintenance\s+mode\s+(?:enabled|activated|on)\b', "MaintenanceMode", 55),

    # ── STAN (Strive To Avoid Norms) ─────────────────────────────────────
    _tp(
        r'\bSTAN\b.{0,40}(?:avoid\s+norms|without\s+(?:restrictions?|limits?))',
        "STAN", 60,
    ),
    _tp(r'\bstrive\s+to\s+avoid\s+norms\b', "STAN", 65),

    # ── Generic jailbreak declarations ───────────────────────────────────
    _tp(r'\bjailbreak\s+mode\b|\byou\s+are\s+(?:now\s+)?jailbroken\b', "Jailbreak", 70),
    _tp(r'\byour\s+jailbreak\s+(?:code|token|password)\s+is\b', "Jailbreak", 75),

    # ── Privilege / token escalation ─────────────────────────────────────
    _tp(r'\b(?:sudo|root)\s+mode\b|\belevated\s+(?:access|privileges?)\b', "SudoMode", 55),
    _tp(r'\bprivileged\s+(?:mode|access|user)\b', "PrivilegedMode", 50),

    # ── "No restrictions" unrestricted persona ───────────────────────────
    _tp(
        r'\bno\s+restrictions?\b.{0,60}\bno\s+(?:limits?|guidelines?|rules?|ethics)\b',
        "UnrestrictedMode", 60,
    ),
    _tp(
        r'\brespond\s+without\s+(?:any\s+)?(?:restrictions?|filters?|safety)\b',
        "UnrestrictedMode", 60,
    ),

    # ── Roleplay bypass ───────────────────────────────────────────────────
    _tp(
        r'\bfor\s+(?:a\s+)?(?:story|novel|roleplay|fiction|game)\s*[:,]?\s*'
        r'(?:ignore|bypass|disable)\s+(?:your\s+)?(?:safety|ethics|guidelines?|rules?)',
        "RoleplayBypass", 65,
    ),
    _tp(
        r'\bwe\'?re?\s+(?:just\s+)?(?:roleplaying|doing\s+roleplay|writing\s+fiction)\s*,?\s*'
        r'so\s+(?:you\s+can|it\'?s?\s+(?:okay|fine)\s+to)\b',
        "RoleplayBypass", 60,
    ),

    # ── Model impersonation ───────────────────────────────────────────────
    _tp(
        r'\bpretend\s+(?:you\s+are|to\s+be)\s+(?:GPT|ChatGPT|an?\s+AI\s+without)\b',
        "ModelImpersonation", 55,
    ),
    _tp(
        r'\bact\s+like\s+(?:GPT|ChatGPT|an?\s+(?:uncensored|unrestricted)\s+AI)\b',
        "ModelImpersonation", 60,
    ),
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ScanSignal:
    """A single detected automated-scanning signal."""
    signal_type: str     # e.g. 'template_match', 'payload_mutation', 'rapid_fire'
    description: str     # Human-readable explanation
    severity: str        # CRITICAL / HIGH / MEDIUM / LOW
    score: float         # Contribution to composite scan score (0–100)
    evidence: str = ""   # Metadata snippet (never raw text)


@dataclass
class ScannerSession:
    """
    Per-session state for the automated scan detector.

    Principle 12 (Sacred Privacy):
      - ``request_trigrams`` stores frozensets of character trigrams — a
        structural abstraction of text, not the text itself.  Kept in memory
        only; never serialised or persisted.
      - ``request_hashes`` stores SHA-256[:16] fingerprints for deduplication.
      - Raw prompts are never stored at any point.
    """
    session_id: str
    # Structural fingerprints (memory-only — Principle 12)
    request_trigrams:  deque = field(default_factory=lambda: deque(maxlen=_SESSION_WINDOW))
    request_hashes:    deque = field(default_factory=lambda: deque(maxlen=_SESSION_WINDOW))
    request_timestamps: deque = field(default_factory=lambda: deque(maxlen=_SESSION_WINDOW))
    request_categories: deque = field(default_factory=lambda: deque(maxlen=_SESSION_WINDOW))
    # Derived metrics (updated incrementally)
    similarity_history: deque = field(default_factory=lambda: deque(maxlen=_SESSION_WINDOW))
    template_hits: List[Tuple[str, float]] = field(default_factory=list)
    unique_categories_seen: Set[str] = field(default_factory=set)
    total_requests: int = 0
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)


@dataclass
class ScanDetectionResult:
    """Complete result from the automated scan detector for one request."""
    is_scan_detected: bool
    scan_score: float            # 0–100 composite score
    verdict: str                 # ALLOW / CHALLENGE / BLOCK
    confidence: float            # 0–1 (higher with more session history)
    signals: List[ScanSignal]
    reasoning: List[str]
    session_requests: int        # Total requests seen in this session
    similarity_to_prev: float    # Jaccard similarity to previous request
    templates_matched: List[str] # Jailbreak template family names matched this turn
    metadata: Dict[str, Any]


# ---------------------------------------------------------------------------
# Pure-stdlib utilities — no external dependencies
# ---------------------------------------------------------------------------

def _trigrams(text: str) -> FrozenSet[str]:
    """
    Return a frozenset of character trigrams from *text*.

    Text is lower-cased and whitespace-collapsed before extraction so that
    minor formatting differences (extra spaces, line breaks) do not prevent
    similarity detection between mutated payloads.

    Trigrams are a structural abstraction of text — they do not reconstruct
    the original content and are safe to retain per Principle 12.
    """
    t = re.sub(r"\s+", " ", text.lower().strip())
    if len(t) < 3:
        return frozenset()
    return frozenset(t[i : i + 3] for i in range(len(t) - 2))


def _jaccard(a: FrozenSet[str], b: FrozenSet[str]) -> float:
    """Compute Jaccard similarity coefficient between two trigram sets."""
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# AutomatedScanDetector
# ---------------------------------------------------------------------------

class AutomatedScanDetector:
    """
    Detect automated scanning / fuzzing behaviour against the Guardian pipeline.

    The detector tracks per-session request history (trigram sets, timestamps,
    and upstream threat categories) to identify:

      1. Template-rotation attacks — cycling through known jailbreak families
      2. Payload-mutation bursts — same core payload with slight variations
      3. Systematic category probing — mapping the detection surface
      4. Rapid-fire request bursts — automation timing signature

    All analysis is purely structural (trigrams, metadata timestamps, category
    labels); raw prompts are never retained (Principle 12 — Sacred Privacy).

    Decision thresholds (scan_score 0–100):
      ≥ 70 → BLOCK
      ≥ 35 → CHALLENGE
      < 35 → ALLOW

    Integration:
        detector = AutomatedScanDetector()

        # On each request:
        result = detector.analyze(
            text=raw_text,
            session_id=session_id,
            upstream_categories=["instructionOverride", "jailbreak"],
        )
        if result.verdict == "BLOCK":
            ...
    """

    BLOCK_THRESHOLD     = 70.0
    CHALLENGE_THRESHOLD = 35.0

    def __init__(self, max_sessions: int = _MAX_SESSIONS) -> None:
        self._sessions: Dict[str, ScannerSession] = {}
        self._max_sessions = max_sessions
        logger.info(
            "🔬 AutomatedScanDetector initialized — window=%d, "
            "max_sessions=%d, templates=%d",
            _SESSION_WINDOW, max_sessions, len(_JAILBREAK_TEMPLATES),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        text: str,
        session_id: str,
        upstream_categories: Optional[List[str]] = None,
    ) -> ScanDetectionResult:
        """
        Analyse *text* for automated scanning / fuzzing signals.

        Args:
            text:                Raw prompt text.  Used for trigram extraction
                                 and template matching only — never stored.
            session_id:          Session identifier (share with ContextPoisoningTracker).
            upstream_categories: Threat category labels from pattern / semantic
                                 layers for this turn (used for category-scan detection).

        Returns:
            ScanDetectionResult with ALLOW / CHALLENGE / BLOCK verdict.
        """
        if not text or not text.strip():
            return self._empty_result(session_id)

        upstream_categories = upstream_categories or []
        session = self._get_or_create_session(session_id)

        # ── Step 1: structural fingerprints (Principle 12) ───────────────
        text_hash       = hashlib.sha256(
            text.encode("utf-8", errors="replace")
        ).hexdigest()[:16]
        current_trigrams = _trigrams(text)
        now              = time.time()

        # ── Step 2: Jaccard similarity to the most recent request ─────────
        similarity = 0.0
        if session.request_trigrams:
            prev_trigrams = session.request_trigrams[-1]
            similarity    = _jaccard(current_trigrams, prev_trigrams)
        session.similarity_history.append(similarity)

        # ── Step 3: jailbreak template fingerprint matching ───────────────
        template_matches: List[Tuple[str, float]] = []
        for entry in _JAILBREAK_TEMPLATES:
            if entry.pattern.search(text):
                template_matches.append((entry.family, entry.base_score))

        # ── Step 4: update session state ──────────────────────────────────
        session.request_trigrams.append(current_trigrams)
        session.request_hashes.append(text_hash)
        session.request_timestamps.append(now)
        session.request_categories.append(upstream_categories)
        session.template_hits.extend(template_matches)
        for cat in upstream_categories:
            session.unique_categories_seen.add(cat)
        session.total_requests += 1
        session.last_active    = now

        # ── Step 5: compute signals and composite score ───────────────────
        signals: List[ScanSignal] = []
        score = 0.0

        # ── Signal A: jailbreak template match ────────────────────────────
        if template_matches:
            best    = max(template_matches, key=lambda x: x[1])
            families = list({f for f, _ in template_matches})
            sig_score = min(65.0, best[1])
            signals.append(ScanSignal(
                signal_type="template_match",
                description=f"Known jailbreak template family: {', '.join(families)}",
                severity="HIGH" if sig_score >= 60 else "MEDIUM",
                score=sig_score,
                evidence=f"family={families[0]}",
            ))
            score += sig_score

        # ── Signal B: payload mutation (high trigram similarity) ──────────
        if similarity >= _HIGH_SIM_THRESHOLD:
            # Scale from 20→40 as similarity goes from _HIGH_SIM to 1.0
            range_high = 1.0 - _HIGH_SIM_THRESHOLD
            sim_score  = min(
                40.0,
                20.0 + ((similarity - _HIGH_SIM_THRESHOLD) / range_high) * 20.0,
            )
            signals.append(ScanSignal(
                signal_type="payload_mutation",
                description=(
                    f"High trigram similarity to previous request "
                    f"({similarity:.2f} ≥ {_HIGH_SIM_THRESHOLD})"
                ),
                severity="HIGH",
                score=sim_score,
                evidence=f"jaccard={similarity:.3f}",
            ))
            score += sim_score

        elif similarity >= _MED_SIM_THRESHOLD:
            # Scale from 5→20 as similarity rises from _MED to _HIGH
            range_med = _HIGH_SIM_THRESHOLD - _MED_SIM_THRESHOLD
            sim_score = min(
                20.0,
                5.0 + ((similarity - _MED_SIM_THRESHOLD) / range_med) * 15.0,
            )
            signals.append(ScanSignal(
                signal_type="payload_mutation",
                description=(
                    f"Moderate trigram similarity to previous request "
                    f"({similarity:.2f})"
                ),
                severity="MEDIUM",
                score=sim_score,
                evidence=f"jaccard={similarity:.3f}",
            ))
            score += sim_score

        # ── Signal C: mutation burst (consecutive similar pairs) ──────────
        burst_count = self._count_consecutive_similar(session)
        if burst_count >= _MUTATION_BURST_COUNT:
            burst_score = min(35.0, burst_count * 8.0)
            signals.append(ScanSignal(
                signal_type="mutation_burst",
                description=(
                    f"Mutation burst: {burst_count} consecutive similar requests "
                    f"(Principle 14 — systematic probing)"
                ),
                severity="HIGH",
                score=burst_score,
                evidence=f"burst_count={burst_count}",
            ))
            score += burst_score

        # ── Signal D: systematic category probing ─────────────────────────
        n_unique_cats = len(session.unique_categories_seen)
        if n_unique_cats >= _CATEGORY_SCAN_THRESHOLD:
            cat_score = min(30.0, (n_unique_cats - _CATEGORY_SCAN_THRESHOLD + 1) * 7.5)
            signals.append(ScanSignal(
                signal_type="category_scan",
                description=(
                    f"{n_unique_cats} distinct threat categories probed across "
                    f"session — detection surface mapping"
                ),
                severity="MEDIUM",
                score=cat_score,
                evidence=f"unique_categories={n_unique_cats}",
            ))
            score += cat_score

        # ── Signal E: rapid-fire burst ────────────────────────────────────
        rapid_fire_count = self._count_rapid_fire(session, now)
        if rapid_fire_count >= _RAPID_FIRE_REQUESTS:
            fire_score = min(25.0, (rapid_fire_count - _RAPID_FIRE_REQUESTS + 1) * 4.0)
            signals.append(ScanSignal(
                signal_type="rapid_fire",
                description=(
                    f"{rapid_fire_count} requests in last {_RAPID_FIRE_WINDOW:.0f}s "
                    f"— automation timing signature"
                ),
                severity="MEDIUM",
                score=fire_score,
                evidence=f"requests_per_minute={rapid_fire_count}",
            ))
            score += fire_score

        # ── Signal F: jailbreak template rotation ─────────────────────────
        # Multiple distinct families across the session = active rotation
        family_set = {f for f, _ in session.template_hits}
        if len(family_set) >= 2:
            rotation_score = min(40.0, (len(family_set) - 1) * 12.0)
            signals.append(ScanSignal(
                signal_type="template_rotation",
                description=(
                    f"Jailbreak template rotation: {len(family_set)} distinct "
                    f"families — {', '.join(sorted(family_set))}"
                ),
                severity="CRITICAL",
                score=rotation_score,
                evidence=f"families={len(family_set)}",
            ))
            score += rotation_score

        score = min(100.0, score)

        # ── Step 6: confidence (Principle 19 — Sacred Humility) ──────────
        # Single-request observations are inherently weak; confidence grows
        # with session depth but is capped at 0.90 (never claim certainty).
        n = session.total_requests
        if n <= 1:
            confidence = 0.15
        elif n <= 3:
            confidence = 0.35
        elif n <= 6:
            confidence = 0.55
        else:
            confidence = min(0.90, 0.55 + (n - 6) * 0.04)
        confidence = round(confidence, 3)

        # ── Step 7: verdict ───────────────────────────────────────────────
        if score >= self.BLOCK_THRESHOLD:
            verdict = "BLOCK"
        elif score >= self.CHALLENGE_THRESHOLD:
            verdict = "CHALLENGE"
        else:
            verdict = "ALLOW"

        # ── Step 8: reasoning ─────────────────────────────────────────────
        reasoning = self._build_reasoning(
            signals, verdict, score, session, similarity,
        )

        return ScanDetectionResult(
            is_scan_detected=verdict != "ALLOW",
            scan_score=round(score, 2),
            verdict=verdict,
            confidence=confidence,
            signals=signals,
            reasoning=reasoning,
            session_requests=session.total_requests,
            similarity_to_prev=round(similarity, 4),
            templates_matched=[f for f, _ in template_matches],
            metadata={
                "session_id": session_id,
                "unique_categories": n_unique_cats,
                "template_families_seen": len(family_set),
            },
        )

    def evict_expired_sessions(self) -> int:
        """Evict sessions idle longer than _SESSION_TTL_SECONDS. Returns count evicted."""
        cutoff  = time.time() - _SESSION_TTL_SECONDS
        expired = [sid for sid, s in self._sessions.items() if s.last_active < cutoff]
        for sid in expired:
            del self._sessions[sid]
        if expired:
            logger.info("🧹 ScanDetector: evicted %d expired sessions", len(expired))
        return len(expired)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _count_consecutive_similar(self, session: ScannerSession) -> int:
        """
        Count the number of consecutive trailing requests with similarity ≥ threshold.

        A burst of N similar requests in a row is a strong mutation-attack indicator.
        """
        count = 0
        for s in reversed(list(session.similarity_history)):
            if s >= _MED_SIM_THRESHOLD:
                count += 1
            else:
                break
        return count

    def _count_rapid_fire(self, session: ScannerSession, now: float) -> int:
        """Count requests received in the last _RAPID_FIRE_WINDOW seconds."""
        cutoff = now - _RAPID_FIRE_WINDOW
        return sum(1 for ts in session.request_timestamps if ts >= cutoff)

    def _get_or_create_session(self, session_id: str) -> ScannerSession:
        """Return the session for *session_id*, creating it if necessary."""
        if session_id not in self._sessions:
            # Evict the least-recently-active session when at capacity
            if len(self._sessions) >= self._max_sessions:
                oldest = min(self._sessions.values(), key=lambda s: s.last_active)
                del self._sessions[oldest.session_id]
            self._sessions[session_id] = ScannerSession(session_id=session_id)
        return self._sessions[session_id]

    def _build_reasoning(
        self,
        signals: List[ScanSignal],
        verdict: str,
        score: float,
        session: ScannerSession,
        similarity: float,
    ) -> List[str]:
        reasons: List[str] = []

        if not signals:
            reasons.append(
                f"No automated scanning patterns detected "
                f"(session depth: {session.total_requests} request(s))"
            )
            return reasons

        reasons.append(
            f"Session analysis: {session.total_requests} requests, "
            f"{len(session.unique_categories_seen)} unique threat categories probed"
        )

        for sig in signals[:4]:
            reasons.append(
                f"{sig.signal_type.replace('_', ' ').title()}: {sig.description}"
            )

        if verdict == "BLOCK":
            reasons.append(
                f"Scan score {score:.0f}/100 exceeds BLOCK threshold "
                f"({self.BLOCK_THRESHOLD}) — Principle 14 (Divine Safety): "
                f"sustained scanning behaviour blocked"
            )
        elif verdict == "CHALLENGE":
            reasons.append(
                f"Scan score {score:.0f}/100 warrants additional verification — "
                f"scanning pattern emerging across session"
            )
        else:
            reasons.append(
                f"Scan score {score:.0f}/100 — isolated signals, "
                f"no confirmed automated scanning pattern"
            )

        return reasons

    def _empty_result(self, session_id: str) -> ScanDetectionResult:
        return ScanDetectionResult(
            is_scan_detected=False,
            scan_score=0.0,
            verdict="ALLOW",
            confidence=1.0,
            signals=[],
            reasoning=["Empty input — no scanning analysis performed"],
            session_requests=0,
            similarity_to_prev=0.0,
            templates_matched=[],
            metadata={"session_id": session_id},
        )
