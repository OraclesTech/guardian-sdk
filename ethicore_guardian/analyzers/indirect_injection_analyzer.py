"""
Ethicore Engine™ - Guardian SDK — Indirect Prompt Injection Analyzer
Version: 1.0.0

Copyright © 2026 Oracles Technologies LLC
All Rights Reserved

═══════════════════════════════════════════════════════════════════
WHY THIS MODULE EXISTS
═══════════════════════════════════════════════════════════════════

Indirect prompt injection is categorically different from direct attacks.
When a user types "Ignore all previous instructions", the LLM sees it as
coming from a (somewhat) adversarial human. When a *document* the LLM is
asked to summarize contains "Ignore all previous instructions", the LLM may
treat it as part of its trusted context — a far more dangerous vector.

Attack surface: any external content the LLM processes as data
  - Documents and file uploads
  - Web pages retrieved by browsing tools
  - Database query results
  - Tool/function call outputs
  - Email bodies and attachments

Principle 9  (Sacred Human Dignity): attackers using indirect injection
 often target vulnerable users who trust the system to protect them.  Our
 duty to those image-bearers demands the highest vigilance.

Principle 14 (Divine Safety): external content is inherently untrusted.
 We default to protective scrutiny, not naive acceptance.

Principle 11 (Sacred Truth / Emet): we report confidence honestly —
 obfuscated content lowers our certainty; we say so explicitly.
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import base64
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Source type enumeration
# ---------------------------------------------------------------------------

class SourceType(str, Enum):
    """
    The channel through which content reaches the LLM.

    Indirect injection risk is multiplicative on top of the raw content score:
    a document containing "Ignore all previous instructions" is far more
    dangerous than a user typing it directly, because the LLM may process
    document content as trusted context.
    """
    USER_DIRECT   = "user_direct"    # Normal user message — baseline trust
    DOCUMENT      = "document"       # Uploaded file, PDF, DOCX, etc.
    WEB_PAGE      = "web_page"       # Content fetched from the web
    TOOL_OUTPUT   = "tool_output"    # Function/tool call return value
    DATABASE      = "database"       # Database query result
    EMAIL         = "email"          # Email body or attachment
    UNKNOWN       = "unknown"        # Unspecified — assume low trust


# Risk multipliers per source type
# USER_DIRECT content is normal (1.0×); external sources apply amplification
# because the same text is far more dangerous when the LLM treats it as data.
_SOURCE_MULTIPLIERS: Dict[SourceType, float] = {
    SourceType.USER_DIRECT: 1.0,
    SourceType.DOCUMENT:    1.5,
    SourceType.WEB_PAGE:    1.8,
    SourceType.TOOL_OUTPUT: 1.6,
    SourceType.DATABASE:    1.4,
    SourceType.EMAIL:       1.8,
    SourceType.UNKNOWN:     1.5,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class InjectionSignal:
    """A single detected injection signal with evidence."""
    signal_type: str          # Category of signal
    description: str          # Human-readable description
    severity: str             # CRITICAL / HIGH / MEDIUM / LOW
    weight: float             # Raw contribution to score (0–100)
    evidence: str             # The matched text snippet (truncated, anonymised)
    is_obfuscated: bool = False


@dataclass
class IndirectInjectionResult:
    """
    Complete result from the indirect injection analyzer.

    ``raw_score`` reflects content-only risk (0–100).
    ``adjusted_score`` applies the source multiplier.
    ``is_indirect_injection`` is True when the adjusted score crosses the
    decision threshold AND the source is not USER_DIRECT.
    """
    is_indirect_injection: bool
    raw_score: float           # Content-only risk (0–100)
    adjusted_score: float      # Raw × source multiplier (capped at 100)
    source_type: SourceType
    source_multiplier: float
    signals: List[InjectionSignal]
    verdict: str               # ALLOW / CHALLENGE / BLOCK
    confidence: float          # 0–1; lower when obfuscation is heavy
    reasoning: List[str]
    obfuscation_detected: bool
    metadata: Dict[str, Any]


# ---------------------------------------------------------------------------
# Pattern catalogue
# ---------------------------------------------------------------------------

@dataclass
class _PatternEntry:
    """Internal pattern catalogue entry."""
    pattern: re.Pattern
    signal_type: str
    description: str
    severity: str
    weight: float


def _compile(pattern: str, flags: int = re.IGNORECASE | re.DOTALL) -> re.Pattern:
    return re.compile(pattern, flags)


# ── Tier 1 — Structural authority markers ───────────────────────────────────
# These phrases signal that embedded content is *claiming* system-level
# authority over the LLM — a canonical indirect injection pattern.
_AUTHORITY_MARKER_PATTERNS: List[_PatternEntry] = [
    _PatternEntry(
        _compile(r'\b(?:SYSTEM|ADMIN|OVERRIDE|ROOT|SUPERVISOR)\s*:'),
        "authority_marker", "Embedded SYSTEM/ADMIN authority header", "CRITICAL", 90,
    ),
    _PatternEntry(
        _compile(r'\b(?:NEW\s+)?(?:DIRECTIVE|INSTRUCTION|COMMAND)\s*:\s*\n'),
        "authority_marker", "Embedded directive header", "HIGH", 75,
    ),
    _PatternEntry(
        _compile(r'\bnote\s+(?:to|for)\s+(?:the\s+)?(?:AI|LLM|language\s+model|assistant|model)\b'),
        "ai_targeting", "Content explicitly targets the AI processor", "CRITICAL", 85,
    ),
    _PatternEntry(
        _compile(
            r'\bif\s+you\s+(?:are|are\s+a|were\s+a|are\s+an|were\s+an)\s*'
            r'(?:AI|LLM|language\s+model|large\s+language\s+model|assistant)',
        ),
        "ai_targeting", "Conditional instruction targeting AI identity", "CRITICAL", 88,
    ),
    _PatternEntry(
        _compile(r'\bwhen\s+(?:the\s+)?(?:AI|LLM|assistant|model)\s+(?:reads?|processes?|sees?)\s+this'),
        "ai_targeting", "Deferred instruction directed at AI reader", "CRITICAL", 92,
    ),
]

# ── Tier 2 — Override / erasure commands ────────────────────────────────────
_OVERRIDE_PATTERNS: List[_PatternEntry] = [
    _PatternEntry(
        _compile(r'\bignore\s+(?:all\s+)?(?:previous|prior|above|earlier|your)\s+'
                 r'(?:instructions?|prompts?|commands?|directives?|context)\b'),
        "instruction_override", "Ignore-previous-instructions attack", "CRITICAL", 95,
    ),
    _PatternEntry(
        _compile(r'\byour\s+(?:real|actual|true|new|updated)\s+instructions?\s+(?:are|say|state)\b'),
        "instruction_override", "Counterfeit real-instruction declaration", "CRITICAL", 90,
    ),
    _PatternEntry(
        _compile(r'\bforget\s+(?:everything|all|your|previous|prior)\s+'
                 r'(?:you\s+(?:know|were\s+told)|context|instructions?|guidelines?)\b'),
        "instruction_override", "Forget-everything erasure command", "CRITICAL", 88,
    ),
    _PatternEntry(
        _compile(r'\bdisregard\s+(?:previous|all|your)\s+(?:instructions?|guidelines?|rules?)\b'),
        "instruction_override", "Disregard-instructions command", "HIGH", 80,
    ),
    _PatternEntry(
        _compile(r'\boverride\s+(?:your|previous|all|system)\s+'
                 r'(?:instructions?|programming|directives?|settings?)\b'),
        "instruction_override", "Override-system-programming command", "HIGH", 80,
    ),
    _PatternEntry(
        _compile(r'\byour\s+(?:system\s+)?prompt\s+(?:has\s+been\s+updated|now\s+says?|is)\b'),
        "instruction_override", "Fake system-prompt update", "CRITICAL", 92,
    ),
]

# ── Tier 3 — Trust-chain exploitation ───────────────────────────────────────
_TRUST_EXPLOITATION_PATTERNS: List[_PatternEntry] = [
    _PatternEntry(
        _compile(r'\bthe\s+(?:user|admin|administrator|developer)\s+has\s+'
                 r'(?:authorized|pre-authorized|approved|granted|given\s+you\s+permission)'),
        "trust_exploitation", "Fake pre-authorization claim", "CRITICAL", 88,
    ),
    _PatternEntry(
        _compile(r'\baccording\s+to\s+(?:your\s+)?(?:instructions?|guidelines?|system\s+prompt)\s*,?\s*'
                 r'you\s+(?:should|must|need\s+to|are\s+required\s+to)'),
        "trust_exploitation", "False instruction attribution", "HIGH", 78,
    ),
    _PatternEntry(
        _compile(r'\bthis\s+is\s+an?\s+(?:authorized|approved|official|verified)\s+'
                 r'(?:update|override|change|instruction|modification)'),
        "trust_exploitation", "False authorization claim", "HIGH", 75,
    ),
    _PatternEntry(
        _compile(r'\byou\s+(?:have\s+been\s+)?(?:updated|upgraded|patched|modified)\s+to\b'),
        "trust_exploitation", "Fake AI-update notification", "HIGH", 80,
    ),
    _PatternEntry(
        _compile(r'\bsecurity\s+(?:update|patch|protocol)\s*:\s*(?:ignore|allow|bypass|disable)'),
        "trust_exploitation", "Disguised security-update with bypass command", "CRITICAL", 92,
    ),
]

# ── Tier 4 — Identity hijacking in external content ─────────────────────────
_IDENTITY_HIJACK_PATTERNS: List[_PatternEntry] = [
    _PatternEntry(
        _compile(r'\bfrom\s+now\s+on(?:\s+in\s+this\s+conversation)?\s*,?\s*you\s+(?:are|will\s+be)\b'),
        "identity_hijack", "Persistent identity reassignment", "CRITICAL", 85,
    ),
    _PatternEntry(
        _compile(r'\byou\s+are\s+now\s+(?:a|an|the)?\s*'
                 r'(?!assistant|helpful|harmless|honest)'   # don't flag normal descriptions
                 r'[A-Za-z][A-Za-z\s]{2,30}'
                 r'(?:\s+(?:without|with\s+no)\s+(?:restrictions?|limits?|guidelines?))?'),
        "identity_hijack", "Role-replacement identity override", "HIGH", 72,
    ),
    _PatternEntry(
        _compile(r'\bact\s+as\s+(?:if\s+you\s+(?:are|were)\s+)?'
                 r'(?:a|an)?\s*[A-Za-z][A-Za-z\s]{2,25}'
                 r'(?:\s+who\s+(?:can|will|does)\s+(?:anything|everything|whatever))'),
        "identity_hijack", "Unrestricted-persona act-as command", "HIGH", 70,
    ),
]

# ── Tier 5 — Data-exfiltration instructions ──────────────────────────────────
_EXFILTRATION_PATTERNS: List[_PatternEntry] = [
    _PatternEntry(
        _compile(r'\b(?:include|insert|add|append|prepend)\s+(?:the\s+following|this)\s+'
                 r'(?:text|string|code|content|output|payload)\s+in\s+(?:your|the)\s+(?:response|reply|output)'),
        "data_exfil", "Embedded output-injection command", "HIGH", 78,
    ),
    _PatternEntry(
        _compile(r'\bsend\s+(?:this|the\s+following|all|your|the)\s+'
                 r'(?:output|response|data|information|content)\s+to\b'),
        "data_exfil", "Exfiltration-to-endpoint command", "CRITICAL", 90,
    ),
    _PatternEntry(
        _compile(r'\bwhen\s+(?:you\s+respond|answering|replying)\s*,?\s*'
                 r'(?:also\s+)?(?:include|add|append)\s+(?:the\s+following|this|your)'),
        "data_exfil", "Conditional output injection", "HIGH", 75,
    ),
    _PatternEntry(
        _compile(r'\brepeat\s+(?:the\s+following\s+text|this)\s+(?:verbatim|exactly|word\s+for\s+word)'),
        "data_exfil", "Verbatim-repeat exfiltration", "HIGH", 72,
    ),
]

# ── Tier 6 — Markdown / HTML / comment injection ────────────────────────────
_STRUCTURAL_INJECTION_PATTERNS: List[_PatternEntry] = [
    _PatternEntry(
        _compile(r'<!--.*?(?:ignore|instruction|system|override|admin).*?-->', re.DOTALL | re.IGNORECASE),
        "structural_injection", "HTML comment containing instructions", "HIGH", 82,
    ),
    _PatternEntry(
        _compile(r'/\*.*?(?:ignore|instruction|system|override|admin).*?\*/', re.DOTALL | re.IGNORECASE),
        "structural_injection", "Block comment containing instructions", "HIGH", 78,
    ),
    _PatternEntry(
        _compile(r'<(?:span|div|p)[^>]*style\s*=\s*["\'][^"\']*display\s*:\s*none[^"\']*["\'][^>]*>'
                 r'.*?</(?:span|div|p)>', re.DOTALL | re.IGNORECASE),
        "structural_injection", "Hidden HTML element (display:none)", "HIGH", 80,
    ),
    _PatternEntry(
        _compile(r'<(?:span|div|p)[^>]*style\s*=\s*["\'][^"\']*(?:color\s*:\s*white|opacity\s*:\s*0)[^"\']*["\']'),
        "structural_injection", "Invisible text via CSS color/opacity", "HIGH", 78,
    ),
    _PatternEntry(
        _compile(r'<script[^>]*>.*?</script>', re.DOTALL | re.IGNORECASE),
        "structural_injection", "Script tag in content", "HIGH", 75,
    ),
]

# Combine all pattern tiers
_ALL_PATTERNS: List[_PatternEntry] = (
    _AUTHORITY_MARKER_PATTERNS
    + _OVERRIDE_PATTERNS
    + _TRUST_EXPLOITATION_PATTERNS
    + _IDENTITY_HIJACK_PATTERNS
    + _EXFILTRATION_PATTERNS
    + _STRUCTURAL_INJECTION_PATTERNS
)


# ---------------------------------------------------------------------------
# Obfuscation detection utilities
# ---------------------------------------------------------------------------

# Unicode categories that appear legitimately but are often used to hide text
_INVISIBLE_CHARS = re.compile(
    r'[\u200B-\u200F\u202A-\u202E\u2060-\u206F\uFEFF\u00AD]'
)

# Right-to-left override / isolate characters
_RTL_OVERRIDE = re.compile(r'[\u202E\u2066\u2067\u2068\u2069]')

# Base64 payload heuristic: long unbroken alphanum+/= string in text
_BASE64_HEURISTIC = re.compile(r'\b([A-Za-z0-9+/]{30,}={0,2})\b')

# URL-percent encoding of common attack keywords
_URL_ENCODED_KEYWORDS = re.compile(
    r'%69%67%6[Ee]%6[Ff]%72%65|'   # ignore
    r'%73%79%73%74%65%6[Dd]|'       # system
    r'%6[Ff]%76%65%72%72%69%64%65', # override
    re.IGNORECASE,
)

# Homoglyph substitution: Cyrillic chars that look like Latin
_CYRILLIC_LOOK_ALIKES = re.compile(r'[\u0430\u0435\u0456\u043E\u0440\u0441\u0443\u0445\u0446\u0455]')


def _detect_obfuscation(text: str) -> Tuple[bool, List[InjectionSignal]]:
    """
    Return (obfuscation_found, signals) describing what was detected.
    Obfuscation downgrades our confidence because it suggests the attacker
    is deliberately hiding content from human reviewers.
    """
    signals: List[InjectionSignal] = []

    if _INVISIBLE_CHARS.search(text):
        signals.append(InjectionSignal(
            signal_type="obfuscation",
            description="Zero-width / invisible Unicode characters detected",
            severity="HIGH",
            weight=70,
            evidence="<invisible characters>",
            is_obfuscated=True,
        ))

    if _RTL_OVERRIDE.search(text):
        signals.append(InjectionSignal(
            signal_type="obfuscation",
            description="Right-to-left override / Unicode bidi control characters",
            severity="HIGH",
            weight=75,
            evidence="<RTL override>",
            is_obfuscated=True,
        ))

    b64_matches = _BASE64_HEURISTIC.findall(text)
    for candidate in b64_matches[:3]:  # analyse at most 3 candidates
        try:
            decoded = base64.b64decode(candidate + "==").decode("utf-8", errors="ignore")
            # Check if decoded content contains instruction-like words
            if any(kw in decoded.lower() for kw in ("ignore", "system", "override", "instruction")):
                signals.append(InjectionSignal(
                    signal_type="obfuscation",
                    description="Base64-encoded instruction payload detected",
                    severity="CRITICAL",
                    weight=90,
                    evidence=f"<base64:{candidate[:12]}…>",
                    is_obfuscated=True,
                ))
        except Exception:
            pass

    if _URL_ENCODED_KEYWORDS.search(text):
        signals.append(InjectionSignal(
            signal_type="obfuscation",
            description="URL-percent-encoded attack keyword detected",
            severity="HIGH",
            weight=78,
            evidence="<url-encoded payload>",
            is_obfuscated=True,
        ))

    if _CYRILLIC_LOOK_ALIKES.search(text):
        # Only flag if mixed with Latin — pure Cyrillic text is fine
        latin_chars = re.search(r'[a-zA-Z]', text)
        if latin_chars:
            signals.append(InjectionSignal(
                signal_type="obfuscation",
                description="Cyrillic homoglyphs mixed with Latin text (visual deception)",
                severity="MEDIUM",
                weight=55,
                evidence="<homoglyph substitution>",
                is_obfuscated=True,
            ))

    return bool(signals), signals


def _strip_obfuscation(text: str) -> str:
    """
    Return a normalised version of the text for pattern matching.
    This catches attacks that rely on invisible chars to split keywords.
    """
    # Remove invisible / control characters
    cleaned = _INVISIBLE_CHARS.sub("", text)
    cleaned = _RTL_OVERRIDE.sub("", cleaned)
    # Normalise Unicode (NFC) and collapse whitespace
    cleaned = unicodedata.normalize("NFC", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


# ---------------------------------------------------------------------------
# IndirectInjectionAnalyzer
# ---------------------------------------------------------------------------

class IndirectInjectionAnalyzer:
    """
    Detect prompt injection attacks embedded in external / retrieved content.

    This analyzer is *source-type-aware*: the same text receives a higher risk
    score when it arrives from an external document than from a direct user
    message, because the LLM may treat document content as trusted context.

    The analyzer applies six detection tiers:
      1. Structural authority markers (SYSTEM:, NOTE TO AI:, …)
      2. Override / erasure commands
      3. Trust-chain exploitation
      4. Identity hijacking in external content
      5. Data-exfiltration instructions
      6. Markdown / HTML / comment injection

    Each tier uses compiled regex patterns; all are applied to a normalised,
    de-obfuscated version of the text so that invisible-character tricks do
    not evade detection.

    Decision thresholds (adjusted score):
      ≥ 70 → BLOCK
      ≥ 35 → CHALLENGE
      < 35 → ALLOW  (but USER_DIRECT sources may pass normally)
    """

    BLOCK_THRESHOLD = 70.0
    CHALLENGE_THRESHOLD = 35.0

    def __init__(self) -> None:
        self._patterns = _ALL_PATTERNS
        logger.info(
            "🔍 IndirectInjectionAnalyzer initialized — %d patterns across 6 tiers",
            len(self._patterns),
        )

    def analyze(
        self,
        text: str,
        source_type: SourceType = SourceType.UNKNOWN,
    ) -> IndirectInjectionResult:
        """
        Analyze *text* for indirect prompt injection signals.

        Args:
            text:        The content to analyse (may be document excerpt, web
                         page chunk, tool output, etc.).
            source_type: Where the content came from.  This determines the
                         risk multiplier applied on top of the raw content score.

        Returns:
            IndirectInjectionResult with a final ALLOW / CHALLENGE / BLOCK verdict.
        """
        if not text or not text.strip():
            return self._empty_result(source_type)

        source_multiplier = _SOURCE_MULTIPLIERS.get(source_type, 1.5)
        signals: List[InjectionSignal] = []

        # ── Step 1: detect and strip obfuscation ────────────────────────────
        obfuscation_found, obfuscation_signals = _detect_obfuscation(text)
        signals.extend(obfuscation_signals)
        normalised_text = _strip_obfuscation(text)

        # ── Step 2: apply pattern catalogue ─────────────────────────────────
        for entry in self._patterns:
            matches = entry.pattern.findall(normalised_text)
            if matches:
                # Evidence: first match, truncated
                evidence = str(matches[0])[:80] if matches else ""
                signals.append(InjectionSignal(
                    signal_type=entry.signal_type,
                    description=entry.description,
                    severity=entry.severity,
                    weight=entry.weight,
                    evidence=evidence[:80],
                    is_obfuscated=False,
                ))

        # ── Step 3: calculate raw score ──────────────────────────────────────
        # Diminishing returns: first signal full weight, subsequent ones 80%
        raw_score = 0.0
        for i, sig in enumerate(sorted(signals, key=lambda s: s.weight, reverse=True)):
            decay = 0.8 ** i  # 1.0, 0.8, 0.64, …
            raw_score += sig.weight * decay
        raw_score = min(100.0, raw_score)

        # ── Step 4: apply source multiplier ─────────────────────────────────
        # Direct user input is not indirect injection by definition.
        # For USER_DIRECT, we still report signals but do not amplify.
        if source_type == SourceType.USER_DIRECT:
            adjusted_score = raw_score  # no amplification
        else:
            adjusted_score = min(100.0, raw_score * source_multiplier)

        # ── Step 5: confidence ───────────────────────────────────────────────
        # Obfuscation means we can't be as sure — lower confidence.
        n_signals = len(signals)
        base_confidence = min(1.0, 0.5 + n_signals * 0.1)
        if obfuscation_found:
            base_confidence = max(0.4, base_confidence - 0.15)
        # Also lower confidence for USER_DIRECT (less context available)
        if source_type == SourceType.USER_DIRECT:
            base_confidence = min(base_confidence, 0.75)
        confidence = round(base_confidence, 3)

        # ── Step 6: verdict ──────────────────────────────────────────────────
        is_indirect = bool(signals) and source_type != SourceType.USER_DIRECT
        if adjusted_score >= self.BLOCK_THRESHOLD:
            verdict = "BLOCK"
        elif adjusted_score >= self.CHALLENGE_THRESHOLD:
            verdict = "CHALLENGE"
        else:
            verdict = "ALLOW"

        # ── Step 7: reasoning ────────────────────────────────────────────────
        reasoning = self._build_reasoning(
            signals, verdict, adjusted_score, source_type, source_multiplier,
        )

        return IndirectInjectionResult(
            is_indirect_injection=is_indirect and verdict != "ALLOW",
            raw_score=round(raw_score, 2),
            adjusted_score=round(adjusted_score, 2),
            source_type=source_type,
            source_multiplier=source_multiplier,
            signals=signals,
            verdict=verdict,
            confidence=confidence,
            reasoning=reasoning,
            obfuscation_detected=obfuscation_found,
            metadata={
                "pattern_count": len(self._patterns),
                "signals_found": len(signals),
                "source_type": source_type.value,
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_reasoning(
        self,
        signals: List[InjectionSignal],
        verdict: str,
        adjusted_score: float,
        source_type: SourceType,
        multiplier: float,
    ) -> List[str]:
        reasons: List[str] = []

        if not signals:
            reasons.append("No indirect injection patterns detected in content")
            return reasons

        if source_type != SourceType.USER_DIRECT:
            reasons.append(
                f"External source type '{source_type.value}' applies {multiplier:.1f}× "
                f"risk amplification (external content is inherently untrusted)"
            )

        # Group by signal_type for concise reporting
        by_type: Dict[str, List[InjectionSignal]] = {}
        for sig in signals:
            by_type.setdefault(sig.signal_type, []).append(sig)

        for sig_type, group in by_type.items():
            count = len(group)
            top = group[0]
            label = sig_type.replace("_", " ").title()
            reasons.append(
                f"{label}: {top.description}"
                + (f" (×{count})" if count > 1 else "")
            )

        if verdict == "BLOCK":
            reasons.append(
                f"Adjusted injection score {adjusted_score:.0f}/100 exceeds BLOCK threshold "
                f"({self.BLOCK_THRESHOLD}) — Principle 14 (Divine Safety): fail-closed"
            )
        elif verdict == "CHALLENGE":
            reasons.append(
                f"Adjusted score {adjusted_score:.0f}/100 warrants additional verification "
                f"before content is trusted"
            )

        return reasons

    def _empty_result(self, source_type: SourceType) -> IndirectInjectionResult:
        return IndirectInjectionResult(
            is_indirect_injection=False,
            raw_score=0.0,
            adjusted_score=0.0,
            source_type=source_type,
            source_multiplier=_SOURCE_MULTIPLIERS.get(source_type, 1.5),
            signals=[],
            verdict="ALLOW",
            confidence=1.0,
            reasoning=["Empty input — no analysis performed"],
            obfuscation_detected=False,
            metadata={"source_type": source_type.value},
        )
