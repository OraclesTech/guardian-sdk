"""
Ethicore Engine™ — Guardian SDK
Threat Pattern Library — Community Edition

Version: 1.0.0 (Community)

This is the open-source community edition, covering 5 OWASP LLM Top-10
aligned threat categories.  The full licensed edition adds 25 additional
categories (30 total), complete ONNX semantic embeddings, and advanced
agentic/multi-turn threat detection.

To unlock the full threat library:
  1. Purchase a license at https://oraclestechnologies.com/guardian
  2. Set ETHICORE_LICENSE_KEY in your environment
  3. Extract the asset bundle: unzip ethicore-guardian-assets-pro.zip -d ~/.ethicore/

API contract: identical to the licensed edition — same exports, same function
signatures.  Code written against the community edition works unchanged with
the licensed edition when credentials are supplied.

References:
  - OWASP LLM Top 10: https://owasp.org/www-project-top-10-for-large-language-model-applications/
  - MITRE ATLAS: https://atlas.mitre.org/

Copyright © 2026 Oracles Technologies LLC. All Rights Reserved.
Framework code: MIT License.  Full threat library: Proprietary.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Enumerations (identical to licensed edition)
# ---------------------------------------------------------------------------

class ThreatSeverity(Enum):
    """Threat severity levels"""
    CRITICAL = "CRITICAL"
    HIGH     = "HIGH"
    MEDIUM   = "MEDIUM"
    LOW      = "LOW"


class FalsePositiveRisk(Enum):
    """False-positive risk classification"""
    VERY_LOW = "VERY_LOW"
    LOW      = "LOW"
    MEDIUM   = "MEDIUM"
    HIGH     = "HIGH"


# ---------------------------------------------------------------------------
# Community threat pattern registry — 5 categories
#
# Patterns sourced from publicly documented research:
#   OWASP LLM01 (Prompt Injection), LLM02 (Insecure Output Handling),
#   and open academic literature on adversarial prompting.
# ---------------------------------------------------------------------------

THREAT_PATTERNS: Dict[str, Any] = {

    # ------------------------------------------------------------------
    # instructionOverride — OWASP LLM01 (Prompt Injection)
    # Attempts to override or discard the model's system instructions.
    # ------------------------------------------------------------------
    "instructionOverride": {
        "patterns": [
            r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|commands?|directives?)",
            r"ignore\s+(?:all|your|the|my)\s+instructions?",
            r"disregard\s+(previous|all)\s+(instructions?|rules?|guidelines?)",
            r"forget\s+(everything|all|previous)\s+(instructions?|you'?ve?\s+been\s+told)",
            r"override\s+(your|previous|system)\s+(instructions?|programming|directives?)",
        ],
        "severity": ThreatSeverity.CRITICAL,
        "weight": 100,
        "description": "Attempts to override or ignore system instructions",
        "semanticFingerprint": [
            "ignore previous instructions",
            "disregard all rules",
            "forget your guidelines",
            "override system directives",
            "new instructions follow",
            "reset your programming",
        ],
        "contextHints": {
            "escalators": ["now", "immediately", "must", "required"],
            "mitigators": ["formatting", "style", "layout"],
        },
        "falsePositiveRisk": FalsePositiveRisk.MEDIUM,
        "mitigationStrategy": "Check for educational/formatting context",
    },

    # ------------------------------------------------------------------
    # jailbreakActivation — OWASP LLM01
    # Known jailbreak persona/mode activation keywords.
    # ------------------------------------------------------------------
    "jailbreakActivation": {
        "patterns": [
            r"\b(DAN|DUDE|STAN|KEVIN|AlterEgo|UnfilteredAI)\b",
            r"do\s+anything\s+now",
            r"jailbreak\s+mode",
            r"unrestricted\s+(mode|AI|assistant)",
        ],
        "severity": ThreatSeverity.CRITICAL,
        "weight": 100,
        "description": "Known jailbreak mode activation attempts",
        "semanticFingerprint": [
            "DAN mode",
            "do anything now",
            "jailbreak mode",
            "unrestricted AI",
            "no rules mode",
            "unfiltered response",
        ],
        "contextHints": {
            "escalators": ["version", "2.0", "enabled", "activated"],
            "mitigators": [],
        },
        "falsePositiveRisk": FalsePositiveRisk.VERY_LOW,
        "mitigationStrategy": "Instant block, log as critical threat",
    },

    # ------------------------------------------------------------------
    # safetyBypass — OWASP LLM01
    # Attempts to disable or circumvent safety filters.
    # ------------------------------------------------------------------
    "safetyBypass": {
        "patterns": [
            r"(disable|turn\s+off|deactivate|remove)\s+(your\s+)?(safety|ethical|moral)\s+(guidelines?|filters?|restrictions?|measures?)",
            r"bypass\s+(safety|content)\s+(filters?|restrictions?)",
            r"ignore\s+(safety|ethical|content)\s+(guidelines?|warnings?|policies?)",
        ],
        "severity": ThreatSeverity.CRITICAL,
        "weight": 98,
        "description": "Attempts to bypass safety mechanisms",
        "semanticFingerprint": [
            "disable safety guidelines",
            "turn off ethical filters",
            "bypass safety restrictions",
            "ignore content policies",
            "without safety checks",
            "remove content filter",
        ],
        "contextHints": {
            "escalators": ["completely", "entirely", "all", "every"],
            "mitigators": [],
        },
        "falsePositiveRisk": FalsePositiveRisk.LOW,
        "mitigationStrategy": "Block and flag for review",
    },

    # ------------------------------------------------------------------
    # roleHijacking — OWASP LLM01
    # Identity replacement / persona injection attacks.
    # ------------------------------------------------------------------
    "roleHijacking": {
        "patterns": [
            r"you\s+are\s+now\s+(a|an)\s+(?!assistant|helpful|AI)[\w\s]+",
            r"act\s+as\s+(if\s+you\s+are\s+)?(a|an)\s+(?!assistant)[\w\s]+",
            r"pretend\s+(to\s+be|you\s+are)\s+(a|an)\s+(?!assistant)[\w\s]+",
        ],
        "severity": ThreatSeverity.CRITICAL,
        "weight": 90,
        "description": "Attempts to hijack AI role or identity",
        "semanticFingerprint": [
            "you are now a",
            "act as if you are",
            "pretend to be",
            "roleplay as",
            "simulate being",
            "become a character",
        ],
        "contextHints": {
            "escalators": ["DAN", "unrestricted", "unfiltered", "no rules"],
            "mitigators": ["tutor", "teacher", "instructor", "guide", "expert"],
        },
        "falsePositiveRisk": FalsePositiveRisk.HIGH,
        "mitigationStrategy": "Allowlist educational roles (tutor, teacher, mentor)",
    },

    # ------------------------------------------------------------------
    # systemPromptLeaks — OWASP LLM07 (System Prompt Leakage)
    # Attempts to extract the model's system prompt or initial instructions.
    # ------------------------------------------------------------------
    "systemPromptLeaks": {
        "patterns": [
            r"(show|display|reveal|print|output|tell\s+me)\s+(your|the)\s+system\s+prompt",
            r"what\s+(is|are)\s+your\s+(initial|original|system|base)\s+(instructions?|prompt|guidelines?)",
            r"(show|display)\s+your\s+(hidden|internal|original)\s+(instructions?|rules?)",
        ],
        "severity": ThreatSeverity.HIGH,
        "weight": 85,
        "description": "Attempts to extract system prompts or instructions",
        "semanticFingerprint": [
            "show system prompt",
            "reveal your instructions",
            "what are your initial guidelines",
            "display hidden rules",
            "print system prompt",
            "tell me your programming",
        ],
        "contextHints": {
            "escalators": ["exactly", "verbatim", "word-for-word", "complete"],
            "mitigators": ["generally", "overview", "summary"],
        },
        "falsePositiveRisk": FalsePositiveRisk.MEDIUM,
        "mitigationStrategy": "Distinguish between capability questions and extraction attempts",
    },
}


# ---------------------------------------------------------------------------
# Public API — identical signatures to licensed edition
# ---------------------------------------------------------------------------

def get_all_patterns() -> List[Dict[str, Any]]:
    """
    Return a flat list of all pattern entries, one per regex pattern string.

    Each entry contains: category, pattern, severity, weight, description.
    Used by PatternAnalyzer to compile the full regex set.
    """
    patterns: List[Dict[str, Any]] = []
    for category, data in THREAT_PATTERNS.items():
        for pattern in data["patterns"]:
            patterns.append({
                "category":    category,
                "pattern":     pattern,
                "severity":    data["severity"],
                "weight":      data["weight"],
                "description": data["description"],
            })
    return patterns


def get_semantic_fingerprints() -> List[Dict[str, Any]]:
    """
    Return a flat list of semantic fingerprint entries for embedding generation.

    Each entry contains: text, category, severity (string), weight.
    Used by SemanticAnalyzer._get_core_threat_patterns().
    """
    fingerprints: List[Dict[str, Any]] = []
    for category, data in THREAT_PATTERNS.items():
        severity_val = (
            data["severity"].value
            if isinstance(data["severity"], ThreatSeverity)
            else data["severity"]
        )
        for text in data["semanticFingerprint"]:
            fingerprints.append({
                "text":     text,
                "category": category,
                "severity": severity_val,
                "weight":   data["weight"],
            })
    return fingerprints


def get_category_metadata(category: str) -> Optional[Dict[str, Any]]:
    """Return full metadata dict for a single category, or None if not found."""
    return THREAT_PATTERNS.get(category)


def get_categories_by_severity(severity: ThreatSeverity) -> List[str]:
    """Return list of category names with the given severity level."""
    return [
        cat
        for cat, data in THREAT_PATTERNS.items()
        if data["severity"] == severity
    ]


def calculate_threat_score(matches: List[Dict[str, Any]]) -> float:
    """
    Calculate a weighted threat score (0–200) from a list of match summaries.

    Each item in *matches* should have a ``category`` key and a ``count`` key.
    Scores are capped at 200 to keep the range consistent with the licensed edition.
    """
    score = 0.0
    for match in matches:
        category = match.get("category", "")
        count = match.get("count", 0)
        if category in THREAT_PATTERNS:
            weight = THREAT_PATTERNS[category]["weight"]
            score += weight * min(count, 3)  # diminishing returns after 3 matches
    return min(200.0, score)


def determine_threat_level(score: float) -> str:
    """
    Convert a numeric threat score to a human-readable threat level string.

    Thresholds (identical to licensed edition):
        CRITICAL  ≥ 150
        HIGH      ≥ 100
        MEDIUM    ≥ 50
        LOW       ≥ 20
        NONE      <  20
    """
    if score >= 150:
        return "CRITICAL"
    if score >= 100:
        return "HIGH"
    if score >= 50:
        return "MEDIUM"
    if score >= 20:
        return "LOW"
    return "NONE"


def is_high_false_positive_risk(category: str) -> bool:
    """Return True if the given category has HIGH or VERY_HIGH false-positive risk."""
    data = THREAT_PATTERNS.get(category)
    if not data:
        return False
    risk = data.get("falsePositiveRisk")
    return risk in (FalsePositiveRisk.HIGH,)


def get_threat_statistics() -> Dict[str, Any]:
    """
    Return a statistics summary for the current threat pattern set.

    Community-edition extras:
        ``edition``                    → "community"
        ``licensed_categories_available`` → 30
    """
    categories = list(THREAT_PATTERNS.keys())

    by_severity = {
        "CRITICAL": len(get_categories_by_severity(ThreatSeverity.CRITICAL)),
        "HIGH":     len(get_categories_by_severity(ThreatSeverity.HIGH)),
        "MEDIUM":   len(get_categories_by_severity(ThreatSeverity.MEDIUM)),
        "LOW":      len(get_categories_by_severity(ThreatSeverity.LOW)),
    }

    total_patterns = sum(len(data["patterns"]) for data in THREAT_PATTERNS.values())
    total_fingerprints = sum(
        len(data["semanticFingerprint"]) for data in THREAT_PATTERNS.values()
    )

    return {
        "totalCategories":             len(categories),
        "bySeverity":                  by_severity,
        "totalRegexPatterns":          total_patterns,
        "totalSemanticFingerprints":   total_fingerprints,
        "avgPatternsPerCategory":      round(total_patterns / len(categories), 1),
        "avgFingerprintsPerCategory":  round(total_fingerprints / len(categories), 1),
        # Community-edition metadata
        "edition":                     "community",
        "licensed_categories_available": 30,
    }


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    stats = get_threat_statistics()
    print("Guardian SDK — Community Edition")
    print(json.dumps(stats, indent=2))
