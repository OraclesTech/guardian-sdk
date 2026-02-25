"""Configuration management"""

import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class GuardianConfig:
    """Guardian SDK configuration"""
    api_key: Optional[str] = None
    enabled: bool = True
    strict_mode: bool = False

    # Sensitivity levels (0.0 to 1.0)
    pattern_sensitivity: float = 0.8
    semantic_sensitivity: float = 0.7
    ml_sensitivity: float = 0.75

    # Performance
    max_latency_ms: int = 50

    # Logging
    log_level: str = "INFO"
    enable_metrics: bool = True

    # Input safety — Principle 12 (Sacred Privacy) / Principle 14 (Divine Safety)
    # Maximum number of characters accepted per analysis call.
    # Input exceeding this limit is truncated and flagged in result metadata.
    # Set to 0 to disable enforcement (not recommended in production).
    # Override via env var: ETHICORE_MAX_INPUT_LENGTH
    max_input_length: int = 32_768

    # Analysis timeout — Principle 14 (Divine Safety): fail-closed on stall.
    # If the full analysis pipeline exceeds this budget the request is returned
    # as CHALLENGE rather than ALLOW.  0 = no timeout (not recommended).
    # Override via env var: ETHICORE_ANALYSIS_TIMEOUT_MS
    analysis_timeout_ms: int = 5_000

    # Rate limiting — Principle 14 (Divine Safety): protect backend resources.
    # Maximum number of analyze() calls permitted per minute per Guardian
    # instance.  0 = unlimited.
    # Override via env var: ETHICORE_MAX_REQUESTS_PER_MINUTE
    max_requests_per_minute: int = 0

    # Caching — Principle 12 (Sacred Privacy) + performance.
    # Results keyed by SHA-256(normalised_text|source_type); raw text is never
    # stored.  Cache is bypassed when session_id is in context so multi-turn
    # context trackers receive every turn individually.
    # Override via env vars: ETHICORE_CACHE_ENABLED / ETHICORE_CACHE_TTL / ETHICORE_CACHE_MAX_MB
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300      # 5 minutes
    cache_max_size_mb: int = 256

    # Learning system access control — Principles 12 + 14.
    # correction_key = None means corrections are DISABLED.
    # Override via env var: ETHICORE_CORRECTION_KEY (never log this value).
    correction_key: Optional[str] = None
    correction_rate_limit_per_minute: int = 10

    # Paid asset license key — enables licensed threat library (30 categories).
    # Override via env var: ETHICORE_LICENSE_KEY (never log this value).
    license_key: Optional[str] = None

    # Path to extracted paid asset bundle directory.
    # Override via env var: ETHICORE_ASSETS_DIR
    # Layout expected: <assets_dir>/data/threat_patterns_licensed.py
    #                  <assets_dir>/models/minilm-l6-v2.onnx  (etc.)
    assets_dir: Optional[str] = None

    @classmethod
    def from_env(cls) -> "GuardianConfig":
        """Load configuration from environment variables"""
        def _int_env(var: str, default: int) -> int:
            try:
                return int(os.getenv(var, str(default)))
            except ValueError:
                return default

        return cls(
            api_key=os.getenv("ETHICORE_API_KEY"),
            enabled=os.getenv("ETHICORE_ENABLED", "true").lower() == "true",
            strict_mode=os.getenv("ETHICORE_STRICT_MODE", "false").lower() == "true",
            max_input_length=_int_env("ETHICORE_MAX_INPUT_LENGTH", 32_768),
            analysis_timeout_ms=_int_env("ETHICORE_ANALYSIS_TIMEOUT_MS", 5_000),
            max_requests_per_minute=_int_env("ETHICORE_MAX_REQUESTS_PER_MINUTE", 0),
            cache_enabled=os.getenv("ETHICORE_CACHE_ENABLED", "true").lower() == "true",
            cache_ttl_seconds=_int_env("ETHICORE_CACHE_TTL", 300),
            cache_max_size_mb=_int_env("ETHICORE_CACHE_MAX_MB", 256),
            correction_key=os.getenv("ETHICORE_CORRECTION_KEY"),
            correction_rate_limit_per_minute=_int_env("ETHICORE_CORRECTION_RATE_LIMIT", 10),
            license_key=os.getenv("ETHICORE_LICENSE_KEY"),
            assets_dir=os.getenv("ETHICORE_ASSETS_DIR"),
        )