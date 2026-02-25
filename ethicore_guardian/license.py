"""
Ethicore Engine™ — Guardian SDK License Validator
Version: 1.0.0

SECURITY NOTE: XOR-obfuscated secret is a lightweight deterrent, not
cryptographic security. Upgrade to Ed25519 for v2 so the private key
never ships. See scripts/_keygen.py for key generation.

Key format: EG-{TIER}-{NONCE8}-{HMAC16}
  Example:   EG-PRO-A3F72B91-WXYZABCD12345678
  Tiers:     PRO, ENT

Copyright © 2026 Oracles Technologies LLC. All Rights Reserved.
"""
from __future__ import annotations

import hashlib
import hmac as _hmac
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# XOR-obfuscated HMAC secret
#
# HOW TO SET THIS UP (one-time, before generating customer keys):
#   1. Generate a random 32-byte secret, e.g.:
#        python -c "import secrets; print(secrets.token_hex(32))"
#   2. Store that raw secret ONLY in scripts/_keygen.py (never commit it).
#   3. XOR-mask it:
#        raw   = bytes.fromhex("your_hex_secret")
#        mask  = _XOR_MASK * (32 // len(_XOR_MASK) + 1)
#        masked = bytes(b ^ m for b, m in zip(raw, mask))
#        print(list(masked))
#   4. Replace the 0x00 placeholders below with the masked byte values.
#
# The XOR mask is NOT secret — its job is only to prevent the raw secret
# from appearing as plain ASCII in a strings(1) scan of the wheel.
# ---------------------------------------------------------------------------
_XOR_MASK = bytes([0x4F, 0x52, 0x41, 0x43, 0x4C, 0x45, 0x53, 0x31])  # "ORACLES1"

_SECRET_MASKED = bytes([
    0x68, 0xAB, 0x09, 0x95, 0x6D, 0xBC, 0x0C, 0x0F,
    0x15, 0x9C, 0x37, 0x97, 0x6B, 0xE8, 0xE3, 0xE3,
    0x5A, 0x1C, 0xA2, 0xDF, 0xD5, 0xB9, 0x9C, 0x26,
    0xF1, 0x5A, 0x5D, 0xE2, 0x52, 0xFD, 0xDF, 0x45,
])
# NOTE: While _SECRET_MASKED is all zeros every computed HMAC equals the
# HMAC of the all-zeros key, meaning NO key you generate will validate
# against a different secret.  This is intentional — fill in real masked
# bytes before generating customer keys with scripts/_keygen.py.

_VALID_TIERS = frozenset({"PRO", "ENT"})

_KEY_RE = re.compile(
    r"^EG-(?P<tier>PRO|ENT)-(?P<nonce>[A-F0-9]{8})-(?P<sig>[A-Z0-9]{16})$",
    re.IGNORECASE,
)


def _recover_secret() -> bytes:
    """XOR-unmask the embedded secret at runtime."""
    mask = _XOR_MASK * (len(_SECRET_MASKED) // len(_XOR_MASK) + 1)
    return bytes(b ^ m for b, m in zip(_SECRET_MASKED, mask))


def _compute_hmac(tier: str, nonce: str) -> str:
    """Compute the 16-char uppercase hex HMAC for a tier+nonce pair."""
    message = f"{tier.upper()}-{nonce.upper()}".encode()
    return _hmac.new(_recover_secret(), message, hashlib.sha256).hexdigest()[:16].upper()


@dataclass
class LicenseInfo:
    """Result of a license key validation."""

    key: str
    tier: str            # "PRO" | "ENT" | "INVALID"
    is_valid: bool
    validated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def is_pro(self) -> bool:
        """Return True if this is a valid PRO-tier license."""
        return self.is_valid and self.tier == "PRO"

    def is_enterprise(self) -> bool:
        """Return True if this is a valid ENT-tier license."""
        return self.is_valid and self.tier == "ENT"


class LicenseValidator:
    """
    HMAC-SHA256 license key validator.

    Validates keys of the form EG-{TIER}-{NONCE8}-{HMAC16} without any
    network calls or external dependencies — pure stdlib.

    Principle 11 (Sacred Truth): validation is deterministic and transparent.
    Principle 14 (Divine Safety): reject invalid keys; never degrade silently.
    """

    def validate(self, key: str) -> LicenseInfo:
        """
        Validate a license key string.

        Args:
            key: License key string, e.g. ``EG-PRO-A3F72B91-WXYZ12345678ABCD``

        Returns:
            :class:`LicenseInfo` with ``is_valid=True`` on success or
            ``tier="INVALID"`` / ``is_valid=False`` on any failure.
        """
        if not key or not isinstance(key, str):
            return LicenseInfo(key=key or "", tier="INVALID", is_valid=False)

        m = _KEY_RE.match(key.strip())
        if not m:
            return LicenseInfo(key=key, tier="INVALID", is_valid=False)

        tier = m.group("tier").upper()
        nonce = m.group("nonce").upper()
        sig = m.group("sig").upper()

        expected = _compute_hmac(tier, nonce)

        # Constant-time comparison — Principle 14 (Divine Safety): prevents
        # timing-side-channel attacks that could reveal the secret key.
        if not _hmac.compare_digest(sig, expected):
            return LicenseInfo(key=key, tier="INVALID", is_valid=False)

        logger.info("License validated: tier=%s (key redacted)", tier)
        return LicenseInfo(key=key, tier=tier, is_valid=True)


# Module-level singleton for convenience
_validator = LicenseValidator()


def validate_license(key: str) -> LicenseInfo:
    """
    Validate a license key using the module-level :class:`LicenseValidator`.

    Convenience wrapper — avoids instantiating a new validator on every call.

    Args:
        key: License key string.

    Returns:
        :class:`LicenseInfo` result.
    """
    return _validator.validate(key)
