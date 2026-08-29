"""
Community-edition activation gate.

The free community edition requires a verified email. Signing up at the portal
(https://portal.oraclestechnologies.com/community) mints an Ed25519-signed
**community token** that the SDK verifies OFFLINE here — no network call at
runtime, preserving the zero-egress guarantee.

Token sources (first match wins):
  1. env  ETHICORE_COMMUNITY_TOKEN
  2. file ~/.ethicore/community_token

Verification is fully self-contained: a small, dependency-free Ed25519 verify
(public-domain reference math) checks the signature against the bundled Guardian
public key. This keeps the community wheel free of any required third-party
dependency (consistent with its pure-Python design).

Paid / hosted-API users (an api_key is present) are NOT gated here.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
from typing import Optional

# Bundled Guardian Ed25519 public key (hex) — the key the hosted API uses to sign
# community tokens (GET /v1/guardian/public-key). Rotating the API signing key
# requires shipping an SDK update with the new value.
_GUARDIAN_PUBLIC_KEY_HEX = "526e268b98783a7f98c5e1b32eacbc74a4fa7c974befb809ab1655cd9bf0d37d"

_TOKEN_ENV = "ETHICORE_COMMUNITY_TOKEN"
_TOKEN_FILE = Path.home() / ".ethicore" / "community_token"
_SIGNUP_URL = "https://portal.oraclestechnologies.com/community"


class CommunityActivationRequired(RuntimeError):
    """Raised when the community edition is used without a valid community token."""


# ─────────────────────────────────────────────────────────────────────────────
# Minimal, dependency-free Ed25519 verify (public-domain reference; verify-only).
# Uses Python's built-in pow() for modular exponentiation (fast, no deep recursion).
# ─────────────────────────────────────────────────────────────────────────────
_b = 256
_q = 2 ** 255 - 19
_d = (-121665 * pow(121666, _q - 2, _q)) % _q
_I = pow(2, (_q - 1) // 4, _q)


def _xrecover(y: int) -> int:
    xx = (y * y - 1) * pow(_d * y * y + 1, _q - 2, _q)
    x = pow(xx, (_q + 3) // 8, _q)
    if (x * x - xx) % _q != 0:
        x = (x * _I) % _q
    if x % 2 != 0:
        x = _q - x
    return x


_By = (4 * pow(5, _q - 2, _q)) % _q
_Bx = _xrecover(_By)
_B = [_Bx % _q, _By % _q]


def _edwards(P, Q):
    x1, y1 = P
    x2, y2 = Q
    x3 = (x1 * y2 + x2 * y1) * pow(1 + _d * x1 * x2 * y1 * y2, _q - 2, _q)
    y3 = (y1 * y2 + x1 * x2) * pow(1 - _d * x1 * x2 * y1 * y2, _q - 2, _q)
    return [x3 % _q, y3 % _q]


def _scalarmult(P, e: int):
    """Iterative double-and-add — no recursion-limit risk."""
    Q = [0, 1]
    if e:
        for i in reversed(range(e.bit_length())):
            Q = _edwards(Q, Q)
            if (e >> i) & 1:
                Q = _edwards(Q, P)
    return Q


def _bit(h: bytes, i: int) -> int:
    return (h[i // 8] >> (i % 8)) & 1


def _encodepoint(P) -> bytes:
    x, y = P
    bits = [(y >> i) & 1 for i in range(_b - 1)] + [x & 1]
    return bytes(sum(bits[i * 8 + j] << j for j in range(8)) for i in range(_b // 8))


def _decodeint(s: bytes) -> int:
    return sum(2 ** i * _bit(s, i) for i in range(0, _b))


def _isoncurve(P) -> bool:
    x, y = P
    return (-x * x + y * y - 1 - _d * x * x * y * y) % _q == 0


def _decodepoint(s: bytes):
    y = sum(2 ** i * _bit(s, i) for i in range(0, _b - 1))
    x = _xrecover(y)
    if x & 1 != _bit(s, _b - 1):
        x = _q - x
    P = [x, y]
    if not _isoncurve(P):
        raise ValueError("decoding point that is not on curve")
    return P


def _ed25519_verify(signature: bytes, message: bytes, public_key: bytes) -> bool:
    try:
        if len(signature) != 64 or len(public_key) != 32:
            return False
        R = _decodepoint(signature[:32])
        A = _decodepoint(public_key)
        S = int.from_bytes(signature[32:], "little")
        # h is the FULL 512-bit SHA-512 digest (little-endian), NOT truncated to 256.
        h = int.from_bytes(hashlib.sha512(_encodepoint(R) + public_key + message).digest(), "little")
        return _scalarmult(_B, S) == _edwards(R, _scalarmult(A, h))
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Token handling
# ─────────────────────────────────────────────────────────────────────────────
def _read_token() -> Optional[str]:
    tok = os.environ.get(_TOKEN_ENV)
    if tok and tok.strip():
        return tok.strip()
    try:
        if _TOKEN_FILE.exists():
            content = _TOKEN_FILE.read_text(encoding="utf-8").strip()
            return content or None
    except Exception:
        pass
    return None


def _canonical(payload: dict) -> bytes:
    # MUST match the API's signing_service._canonical exactly.
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def verify_token(token: str) -> bool:
    """True iff `token` is a valid, signed community token for the bundled key."""
    try:
        padded = token + "=" * (-len(token) % 4)
        obj = json.loads(base64.urlsafe_b64decode(padded))
        payload = obj["payload"]
        if payload.get("typ") != "community":
            return False
        sig = bytes.fromhex(obj["sig"])
        pub = bytes.fromhex(_GUARDIAN_PUBLIC_KEY_HEX)
        return _ed25519_verify(sig, _canonical(payload), pub)
    except Exception:
        return False


def is_activated() -> bool:
    token = _read_token()
    return bool(token and verify_token(token))


def require_activation() -> None:
    """Raise CommunityActivationRequired unless a valid community token is present."""
    if is_activated():
        return
    raise CommunityActivationRequired(
        "\n"
        "  Ethicore Engine® Guardian — Community Edition requires a free, verified email.\n"
        "\n"
        f"  1. Get your community token (takes 30 seconds): {_SIGNUP_URL}\n"
        "  2. Activate it, either:\n"
        "       export ETHICORE_COMMUNITY_TOKEN=<your-token>\n"
        "     or save it to ~/.ethicore/community_token\n"
        "\n"
        "  Have a hosted API key? Pass api_key=... (or set ETHICORE_API_KEY) to use\n"
        "  the full engine — API keys are not gated.\n"
    )
