#!/usr/bin/env python3
"""
Ethicore Engine™ Guardian SDK — Secret Masker Utility

Computes the XOR-masked bytes to embed in ethicore_guardian/license.py
from your raw 32-byte HMAC secret.

This script is SAFE TO KEEP — it contains only the public XOR mask
("ORACLES1"), NOT the actual secret.  Only scripts/_keygen.py holds
the raw secret.

Usage:
    python scripts/mask_secret.py <64-char-hex-string>

Example:
    # Step 1: generate a secret (copy the output)
    python -c "import secrets; print(secrets.token_hex(32))"

    # Step 2: compute the masked bytes for license.py
    python scripts/mask_secret.py a1b2c3d4e5f60718293a4b5c6d7e8f90...

Copyright © 2026 Oracles Technologies LLC. All Rights Reserved.
"""
from __future__ import annotations

import sys

# Public XOR mask — same value as _XOR_MASK in license.py
_XOR_MASK = bytes([0x4F, 0x52, 0x41, 0x43, 0x4C, 0x45, 0x53, 0x31])  # "ORACLES1"


def mask(raw: bytes) -> bytes:
    mask_repeated = _XOR_MASK * (len(raw) // len(_XOR_MASK) + 1)
    return bytes(b ^ m for b, m in zip(raw, mask_repeated))


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/mask_secret.py <64-char-hex-string>")
        print()
        print("Generate a secret first:")
        print("  python -c \"import secrets; print(secrets.token_hex(32))\"")
        return 1

    hex_input = sys.argv[1].strip()

    if len(hex_input) != 64:
        print(f"ERROR: Expected 64 hex chars (32 bytes), got {len(hex_input)} chars.")
        return 1

    try:
        raw = bytes.fromhex(hex_input)
    except ValueError as e:
        print(f"ERROR: Invalid hex string — {e}")
        return 1

    masked = mask(raw)

    print()
    print("=" * 60)
    print("MASKED BYTES — paste into ethicore_guardian/license.py")
    print("=" * 60)
    print()
    print("_SECRET_MASKED = bytes([")
    for i in range(0, 32, 8):
        chunk = masked[i:i+8]
        line = ", ".join(f"0x{b:02X}" for b in chunk)
        print(f"    {line},")
    print("])")
    print()
    print("=" * 60)
    print("RAW SECRET — paste into scripts/_keygen.py  (KEEP PRIVATE)")
    print("=" * 60)
    print()
    print(f'_REAL_SECRET = bytes.fromhex("{hex_input}")')
    print()
    print("Next steps:")
    print("  1. Replace _SECRET_MASKED in ethicore_guardian/license.py")
    print("  2. Replace _REAL_SECRET in scripts/_keygen.py")
    print("  3. Generate a test key:  python scripts/_keygen.py PRO")
    print("  4. Verify it validates:  python -c \"from ethicore_guardian.license import validate_license; print(validate_license('EG-PRO-...')  )\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
