"""
Community activation gate — offline Ed25519 community-token verification.

Validates the dependency-free Ed25519 verify against `cryptography` signatures,
the token-verification logic, and the hard gate on community_guardian.Guardian().
"""
from __future__ import annotations

import base64
import json
import time

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

import ethicore_guardian.activation as activation


def _keypair():
    pk = Ed25519PrivateKey.generate()
    pub_hex = pk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    return pk, pub_hex


def _canonical(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _mint(pk, payload: dict) -> str:
    body = {"payload": payload, "sig": pk.sign(_canonical(payload)).hex()}
    return base64.urlsafe_b64encode(
        json.dumps(body, separators=(",", ":")).encode()
    ).decode().rstrip("=")


def test_ed25519_verify_matches_cryptography():
    """The pure-Python verify must agree with cryptography's Ed25519 signatures."""
    pk, pub_hex = _keypair()
    msg = b"ethicore community activation"
    sig = pk.sign(msg)
    pub = bytes.fromhex(pub_hex)
    assert activation._ed25519_verify(sig, msg, pub) is True
    assert activation._ed25519_verify(sig, msg + b"!", pub) is False          # wrong message
    flipped = sig[:-1] + bytes([sig[-1] ^ 0x01])
    assert activation._ed25519_verify(flipped, msg, pub) is False             # tampered sig


def test_verify_token_valid_wrong_type_tampered(monkeypatch):
    pk, pub_hex = _keypair()
    monkeypatch.setattr(activation, "_GUARDIAN_PUBLIC_KEY_HEX", pub_hex)
    payload = {"typ": "community", "email": "dev@example.com", "iat": int(time.time()), "ver": 1}
    token = _mint(pk, payload)
    assert activation.verify_token(token) is True
    assert activation.verify_token(_mint(pk, {**payload, "typ": "paid"})) is False
    assert activation.verify_token(token[:-6] + "ABCDEF") is False
    assert activation.verify_token("not-a-real-token") is False


def test_verify_token_rejects_wrong_key(monkeypatch):
    signer, _ = _keypair()
    _, other_pub = _keypair()
    monkeypatch.setattr(activation, "_GUARDIAN_PUBLIC_KEY_HEX", other_pub)
    token = _mint(signer, {"typ": "community", "email": "x@y.com", "iat": 1, "ver": 1})
    assert activation.verify_token(token) is False


def test_read_token_from_env(monkeypatch):
    monkeypatch.setenv("ETHICORE_COMMUNITY_TOKEN", "  some-token  ")
    assert activation._read_token() == "some-token"


def test_hard_gate_raises_without_token(monkeypatch):
    monkeypatch.setattr(activation, "is_activated", lambda: False)   # override autouse bypass
    monkeypatch.delenv("ETHICORE_API_KEY", raising=False)
    monkeypatch.delenv("ETHICORE_COMMUNITY_TOKEN", raising=False)
    with pytest.raises(activation.CommunityActivationRequired):
        activation.require_activation()
    from ethicore_guardian.community_guardian import Guardian
    with pytest.raises(activation.CommunityActivationRequired):
        Guardian()


def test_gate_allows_when_activated(monkeypatch):
    monkeypatch.setattr(activation, "is_activated", lambda: True)
    activation.require_activation()  # must not raise
    from ethicore_guardian.community_guardian import Guardian
    Guardian()  # constructs fine


def test_api_key_users_are_exempt(monkeypatch):
    monkeypatch.setattr(activation, "is_activated", lambda: False)   # gate would block community
    from ethicore_guardian.community_guardian import Guardian
    Guardian(api_key="eg-sk-test-key")  # api_key present → not gated
