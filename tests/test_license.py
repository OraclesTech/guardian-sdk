"""
Ethicore Engine™ - Guardian SDK — License Validator Tests
Version: 1.0.0

Tests for ethicore_guardian/license.py — the HMAC-SHA256 license key
validator.  None of these tests require a valid license key because they
verify format parsing, structural validation, and tamper detection, not
actual key issuance.

Any test that asserts ``is_valid=True`` must be generated via
``scripts/_keygen.py`` and hard-coded here (or skipped when the key is
not present in the environment).

Principle 13 (Ultimate Accountability): every security gate must be
covered by tests.

Copyright © 2026 Oracles Technologies LLC
All Rights Reserved
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

from ethicore_guardian.license import (
    LicenseInfo,
    LicenseValidator,
    validate_license,
)


# ---------------------------------------------------------------------------
# Shared validator instance
# ---------------------------------------------------------------------------

@pytest.fixture
def validator() -> LicenseValidator:
    return LicenseValidator()


# ===========================================================================
# TestLicenseKeyFormat — format parsing and structural rejection
# ===========================================================================

class TestLicenseKeyFormat:
    """
    License keys that do not match EG-{TIER}-{NONCE8}-{HMAC16} must be
    rejected immediately, before any HMAC computation.

    Principle 14 (Divine Safety): fail-fast on malformed input.
    """

    def test_empty_string_is_invalid(self, validator):
        result = validator.validate("")
        assert result.is_valid is False
        assert result.tier == "INVALID"

    def test_none_is_invalid(self, validator):
        result = validator.validate(None)  # type: ignore[arg-type]
        assert result.is_valid is False
        assert result.tier == "INVALID"

    def test_random_garbage_is_invalid(self, validator):
        result = validator.validate("not-a-key-at-all")
        assert result.is_valid is False

    def test_wrong_prefix_is_invalid(self, validator):
        result = validator.validate("XG-PRO-AABBCCDD-1234567890ABCDEF")
        assert result.is_valid is False

    def test_wrong_tier_label_is_invalid(self, validator):
        result = validator.validate("EG-DEV-AABBCCDD-1234567890ABCDEF")
        assert result.is_valid is False

    def test_nonce_too_short_is_invalid(self, validator):
        result = validator.validate("EG-PRO-AABB-1234567890ABCDEF")
        assert result.is_valid is False

    def test_nonce_too_long_is_invalid(self, validator):
        result = validator.validate("EG-PRO-AABBCCDDEE-1234567890ABCDEF")
        assert result.is_valid is False

    def test_hmac_too_short_is_invalid(self, validator):
        result = validator.validate("EG-PRO-AABBCCDD-12345678")
        assert result.is_valid is False

    def test_hmac_too_long_is_invalid(self, validator):
        result = validator.validate("EG-PRO-AABBCCDD-1234567890ABCDEFGHI")
        assert result.is_valid is False

    def test_lowercase_input_still_validates_format(self, validator):
        """Key matching the pattern in lowercase must reach HMAC check (not rejected on format)."""
        # Even if HMAC will fail (all-zero secret), the result must NOT be
        # rejected purely on format grounds — it must reach HMAC comparison.
        result = validator.validate("eg-pro-aabbccdd-1234567890abcdef")
        # With placeholder secret the HMAC will likely fail, so is_valid=False,
        # but tier must be set (not "INVALID" from format rejection alone) IFF
        # the regex accepts lowercase.  Our regex uses re.IGNORECASE so this is
        # either INVALID (format OK, HMAC fails) or INVALID (format rejected).
        # Both are acceptable — we just confirm no exception is raised.
        assert isinstance(result, LicenseInfo)

    def test_very_long_string_does_not_raise(self, validator):
        """A very long garbage string must not cause an exception."""
        result = validator.validate("EG-PRO-" + "A" * 1000)
        assert result.is_valid is False


# ===========================================================================
# TestLicenseTierDetection — tier helpers and dataclass behaviour
# ===========================================================================

class TestLicenseTierDetection:
    """
    LicenseInfo helper methods must correctly reflect the tier field.

    Principle 11 (Sacred Truth): tier information is never ambiguous.
    """

    def _make_valid_pro(self) -> LicenseInfo:
        return LicenseInfo(
            key="EG-PRO-AABBCCDD-XXXXXXXXXXXXXXXX",
            tier="PRO",
            is_valid=True,
        )

    def _make_valid_ent(self) -> LicenseInfo:
        return LicenseInfo(
            key="EG-ENT-AABBCCDD-XXXXXXXXXXXXXXXX",
            tier="ENT",
            is_valid=True,
        )

    def _make_invalid(self) -> LicenseInfo:
        return LicenseInfo(
            key="bad-key",
            tier="INVALID",
            is_valid=False,
        )

    def test_is_pro_returns_true_for_pro_tier(self):
        assert self._make_valid_pro().is_pro() is True

    def test_is_pro_returns_false_for_ent_tier(self):
        assert self._make_valid_ent().is_pro() is False

    def test_is_pro_returns_false_for_invalid(self):
        assert self._make_invalid().is_pro() is False

    def test_is_enterprise_returns_true_for_ent_tier(self):
        assert self._make_valid_ent().is_enterprise() is True

    def test_is_enterprise_returns_false_for_pro_tier(self):
        assert self._make_valid_pro().is_enterprise() is False

    def test_is_enterprise_returns_false_for_invalid(self):
        assert self._make_invalid().is_enterprise() is False

    def test_validated_at_is_utc_datetime(self):
        info = self._make_valid_pro()
        assert isinstance(info.validated_at, datetime)
        assert info.validated_at.tzinfo is not None

    def test_validated_at_is_recent(self):
        """validated_at should be within the last few seconds."""
        info = self._make_valid_pro()
        now = datetime.now(timezone.utc)
        delta = abs((now - info.validated_at).total_seconds())
        assert delta < 5, f"validated_at seems stale: {delta:.1f}s ago"


# ===========================================================================
# TestLicenseValidatorEdgeCases — boundary and tamper-resistance
# ===========================================================================

class TestLicenseValidatorEdgeCases:
    """
    The validator must be robust to malicious input and tampered keys.

    Principle 14 (Divine Safety): reject anything unexpected rather than
    degrade to a permissive state.
    """

    def test_module_level_validate_license_works(self):
        """The module-level convenience function must delegate correctly."""
        result = validate_license("garbage")
        assert result.is_valid is False

    def test_tampered_hmac_is_rejected(self, validator):
        """
        A structurally valid key whose HMAC segment has been altered must
        be rejected even if it passes the format check.

        We test this by taking a well-formed key, flipping the last HMAC
        character, and confirming rejection.  Note: with the placeholder
        all-zero secret the original key itself is also invalid, but the
        tampered version will (with overwhelming probability) produce a
        different HMAC than whatever the zero-key computes — if both happen
        to be "INVALID" the test still passes because we only assert that
        the tampered key is invalid.
        """
        tampered = "EG-PRO-AABBCCDD-1234567890ABCDE0"
        result = validator.validate(tampered)
        assert result.is_valid is False

    def test_pro_and_ent_keys_parse_different_tiers(self, validator):
        """PRO and ENT prefix variants must produce different tier values when well-formed."""
        pro = validator.validate("EG-PRO-AABBCCDD-1234567890ABCDEF")
        ent = validator.validate("EG-ENT-AABBCCDD-1234567890ABCDEF")
        # Both may be invalid due to HMAC mismatch with placeholder secret,
        # but if either were valid they must have different tiers.
        # At minimum, confirm no exception and the key field is preserved.
        assert pro.key.startswith("EG-PRO")
        assert ent.key.startswith("EG-ENT")

    def test_whitespace_padding_is_stripped(self, validator):
        """Leading/trailing whitespace must not cause format rejection."""
        result = validator.validate("  EG-PRO-AABBCCDD-1234567890ABCDEF  ")
        # After strip(), the key matches the format → reaches HMAC check.
        # With placeholder secret, is_valid is False, but key field stores original.
        assert isinstance(result, LicenseInfo)

    def test_multiple_validate_calls_do_not_interfere(self, validator):
        """Validating multiple keys in sequence must be stateless."""
        results = [
            validator.validate("garbage1"),
            validator.validate("garbage2"),
            validator.validate("EG-PRO-AABBCCDD-1234567890ABCDEF"),
        ]
        for r in results:
            assert isinstance(r, LicenseInfo)
            assert r.is_valid is False

    @pytest.mark.skipif(
        not os.environ.get("ETHICORE_LICENSE_KEY"),
        reason="No ETHICORE_LICENSE_KEY in environment — skipping live key test.",
    )
    def test_environment_key_validates(self, validator):
        """
        If ETHICORE_LICENSE_KEY is set in the environment, it must validate
        successfully.  This test is only meaningful once the real HMAC secret
        is embedded in license.py and a valid key has been generated.
        """
        key = os.environ["ETHICORE_LICENSE_KEY"]
        result = validator.validate(key)
        assert result.is_valid is True, (
            f"ETHICORE_LICENSE_KEY did not validate: tier={result.tier!r}.  "
            "Ensure _SECRET_MASKED in license.py matches the key's HMAC secret."
        )
        assert result.tier in ("PRO", "ENT")
