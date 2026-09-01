"""PII redaction. A lending system's logs are a liability."""

from __future__ import annotations

import pytest

from src.utils.pii import MASK, redact, redact_field, redact_text, summarise


class TestPatterns:
    @pytest.mark.parametrize(
        "value",
        [
            "ABCDE1234F",  # PAN
            "22AAAAA0000A1Z5",  # GSTIN
            "9876543210",  # mobile
            "+91 9876543210",
            "2345 6789 0123",  # Aadhaar
            "HDFC0001234",  # IFSC
            "borrower@example.com",
            "123456789012345",  # bank account
        ],
    )
    def test_identifiers_are_masked(self, value: str) -> None:
        assert MASK in redact_text(value)

    def test_phone_is_matched_before_the_broad_account_pattern(self) -> None:
        """Pattern order is load-bearing: the 9-18 digit account run would
        otherwise swallow a 10-digit mobile and mislabel it."""
        assert redact_text("call 9876543210") == "call [REDACTED]"

    def test_ordinary_numbers_survive(self) -> None:
        """Over-redaction is preferred, but not to the point of erasing metrics."""
        assert redact_text("score 78.5 limit 4000") == "score 78.5 limit 4000"


class TestKeyAwareness:
    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("consent_handle", "anything"),
            ("api_key", "sk-live"),
            ("gstin", "not-even-a-gstin"),
        ],
    )
    def test_sensitive_keys_are_masked_wholesale(self, key: str, value: str) -> None:
        assert redact({key: value}) == {key: MASK}

    def test_msme_id_survives_redaction(self) -> None:
        """The IFSC pattern (four letters, a zero, six alphanumerics) matches
        ``MSME0000042`` exactly. Without the safe-key exemption the primary log
        correlation key would be masked out of the logs it exists to correlate."""
        assert redact({"msme_id": "MSME0000042"}) == {"msme_id": "MSME0000042"}
        assert redact_field("msme_id", "MSME0000042") == "MSME0000042"

    def test_bare_msme_id_string_is_still_redacted(self) -> None:
        """Without key context the conservative default still applies."""
        assert redact_text("MSME0000042") == MASK

    def test_nested_structures_are_walked(self) -> None:
        payload = {"borrower": {"pan": "ABCDE1234F", "msme_id": "MSME0000001"}}
        result = redact(payload)
        assert result["borrower"]["pan"] == MASK
        assert result["borrower"]["msme_id"] == "MSME0000001"

    def test_recursion_is_bounded(self) -> None:
        deep: dict = {"a": {}}
        node = deep["a"]
        for _ in range(20):
            node["a"] = {}
            node = node["a"]
        assert redact(deep) is not None  # must terminate


class TestSummarise:
    def test_frames_become_shapes(self) -> None:
        import pandas as pd

        assert summarise(pd.DataFrame({"a": [1, 2, 3]})) == {
            "type": "DataFrame",
            "shape": (3, 1),
        }

    def test_lists_become_counts(self) -> None:
        assert summarise([1, 2, 3])["size"] == 3

    def test_safe_key_passes_through(self) -> None:
        assert summarise("MSME0000042", key="msme_id") == "MSME0000042"
