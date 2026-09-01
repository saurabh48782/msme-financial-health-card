"""Redaction of Indian borrower identifiers before anything is logged or stored.

A lending system's logs are a liability. Every payload that reaches a log line, a
persisted audit row or a rendered template passes through :func:`redact` first.
"""

from __future__ import annotations

import re
from typing import Any

MASK = "[REDACTED]"
_MAX_STRING = 512

# Ordered most-specific first.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # GSTIN: 2-digit state + 10-char PAN + entity digit + 'Z' + checksum
    ("gstin", re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z]\d[A-Z]\w\b")),
    # PAN: AAAAA9999A
    ("pan", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),
    ("email", re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b")),
    ("ifsc", re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b")),
    # Aadhaar: 12 digits, optionally space/hyphen grouped in 4s
    ("aadhaar", re.compile(r"\b[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}\b")),
    # Indian mobile: optional +91/0 prefix, then 6-9 followed by 9 digits
    ("mobile", re.compile(r"(?:\+91[-\s]?|\b0)?\b[6-9]\d{9}\b")),
    # Bank account: broad digit run. MUST stay last.
    ("account", re.compile(r"\b\d{9,18}\b")),
)

# Keys whose value is an internal identifier that we *want* in logs and audit
# rows. Without this exemption the IFSC pattern (four letters, a zero, six
# alphanumerics) swallows every MSME ID in this dataset — `MSME0000042` matches it
# exactly — and the primary correlation key disappears from the logs it exists to
# correlate. Note what is absent: `consent_handle` is bearer-equivalent, and GSTIN
# and PAN are borrower identity, so none of them are here.
SAFE_KEYS = frozenset(
    {
        "application_id",
        "borrower_id",
        "consent_id",
        "feature_snapshot_hash",
        "lsp_id",
        "model_version",
        "msme_id",
        "msme_ids",
        "offer_id",
        "policy_version",
        "request_id",
        "rubric_version",
        "trace_id",
    }
)

# Keys whose value is replaced wholesale regardless of what it looks like.
_SENSITIVE_KEYS = frozenset(
    {
        "aadhaar",
        "aadhaar_number",
        "account_no",
        "account_number",
        "api_key",
        "authorization",
        "bank_account",
        "consent_handle",
        "dsn",
        "email",
        "gstin",
        "ifsc",
        "mobile",
        "pan",
        "pan_number",
        "password",
        "phone",
        "secret",
        "token",
        "x-api-key",
    }
)


def redact_text(value: str) -> str:
    """Mask every identifier pattern in ``value``."""
    for _, pattern in _PATTERNS:
        value = pattern.sub(MASK, value)
    return value


def redact(value: Any, *, _depth: int = 0) -> Any:
    """Recursively redact a payload, masking sensitive keys wholesale.

    Structure is preserved so a redacted payload is still useful for debugging.
    Long strings are truncated: a request body in a log line is noise, and the
    truncation marker tells the reader something was dropped.
    """
    if _depth > 8:
        return MASK
    if isinstance(value, str):
        cleaned = redact_text(value)
        if len(cleaned) > _MAX_STRING:
            return f"{cleaned[:_MAX_STRING]}...<{len(cleaned)} chars>"
        return cleaned
    if isinstance(value, dict):
        return {key: redact_field(key, child, _depth=_depth + 1) for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        rendered = [redact(child, _depth=_depth + 1) for child in value]
        return type(value)(rendered) if isinstance(value, (list, tuple)) else set(rendered)
    return value


def redact_field(key: Any, value: Any, *, _depth: int = 0) -> Any:
    """Redact one key/value pair, honouring the sensitive and safe key lists."""
    if isinstance(key, str):
        lowered = key.lower()
        if lowered in _SENSITIVE_KEYS:
            return MASK
        if lowered in SAFE_KEYS:
            return value
    return redact(value, _depth=_depth)


def summarise(value: Any, *, key: str | None = None, preview: int = 3) -> Any:
    """Reduce a large value to something a log line can carry.

    Used by the tracing decorator: a 50,000-row frame becomes a count, not a dump.
    """
    if isinstance(value, dict):
        keys = sorted(str(k) for k in value)
        return {"type": "dict", "size": len(value), "keys": keys[:preview]}
    if isinstance(value, (list, tuple, set)):
        return {"type": type(value).__name__, "size": len(value)}
    if hasattr(value, "shape"):
        return {"type": type(value).__name__, "shape": tuple(value.shape)}
    if isinstance(value, str) and len(value) > 64:
        return {"type": "str", "size": len(value)}
    if key is not None:
        return redact_field(key, value)
    return redact(value)
