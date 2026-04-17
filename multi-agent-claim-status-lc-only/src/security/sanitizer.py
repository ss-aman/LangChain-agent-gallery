"""Input sanitisation – prevents XSS, SQL-injection patterns, and ReDoS."""
from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Optional


# ── Compile once at import time ─────────────────────────────────────────────

# Patterns that indicate SQL injection attempts
_SQL_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE|UNION)\b)",
        r"(--|;|\/\*|\*\/)",
        r"(\bOR\b\s+\d+\s*=\s*\d+)",
        r"(\bAND\b\s+\d+\s*=\s*\d+)",
        r"('.*?'--)",
        r"(\bXP_\w+)",
        r"(SLEEP\s*\(\s*\d+\s*\))",
        r"(BENCHMARK\s*\()",
        r"(LOAD_FILE\s*\()",
        r"(INTO\s+OUTFILE)",
    ]
]

# Patterns for XSS detection
_XSS_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"<script[^>]*>.*?</script>",
        r"javascript\s*:",
        r"on\w+\s*=",           # onclick=, onload=, etc.
        r"<iframe",
        r"<object",
        r"<embed",
        r"<form",
        r"data\s*:",
        r"vbscript\s*:",
    ]
]

# Claim number format: CLM followed by 6-12 digits
_CLAIM_NUMBER_RE = re.compile(r"^CLM\d{6,12}$", re.IGNORECASE)

# Policy number: alphanumeric, dashes allowed
_POLICY_NUMBER_RE = re.compile(r"^[A-Z0-9\-]{6,20}$", re.IGNORECASE)

# Date: YYYY-MM-DD
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass
class SanitizationResult:
    is_safe: bool
    cleaned_value: str
    threat_type: Optional[str] = None
    detail: Optional[str] = None


class InputSanitizer:
    MAX_QUERY_LENGTH = 2000
    MAX_FIELD_LENGTH = 512

    # ── Public API ─────────────────────────────────────────────────────────────

    @classmethod
    def sanitize_user_query(cls, text: str) -> SanitizationResult:
        """Full pipeline for free-text user queries."""
        if not isinstance(text, str):
            return SanitizationResult(False, "", "TYPE_ERROR", "Input must be a string")

        if len(text) > cls.MAX_QUERY_LENGTH:
            return SanitizationResult(
                False, "", "INPUT_TOO_LONG",
                f"Query exceeds {cls.MAX_QUERY_LENGTH} characters"
            )

        stripped = text.strip()

        xss = cls._detect_xss(stripped)
        if xss:
            return SanitizationResult(False, "", "XSS", xss)

        sqli = cls._detect_sql_injection(stripped)
        if sqli:
            return SanitizationResult(False, "", "SQL_INJECTION", sqli)

        cleaned = html.escape(stripped)
        return SanitizationResult(True, cleaned)

    @classmethod
    def sanitize_claim_number(cls, value: str) -> SanitizationResult:
        v = value.strip().upper()
        if not _CLAIM_NUMBER_RE.match(v):
            return SanitizationResult(
                False, "", "INVALID_FORMAT",
                "Claim number must match CLMxxxxxx (6-12 digits)"
            )
        return SanitizationResult(True, v)

    @classmethod
    def sanitize_date(cls, value: str) -> SanitizationResult:
        v = value.strip()
        if not _DATE_RE.match(v):
            return SanitizationResult(False, "", "INVALID_DATE", "Expected YYYY-MM-DD")
        return SanitizationResult(True, v)

    @classmethod
    def sanitize_policy_number(cls, value: str) -> SanitizationResult:
        v = value.strip().upper()
        if not _POLICY_NUMBER_RE.match(v):
            return SanitizationResult(
                False, "", "INVALID_FORMAT", "Invalid policy number format"
            )
        return SanitizationResult(True, v)

    # ── Internals ──────────────────────────────────────────────────────────────

    @staticmethod
    def _detect_sql_injection(text: str) -> Optional[str]:
        for pattern in _SQL_INJECTION_PATTERNS:
            m = pattern.search(text)
            if m:
                return f"Suspicious SQL pattern detected: {m.group()}"
        return None

    @staticmethod
    def _detect_xss(text: str) -> Optional[str]:
        for pattern in _XSS_PATTERNS:
            m = pattern.search(text)
            if m:
                return f"Suspicious HTML/script pattern detected: {m.group()}"
        return None
