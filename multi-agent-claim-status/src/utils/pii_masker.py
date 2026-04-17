"""Lightweight PII masking for display / logging purposes."""
from __future__ import annotations

import re


class PIIMasker:
    @staticmethod
    def mask_email(email: str) -> str:
        parts = email.split("@")
        if len(parts) != 2:
            return "***@***.***"
        local, domain = parts
        visible = local[:2] if len(local) > 2 else local[0]
        return f"{visible}***@{domain}"

    @staticmethod
    def mask_name(name: str) -> str:
        parts = name.split()
        if not parts:
            return "***"
        masked = [p[0] + "***" if len(p) > 1 else "***" for p in parts]
        return " ".join(masked)

    @staticmethod
    def mask_policy_number(policy: str) -> str:
        if len(policy) <= 4:
            return "****"
        return "*" * (len(policy) - 4) + policy[-4:]

    @staticmethod
    def mask_claim_amount(amount: float) -> str:
        """Show magnitude bracket only, not exact value."""
        if amount < 1_000:
            return "< $1,000"
        if amount < 10_000:
            return "$1,000 – $10,000"
        if amount < 100_000:
            return "$10,000 – $100,000"
        return "> $100,000"

    @staticmethod
    def sanitize_for_log(text: str) -> str:
        """Remove common PII patterns from arbitrary text before logging."""
        text = re.sub(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "[EMAIL]", text
        )
        text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]", text)
        text = re.sub(r"\b(?:\d[ -]?){13,16}\b", "[CARD]", text)
        text = re.sub(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE]", text)
        return text
