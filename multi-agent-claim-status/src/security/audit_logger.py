"""Structured audit logger with PII masking.

Uses structlog for JSON-formatted, machine-parseable output.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Optional

import structlog

from src.config import get_settings

settings = get_settings()

# ── PII masking patterns ───────────────────────────────────────────────────────

_PII_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "***@***.***"),
    (re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"), "***-***-****"),     # phone
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "***-**-****"),                  # SSN
    (re.compile(r"\b(?:\d[ -]?){13,16}\b"), "****-****-****-****"),          # card
]


def _mask_pii(value: str) -> str:
    for pattern, replacement in _PII_RULES:
        value = pattern.sub(replacement, value)
    return value


def _sanitize_event(event_dict: dict[str, Any]) -> dict[str, Any]:
    """Walk all string values and mask PII before serialization."""
    sanitized: dict[str, Any] = {}
    for k, v in event_dict.items():
        if isinstance(v, str):
            sanitized[k] = _mask_pii(v)
        elif isinstance(v, dict):
            sanitized[k] = _sanitize_event(v)
        else:
            sanitized[k] = v
    return sanitized


# ── Configure structlog once at module load ────────────────────────────────────

os.makedirs(os.path.dirname(settings.AUDIT_LOG_FILE), exist_ok=True)

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.CallsiteParameterAdder(
            [structlog.processors.CallsiteParameter.FILENAME,
             structlog.processors.CallsiteParameter.LINENO]
        ),
        _sanitize_event,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(
        getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    ),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

_logger = structlog.get_logger("audit")


class AuditLogger:
    """Domain-level audit events. Each method maps to a specific security event."""

    @staticmethod
    def log_auth_attempt(
        *,
        username: str,
        success: bool,
        ip_address: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> None:
        _logger.info(
            "AUTH_ATTEMPT",
            username=username,
            success=success,
            ip_address=ip_address,
            reason=reason,
        )

    @staticmethod
    def log_token_validation(
        *,
        user_id: Optional[str],
        success: bool,
        jti: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> None:
        _logger.info(
            "TOKEN_VALIDATION",
            user_id=user_id,
            success=success,
            jti=jti,
            reason=reason,
        )

    @staticmethod
    def log_claim_query(
        *,
        user_id: str,
        session_id: str,
        query: str,
        intent: Optional[str] = None,
        claim_number: Optional[str] = None,
        success: bool = True,
        error: Optional[str] = None,
    ) -> None:
        _logger.info(
            "CLAIM_QUERY",
            user_id=user_id,
            session_id=session_id,
            query=query[:200],          # truncate to avoid log bloat
            intent=intent,
            claim_number=claim_number,
            success=success,
            error=error,
        )

    @staticmethod
    def log_security_event(
        *,
        event_type: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        details: Optional[str] = None,
        severity: str = "MEDIUM",       # LOW | MEDIUM | HIGH | CRITICAL
    ) -> None:
        _logger.warning(
            "SECURITY_EVENT",
            event_type=event_type,
            user_id=user_id,
            ip_address=ip_address,
            details=details,
            severity=severity,
        )

    @staticmethod
    def log_rate_limit(
        *,
        identifier: str,
        tier: str,
        ip_address: Optional[str] = None,
    ) -> None:
        _logger.warning(
            "RATE_LIMITED",
            identifier=identifier,
            tier=tier,
            ip_address=ip_address,
        )
