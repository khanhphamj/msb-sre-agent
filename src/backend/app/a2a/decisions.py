"""Approve/reject words of the A2A HITL protocol — shared by the server (parses them) and the client (refuses to
relay them unless a human approved the tool call). Kept dependency-free to avoid import cycles."""

from __future__ import annotations

APPROVE_WORDS = {"approve", "approved", "yes", "ok", "đồng ý", "duyệt"}
REJECT_WORDS = {"reject", "rejected", "no", "từ chối", "không"}


def parse_decision(text: str) -> tuple[bool, str] | None:
    """'approve' / 'reject: reason' ⇒ (approve?, reason); anything else ⇒ None (normal chat)."""
    t = text.strip()
    if t.lower() in APPROVE_WORDS:
        return True, ""
    head, _, reason = t.partition(":")
    if head.strip().lower() in REJECT_WORDS:
        return False, reason.strip()
    return None
