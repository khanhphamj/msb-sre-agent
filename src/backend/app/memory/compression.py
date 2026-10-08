"""Context compression — keeps the prompt under the token budget in long conversations.

Two-tier strategy:
1. Rolling summary (node `compress`, stored in state): when total tokens > CONTEXT_MAX_TOKENS, summarize
   the old messages (except the last CONTEXT_KEEP_LAST) into `state.summary`, then remove them from the
   checkpoint with RemoveMessage. Smaller checkpoint => faster AgentBase Memory reads/writes.
2. Hard trim (`fit_to_budget`, not persisted): right before calling the LLM, trim if still over
   CONTEXT_HARD_LIMIT_TOKENS (e.g. a huge tool output within the same turn).

Never cuts between an AIMessage(tool_calls) ↔ ToolMessage pair.
"""

from __future__ import annotations

from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)
from langchain_core.messages.utils import count_tokens_approximately

from app.config import Settings
from app.llm import get_llm
from app.observability import tracing

SUMMARY_PROMPT = """You are a conversation context compressor. Update the summary below with the
new messages. Keep: the user's goals, important facts/numbers, settled decisions,
important tool results, unfinished tasks. Drop greetings and unnecessary detail. Be concise, use bullet points,
and write in the same language as the conversation.

Current summary:
{summary}
"""


def _safe_cut_index(messages: list[AnyMessage], keep_last: int) -> int:
    cut = max(len(messages) - keep_last, 0)
    # Don't let the tail start with a ToolMessage (orphaned tool_call)
    while cut > 0 and isinstance(messages[cut], ToolMessage):
        cut -= 1
    return cut


TRUNCATION_MARKER = "\n…[truncated {n} chars to fit the context budget]…\n"


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head, tail = int(max_chars * 0.7), int(max_chars * 0.2)
    return text[:head] + TRUNCATION_MARKER.format(n=len(text) - head - tail) + text[-tail:]


def truncate_tool_outputs(messages: list[AnyMessage], max_tokens: int) -> list[AnyMessage]:
    """Truncate (keep head + tail) oversized ToolMessages — one huge tool output must not sink the whole turn."""
    max_chars = max(max_tokens, 1) * 4  # count_tokens_approximately ≈ 4 chars/token
    out: list[AnyMessage] = []
    for m in messages:
        if isinstance(m, ToolMessage) and isinstance(m.content, str) and len(m.content) > max_chars:
            m = m.model_copy(update={"content": _truncate_text(m.content, max_chars)})
        out.append(m)
    return out


def drop_placeholder_tool_messages(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Drop placeholder ToolMessages inserted by AgentBaseMemoryEvents ("…interrupted before completion.") when a
    real ToolMessage exists for the same tool_call_id (after HITL resume) — so the LLM doesn't see a fake error."""
    from app.hitl import is_placeholder_tool_message

    real = {
        m.tool_call_id
        for m in messages
        if isinstance(m, ToolMessage) and not is_placeholder_tool_message(m)
    }
    return [
        m
        for m in messages
        if not (
            isinstance(m, ToolMessage) and is_placeholder_tool_message(m) and m.tool_call_id in real
        )
    ]


async def summarize_if_needed(
    messages: list[AnyMessage], summary: str, settings: Settings
) -> dict | None:
    """Return a state update {summary, messages: [RemoveMessage...]} or None if not needed/not compressible.

    Summarizer errors must NOT break the chat turn (found in review: history larger than the small model's context ⇒
    400 forever): catch ⇒ skip compression this turn (WARNING); the hard trim before the LLM call still protects.
    """
    tokens_before = count_tokens_approximately(messages)
    meta = {
        "tokens_before": tokens_before,
        "threshold": settings.context_max_tokens,
        "keep_last": settings.context_keep_last,
        "message_count": len(messages),
    }
    with tracing.step("context.compress", metadata=meta) as st:
        if tokens_before <= settings.context_max_tokens:
            st.set(output={"decision": "skipped_under_budget"})
            return None
        cut = _safe_cut_index(messages, settings.context_keep_last)
        if cut == 0:
            st.set(output={"decision": "skipped_nothing_to_cut"}, level="WARNING")
            return None
        # The summarizer input must fit the budget too: truncate big tool outputs, then summarize only the
        # OLDEST part that fits and remove exactly that part — never delete messages the summarizer didn't see.
        # The rest stays in history and is compressed on a later turn.
        budget = settings.context_hard_limit_tokens
        old_in = truncate_tool_outputs(messages[:cut], max(budget // 4, 256))
        full_cut = cut
        while cut > 0 and count_tokens_approximately(old_in[:cut]) > budget:
            cut -= 1
        while cut > 0 and isinstance(
            messages[cut], ToolMessage
        ):  # keep tool_call ↔ result together
            cut -= 1
        if cut == 0:
            st.set(output={"decision": "skipped_oldest_message_over_budget"}, level="WARNING")
            return None
        old, old_in = messages[:cut], old_in[:cut]
        meta["partial"] = cut < full_cut
        try:
            result = await get_llm("summarize").ainvoke(
                [
                    SystemMessage(SUMMARY_PROMPT.format(summary=summary or "(empty)")),
                    *old_in,
                    HumanMessage("Return the updated summary."),
                ],
                config={"run_name": "context.summarize", "tags": ["compression"]},
            )
        except Exception as e:  # noqa: BLE001
            st.set(
                output={"decision": "skipped_summarizer_error"},
                level="WARNING",
                status_message=f"{type(e).__name__}: {str(e)[:200]}",
            )
            return None
        new_summary = str(result.content).strip()
        if not new_summary:
            st.set(output={"decision": "skipped_empty_summary"}, level="WARNING")
            return None
        tokens_after = count_tokens_approximately(messages[cut:]) + count_tokens_approximately(
            [SystemMessage(new_summary)]
        )
        st.set(
            output={"decision": "summarized", "summary": new_summary},
            metadata={
                **meta,
                "removed_messages": len(old),
                "tokens_after": tokens_after,
                "compression_ratio": round(tokens_after / max(tokens_before, 1), 3),
            },
        )
        return {"summary": new_summary, "messages": [RemoveMessage(id=m.id) for m in old if m.id]}


def fit_to_budget(messages: list[AnyMessage], settings: Settings) -> list[AnyMessage]:
    """Ensure the prompt is ≤ CONTEXT_HARD_LIMIT_TOKENS while NEVER losing the last HumanMessage.

    1) drop the bridge's placeholder ToolMessages; 2) truncate oversized tool outputs (keep head + tail);
    3) trim old messages; 4) if still over (current turn too large): system + truncated current turn;
    5) if the user's message alone is still too large: keep its head + tail.
    """
    limit = settings.context_hard_limit_tokens
    messages = drop_placeholder_tool_messages(messages)
    tokens = count_tokens_approximately(messages)
    if tokens <= limit:
        return messages
    work = truncate_tool_outputs(messages, max(limit // 4, 256))
    trimmed = work
    if count_tokens_approximately(work) > limit:
        trimmed = trim_messages(
            work,
            max_tokens=limit,
            token_counter=count_tokens_approximately,
            strategy="last",
            start_on="human",
            include_system=True,
            allow_partial=False,
        )
    if not any(isinstance(m, HumanMessage) for m in trimmed):
        # The current turn (from the last HumanMessage) alone exceeds the budget ⇒ keep it, truncate tool outputs hard
        system = [m for m in work[:1] if isinstance(m, SystemMessage)]
        last_h = max(i for i, m in enumerate(work) if isinstance(m, HumanMessage))
        turn = truncate_tool_outputs(
            work[last_h:], max(limit // (4 * max(len(work) - last_h, 1)), 64)
        )
        trimmed = system + turn
        over = count_tokens_approximately(trimmed) - limit
        human_i = next((i for i, m in enumerate(trimmed) if isinstance(m, HumanMessage)), None)
        if (
            over > 0
            and human_i is not None
            and isinstance(trimmed[human_i].content, str)
            # only when the user's message itself is the problem (not AI text / tool args in this turn)
            and count_tokens_approximately([trimmed[human_i]]) > limit // 2
        ):
            # The user's own message alone exceeds the budget (pasted document…) ⇒ keep head + tail of it,
            # otherwise the LLM call fails with a context-length 400 on every retry.
            h = trimmed[human_i]
            keep = max(len(h.content) - over * 4 - 400, 256)  # ≈4 chars/token + room for the marker
            trimmed[human_i] = h.model_copy(update={"content": _truncate_text(h.content, keep)})
    tracing.event(
        "context.hard_trim",
        level="WARNING",
        metadata={
            "tokens_before": tokens,
            "tokens_after": count_tokens_approximately(trimmed),
            "limit": limit,
            "dropped_messages": len(messages) - len(trimmed),
        },
    )
    return trimmed
