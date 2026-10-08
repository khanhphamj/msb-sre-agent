"""Per-user isolation — every data path: short-term, long-term, HITL, feedback, user_id."""

from __future__ import annotations

import pytest
from greennode_agentbase import RequestContext
from greennode_agentbase.exceptions import GreenNodeRequestError
from langchain_core.messages import AIMessage

from app import service
from app.auth.inbound import validate_user_id
from app.config import get_settings


def ctx(user: str, session: str = "shared-session") -> RequestContext:
    return RequestContext(session_id=session, user_id=user)


async def test_short_term_same_session_id_different_users(fake_llm):
    fake_llm(AIMessage("ok"))
    await service.handle({"message": "my PIN is 9999"}, ctx("alice"))
    model = fake_llm(AIMessage("ok"))
    await service.handle({"message": "what was the PIN again?"}, ctx("bob"))
    assert not any("9999" in str(m.content) for m in model.calls[-1])


async def test_long_term_facts_not_shared(fake_llm):
    fake_llm(
        AIMessage(
            "",
            tool_calls=[
                {"name": "remember", "args": {"fact": "Alice is allergic to shrimp"}, "id": "r1"}
            ],
        ),
        AIMessage("noted"),
    )
    await service.handle({"message": "remember: I am allergic to shrimp"}, ctx("alice", "s-a"))
    model = fake_llm(AIMessage("I don't know"))
    await service.handle({"message": "am I allergic to shrimp?"}, ctx("bob", "s-b"))
    system = model.calls[-1][0].content
    assert "allergic to shrimp" not in system  # bob's auto-recall doesn't see alice's fact


async def test_hitl_cannot_be_resumed_by_other_user(fake_llm, monkeypatch):
    monkeypatch.setenv("HITL_TOOLS", '["remember"]')
    get_settings.cache_clear()
    fake_llm(AIMessage("", tool_calls=[{"name": "remember", "args": {"fact": "x"}, "id": "c1"}]))
    out = await service.handle({"message": "remember x"}, ctx("alice", "hitl-s"))
    assert out["status"] == "interrupted"
    with pytest.raises(GreenNodeRequestError) as e:  # bob uses alice's exact session id
        await service.handle(
            {"type": "resume", "decisions": [{"tool_call_id": "c1", "action": "approve"}]},
            ctx("bob", "hitl-s"),
        )
    assert e.value.status_code == 409


async def test_feedback_token_bound_to_user(fake_llm):
    token = service.feedback_token("alice", "trace-1")
    ok = await service.handle(
        {"type": "feedback", "trace_id": "trace-1", "feedback_token": token, "score": 1},
        ctx("alice"),
    )
    assert ok["status"] in ("success", "ignored")
    for payload in (
        {"type": "feedback", "trace_id": "trace-1", "feedback_token": token, "score": -1},
        {"type": "feedback", "trace_id": "trace-1", "score": -1},
    ):
        with pytest.raises(GreenNodeRequestError) as e:
            await service.handle(payload, ctx("bob"))
        assert e.value.status_code == 403


@pytest.mark.parametrize("bad", ["a/b", "../alice", "alice/../bob", "x y", "", "/root", "a" * 200])
def test_invalid_user_ids_rejected(bad):
    with pytest.raises(GreenNodeRequestError):
        validate_user_id(bad)


@pytest.mark.parametrize("good", ["alice", "user-123", "1910e858-f6aa-4195", "a.b@vng.com.vn"])
def test_valid_user_ids(good):
    assert validate_user_id(good) == good


async def test_namespace_injection_blocked_end_to_end(fake_llm):
    fake_llm(AIMessage("ok"))
    with pytest.raises(GreenNodeRequestError) as e:
        await service.handle({"message": "hi"}, ctx("alice/../bob"))
    assert e.value.status_code == 400


@pytest.mark.parametrize(
    "bad_session",
    ["../../victim/sessions/s1", "a/b", "s_x", "a::b", "..", "x%2F..", "s?x=1", "", "a" * 200],
)
async def test_session_path_traversal_blocked(fake_llm, bad_session):
    fake_llm(AIMessage("ok"))
    with pytest.raises(GreenNodeRequestError) as e:
        await service.handle({"message": "hi"}, ctx("alice", bad_session))
    assert e.value.status_code == 400
    with pytest.raises(GreenNodeRequestError):
        await service.run_chat("hi", user_id="alice", session_id=bad_session)


def test_unsafe_idp_subject_mapped_not_rejected():
    from app.auth.inbound import user_id_from_subject, validate_user_id

    mapped = user_id_from_subject("auth0|abc123", "https://tenant.auth0.com/")
    assert mapped.startswith("u-") and validate_user_id(mapped) == mapped
    assert mapped == user_id_from_subject("auth0|abc123", "https://tenant.auth0.com/")  # stable
    assert mapped != user_id_from_subject(
        "auth0|abc123", "https://other/"
    )  # different IdP ⇒ different user
    assert user_id_from_subject("alice", "x") == "alice"


async def test_double_resume_runs_approved_tool_once(fake_llm, monkeypatch):
    import asyncio

    monkeypatch.setenv("HITL_TOOLS", '["remember"]')
    get_settings.cache_clear()
    runs = []
    import app.memory.long_term as lt

    orig_save = lt.InMemoryLTM.save

    async def counting_save(self, user_id, fact):
        runs.append(fact)
        await asyncio.sleep(0.05)
        return await orig_save(self, user_id, fact)

    monkeypatch.setattr(lt.InMemoryLTM, "save", counting_save)
    fake_llm(
        AIMessage("", tool_calls=[{"name": "remember", "args": {"fact": "x"}, "id": "c1"}]),
        AIMessage("done"),
    )
    out = await service.handle({"message": "remember x"}, ctx("alice", "dbl-1"))
    assert out["status"] == "interrupted"
    body = {"type": "resume", "decisions": [{"tool_call_id": "c1", "action": "approve"}]}
    results = await asyncio.gather(
        service.handle(body, ctx("alice", "dbl-1")),
        service.handle(body, ctx("alice", "dbl-1")),
        return_exceptions=True,
    )
    assert runs == ["x"]  # side-effecting tool runs only once
    codes = sorted(getattr(r, "status_code", 200) for r in results)
    assert codes == [200, 409]
