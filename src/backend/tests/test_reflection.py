from __future__ import annotations

from greennode_agentbase import RequestContext
from langchain_core.messages import AIMessage

from app import service
from app.config import get_settings
from evals.evaluators import expected_tools, item_passed, no_error


async def test_reflection_retries_and_removes_bad_answer(fake_llm, monkeypatch):
    monkeypatch.setenv("REFLECTION_ENABLED", "true")
    monkeypatch.setenv("REFLECTION_MAX_RETRIES", "1")
    get_settings.cache_clear()
    model = fake_llm(
        AIMessage("bad answer"),
        AIMessage('{"pass": false, "score": 0.2, "critique": "missing figures"}'),
        AIMessage("good answer"),
        AIMessage('{"pass": true, "score": 0.9, "critique": ""}'),
    )
    out = await service.handle(
        {"message": "question"}, RequestContext(session_id="r1", user_id="u1")
    )
    assert out["response"] == "good answer"
    # The 2nd agent call receives the critique in the system prompt
    assert "missing figures" in model.calls[2][0].content
    # The bad answer has been removed from history
    assert not any(getattr(m, "content", "") == "bad answer" for m in model.calls[-1])


def test_evaluators():
    out = {"status": "success", "response": "x", "tools_used": ["get_current_time"]}
    evals = [
        no_error(output=out),
        expected_tools(output=out, metadata={"expected_tools": ["get_current_time"]}),
    ]
    assert item_passed(evals)
    bad = expected_tools(output=out, metadata={"expected_tools": ["remember"]})
    assert bad.value == 0.0


def test_judge_output_parsed_defensively():
    from app.reflection import _parse

    assert _parse('{"pass": "false", "score": 0.2, "critique": "missing"}')["pass"] is False
    assert _parse('{"pass": false, "score": null}')["score"] == 1.0  # null ⇒ default, no crash
    assert _parse("not json")["pass"] is True
    assert _parse('{"score": "abc"}')["score"] == 1.0
    assert _parse('{"score": 7}')["score"] == 1.0  # clamped to [0, 1]


async def test_fail_verdict_without_critique_still_retries(fake_llm, monkeypatch):
    """Judge says fail but gives no critique ⇒ must retry, never end the turn with an empty reply."""
    monkeypatch.setenv("REFLECTION_ENABLED", "true")
    monkeypatch.setenv("REFLECTION_MAX_RETRIES", "1")
    get_settings.cache_clear()
    fake_llm(
        AIMessage("bad answer"),
        AIMessage('{"pass": false, "score": 0.4}'),  # no critique
        AIMessage("good answer"),
        AIMessage('{"pass": true, "score": 0.9}'),
    )
    out = await service.handle({"message": "q"}, RequestContext(session_id="r2", user_id="u1"))
    assert out["response"] == "good answer"


# --- eval CI gate: crashing items / judges must count as failures, never vanish
async def test_eval_task_never_raises(monkeypatch):
    from evals import run_eval

    async def boom(*a, **k):
        raise ConnectionError("LLM 502")

    monkeypatch.setattr(run_eval, "run_chat", boom)
    out = await run_eval.make_task("approve")(item={"input": {"message": "hi"}})
    assert out["status"] == "error" and "LLM 502" in out["error"]
    assert no_error(output=out).value == 0.0


async def test_judge_error_scores_zero_instead_of_disappearing(monkeypatch):
    from evals import evaluators

    class _Broken:
        async def ainvoke(self, *a, **k):
            raise TimeoutError("judge down")

    monkeypatch.setattr(evaluators, "get_llm", lambda task: _Broken())
    ev = await evaluators.llm_judge_correctness(
        input={"message": "q"}, output={"response": "a"}, expected_output="a"
    )
    assert ev.name == "correctness" and ev.value == 0.0
    ok = no_error(output={"status": "success", "response": "a"})
    assert not item_passed([ok, ev])


def test_eval_config_error_exit_code(monkeypatch):
    import sys

    from evals import run_eval

    def bad_settings():
        raise ValueError("LLM_MODEL and LLM_API_KEY are required")

    monkeypatch.setattr(run_eval, "get_settings", bad_settings)
    monkeypatch.setattr(sys, "argv", ["run_eval", "--data", "x.jsonl"])
    assert run_eval.main() == run_eval.EXIT_CONFIG_ERROR


def test_gate_fails_when_langfuse_drops_crashed_items(monkeypatch):
    """Langfuse returns results only for the items that did not raise and computes pass_rate over THOSE
    (here 1/1 = 100%). The CI gate must divide by the dataset size: 1 passed of 4 ⇒ 25% < 50% ⇒ exit 1."""
    import sys
    from types import SimpleNamespace

    from langfuse import Evaluation

    from evals import run_eval

    passed = SimpleNamespace(evaluations=[no_error(output={"status": "success", "response": "x"})])
    result = SimpleNamespace(
        item_results=[passed],
        run_evaluations=[Evaluation(name="pass_rate", value=1.0)],
        format=lambda: "",
    )
    dataset = SimpleNamespace(items=[object()] * 4, run_experiment=lambda **kw: result)
    client = SimpleNamespace(get_dataset=lambda name: dataset)
    monkeypatch.setattr(run_eval.tracing, "init_tracing", lambda s: None)
    monkeypatch.setattr(run_eval.tracing, "get_client", lambda: client)
    monkeypatch.setattr(run_eval.tracing, "shutdown_tracing", lambda: None)
    monkeypatch.setattr(sys, "argv", ["run_eval", "--dataset", "d", "--min-pass-rate", "0.5"])
    assert run_eval.main() == run_eval.EXIT_GATE_FAILED
