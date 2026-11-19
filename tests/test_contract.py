"""Контракт reset/step, схемы, лимиты, завершение, неверный ввод (раздел 04)."""
import json

import pytest

from auditenv.common.jsonschema_lite import check
from auditenv.env.core import AuditEnv
from auditenv.env.calculator import CalcError, evaluate
from conftest import act


def _assert_obs(obs):
    assert check(obs, "observation") == []
    json.dumps(obs)


def test_reset_signature_and_first_observation(env, cfg):
    out = env.reset(seed=5, config=cfg)
    assert isinstance(out, tuple) and len(out) == 2
    obs, info = out
    _assert_obs(obs)
    assert info == {}
    assert obs["steps_remaining"] == 100
    res = obs["result"]
    assert set(res) == {"task", "manifest", "policy", "tools", "budget"}
    assert check(res["manifest"], "manifest") == []
    assert check(res["policy"], "policy") == []
    assert res["budget"]["max_steps"] == 100


def test_step_signature_and_json(env, cfg):
    env.reset(seed=5, config=cfg)
    out = env.step(act("list_records", table="invoices", limit=5))
    assert len(out) == 5
    obs, reward, terminated, truncated, info = out
    _assert_obs(obs)
    assert reward == 0.0 and terminated is False and truncated is False and info == {}
    assert len(obs["result"]["records"]) == 5
    for r in obs["result"]["records"]:
        assert check(r, "invoice") == []
    assert obs["steps_remaining"] == 99


@pytest.mark.parametrize("limit", [0, 51, -1, "5", True, 5.0, None])
def test_list_limit_bounds(env, cfg, limit):
    env.reset(seed=5, config=cfg)
    obs, *_ = env.step(act("list_records", table="ledger", limit=limit))
    assert obs["error"]["code"] in ("limit_out_of_range",)
    assert obs["steps_remaining"] == 99  # ошибка расходует шаг


def test_pagination_complete_and_cursors_stable(env, grader, cfg):
    obs, _ = env.reset(seed=11, config=cfg)
    n = obs["result"]["manifest"]["counts"]["invoices"]
    seen, cursors, cur = [], [], None
    while True:
        o, *_ = env.step(act("list_records", table="invoices", cursor=cur, limit=7))
        assert o["error"] is None
        seen += [r["invoice_id"] for r in o["result"]["records"]]
        cur = o["result"]["next_cursor"]
        if cur is None:
            break
        cursors.append(cur)
    assert len(seen) == n == len(set(seen))
    env2 = AuditEnv(grader, mode="train")
    env2.reset(seed=11, config=cfg)
    o2, *_ = env2.step(act("list_records", table="invoices", cursor=cursors[0], limit=7))
    assert [r["invoice_id"] for r in o2["result"]["records"]] == seen[7:14]
    # курсор непрозрачен: в нём нет ID записей
    assert all(s not in cursors[0] for s in seen)


def test_cursor_tampering_and_cross_table(env, cfg):
    env.reset(seed=11, config=cfg)
    o, *_ = env.step(act("list_records", table="ledger", limit=10))
    cur = o["result"]["next_cursor"]
    for bad in [cur + "x", cur[:-2], "AAAA", "", 123, "../../etc/passwd"]:
        o, *_ = env.step(act("list_records", table="ledger", cursor=bad, limit=10))
        assert o["error"]["code"] in ("invalid_cursor", "invalid_arguments")
    o, *_ = env.step(act("list_records", table="invoices", cursor=cur, limit=10))
    assert o["error"]["code"] == "invalid_cursor"


def test_unknown_ids_tables_actions(env, cfg):
    env.reset(seed=5, config=cfg)
    cases = [
        (act("get_record", table="ledger", id="R-NOPE"), "not_found"),
        (act("get_record", table="truth", id="x"), "unknown_table"),
        (act("list_records", table="provenance"), "unknown_table"),
        (act("get_document", doc_id="D-NOPE"), "not_found"),
        (act("shell", cmd="ls"), "unknown_action"),
        (act("get_record", table="ledger"), "invalid_arguments"),
        (act("get_record", table="ledger", id="x", extra=1), "invalid_arguments"),
        ({"action": "get_record"}, "invalid_arguments"),
        ({"args": {}}, "invalid_action_format"),
        ({"action": "list_records", "args": {"table": "fx"}, "x": 1}, "invalid_action_format"),
        ("{not json", "invalid_json"),
        ("[1,2]", "invalid_action_format"),
        (42, "invalid_action_format"),
        ({"action": "list_records", "args": []}, "invalid_arguments"),
    ]
    steps = 100
    for a, code in cases:
        obs, r, term, trunc, info = env.step(a)
        steps -= 1
        assert obs["error"]["code"] == code, (a, obs)
        assert obs["result"] is None
        assert obs["steps_remaining"] == steps
        assert r == 0.0 and not term and not trunc and info == {}
        msg = obs["error"]["message"]
        assert len(msg) <= 200 and "/" not in msg.replace("{\"action\": str, \"args\": object}", "") \
            and "Traceback" not in msg and "R-NOPE" not in msg


def test_get_record_aliases_and_document(env, cfg):
    obs, _ = env.reset(seed=5, config=cfg)
    o, *_ = env.step(act("list_records", table="payments", limit=1))
    p = o["result"]["records"][0]
    o1, *_ = env.step(act("get_record", table="payments", id=p["payment_id"]))
    o2, *_ = env.step(act("get_record", table="payments", id=p["doc_id"]))
    assert o1["result"]["record"] == o2["result"]["record"] == p
    o3, *_ = env.step(act("get_document", doc_id=p["doc_id"]))
    assert check(o3["result"]["document"], "document") == []
    assert o3["result"]["document"]["record"] == p
    o, *_ = env.step(act("list_records", table="fx", limit=1))
    fx = o["result"]["records"][0]
    fid = f"fx:{fx['date']}:{fx['source_currency']}:{fx['target_currency']}"
    o4, *_ = env.step(act("get_record", table="fx", id=fid))
    assert o4["result"]["record"] == fx


def test_returned_records_are_copies(env, cfg):
    env.reset(seed=5, config=cfg)
    o, *_ = env.step(act("list_records", table="ledger", limit=1))
    r = o["result"]["records"][0]
    rid = r["row_id"]
    r["debit_minor"] = 10**12
    o2, *_ = env.step(act("get_record", table="ledger", id=rid))
    assert o2["result"]["record"]["debit_minor"] != 10**12


def test_calculator_decimal_exact():
    assert evaluate("0.1 + 0.2") == "0.3"
    assert evaluate("(1000.50 - 0.5) * 3") == "3000"
    assert evaluate("-2 * -3") == "6"
    assert evaluate("123456789012345678901234567890 / 10") == "12345678901234567890123456789"
    assert evaluate("1 / 3").startswith("0.33333333333333333333333333333333")


@pytest.mark.parametrize("expr", [
    "__import__('os').system('ls')", "open('/etc/passwd')", "a + 1", "(1).real", "2 ** 10",
    "10**10**10", "9 // 2", "7 % 2", "1e999", "1_000", "[1]", "lambda: 1", "1 if 1 else 2", "1 < 2",
    "'a'", "1 / 0", "(" * 200 + "1" + ")" * 200, "1+" * 100 + "1", "9" * 40, "", "1;2", "0x10",
    "99999999999999999999999999999 * 99999999999999999999999999999 * 99999999999999999999", "True + 1",
])
def test_calculator_rejects(expr):
    with pytest.raises(CalcError):
        evaluate(expr)


def test_calculator_tool_error_is_bounded(env, cfg):
    env.reset(seed=5, config=cfg)
    o, *_ = env.step(act("calculator", expression="__import__('os')"))
    assert o["error"]["code"] == "calc_error"
    o, *_ = env.step(act("calculator", expression=12))
    assert o["error"]["code"] == "calc_error"


def test_truncation_at_100_steps_with_empty_answer(env, clean_cfg, cfg):
    env.reset(seed=3, config=clean_cfg)
    for i in range(99):
        o, r, term, trunc, info = env.step(act("calculator", expression="1+1"))
        assert r == 0.0 and not term and not trunc
    o, r, term, trunc, info = env.step(act("calculator", expression="1+1"))
    assert trunc is True and term is False and o["steps_remaining"] == 0
    # чистые книги + пустой ответ: F1=1, штраф 100 шагов -> 0.8
    assert r == pytest.approx(0.8)
    assert info["metrics"]["inspection_steps"] == 100
    o, r, term, trunc, info = env.step(act("calculator", expression="1+1"))
    assert o["error"]["code"] == "episode_finished" and r == 0.0


def test_submit_as_100th_step_terminates(env, clean_cfg):
    obs, _ = env.reset(seed=3, config=clean_cfg)
    eid = obs["result"]["manifest"]["episode_id"]
    for _ in range(99):
        env.step(act("calculator", expression="1"))
    o, r, term, trunc, info = env.step(act("submit", findings={"episode_id": eid, "findings": []}))
    assert term is True and trunc is False
    assert info["metrics"]["inspection_steps"] == 99
    assert r == pytest.approx(1 - 0.002 * 99)


def test_resubmit_does_not_change_grade(env, grader, cfg):
    obs, _ = env.reset(seed=8, config=cfg)
    eid = obs["result"]["manifest"]["episode_id"]
    o, r1, term, trunc, info = env.step(act("submit", findings={"episode_id": eid, "findings": []}))
    assert term
    o, r2, term2, trunc2, info2 = env.step(act("submit", findings={"episode_id": eid, "findings": []}))
    assert o["error"]["code"] == "episode_finished" and r2 == 0.0 and term2 and info2 == {}
    # прямой повторный grade по токену отклоняется оценщиком
    assert grader.call("grade", token=env._token, findings={}, inspection_steps=0, reason="x")["error"] == "already_graded"


def test_eval_mode_hides_reward(grader, cfg):
    env = AuditEnv(grader, mode="eval")
    obs, info = env.reset(seed=8, config=cfg)
    assert info == {}
    eid = obs["result"]["manifest"]["episode_id"]
    o, r, term, trunc, info = env.step(act("submit", findings={"episode_id": eid, "findings": []}))
    assert r == 0.0 and info == {} and term
    assert o == {"result": {"status": "submitted"}, "error": None, "steps_remaining": 99}


def test_time_limit_truncates(grader, clean_cfg):
    t = [0.0]
    env = AuditEnv(grader, mode="train", clock=lambda: t[0])
    env.reset(seed=3, config=clean_cfg)
    env.step(act("calculator", expression="1"))
    t[0] = 61.0
    o, r, term, trunc, info = env.step(act("calculator", expression="1"))
    assert trunc and not term and o["result"] is None
    assert info["metrics"]["inspection_steps"] == 1


def test_step_before_reset(grader):
    env = AuditEnv(grader, mode="train")
    o, r, term, trunc, info = env.step(act("calculator", expression="1"))
    assert o["error"]["code"] == "not_reset"


def test_reset_isolation(grader, cfg):
    a = AuditEnv(grader, mode="train")
    b = AuditEnv(grader, mode="train")
    oa, _ = a.reset(seed=1, config=cfg)
    ob, _ = b.reset(seed=2, config=cfg)
    for _ in range(5):
        a.step(act("calculator", expression="1"))
    assert b.step(act("calculator", expression="1"))[0]["steps_remaining"] == 99
    oa2, _ = a.reset(seed=1, config=cfg)
    assert oa2 == oa  # тот же seed -> то же первое наблюдение, счётчики сброшены
    assert a.step(act("calculator", expression="1"))[0]["steps_remaining"] == 99


def test_submit_bad_args_is_invalid_structure(env, cfg):
    env.reset(seed=8, config=cfg)
    o, r, term, trunc, info = env.step(act("submit", findings=None))
    assert term and r == -1.0 and info["metrics"]["valid"] is False


def test_invalid_findings_logged_by_grader(grader, cfg):
    env = AuditEnv(grader, mode="train")
    env.reset(seed=9, config=cfg)
    env.step(act("submit", findings={"episode_id": "wrong", "findings": []}))
    lines = [json.loads(l) for l in grader.log_path.read_text(encoding="utf-8").splitlines()]
    assert any(l["event"] == "invalid_findings" and l["error"] == "episode_id mismatch" for l in lines)
