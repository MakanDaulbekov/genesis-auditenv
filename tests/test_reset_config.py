"""P0.2: reset(seed) без config / с частичным config; понятные ошибки битого config."""
import json

import pytest

from auditenv.env.core import AuditEnv, EpisodeOpenError
from auditenv.generator.config import ConfigError, load_default, resolve_config, validate_config
from conftest import act


def manifest(obs):
    return obs["result"]["manifest"]


def test_reset_seed_only(grader, cfg):
    env = AuditEnv(grader, "train")
    a, _ = env.reset(seed=12345)
    b, _ = AuditEnv(grader, "train").reset(seed=12345, config=None)
    c, _ = AuditEnv(grader, "train").reset(seed=12345, config={})
    d, _ = AuditEnv(grader, "train").reset(seed=12345, config=cfg)  # полный дефолт = то же самое
    assert a == b == c == d
    assert manifest(a)["config_version"] == load_default()["config_version"] == "default-1.0.0"


def test_reset_partial_config_merges(grader):
    env = AuditEnv(grader, "train")
    obs, _ = env.reset(seed=12345, config={"clean": True})
    o, r, *_ = env.step(act("submit", findings={"episode_id": manifest(obs)["episode_id"], "findings": []}))
    assert r == 1.0  # чистая компания: пустой ответ — максимум
    obs2, _ = AuditEnv(grader, "train").reset(seed=12345, config={"currencies": {"base": ["JPY"]}})
    assert manifest(obs2)["base_currency"] == "JPY" and manifest(obs2)["currency_exponents"]["JPY"] == 0


def test_reset_without_seed_is_deterministic(grader):
    e1, e2 = AuditEnv(grader, "train"), AuditEnv(grader, "train")
    assert e1.reset()[0] == e2.reset()[0]          # оба — авто-seed 0
    assert manifest(e1.reset(seed=0)[0]) == manifest(AuditEnv(grader, "train").reset()[0])


@pytest.mark.parametrize("bad,needle", [
    ({"errors": {"fx_wrong_date_or_rate": "2"}}, "config.errors.fx_wrong_date_or_rate"),
    ({"errors": {"typo_error": 1}}, "config.errors.typo_error"),
    ({"size": {"n_invoices": [50, 10]}}, "config.size.n_invoices"),
    ({"currencies": {"base": ["XYZ"]}}, "config.currencies.base"),
    ({"clean": "yes"}, "config.clean"),
    ({"unknown_section": 1}, "неизвестные поля"),
    ({"errors": {"duplicate_invoice": 13}}, "config.errors.duplicate_invoice"),
    ({"interactions": {"partial_fx_boundary": 4}}, "config.interactions.partial_fx_boundary"),
])
def test_bad_config_has_clear_reason(grader, bad, needle):
    with pytest.raises(EpisodeOpenError) as ei:
        AuditEnv(grader, "train").reset(seed=1, config=bad)
    msg = str(ei.value)
    assert needle in msg and "Traceback" not in msg and "/home" not in msg and ".py" not in msg


def test_bad_seed_has_clear_reason(grader):
    for s in ("7", 1.5, True):
        with pytest.raises(EpisodeOpenError) as ei:
            AuditEnv(grader, "train").reset(seed=s)
        assert "seed" in str(ei.value)


def test_split_configs_are_valid():
    from auditenv.cli import split_configs
    for f in ("configs/splits/train.json", "configs/splits/dev.json"):
        for _, c in split_configs(f)[1]:
            validate_config(c)
    validate_config(load_default())
    assert resolve_config({"clean": True})["clean"] is True
    with pytest.raises(ConfigError):
        resolve_config([1, 2])


def test_infeasible_config_reports_generation_failure(grader):
    # Сценарий «частичный платёж в ДРУГОЙ валюте» невозможен, если иностранных валют нет вовсе.
    # (Прежний пример 3+3 сценария при 0% иностранных клиентов теперь выполним: генератор добирает
    # кандидатов — см. tests/test_robustness.py::test_extreme_configs_generate_and_validate.)
    with pytest.raises(EpisodeOpenError) as ei:
        AuditEnv(grader, "train").reset(seed=1, config={"interactions": {"partial_fx_boundary": 1, "duplicate_cutoff": 0},
                                                        "currencies": {"base": ["USD"], "foreign": ["USD"]}})
    assert "не удалось построить эпизод" in str(ei.value)
