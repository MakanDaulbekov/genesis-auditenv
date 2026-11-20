"""Оркестратор генерации: мир -> рендер «до» -> мутации -> рендер «после» -> truth/provenance.

Ключ эпизода = SHA-256(версия генератора, секрет оценщика, seed, канонический config).
Секрет (AUDITENV_GRADER_SECRET) задаётся организаторами только в процессе оценщика на
закрытом прогоне: без него маленькие seed можно перебрать открытым генератором.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from ..common.drbg import DRBG, derive_key
from .mutations import ERROR_TYPES, MUTATIONS, SCENARIOS
from .render import canonical_json, json_bytes, public_files, render
from .truth import build_truth, diff_tables
from .world import GENERATOR_VERSION, GenerationError, _rand_range, build_world, post_all

MAX_ISSUES = 12
MAX_ATTEMPTS = 25


@dataclass
class Episode:
    episode_id: str
    public: dict[str, bytes]     # имя файла -> байты (видит среда)
    private: dict[str, bytes]    # truth.json, provenance.json (только оценщик)
    tables: dict                 # открытые таблицы как объекты
    truth: dict
    provenance: dict

    def __repr__(self) -> str:   # никогда не печатаем закрытое содержимое
        return f"<Episode {self.episode_id}>"


def episode_key(seed: int, cfg: dict, secret: str = "") -> bytes:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be int")
    return derive_key("auditenv", GENERATOR_VERSION, secret, seed, canonical_json(cfg))


def _plan(rng: DRBG, cfg: dict) -> tuple[list[str], list[str], int]:
    """-> (сценарии, одиночные ошибки, сколько одиночных отброшено сверх MAX_ISSUES)."""
    if cfg.get("clean"):
        return [], [], 0
    scen = []
    for name in sorted(cfg.get("interactions", {})):
        scen += [name] * _rand_range(rng, cfg["interactions"][name])
    singles = []
    for t in ERROR_TYPES:
        singles += [t] * _rand_range(rng, cfg["errors"].get(t, 0))
    # не более MAX_ISSUES: сценарий даёт 2 issue. Если одиночных больше, чем помещается, они урезаются
    # ПОРОВНУ по типам (по кругу), чтобы ни один заказанный тип не пропал целиком; иначе порядок прежний.
    room = MAX_ISSUES - 2 * len(scen)
    if room < 0:
        raise GenerationError("too many scenarios")
    if len(singles) <= room:
        return scen, singles, 0
    left = {t: singles.count(t) for t in ERROR_TYPES}
    kept = {t: 0 for t in ERROR_TYPES}
    while sum(kept.values()) < room:
        for t in ERROR_TYPES:
            if left[t] and sum(kept.values()) < room:
                left[t] -= 1
                kept[t] += 1
    return scen, [t for t in ERROR_TYPES for _ in range(kept[t])], len(singles) - room


def _attempt(key: bytes, cfg: dict) -> tuple[dict, dict, dict, list, list, dict]:
    world = build_world(key, cfg)
    before_entries = post_all(world)
    before = render(world, before_entries)
    rng = DRBG(derive_key(key, "mutations"))
    scen, singles, dropped = _plan(rng, cfg)
    log: list[dict] = []
    scenarios: list[dict] = []
    for i, name in enumerate(scen):
        group = f"G{i + 1}"
        entries = SCENARIOS[name](world, rng, group)
        log += entries
        scenarios.append({"group": group, "scenario": name})
    for t in singles:
        log.append(MUTATIONS[t](world, rng, {}))
    if len(log) > MAX_ISSUES:
        raise GenerationError("too many issues")
    after_entries = post_all(world)
    after = render(world, after_entries)
    plan = {"planned_scenarios": scen, "planned_singles": singles}
    if dropped:
        plan["dropped_over_max_issues"] = dropped  # записано честно: config просил больше 12 ошибок
    return world, before, after, log, scenarios, plan


def generate_episode(seed: int, cfg: dict, secret: str = "") -> Episode:
    key0 = episode_key(seed, cfg, secret)
    key = key0
    last_err = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            world, before, after, log, scenarios, plan = _attempt(key, cfg)
            break
        except GenerationError as e:  # детерминированный повтор с производным ключом
            last_err = e
            key = derive_key(key0, "retry", attempt)
    else:
        raise GenerationError(f"generation failed after {MAX_ATTEMPTS} attempts: {last_err}")

    truth = build_truth(key, world["episode_id"], log, scenarios)
    pub = public_files(world, after, cfg["config_version"])
    diff = diff_tables(before, after)
    provenance = {
        "schema_version": truth["schema_version"],
        "episode_id": world["episode_id"],
        "generator_version": GENERATOR_VERSION,
        "config_version": cfg["config_version"],
        "seed": seed,
        "secret_used": bool(secret),
        "attempt": attempt,
        "config": cfg,
        "plan": plan,
        "mutations": log,
        "scenarios": scenarios,
        "distractors": world["distractors"],
        "diff": diff,
    }
    private = {"truth.json": json_bytes(truth), "provenance.json": json_bytes(provenance)}
    return Episode(world["episode_id"], pub, private, after, truth, provenance)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
