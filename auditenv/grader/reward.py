"""Общая терминальная награда раздела 05. Вычисляется точно (Fraction), наружу — float.

reward = clip(2*F1 - 1 - 0.002*inspection_steps, -1, 1)
P = TP/(TP+FP), R = TP/(TP+FN), F1 = 2TP/(2TP+FP+FN); нулевой знаменатель -> 0;
эталон и выводы пусты -> F1 = 1; эталон пуст, выводы есть -> F1 = 0.
Неверная структура -> reward = -1 (метрики не считаются).
inspection_steps = число шагов эпизода до submit (все действия, включая ошибочные, кроме
самого submit). При truncation submit не было, inspection_steps = число сделанных шагов.
"""
from __future__ import annotations

from fractions import Fraction

from .matcher import InvalidFindings, max_matching, normalize

STEP_COST = Fraction(2, 1000)


def _div(a: int, b: int) -> Fraction:
    return Fraction(0) if b == 0 else Fraction(a, b)


def metrics(tp: int, fp: int, fn: int) -> dict:
    p = _div(tp, tp + fp)
    r = _div(tp, tp + fn)
    if tp + fp + fn == 0:
        f1 = Fraction(1)
    else:
        f1 = _div(2 * tp, 2 * tp + fp + fn)
    return {"precision": p, "recall": r, "f1": f1}


def shared_reward(f1: Fraction, inspection_steps: int) -> Fraction:
    r = 2 * f1 - 1 - STEP_COST * inspection_steps
    return max(Fraction(-1), min(Fraction(1), r))


TYPES = ("duplicate_invoice", "missing_or_partial_allocation", "fx_wrong_date_or_rate", "period_cutoff_error")


def _by_type(findings, issues, pairs) -> dict:
    """Диагностика по типам (только для журнала/отчёта оценщика, не для политики)."""
    out = {t: {"tp": 0, "fp": 0, "fn": 0} for t in TYPES}
    matched_f = {fi for fi, _ in pairs}
    matched_t = {iid for _, iid in pairs}
    for f in findings:
        out.setdefault(f.type if f.type in TYPES else "unknown_type", {"tp": 0, "fp": 0, "fn": 0})
        out[f.type if f.type in TYPES else "unknown_type"]["tp" if f.index in matched_f else "fp"] += 1
    for t in issues:
        if t["issue_id"] not in matched_t:
            out[t["type"]]["fn"] += 1
    return out


def grade(raw_findings, truth: dict, alias: dict, tolerance: int, inspection_steps: int,
          known_ids: frozenset | None = None) -> dict:
    """Полная оценка одного эпизода. Возвращает только JSON-совместимые значения.
    known_ids — все ID открытых записей эпизода (matcher.public_ids): несуществующий evidence_id -> FP."""
    if isinstance(inspection_steps, bool) or not isinstance(inspection_steps, int) or inspection_steps < 0:
        raise ValueError("inspection_steps must be a non-negative int")
    try:
        findings = normalize(raw_findings, truth["episode_id"], alias, known_ids)
    except InvalidFindings as e:
        return {"valid": False, "error": str(e), "reward": -1.0, "reward_exact": "-1",
                "tp": 0, "fp": 0, "fn": len(truth["issues"]), "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "inspection_steps": inspection_steps, "n_findings": None, "matches": [],
                "by_type": _by_type([], truth["issues"], [])}
    pairs = max_matching(findings, truth["issues"], tolerance)
    by_type = _by_type(findings, truth["issues"], pairs)
    tp = len(pairs)
    fp = len(findings) - tp
    fn = len(truth["issues"]) - tp
    m = metrics(tp, fp, fn)
    r = shared_reward(m["f1"], inspection_steps)
    return {"valid": True, "error": None, "reward": float(r), "reward_exact": f"{r.numerator}/{r.denominator}",
            "tp": tp, "fp": fp, "fn": fn,
            "precision": float(m["precision"]), "recall": float(m["recall"]), "f1": float(m["f1"]),
            "inspection_steps": inspection_steps, "n_findings": len(findings), "by_type": by_type,
            "matches": [{"finding_index": fi, "issue_id": iid} for fi, iid in pairs]}


def oracle_findings(truth: dict) -> dict:
    """Ответ oracle: эталон с первым допустимым набором доказательств (только у оценщика)."""
    return {"episode_id": truth["episode_id"], "findings": [
        {"type": t["type"], "affected_ids": list(t["affected_ids"]),
         "claimed_amount_minor": t["expected_amount_minor"], "currency": t["currency"],
         "evidence_ids": list(t["evidence_sets"][0]), "explanation": "oracle"}
        for t in sorted(truth["issues"], key=lambda t: t["issue_id"])]}
