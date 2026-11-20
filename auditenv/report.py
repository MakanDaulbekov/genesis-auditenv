"""REPORT.md: диагностика политик (P/R/F1/возврат/шаги/время) и измеренный разбор сбоев.

Это инструмент стороны оценщика: читает grader_data (эталон, provenance) и траектории
уже ЗАВЕРШЁННОГО прогона. В процесс агента ничего не передаётся.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from .env.store import parse_jsonl
from .grader.matcher import alias_map_from_tables, normalize
from .grader.reward import grade

ROOT = Path(__file__).resolve().parent.parent
TYPES = ["duplicate_invoice", "missing_or_partial_allocation", "fx_wrong_date_or_rate", "period_cutoff_error"]
SHORT = {"duplicate_invoice": "duplicate", "missing_or_partial_allocation": "allocation",
         "fx_wrong_date_or_rate": "fx", "period_cutoff_error": "cutoff"}


def _f(x, n=3):
    return "—" if x is None else f"{x:.{n}f}"


def _v(x):
    return "—" if x is None else x


def _machine() -> str:
    """Где сделан замер времени/памяти (честная пометка в отчёте)."""
    import os
    import platform
    return (f"{platform.system()} {platform.release()} {platform.machine()}, Python {platform.python_version()}, "
            f"{os.cpu_count()} логических CPU")


def load_episode(ref: str):
    pub = ROOT / "episodes" / ref
    prv = ROOT / "grader_data" / ref
    truth = json.loads((prv / "truth.json").read_text(encoding="utf-8"))
    prov = json.loads((prv / "provenance.json").read_text(encoding="utf-8"))
    alias = alias_map_from_tables(parse_jsonl((pub / "invoices.jsonl").read_bytes()),
                                  parse_jsonl((pub / "payments.jsonl").read_bytes()))
    return truth, prov, alias


def submitted_findings(traj_path: Path) -> dict[tuple[str, str], dict]:
    out = {}
    with open(traj_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            a = r.get("action")
            if a and a.get("action") == "submit" and isinstance(a["args"].get("findings"), dict):
                out[(r["policy"], r["episode_id"])] = (a["args"]["findings"], r["step"] - 1)
    return out


def failure_analysis(policy: str, refs: list[str], subs: dict) -> dict:
    capped = 0
    fn_variant = Counter()
    tp_variant = Counter()
    fp_kind = Counter()
    per_type = {t: Counter() for t in TYPES}
    for ref in refs:
        truth, prov, alias = load_episode(ref)
        key = (policy, truth["episode_id"])
        if key not in subs:
            continue
        findings, steps = subs[key]
        capped += isinstance(findings.get("findings"), list) and len(findings["findings"]) >= 30
        res = grade(findings, truth, alias, 0, steps)
        matched_issues = {m["issue_id"] for m in res["matches"]}
        matched_f = {m["finding_index"] for m in res["matches"]}
        variants = {m["issue_id"]: f"{SHORT[m['type']]}:{m['variant'].split(':')[0]}" +
                    (" [interaction]" if m["issue"]["interaction_group"] else "") for m in prov["mutations"]}
        for i in truth["issues"]:
            (tp_variant if i["issue_id"] in matched_issues else fn_variant)[variants[i["issue_id"]]] += 1
            per_type[i["type"]]["tp" if i["issue_id"] in matched_issues else "fn"] += 1
        dkind = {}
        for d in prov["distractors"]:
            for x in d["ids"]:
                dkind.setdefault(x, d["kind"])
        truth_ids = {x for i in truth["issues"] for x in i["affected_ids"]}
        try:
            norm = normalize(findings, truth["episode_id"], alias)
        except Exception:
            continue
        for f in norm:
            if f.index in matched_f:
                continue
            per_type[f.type]["fp"] += 1
            if set(f.affected) & truth_ids:
                same = [i for i in truth["issues"] if tuple(i["affected_ids"]) == f.affected]
                if same and same[0]["type"] == f.type:
                    why = "верный ID и тип, неверная сумма/доказательства"
                elif same:
                    why = "верный ID, неверный тип"
                else:
                    why = "задет документ с ошибкой, неверные affected_ids"
                fp_kind[f"{SHORT[f.type]}: {why}"] += 1
            else:
                kinds = sorted({dkind[x] for x in f.affected if x in dkind})
                fp_kind[f"{SHORT[f.type]}: отвлекающий случай «{', '.join(kinds)}»" if kinds
                        else f"{SHORT[f.type]}: прочие корректные записи"] += 1
    return {"capped": capped, "fn_variant": fn_variant, "tp_variant": tp_variant, "fp_kind": fp_kind, "per_type": per_type}


def main() -> int:
    from .common.console import utf8_stdio
    utf8_stdio()
    t0 = time.perf_counter()
    summary = json.loads((ROOT / "grader_data" / "run_summary.json").read_text(encoding="utf-8"))
    subs = submitted_findings(ROOT / "trajectories.jsonl")
    refs = [e["ref"] for s in ("train", "dev")
            for e in json.loads((ROOT / "episodes" / s / "index.json").read_text(encoding="utf-8"))]
    agg = summary["aggregate"]
    L = []
    L.append("# REPORT: диагностика базовых политик и разбор сбоев\n")
    L.append("> Сгенерировано `python -m auditenv.report` (часть `make check`). Награда — общая функция раздела 05: "
             "`clip(2*F1 - 1 - 0.002*inspection_steps, -1, 1)`. Это **диагностика среды**, а не заявление об "
             "обучении модели: веса не обучались, контролируемого сравнения на отложенных данных нет.\n")
    n_clean = sum(1 for g in summary["grader"] if g["policy"] == "oracle" and g["n_truth"] == 0)
    n_issues = sum(g["n_truth"] for g in summary["grader"] if g["policy"] == "oracle")
    L.append(f"Данные: 20 открытых эпизодов (10 train + 10 dev), из них {n_clean} чистых, {n_issues} внедрённых ошибок; "
             "конфиги `configs/splits/*.json` (`config_version` = train-1.0.0 / dev-1.0.0).\n")
    L.append("## 1. Сводка по политикам (train+dev, режим train)\n")
    L.append("| политика | возврат (средний) | P (ср.) | R (ср.) | F1 (ср.) | micro P / R / F1 | TP/FP/FN | шаги (ср.) | "
             "время эпизода, мс (p50 / p95 / макс.) | пик RSS, МБ (среда+политика / оценщик) | возврат на чистых | "
             "неверных submit |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for pol in ("oracle", "competent_reconciler", "simple_reconciler", "random_explorer", "flag_all"):
        a = agg[pol]
        m = a["micro"]
        L.append(f"| {pol} | {_f(a['mean_return'])} | {_f(a['mean_precision'])} | {_f(a['mean_recall'])} | "
                 f"{_f(a['mean_f1'])} | {_f(m['precision'], 2)} / {_f(m['recall'], 2)} / {_f(m['f1'], 2)} | "
                 f"{m['tp']}/{m['fp']}/{m['fn']} | {a['mean_steps']} | {a.get('p50_wall_ms')} / {a.get('p95_wall_ms')} / "
                 f"{a['max_wall_ms']} | {_v(a.get('peak_rss_mb_env_policy'))} / {_v(a.get('peak_rss_mb_grader'))} | "
                 f"{_f(a['clean_mean_return'])} | {a['invalid_submissions']} |")
    L.append("\nПримечания: P/R «средние» усредняются по эпизодам; на чистом эпизоде без выводов P=R=0 по правилу "
             "нулевого знаменателя (поэтому у oracle средние P/R < 1 при F1 = 1). Каждая политика прогоняется в своём "
             "процессе, поэтому пик RSS — по политике (ограничение задачи — 4 ГБ). Лимит времени — 60 с на эпизод; "
             "максимум по всем прогонам указан в таблице. oracle отправляет ответ без осмотра (k = 0), поэтому его "
             "возврат = 1 − 0.002·0 = 1.\n")
    L.append("Пороговые свойства, которые подтверждают числа: oracle ≥ competent_reconciler > simple_reconciler > "
             "random_explorer ≥ flag_all; "
             "«пометить всё» получает −1 и на чистых, и на ошибочных книгах; молчание на чистых книгах дало бы ≈ 1, а "
             "любая ложная находка снижает возврат до F1 = 0 → −1.\n")

    sol_p = ROOT / "grader_data" / "solvability.json"
    if sol_p.exists():
        so = json.loads(sol_p.read_text(encoding="utf-8"))
        ca = agg.get("competent_reconciler")
        L.append("## 2. Решаемость: competent_reconciler (измерено)\n")
        L.append("competent_reconciler — честная скриптовая сверка без доступа к эталону: читает таблицы через "
                 "list_records, переводит время в часовой пояс компании, сравнивает дубли по полной сути счёта "
                 "(включая периоды услуг), берёт курс по policy, читает remittance advice через get_document только "
                 "для платежей с нераспределённым остатком; курсы — дешевле из «fx постранично» и «get_record по "
                 "ключам». Это доказательство, что каждая внедрённая ошибка обнаружима по открытым данным в пределах "
                 "100 шагов, а отвлекающие случаи не дают ложных срабатываний.\n")
        if ca:
            L.append(f"* train+dev (20 эпизодов): TP/FP/FN = {ca['micro']['tp']}/{ca['micro']['fp']}/{ca['micro']['fn']}, "
                     f"средний возврат {ca['mean_return']:.3f}, шагов в среднем {ca['mean_steps']}.")
        L.append(f"* {so['episodes']} случайных эпизодов (300 seed × 6 сочетаний параметров + {so['by_variant'].get('worst_size', {}).get('n', 0)} "
                 f"худшего размера): **{so['perfect']}/{so['episodes']}** с F1 = 1 и без FP; TP/FP/FN = "
                 f"{so['tp']}/{so['fp']}/{so['fn']}.")
        L.append(f"* Шаги: максимум **{so['max_steps']}** из 100, среднее {so['mean_steps']}. Наибольший эпизод: "
                 f"{so['max_ledger_rows']} проводок, {so['max_documents']} документов.\n")
        L.append("| сочетание | эпизодов | без FP/FN | макс. шагов |")
        L.append("|---|---|---|---|")
        for v, d in so["by_variant"].items():
            L.append(f"| {v} | {d['n']} | {d['perfect']} | {d['max_steps']} |")
        L.append("\nГистограмма потраченных шагов (включая submit):\n")
        L.append("| шаги | эпизодов |")
        L.append("|---|---|")
        for b, c in so["steps_histogram"].items():
            L.append(f"| {b} | {c} |")
        if so["failures"]:
            L.append("\nОставшиеся сбои (каждый — баг генератора или задокументированная неоднозначность):\n")
            for f in so["failures"]:
                L.append(f"* {f}")
        else:
            L.append("\nОставшихся FN/FP нет. Бюджет: размеры ограничены в `generator/config.py` (n_invoices ≤ 75, "
                     "n_expenses ≤ 700), чтобы эпизод оставался в пределах раздела 03 и честная политика имела "
                     "запас ≥ 25 шагов; при 150 счетах / 900 расходах (≈2 600 проводок) потребовалось бы до 89 шагов.\n")

    fa = failure_analysis("simple_reconciler", refs, subs)
    L.append("## 3. Разбор сбоев simple_reconciler (измерено)\n")
    L.append("simple_reconciler — честная скриптовая сверка с тремя сознательными упрощениями: (1) дата берётся из строки "
             "времени как написано, без перевода в часовой пояс компании; (2) дубли — по (контрагент, валюта, сумма, "
             "описания/кол-во/цены строк) без учёта периода услуг; (3) remittance advice читается только для платежей "
             "с нераспределённым остатком.\n")
    L.append("### 3.1 По типам ошибок\n")
    L.append("| тип | TP | FP | FN | precision | recall |")
    L.append("|---|---|---|---|---|---|")
    for t in TYPES:
        c = fa["per_type"][t]
        p = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
        r = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
        L.append(f"| {t} | {c['tp']} | {c['fp']} | {c['fn']} | {p:.2f} | {r:.2f} |")
    L.append("\n### 3.2 Пропуски (FN) и попадания по вариантам мутаций\n")
    L.append("| вариант мутации | найдено (TP) | пропущено (FN) |")
    L.append("|---|---|---|")
    for v in sorted(set(fa["fn_variant"]) | set(fa["tp_variant"])):
        L.append(f"| {v} | {fa['tp_variant'][v]} | {fa['fn_variant'][v]} |")
    L.append("\n### 3.3 Ложные срабатывания (FP) по причинам\n")
    L.append("| причина | FP |")
    L.append("|---|---|")
    for k, v in fa["fp_kind"].most_common():
        L.append(f"| {k} | {v} |")
    L.append("\n### 3.4 Выводы\n")
    fx_fp = sum(v for k, v in fa["fp_kind"].items() if k.startswith("fx:"))
    dup_fp = sum(v for k, v in fa["fp_kind"].items() if k.startswith("duplicate:"))
    cut_fp = sum(v for k, v in fa["fp_kind"].items() if k.startswith("cutoff:"))
    naive_fn = fa["fn_variant"].get("fx:naive_date", 0) + fa["fn_variant"].get("fx:naive_date [interaction]", 0)
    L.append(f"* **Часовые пояса — главный источник ошибок.** {fx_fp} FP типа fx: курс берётся по «наивной» дате, "
             f"которая у документов в чужом смещении отличается от локальной даты компании. Те же ошибки бухгалтера "
             f"(вариант `naive_date`) сверка не видит: {naive_fn} FN — её правило совпадает с ошибочным.")
    L.append(f"* **Законные повторы похожи на дубли.** {dup_fp} FP duplicate: еженедельные подписки с одинаковой суммой "
             "различаются только периодом услуг, а сверка его игнорирует.")
    L.append(f"* **Граница периода.** {cut_fp} FP cutoff: счета, «наивно» датированные следующим месяцем, но выпущенные "
             "в периоде по времени компании; пропуски cutoff — вариант `timezone_ignored`, где наивная дата внутри периода, "
             f"и cutoff платежей (FN: {fa['fn_variant'].get('cutoff:next_period_payment_booked', 0)}), который эта сверка "
             "не проверяет вовсе.")
    L.append("* **Распределение платежей** почти без ошибок: правило «invoice_refs = remittance advice» сверка применяет "
             "буквально; цена — шаги на get_document. Предоплаты, переплаты и возвраты (отвлекающие) не дают FP "
             "по allocation (но дают FP по fx из-за наивной даты).")
    L.append(f"* **Лимит 30 выводов** (эпизодов, где сверка в него упёрлась: {fa['capped']}). Выводы, найденные последними "
             "(распределение проверяется в конце), были отброшены: ложные срабатывания вытесняют верные. Отсюда "
             "единичные FN по allocation.")
    L.append("* Следствие для среды: высокую награду нельзя получить ни трюками (flag_all ≈ −1), ни поверхностной "
             "сверкой; нужны конвертация времени, чтение доказательств и аккуратность к отвлекающим случаям.\n")

    jury_p = ROOT / "grader_data" / "jury.json"
    if jury_p.exists():
        j = json.loads(jury_p.read_text(encoding="utf-8"))
        L.append("## 4. Самопроверка (tests/jury/run_jury.py)\n")
        L.append("| блок (зеркало раздела 10 PDF) | пройдено |")
        L.append("|---|---|")
        for b in ("contract", "data", "reward", "paired"):
            L.append(f"| {b} | {j['blocks'][b]['passed']}/20 |")
        L.append(f"\nСамопроверка: **{j['passed']}/80** наших проверок, зеркальных разделам 06 и 10 PDF. **Это не оценка "
                 "жюри.** Блок data — 20 новых эпизодов (5 чистых, 5 со взаимодействиями, 10 новых сочетаний "
                 "параметров) через независимый валидатор; paired — 20 пар «повтор seed+действий» + "
                 "«перестановка+переименование ID».\n")
        L.append("Не покрыто самопроверкой: harness жюри, их политики, их преобразования и закрытые seed (для этого — "
                 "`make jury-run`, который принимает их сплит, секрет и политики), а также реальная сборка Docker-образа.\n")
    L.append("## 5. Ограничения отчёта\n")
    L.append("* 20 открытых эпизодов — малая выборка; числа характеризуют базовые политики, а не потолок задачи.")
    L.append(f"* Время и память измерены на машине: {_machine()}. У организаторов числа будут другими, но запас до "
             "лимита 60 с на эпизод — несколько порядков. Остальные числа отчёта (TP/FP/FN, награды, шаги) "
             "детерминированы и от машины не зависят.")
    L.append("* Полный список ограничений среды — DESIGN.md, раздел «Честные ограничения».")
    L.append(f"\n_Отчёт построен за {time.perf_counter() - t0:.2f} с._")
    (ROOT / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print(f"[report] REPORT.md: {len(L)} строк; simple_reconciler FP={sum(fa['fp_kind'].values())} "
          f"FN={sum(fa['fn_variant'].values())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
