"""Время: ISO-8601 со смещением, фиксированные смещения (без DST и tzdata)."""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone

_OFFSET_RE = re.compile(r"^([+-])(\d{2}):(\d{2})$")
_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$")


def parse_offset(offset: str) -> timezone:
    m = _OFFSET_RE.match(offset)
    if not m:
        raise ValueError("bad offset")
    sign = 1 if m.group(1) == "+" else -1
    minutes = int(m.group(2)) * 60 + int(m.group(3))
    return timezone(sign * timedelta(minutes=minutes))


def is_iso_ts(s: object) -> bool:
    return isinstance(s, str) and bool(_TS_RE.match(s))


def parse_ts(s: str) -> datetime:
    if not is_iso_ts(s):
        raise ValueError("timestamp must be ISO-8601 with offset, e.g. 2031-03-31T23:00:00+05:00")
    return datetime.fromisoformat(s)


def format_ts(dt: datetime, offset: str) -> str:
    """Отрендерить момент времени в заданном смещении: YYYY-MM-DDTHH:MM:SS+HH:MM."""
    local = dt.astimezone(parse_offset(offset))
    return local.strftime("%Y-%m-%dT%H:%M:%S") + offset


def local_date(ts: str, offset: str) -> date:
    """Календарная дата момента ts в часовом поясе компании."""
    return parse_ts(ts).astimezone(parse_offset(offset)).date()


def period_bounds(period_start: str, period_end: str, offset: str) -> tuple[datetime, datetime]:
    """Период = [period_start 00:00, period_end+1 00:00) в смещении компании."""
    tz = parse_offset(offset)
    ps = date.fromisoformat(period_start)
    pe = date.fromisoformat(period_end)
    start = datetime(ps.year, ps.month, ps.day, tzinfo=tz)
    end = datetime(pe.year, pe.month, pe.day, tzinfo=tz) + timedelta(days=1)
    return start, end


def in_period(ts: str, period_start: str, period_end: str, offset: str) -> bool:
    start, end = period_bounds(period_start, period_end, offset)
    t = parse_ts(ts)
    return start <= t < end
