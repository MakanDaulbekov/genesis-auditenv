"""UTF-8 для stdout/stderr точек входа CLI.

Сообщения на русском. На Linux/Docker поток уже UTF-8, и вызов ничего не меняет. На Windows при
перенаправлении в файл/канал Python берёт кодировку локали (cp1252 и т.п.), и print() падал бы
с UnicodeEncodeError. Вызывается в начале main() каждой команды; бинарный протокол оценщика
(sys.stdout.buffer) не затрагивается.
"""
from __future__ import annotations

import sys


def utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        enc = (getattr(stream, "encoding", None) or "").lower().replace("-", "")
        if enc != "utf8" and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass
