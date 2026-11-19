"""Узкие инструменты расследования. Ошибки — обезличенные коды без путей и стектрейсов."""
from __future__ import annotations

from .calculator import CalcError, evaluate
from .store import MAX_PAGE, TABLES, ReadOnlyStore

ERRORS = {
    "invalid_json": "Действие не является корректным JSON.",
    "invalid_action_format": "Ожидается объект {\"action\": str, \"args\": object}.",
    "unknown_action": "Неизвестное действие.",
    "invalid_arguments": "Неверные аргументы действия.",
    "unknown_table": "Неизвестная таблица.",
    "not_found": "Запись не найдена.",
    "invalid_cursor": "Неверный курсор.",
    "limit_out_of_range": "limit должен быть целым числом от 1 до 50.",
    "calc_error": "Выражение не допускается или не вычислимо.",
    "episode_finished": "Эпизод завершён; вызовите reset.",
    "not_reset": "Сначала вызовите reset.",
}

ACTION_ARGS = {
    "list_records": ({"table"}, {"table", "cursor", "limit"}),
    "get_record": ({"table", "id"}, {"table", "id"}),
    "get_document": ({"doc_id"}, {"doc_id"}),
    "calculator": ({"expression"}, {"expression"}),
    "submit": ({"findings"}, {"findings"}),
}

TOOLS_DESCRIPTION = [
    {"action": "list_records", "args": {"table": "ledger|invoices|payments|fx (обязателен)",
                                        "cursor": "str|null — необязателен; null/нет = с начала таблицы",
                                        "limit": "int 1..50 — необязателен; по умолчанию 50"},
     "returns": "{table, records, next_cursor}; next_cursor=null — конец таблицы; строки fx содержат fx_id",
     "notes": "лишние аргументы -> invalid_arguments"},
    {"action": "get_record", "args": {"table": "ledger|invoices|payments|fx",
                                      "id": "row_id | invoice_id/doc_id | payment_id/doc_id | fx:<date>:<src>:<tgt>"},
     "returns": "{table, record}"},
    {"action": "get_document", "args": {"doc_id": "doc_id of an invoice or payment"},
     "returns": "{document} — source document incl. remittance advice / PO / memo"},
    {"action": "calculator", "args": {"expression": "decimal arithmetic: + - * / ( ) only"},
     "returns": "{value} as decimal string"},
    {"action": "submit", "args": {"findings": "findings.json object"}, "returns": "{status}; ends the episode"},
]


class ToolError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def err(code: str) -> dict:
    return {"code": code, "message": ERRORS[code]}


def check_args(action: str, args: dict) -> None:
    required, allowed = ACTION_ARGS[action]
    if not isinstance(args, dict) or not required.issubset(args) or not set(args).issubset(allowed):
        raise ToolError("invalid_arguments")


def run_tool(store: ReadOnlyStore, action: str, args: dict) -> dict:
    """Выполняет не-submit действие. Возвращает result или бросает ToolError."""
    check_args(action, args)
    if action == "list_records":
        table = args["table"]
        if not isinstance(table, str):
            raise ToolError("invalid_arguments")
        if not store.has_table(table):
            raise ToolError("unknown_table")
        limit = args.get("limit", MAX_PAGE)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_PAGE:
            raise ToolError("limit_out_of_range")
        cursor = args.get("cursor")
        if cursor is None:
            offset = 0
        else:
            offset = store.parse_cursor(table, cursor)
            if offset is None:
                raise ToolError("invalid_cursor")
        records, nxt = store.page(table, offset, limit)
        return {"table": table, "records": records,
                "next_cursor": None if nxt is None else store.make_cursor(table, nxt)}
    if action == "get_record":
        table, ident = args["table"], args["id"]
        if not isinstance(table, str) or not isinstance(ident, str) or len(ident) > 128:
            raise ToolError("invalid_arguments")
        if not store.has_table(table):
            raise ToolError("unknown_table")
        rec = store.get(table, ident)
        if rec is None:
            raise ToolError("not_found")
        return {"table": table, "record": rec}
    if action == "get_document":
        doc_id = args["doc_id"]
        if not isinstance(doc_id, str) or len(doc_id) > 128:
            raise ToolError("invalid_arguments")
        doc = store.document(doc_id)
        if doc is None:
            raise ToolError("not_found")
        return {"document": doc}
    if action == "calculator":
        try:
            return {"value": evaluate(args["expression"])}
        except CalcError:
            raise ToolError("calc_error") from None
    raise ToolError("unknown_action")


assert set(TABLES) == {"ledger", "invoices", "payments", "fx"}
