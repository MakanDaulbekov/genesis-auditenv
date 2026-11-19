"""Пример внешней политики для жюри (≈15 строк): прочитать счета и отправить пустой ответ.

Запуск: python -m auditenv.cli run --policies examples.my_policy:SilentAuditor --splits dev
        python -m auditenv.agent_host --policy examples.my_policy:SilentAuditor --split dev
"""


class SilentAuditor:
    def reset(self, observation: dict) -> None:
        self.episode_id = observation["result"]["manifest"]["episode_id"]
        self.cursor, self.pages = None, 0

    def act(self, observation: dict) -> dict:
        res = observation["result"] or {}
        if self.pages and res.get("next_cursor") is None or self.pages >= 3:
            return {"action": "submit", "args": {"findings": {"episode_id": self.episode_id, "findings": []}}}
        self.cursor = res.get("next_cursor") if self.pages else None
        self.pages += 1
        return {"action": "list_records", "args": {"table": "invoices", "cursor": self.cursor, "limit": 50}}
