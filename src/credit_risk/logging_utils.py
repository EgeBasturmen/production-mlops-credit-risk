# src/credit_risk/logging_utils.py
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    """
    logging record -> JSON line
    - message field'ını da koyar
    - extra ile gelen alanları da ekler
    """

    def format(self, record: logging.LogRecord) -> str:
        base = {
            "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # record.__dict__ içinde standart logging alanları + extra alanlar var.
        # Biz standart olanları ayıklayıp extra'ları ekliyoruz.
        reserved = {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
        }

        for k, v in record.__dict__.items():
            if k in reserved:
                continue
            if k.startswith("_"):
                continue
            # ör: event, request_id, path, status_code, latency_ms ...
            base[k] = v

        if record.exc_info:
            base["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(base, ensure_ascii=False)
