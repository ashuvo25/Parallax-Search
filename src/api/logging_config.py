"""JSON logging: one JSON object per line, to the console and logs/api.jsonl."""
import json
import logging
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG_FILE = ROOT / "logs" / "api.jsonl"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data = {
            "time": datetime.fromtimestamp(record.created, timezone.utc)
                            .isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        fields = getattr(record, "fields", None)  # passed with extra={"fields": {...}}
        if fields:
            data.update(fields)
        if record.exc_info:
            data["exception"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False, default=str)


def setup_logging() -> logging.Logger:
    """Configure the 'api' logger once. Child loggers like 'api.requests' inherit it."""
    logger = logging.getLogger("api")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False

    LOG_FILE.parent.mkdir(exist_ok=True)
    formatter = JsonFormatter()
    console = logging.StreamHandler()
    # rotate at 5 MB, keep 3 old files, so the log can't fill the disk
    file = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    for handler in (console, file):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger