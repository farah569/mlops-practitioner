import logging
import json
import contextvars
from datetime import datetime

correlation_id_var = contextvars.ContextVar("correlation_id", default="SYSTEM")


class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": correlation_id_var.get(),
        }
        return json.dumps(log_record)


def setup_logging():
    logger = logging.getLogger("prodml")
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(handler)
    return logger
