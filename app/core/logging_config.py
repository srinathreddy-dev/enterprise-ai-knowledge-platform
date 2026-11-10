
import json
import logging
from datetime import datetime, timezone


class StructuredFormatter(logging.Formatter):
    """Format application logs as structured JSON."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(
                record.created,
                tz=timezone.utc,
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        optional_fields = [
            "http_method",
            "http_path",
            "status_code",
            "duration_ms",
        ]

        for field in optional_fields:
            value = getattr(record, field, None)
            if value is not None:
                log_entry[field] = value

        if record.exc_info:
            log_entry["exception"] = self.formatException(
                record.exc_info
            )

        return json.dumps(log_entry)


def configure_logging() -> None:
    """Configure structured JSON logging for the application."""

    handler = logging.StreamHandler()
    handler.setFormatter(StructuredFormatter())

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
