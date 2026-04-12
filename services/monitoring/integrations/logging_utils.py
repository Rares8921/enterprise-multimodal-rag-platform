import logging
import os

try:
    from pythonjsonlogger import jsonlogger
except Exception:  # pragma: no cover
    jsonlogger = None


def configure_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()

    root = logging.getLogger()
    root.setLevel(level)

    # Avoid duplicate handlers if reloaded
    if root.handlers:
        return

    handler = logging.StreamHandler()

    if jsonlogger is not None and os.getenv("LOG_JSON", "true").lower() in {"1", "true", "yes"}:
        formatter = jsonlogger.JsonFormatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s %(filename)s %(lineno)d"
        )
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

    handler.setFormatter(formatter)
    root.addHandler(handler)
