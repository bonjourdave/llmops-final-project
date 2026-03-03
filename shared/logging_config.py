import logging
import sys


def configure_logging(log_level: str = "INFO") -> None:
    """
    Production-style structured logging:
    - Writes to stdout (Docker / Cloud Run friendly)
    - Format: timestamp  level  logger  message
    - Aligns uvicorn loggers to the same level and format
    """
    level = getattr(logging, (log_level or "INFO").upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    # Remove existing handlers to prevent duplicate output on hot-reload
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    root.addHandler(handler)

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).setLevel(level)
