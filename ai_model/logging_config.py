"""Application logging; keep Lightning's training metrics in its CSV logger."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys
from typing import Optional


def configure_logging(log_dir: Optional[Path] = None) -> None:
    """Send application events to stderr and, when configured, a bounded file."""
    logger = logging.getLogger("ai_model")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in logger.handlers[:]:
        if handler.name in {"ai_model.console", "ai_model.file"}:
            logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler(sys.stderr)
    console.set_name("ai_model.console")
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / "app.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.set_name("ai_model.file")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
