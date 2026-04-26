"""Logger setup — UTF-8 safe trên Windows console."""
from __future__ import annotations

import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Trả về logger với UTF-8 stream handler (fix lỗi cp1252 trên Windows)."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    # Reconfig stdout to UTF-8 nếu chưa
    if hasattr(sys.stdout, "reconfigure") and sys.stdout.encoding and \
            sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                          datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger
