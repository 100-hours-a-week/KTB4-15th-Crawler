"""stderr logs with explicit Korean time, independent of container timezone."""

import logging
from datetime import datetime

from .apply import KST


class KstFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        return datetime.fromtimestamp(record.created, KST).isoformat(timespec="milliseconds")


def configure_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(KstFormatter("%(asctime)s %(levelname)s %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler])
