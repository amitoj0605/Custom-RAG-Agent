import logging
from datetime import datetime

logger = logging.getLogger("agent_logger")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def log(message: str):

    timestamp = datetime.now().strftime("%H:%M:%S")

    formatted = f"[{timestamp}] {message}"

    logger.info(formatted)

    return formatted