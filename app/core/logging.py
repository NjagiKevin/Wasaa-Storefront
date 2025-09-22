import logging
from logging.handlers import RotatingFileHandler
import os

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../logs")
os.makedirs(LOG_DIR, exist_ok=True)

log_file = os.path.join(LOG_DIR, "app.log")

formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s"
)

# Rotating file handler (max 5MB per file, keep 3 backups)
file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3)
file_handler.setFormatter(formatter)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

logger = logging.getLogger("wasaa_app")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

def get_logger(name: str = None) -> logging.Logger:
    if name:
        return logging.getLogger(name)
    return logger
