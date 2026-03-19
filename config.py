import os
import logging
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# --- Configuration ---
WS_HOST = os.getenv("ZENITH_HOST", "0.0.0.0")
WS_PORT = int(os.getenv("ZENITH_PORT", "8000"))
CORS_ORIGINS = os.getenv("ZENITH_CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
MODEL_DIR = Path(os.getenv("ZENITH_MODEL_DIR", "."))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
LOG_LEVEL = os.getenv("ZENITH_LOG_LEVEL", "INFO").upper()

# --- Logging ---
def setup_logging(name: str = "zenith") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(name)-18s | %(levelname)-5s | %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    return logger
