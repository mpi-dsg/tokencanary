"""Single source of truth for locating and loading the project .env file.

The code lives in audit/ and the .env sits at the repository root, one level
up, so that scripts can be run as `python audit/<script>.py` from the root with
data paths like `results/...` resolving against the working directory."""
import os

from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")


ZAI_KEY_PATH = os.path.expanduser("~/.zai")


def ensure_loaded():
    if not os.path.exists(ENV_PATH):
        raise RuntimeError(f".env not found at {ENV_PATH}")
    load_dotenv(ENV_PATH)
    if "ZAI_API_KEY" not in os.environ and os.path.exists(ZAI_KEY_PATH):
        with open(ZAI_KEY_PATH) as f:
            os.environ["ZAI_API_KEY"] = f.read().strip()
