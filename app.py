"""Root entrypoint for Streamlit app."""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Execute the main application code from app/app.py
with open(ROOT_DIR / "app" / "app.py", "r", encoding="utf-8") as f:
    code = f.read()

exec(code)
