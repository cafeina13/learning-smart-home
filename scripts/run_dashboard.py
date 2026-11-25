"""Start the dashboard on http://127.0.0.1:8000 (local only).

    .venv\\Scripts\\python.exe scripts\\run_dashboard.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn  # noqa: E402

if __name__ == "__main__":
    uvicorn.run("dashboard.app:app", host="127.0.0.1", port=8000, log_level="warning")
