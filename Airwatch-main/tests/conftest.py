import os
import sys
import tempfile
from pathlib import Path

# Must run before any `app` import so every test uses a throwaway DB.
os.environ["AIRWATCH_DB_URL"] = f"sqlite:///{Path(tempfile.mkdtemp()) / 'test.db'}"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
