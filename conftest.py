"""Root conftest: ensures src package is importable."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
