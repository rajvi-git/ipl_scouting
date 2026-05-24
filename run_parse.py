"""Parse Cricsheet JSON only. Run from project root: python run_parse.py"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.parse_cricsheet import main

if __name__ == "__main__":
    main()
