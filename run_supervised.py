"""Entry point: build training pairs and train supervised IPL impact/tier models."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.build_training_table import main as build_training_table_main
from src.models.train_supervised import main as train_supervised_main


if __name__ == "__main__":
    build_training_table_main()
    train_supervised_main()
