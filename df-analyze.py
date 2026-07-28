from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent  # isort: skip
SRC = Path(__file__).resolve().parent / "src"  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
sys.path.append(str(SRC))  # isort: skip
# fmt: on

from df_analyze.runtime.bootstrap import bootstrap

bootstrap("df-analyze", Path(__file__), ROOT)

# Import torch before transformers; some transformer builds require this order.
import torch  # noqa: F401, E402  # type: ignore

from src.df_analyze._main import main

if __name__ == "__main__":
    main()
