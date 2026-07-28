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
#
# Historical context:
# https://github.com/huggingface/transformers/issues/5281#issuecomment-2365359156
# "Segmentation fault when trying to load models" (#5281)
#
# A user who encountered the same problem reported:
# > My solution is just to import torch before import the transformers
#
# Keep this import above df_analyze._main, which can import transformers-backed
# model modules. The bootstrap call remains first so a missing torch dependency
# can be diagnosed or installed before this import is attempted.
import torch  # noqa: F401  # type: ignore

from src.df_analyze._main import main

if __name__ == "__main__":
    main()
