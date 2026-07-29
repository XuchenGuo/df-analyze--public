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

bootstrap("df-embed", Path(__file__), ROOT)

# fmt: off
# Stupid insane Python import garbage
# https://github.com/huggingface/transformers/issues/5281#issuecomment-2365359156
"""
# Segmentation fault when trying to load models #5281

andr2w commented on Jan 25, 2023:

> I come across the same problem too.
>
> My solution is just to import torch before import the transformers
"""
import torch  # noqa  # type: ignore
# fmt: on

# The bootstrap call must remain before the torch import so a missing dependency
# can be diagnosed or installed. Torch still remains before the embedding entry
# point, which imports transformers-backed modules.

from src.df_analyze.embedding.main import main

if __name__ == "__main__":
    main()
