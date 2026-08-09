import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
sys.path.append(str(SRC))


def main() -> None:
    import torch  # noqa: F401

    from df_analyze._main import main as run

    run()
