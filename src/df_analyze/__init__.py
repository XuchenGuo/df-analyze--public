import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
sys.path.append(str(SRC))

# Historical entry-point scaffold retained for reference and compatibility notes:
# from df_analyze._main import main
# def main() -> int:
#     print("Hello from rye-learn!")
#     return 0


def main() -> None:
    from df_analyze.runtime.bootstrap import bootstrap

    script = ROOT / "df-analyze.py"
    bootstrap(
        "df-analyze",
        script if script.exists() else None,
        ROOT,
        module="df_analyze",
    )

    from df_analyze._main import main as run

    run()
