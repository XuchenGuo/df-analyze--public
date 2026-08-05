from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent.parent


def main() -> None:
    """Run the installed ``df-embed`` command with runtime bootstrapping."""
    from df_analyze.runtime.bootstrap import bootstrap

    script = ROOT / "df-embed.py"
    bootstrap(
        "df-embed",
        script if script.exists() else None,
        ROOT,
        module="df_analyze.embed_entrypoint",
    )

    # Transformers can crash during native-library loading when imported before
    # torch on some systems, so preserve the source script's import order.
    import torch  # noqa: F401

    from df_analyze.embedding.main import main as run

    run()


if __name__ == "__main__":
    main()
