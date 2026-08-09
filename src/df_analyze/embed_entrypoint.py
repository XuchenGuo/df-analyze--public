from __future__ import annotations


def main() -> None:
    """Run the installed ``df-embed`` command."""
    import torch  # noqa: F401

    from df_analyze.embedding.main import main as run

    run()


if __name__ == "__main__":
    main()
