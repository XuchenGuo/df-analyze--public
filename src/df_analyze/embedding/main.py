from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

from df_analyze.embedding.cli import EmbeddingOptions, make_parser
from df_analyze.embedding.datasets import (
    dataset_from_opts,
)
from df_analyze.embedding.download import (
    dl_models_from_opts,
    error_if_download_needed,
)
from df_analyze.embedding.embed import (
    get_embeddings,
    get_model,
)


def main() -> None:
    """Do embedding logic here"""
    parser = make_parser()
    opts = EmbeddingOptions.from_parser(parser)
    error_if_download_needed(opts)
    dl_models_from_opts(opts)
    if opts.any_download:
        return
    ds = dataset_from_opts(opts)
    model, processor = get_model(opts.modality, runtime=opts.runtime)
    df = get_embeddings(
        ds=ds,  # type: ignore
        processor=processor,  # type: ignore
        model=model,  # type: ignore
        batch_size=opts.batch_size,
        load_limit=opts.limit_samples,
        runtime=opts.runtime,
    )

    # print(df)
    # print(opts)
    df.to_parquet(opts.outpath)
    print(f"Saved embeddings to {opts.outpath}")
