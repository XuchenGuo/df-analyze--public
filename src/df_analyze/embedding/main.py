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
from df_analyze.runtime.hardware import (
    DeviceIntent,
    RuntimeComponent,
    device_reason_text,
    format_device_plan,
    is_cuda_runtime_error,
    release_accelerator_memory,
    validate_cuda_request,
)


def main() -> None:
    """Do embedding logic here"""
    parser = make_parser()
    opts = EmbeddingOptions.from_parser(parser)
    error_if_download_needed(opts)
    dl_models_from_opts(opts)
    if opts.any_download:
        return
    components = {"embedding": RuntimeComponent.Embedding}
    runtime = opts.runtime
    validate_cuda_request(runtime, components)
    print(format_device_plan(runtime, components))
    print()
    ds = dataset_from_opts(opts)
    model = None
    processor = None
    decision = runtime.decision_for(RuntimeComponent.Embedding)
    print(
        f"[device] Embedding: {decision.resolved.upper()} "
        f"({device_reason_text(decision)})"
    )
    try:
        model, processor = get_model(opts.modality, runtime=runtime)
        df = get_embeddings(
            ds=ds,  # type: ignore
            processor=processor,  # type: ignore
            model=model,  # type: ignore
            batch_size=opts.batch_size,
            load_limit=opts.limit_samples,
            runtime=runtime,
        )
    except Exception as error:
        cuda_failure = (
            decision.resolved == "cuda" and is_cuda_runtime_error(error)
        )
        if (
            runtime.intent is DeviceIntent.CUDA
            and cuda_failure
        ):
            raise RuntimeError(
                "Embedding failed on CUDA while --device cuda is strict. "
                "CPU fallback is disabled; use --device auto to allow one "
                "complete CPU retry."
            ) from error
        if not (
            runtime.intent is DeviceIntent.Auto
            and cuda_failure
        ):
            raise
        print(
            "Embedding encountered a CUDA runtime failure. Retrying the complete "
            "embedding task on CPU once."
        )
        model = None
        processor = None
        release_accelerator_memory()
        runtime.record_cpu_fallback(
            RuntimeComponent.Embedding,
            f"cuda_runtime_fallback:{type(error).__name__}",
        )
        final_decision = runtime.decision_for(RuntimeComponent.Embedding)
        print(
            f"[device] Embedding: {final_decision.resolved.upper()} "
            f"({device_reason_text(final_decision)})"
        )
        model, processor = get_model(opts.modality, runtime=runtime)
        df = get_embeddings(
            ds=ds,  # type: ignore
            processor=processor,  # type: ignore
            model=model,  # type: ignore
            batch_size=opts.batch_size,
            load_limit=opts.limit_samples,
            runtime=runtime,
        )

    # print(df)
    # print(opts)
    df.to_parquet(opts.outpath)
    print(f"Saved embeddings to {opts.outpath}")
