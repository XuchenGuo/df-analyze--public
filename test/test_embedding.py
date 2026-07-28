from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
SRC = ROOT / "src"  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
sys.path.append(str(SRC))  # isort: skip
# fmt: on

from tempfile import TemporaryDirectory

import pytest
from pytest import CaptureFixture
from tqdm import tqdm
from transformers.models.siglip.modeling_siglip import SiglipModel
from transformers.models.siglip.processing_siglip import SiglipProcessor
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaModel
from transformers.models.xlm_roberta.tokenization_xlm_roberta_fast import (
    XLMRobertaTokenizerFast,
)

from df_analyze.embedding import download as download_module
from df_analyze.embedding.cli import EmbeddingModality, EmbeddingOptions
from df_analyze.embedding.datasets import (
    EmbeddingDataset,
    VisionDataset,
    dataset_from_opts,
)
from df_analyze.embedding.download import (
    INTFLOAT_MODEL_FILES,
    INTFLOAT_TOKENIZER_FILES,
    SIGLIP_MODEL_FILES,
    SIGLIP_PREPROCESSOR_FILES,
    dl_models_from_opts,
    error_if_download_needed,
)
from df_analyze.embedding.embed import (
    get_embeddings,
    get_model,
    get_nlp_embeddings,
    get_vision_embeddings,
)
from df_analyze.embedding.testing import (
    NLPTestingDataset,
    VisionTestingDataset,
    cluster_nlp_sanity_check,
    cluster_vision_sanity_check,
    vision_padding_check,
)

MACOS_NLP_RUNTIMES = ROOT / "nlp_embed_runtimes.parquet"
MACOS_VISION_RUNTIMES = ROOT / "vision_embed_runtimes.parquet"
NIAGARA_NLP_RUNTIMES = ROOT / "nlp_embed_runtimes_niagara.parquet"
NIAGARA_VISION_RUNTIMES = ROOT / "vision_embed_runtimes_niagara.parquet"
NLP_TEST_DATA = ROOT / "data/testing/embedding/NLP"
HAS_NLP_TEST_DATA = any(path.is_dir() for path in NLP_TEST_DATA.glob("*"))
VISION_TEST_DATA = ROOT / "data/testing/embedding/vision"
HAS_VISION_TEST_DATA = any(path.is_dir() for path in VISION_TEST_DATA.glob("*"))
HAS_NLP_MODEL = all(
    path.is_file() for path in [*INTFLOAT_MODEL_FILES, *INTFLOAT_TOKENIZER_FILES]
)
HAS_VISION_MODEL = all(
    path.is_file() for path in [*SIGLIP_MODEL_FILES, *SIGLIP_PREPROCESSOR_FILES]
)


@pytest.mark.fast
def test_vision_random(capsys: CaptureFixture) -> None:
    with capsys.disabled():
        for _ in tqdm(range(10), desc="Creating random Vision data", disable=True):
            with TemporaryDirectory() as tempdir:
                VisionDataset.random(tempdir=tempdir)


@pytest.mark.fast
@pytest.mark.skipif(not HAS_NLP_TEST_DATA, reason="NLP test data is not installed")
def test_main_ds_nlp_loading(capsys: CaptureFixture) -> None:
    test_dses = NLPTestingDataset.get_all()
    dses = [ds.to_embedding_dataset() for ds in test_dses]
    for ds in dses:
        if ds.name == "go_emotions":
            continue  # multilabel
        try:
            ds.load(limit=1000)
        except Exception as e:
            raise ValueError(f"Got error for ds: {ds.name} @ {ds.datapath}") from e


@pytest.mark.fast
def test_main_ds_vision_loading(capsys: CaptureFixture) -> None:
    test_dses = VisionTestingDataset.get_all_cls()
    dses = [ds.to_embedding_dataset() for ds in test_dses]
    for ds in dses:
        if ds.name == "rare-species":
            continue  # text labels
        try:
            ds.load(limit=1000)
        except Exception as e:
            raise ValueError(f"Got error for ds: {ds.name} @ {ds.datapath}") from e


@pytest.mark.fast
@pytest.mark.skipif(
    not (HAS_NLP_TEST_DATA and HAS_NLP_MODEL),
    reason="NLP test data or model is not installed",
)
def test_nlp_embed(capsys: CaptureFixture) -> None:
    model, tokenizer = get_model(EmbeddingModality.NLP)
    assert isinstance(model, XLMRobertaModel)
    assert isinstance(tokenizer, XLMRobertaTokenizerFast)
    with capsys.disabled():
        for ds in tqdm(NLPTestingDataset.get_all(), desc="Embedding NLP data"):
            if ds.name == "go_emotions":
                continue  # multilabel
            ds = ds.to_embedding_dataset()
            df = get_nlp_embeddings(
                ds=ds,
                tokenizer=tokenizer,
                model=model,
                batch_size=2,
                num_texts=4,
                load_limit=1000,
            )
            assert len(df) == 4
            assert df.shape[1] == 1024 + 1


def _main_loop(ds: EmbeddingDataset, tempdir: str, modality: EmbeddingModality) -> None:
    out = Path(tempdir) / f"{ds.name}_test_embed_out.parquet"
    opts = EmbeddingOptions(
        datapath=ds.datapath,
        modality=EmbeddingModality(modality.value),
        name=ds.name,
        outpath=out,
        limit_samples=8,
        batch_size=2,
    )
    assert opts.outpath is not None
    error_if_download_needed(opts)
    dl_models_from_opts(opts)
    ds = dataset_from_opts(opts)
    model, processor = get_model(opts.modality)
    df = get_embeddings(  # noqa: F841
        ds=ds,  # type: ignore
        processor=processor,  # type: ignore
        model=model,  # type: ignore
        batch_size=opts.batch_size,
        load_limit=opts.limit_samples,
    )

    # print(df)
    # print(opts)
    # df.to_parquet(opts.outpath)
    # print(f"Saved embeddings to {opts.outpath}")


@pytest.mark.slow
@pytest.mark.skipif(not HAS_VISION_MODEL, reason="Vision model is not installed")
def test_vision_embed(capsys: CaptureFixture) -> None:
    model, processor = get_model(EmbeddingModality.Vision)
    assert isinstance(model, SiglipModel)
    assert isinstance(processor, SiglipProcessor)
    with capsys.disabled():
        for ds in tqdm(VisionTestingDataset.get_all_cls(), desc="Embedding vision data"):
            # if ds.name != "nsfw_detect":
            #     continue
            if ds.name == "rare-species":
                continue  # text labels
            ds = ds.to_embedding_dataset()
            df = get_vision_embeddings(
                ds=ds,
                processor=processor,
                model=model,
                batch_size=2,
                num_imgs=4,
                load_limit=1000,
            )
            assert len(df) == 4
            assert df.shape[1] == 1152 + 1


@pytest.mark.slow
@pytest.mark.skipif(not HAS_VISION_MODEL, reason="Vision model is not installed")
def test_main_vision(capsys: CaptureFixture) -> None:
    with capsys.disabled():
        dses = VisionTestingDataset.get_all_cls()
        dses = [ds.to_embedding_dataset() for ds in dses]
        with TemporaryDirectory() as tempdir:
            # we test some random images since we have less of these datasets
            dses = dses + [VisionDataset.random(tempdir=tempdir) for _ in range(10)]
            for ds in tqdm(dses, desc="Processing vision datasets"):
                if ds.name == "rare-species":
                    continue  # string labels
                try:
                    _main_loop(ds=ds, tempdir=tempdir, modality=EmbeddingModality.Vision)
                except Exception as e:
                    raise RuntimeError(f"Got exception for dataset: {ds.name}") from e


@pytest.mark.fast
@pytest.mark.skipif(
    not (HAS_NLP_TEST_DATA and HAS_NLP_MODEL),
    reason="NLP test data or model is not installed",
)
def test_main_nlp(capsys: CaptureFixture) -> None:
    with capsys.disabled():
        for ds in tqdm(NLPTestingDataset.get_all(), desc="Processing NLP datasets"):
            if ds.name == "go_emotions":
                continue  # multilabel
            ds = ds.to_embedding_dataset()
            with TemporaryDirectory() as tempdir:
                try:
                    _main_loop(ds=ds, tempdir=tempdir, modality=EmbeddingModality.NLP)
                except Exception as e:
                    raise RuntimeError(f"Got exception for dataset: {ds.name}") from e


@pytest.mark.fast
def test_download_models(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    present = tmp_path / "present"
    missing = tmp_path / "missing"
    present.touch()
    calls = []
    monkeypatch.setattr(download_module, "INTFLOAT_MODEL_FILES", [present])
    monkeypatch.setattr(download_module, "INTFLOAT_TOKENIZER_FILES", [missing])
    monkeypatch.setattr(download_module, "SIGLIP_MODEL_FILES", [present])
    monkeypatch.setattr(download_module, "SIGLIP_PREPROCESSOR_FILES", [missing])
    monkeypatch.setattr(
        download_module,
        "download_nlp_intfloat_ml_model",
        lambda force=False: calls.append(("nlp", force)),
    )
    monkeypatch.setattr(
        download_module,
        "download_siglip_model",
        lambda force=False: calls.append(("vision", force)),
    )

    download_module.download_models()

    assert calls == [("nlp", False), ("vision", False)]


@pytest.mark.med
@pytest.mark.skipif(
    not (HAS_VISION_TEST_DATA and HAS_VISION_MODEL),
    reason="Vision test data or model is not installed",
)
def test_vision_padding(capsys: CaptureFixture) -> None:
    with capsys.disabled():
        vision_padding_check()


@pytest.mark.slow
@pytest.mark.skipif(
    not (HAS_VISION_TEST_DATA and HAS_VISION_MODEL),
    reason="Vision test data or model is not installed",
)
def test_cluster_sanity_vision(capsys: CaptureFixture) -> None:
    with capsys.disabled():
        cluster_vision_sanity_check(n_samples=32)


@pytest.mark.slow
@pytest.mark.skipif(
    not (HAS_NLP_TEST_DATA and HAS_NLP_MODEL),
    reason="NLP test data or model is not installed",
)
def test_cluster_sanity_nlp(capsys: CaptureFixture) -> None:
    with capsys.disabled():
        cluster_nlp_sanity_check(n_samples=128)


def embed_nlp_datasets() -> None:
    """
    NLP
    ...
                          ds  n_samp  batch  total_s     est_h
     tweet_topic_single_2020    3586      8   10.332  0.040202
     tweet_topic_single_2021    3398      8   11.773  0.043408
    financial-classification    5057      8    8.510  0.046696
             rotten_tomatoes   10662      8    9.551  0.110496
             toxic-chat_0124   10165     40   31.481  0.347227
             toxic-chat_1123   10165     40   35.919  0.396177
    multiclass-sent-analysis   41643     40   11.138  0.503277
    """
    dsnames = [
        "tweet_topic_single_2020",
        "tweet_topic_single_2021",
        "rotten_tomatoes",
        "toxic-chat_0124",
        "toxic-chat_1123",
    ]
    dses = [ds for ds in NLPTestingDataset.get_all() if ds.name in dsnames]
    for ds in dses:
        print(ds.datafiles["all"])


def embed_vision_datasets() -> None:
    """
    Vision
    ......
                                   n_samp  batch  total_s     est_h
    Brain-Tumor-Classification        394      8   35.988  0.030771
    garbage_detection                 640     16   34.828  0.048372
    Visual_Emotional_Analysis         800     16   34.573  0.060023
    data-food-classification         1400     16   36.419  0.110648
    Anime-dataset                    1882      8   38.831  0.158594
    fast_food_image_classification   3000     16   35.165  0.228939
    GenAI-Bench-1600                 9600      8   37.869  0.788938

    """

    dsnames = ["fast_food_image_classification", "Anime-dataset"]
    dses = [ds for ds in VisionTestingDataset.get_all_cls() if ds.name in dsnames]
    for ds in dses:
        print(ds.root)


if __name__ == "__main__":
    # download_models(force=True)
    # embed_nlp_datasets()
    embed_vision_datasets()
