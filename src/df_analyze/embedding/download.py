from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

from typing import cast

from transformers import AutoModel, AutoProcessor, AutoTokenizer
from transformers.models.siglip.modeling_siglip import SiglipModel
from transformers.models.siglip.processing_siglip import SiglipProcessor
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaModel
from transformers.models.xlm_roberta.tokenization_xlm_roberta_fast import (
    XLMRobertaTokenizerFast,
)

from df_analyze.embedding.cli import EmbeddingModality, EmbeddingOptions

INTFLOAT_MULTILINGUAL_MODEL = ROOT / "downloaded_models/intfloat_multi_large/model"
INTFLOAT_MODEL_FILES = [
    INTFLOAT_MULTILINGUAL_MODEL / "config.json",
    INTFLOAT_MULTILINGUAL_MODEL / "model.safetensors",
]


INTFLOAT_MULTILINGUAL_TOKENIZER = (
    ROOT / "downloaded_models/intfloat_multi_large/tokenizer"
)
INTFLOAT_TOKENIZER_FILES = [
    INTFLOAT_MULTILINGUAL_TOKENIZER / "special_tokens_map.json",
    INTFLOAT_MULTILINGUAL_TOKENIZER / "tokenizer_config.json",
    INTFLOAT_MULTILINGUAL_TOKENIZER / "tokenizer.json",
]

SIGLIP_MODEL = ROOT / "downloaded_models/siglip_so400m_patch14_384/model"
SIGLIP_MODEL_FILES = [
    SIGLIP_MODEL / "config.json",
    SIGLIP_MODEL / "model.safetensors",
]

SIGLIP_PREPROCESSOR = ROOT / "downloaded_models/siglip_so400m_patch14_384/preprocessor"
SIGLIP_PREPROCESSOR_FILES = [
    SIGLIP_PREPROCESSOR / "preprocessor_config.json",
    SIGLIP_PREPROCESSOR / "special_tokens_map.json",
    SIGLIP_PREPROCESSOR / "spiece.model",
    SIGLIP_PREPROCESSOR / "tokenizer_config.json",
]


def download_nlp_intfloat_ml_model(force: bool = False) -> None:
    INTFLOAT_MULTILINGUAL_MODEL.mkdir(exist_ok=True, parents=True)
    INTFLOAT_MULTILINGUAL_TOKENIZER.mkdir(exist_ok=True, parents=True)
    model = cast(
        XLMRobertaModel,
        AutoModel.from_pretrained(
            "intfloat/multilingual-e5-large", force_download=force
        ),
    )
    model.save_pretrained(INTFLOAT_MULTILINGUAL_MODEL)
    print(f"Saved model to {INTFLOAT_MULTILINGUAL_MODEL}")

    tokenizer = cast(
        XLMRobertaTokenizerFast,
        AutoTokenizer.from_pretrained(
            "intfloat/multilingual-e5-large", force_download=force
        ),
    )
    tokenizer.save_pretrained(INTFLOAT_MULTILINGUAL_TOKENIZER)
    print(f"Saved tokenizer to {INTFLOAT_MULTILINGUAL_MODEL}")


def load_nlp_intfloat_ml_model_offline() -> (
    tuple[XLMRobertaModel, XLMRobertaTokenizerFast]
):
    model = cast(
        XLMRobertaModel,
        AutoModel.from_pretrained(INTFLOAT_MULTILINGUAL_MODEL, local_files_only=True),
    )
    tokenizer = cast(
        XLMRobertaTokenizerFast,
        AutoTokenizer.from_pretrained(
            INTFLOAT_MULTILINGUAL_TOKENIZER, local_files_only=True
        ),
    )
    return model, tokenizer


def download_siglip_model(force: bool = False) -> None:
    SIGLIP_MODEL.mkdir(exist_ok=True, parents=True)
    SIGLIP_PREPROCESSOR.mkdir(exist_ok=True, parents=True)
    model = cast(
        SiglipModel,
        AutoModel.from_pretrained(
            "google/siglip-so400m-patch14-384", force_download=force
        ),
    )
    model.save_pretrained(SIGLIP_MODEL)
    print(f"Saved model to {SIGLIP_MODEL}")

    processor = cast(
        SiglipProcessor,
        AutoProcessor.from_pretrained(
            "google/siglip-so400m-patch14-384", force_download=force
        ),
    )
    processor.save_pretrained(SIGLIP_PREPROCESSOR)
    print(f"Saved preprocessor to {SIGLIP_PREPROCESSOR}")


def load_siglip_offline() -> tuple[SiglipModel, SiglipProcessor]:
    model = cast(
        SiglipModel,
        AutoModel.from_pretrained(SIGLIP_MODEL, local_files_only=True),
    )
    tokenizer = cast(
        SiglipProcessor,
        AutoProcessor.from_pretrained(SIGLIP_PREPROCESSOR, local_files_only=True),
    )
    return model, tokenizer


def error_if_download_needed(opts: EmbeddingOptions) -> None:
    """
    Notes
    -----
    There are maybe a few cases to check here, but all we REALLY want to do
    is to warn the user in case a download is needed and has never been done.
    The download function is a noop if the models are already present, so no
    need to worry about that case.
    """
    if opts.any_download:
        return

    no_nlp = not all(
        file.exists() for file in [*INTFLOAT_MODEL_FILES, *INTFLOAT_TOKENIZER_FILES]
    )
    no_vision = not all(
        file.exists() for file in [*SIGLIP_MODEL_FILES, *SIGLIP_PREPROCESSOR_FILES]
    )

    if opts.modality is EmbeddingModality.NLP and no_nlp:
        raise FileNotFoundError(
            "Could not find all NLP model files. You will need to re-run\n"
            "\n"
            "    python df-embed.py --modality nlp --download\n"
            "\n"
            "or, if there is some issue with the current files, then\n"
            "\n"
            "    python df-embed.py --modality nlp --force-download\n"
            "\n"
            "once in order for the embedding functionality to work."
        )

    if opts.modality is EmbeddingModality.Vision and no_vision:
        raise FileNotFoundError(
            "Could not find all vision model files. You will need to re-run\n"
            "\n"
            "    python df-embed.py --modality vision --download\n"
            "\n"
            "or, if there is some issue with the current files, then\n"
            "\n"
            "    python df-embed.py --modality vision --force-download\n"
            "\n"
            "once in order for the embedding functionality to work."
        )


def download_models(nlp: bool = True, vision: bool = True, force: bool = False) -> None:
    if force:
        if nlp:
            download_nlp_intfloat_ml_model(force=True)
        if vision:
            download_siglip_model(force=True)
        return

    nlp_files = [*INTFLOAT_MODEL_FILES, *INTFLOAT_TOKENIZER_FILES]
    vision_files = [*SIGLIP_MODEL_FILES, *SIGLIP_PREPROCESSOR_FILES]
    if nlp and not all(file.exists() for file in nlp_files):
        download_nlp_intfloat_ml_model()
    if vision and not all(file.exists() for file in vision_files):
        download_siglip_model()


def dl_models_from_opts(opts: EmbeddingOptions) -> None:
    """This is a noop if models are present"""
    nlp = opts.modality is EmbeddingModality.NLP
    vision = opts.modality is EmbeddingModality.Vision
    force = opts.force_download
    download_models(nlp=nlp, vision=vision, force=force)
