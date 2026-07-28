from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
from pathlib import Path

import pandas as pd
import pytest
from pytest import CaptureFixture
from tqdm import tqdm
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaModel
from transformers.models.xlm_roberta.tokenization_xlm_roberta_fast import (
    XLMRobertaTokenizerFast,
)

from df_analyze.embedding.cli import EmbeddingModality
from df_analyze.embedding.embed import (
    get_model,
    get_nlp_tokenizations,
)
from df_analyze.embedding.testing import (
    NLPTestingDataset,
)
from pandas import DataFrame
import re

NLP_TEST_DATA = ROOT / "data/testing/embedding/NLP"
HAS_NLP_TEST_DATA = any(path.is_dir() for path in NLP_TEST_DATA.glob("*"))


@pytest.mark.skipif(not HAS_NLP_TEST_DATA, reason="NLP test data is not installed")
def test_nlp_embed(capsys: CaptureFixture) -> None:
    model, tokenizer = get_model(EmbeddingModality.NLP)
    assert isinstance(model, XLMRobertaModel)
    assert isinstance(tokenizer, XLMRobertaTokenizerFast)
    with capsys.disabled():
        infos = []
        for ds in tqdm(NLPTestingDataset.get_all(), desc="Tokenizing NLP data"):
            if ds.name == "go_emotions":
                continue  # multilabel
            ds = ds.to_embedding_dataset()
            df = get_nlp_tokenizations(
                ds=ds,
                tokenizer=tokenizer,
                num_texts=5000,
                load_limit=5000,
            )
            df["n_word"] = df["text"].apply(lambda s: len(re.split(r"\s+", s)))
            df["n_char"] = df["text"].apply(lambda s: len(s))
            df["n_token"] = df["mask"].apply(lambda x: len(x))
            meds = df.iloc[:, 3:].median().to_frame().T.rename(columns=lambda col: f"{col}_med")
            token_rate = (df["n_token"] /  df["n_word"]).mean()
            info = DataFrame({"ds": ds.name, "tokens/word": token_rate}, index=[0])
            info = pd.concat([info, meds], axis=1)
            infos.append(info)
        info = pd.concat(infos, axis=0, ignore_index=True)
        with pd.option_context("display.max_rows", 100):
            print(
                info.sort_values(by="tokens/word", ascending=True)
                .to_markdown(
                    tablefmt="simple",
                    index=False,
                    floatfmt=["0.0f", "0.2f", "0.0f", "0.0f", "0.0f"],
                )
            )

@pytest.mark.skipif(not HAS_NLP_TEST_DATA, reason="NLP test data is not installed")
def test_nlp_truncation(capsys: CaptureFixture) -> None:
    longest_ds = "readability_fineweb"
    model, tokenizer = get_model(EmbeddingModality.NLP)
    dses = NLPTestingDataset.get_all()
    ds = sorted(filter(lambda ds: ds.name == longest_ds, dses))[0]
    texts = ds.to_embedding_dataset().X().tolist()
    texts = sorted(texts, key=lambda s: len(s), reverse=True)
    text = texts[1]
    ch = len(text)
    texts = [
        text + text + text,
        text + text,
        text,
        text[:int(0.9*ch)],
        text[:int(0.8*ch)],
        text[:int(0.7*ch)],
        text[:int(0.5*ch)],
    ]
    bd = tokenizer(
        texts, max_length=512, padding=True, truncation=True, return_tensors="pt"
    )
    # very clearly chop off the right of the text, i.e. use from beginning up
    # to first 512 tokens
    print(bd["input_ids"])
    print(bd["input_ids"].shape)
