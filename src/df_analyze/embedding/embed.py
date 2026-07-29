from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
from pathlib import Path
from typing import Optional, Union, cast, overload

import pandas as pd
import torch
from pandas import DataFrame, Series
from torch import Tensor
from tqdm import tqdm
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.siglip.modeling_siglip import SiglipModel
from transformers.models.siglip.processing_siglip import SiglipProcessor
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaModel
from transformers.models.xlm_roberta.tokenization_xlm_roberta_fast import (
    XLMRobertaTokenizerFast,
)

from df_analyze.embedding.cli import EmbeddingModality
from df_analyze.embedding.datasets import NLPDataset, VisionDataset
from df_analyze.embedding.download import (
    load_nlp_intfloat_ml_model_offline,
    load_siglip_offline,
)
from df_analyze.embedding.utils import avg_pool, batched
from df_analyze.runtime.hardware import RuntimeComponent, RuntimePolicy, get_runtime


def _embedding_device(runtime: Optional[RuntimePolicy]) -> str:
    return (runtime or get_runtime()).device_for(RuntimeComponent.Embedding)


def get_model(
    modality: EmbeddingModality,
    runtime: Optional[RuntimePolicy] = None,
) -> Union[
    tuple[XLMRobertaModel, XLMRobertaTokenizerFast], tuple[SiglipModel, SiglipProcessor]
]:
    # no idea WTF is going on here, why I can't compare enums properly...
    # Normalize the enum value before identity comparison because callers can
    # supply an equivalent EmbeddingModality instance from another import path.
    device = _embedding_device(runtime)
    if EmbeddingModality(modality.value) is EmbeddingModality.NLP:
        model, processor = load_nlp_intfloat_ml_model_offline()
    elif EmbeddingModality(modality.value) is EmbeddingModality.Vision:
        model, processor = load_siglip_offline()
    else:
        raise ValueError(f"Unrecognized modality: {modality}")
    model.to(device)
    model.eval()
    return model, processor


def get_nlp_tokenizations(
    ds: NLPDataset,
    tokenizer: XLMRobertaTokenizerFast,
    load_limit: Optional[int] = None,
    num_texts: Optional[int] = None,
) -> DataFrame:
    X = ds.X(limit=load_limit)
    y = ds.y(limit=load_limit)

    # Each input text should start with "query: " or "passage: ", even for
    # non-English texts. For tasks other than retrieval, you can simply use
    # the "query: " prefix. See also the discussion for this model on HF, but
    # the TL;DR is basically that it doesn't matter too much which you prepend,
    # but empirically, using "query: " seems to give better performance overall
    all_texts = X.apply(lambda text: f"query: {text}").tolist()
    if num_texts is not None:
        all_texts = all_texts[:num_texts]
        y = y.iloc[:num_texts]

    # Tokenize the input texts
    # batch_dict is:
    # { input_ids: Tensor[B, N], attention_mask: Tensor[B, N] }
    #
    # B = batch size, i.e. number of samples
    # N = max number of tokens in longest sequence of tokens in batch
    #
    # input_ids has the tokens as integers (ones seem to be padding), i.e. is
    # the tokenization of the input. This of course will only be loosely
    # correlated with word count or etc.
    #
    # attention_mask is a boolean where mask[i, :] is such that for input i,
    # mask[i] is all ones up to number of non-masking tokens in input_ids
    all_tokenizations = []
    with torch.no_grad():
        for i, text in enumerate(all_texts):
            bd = batch_dict = tokenizer(
                text,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            mask = batch_dict["attention_mask"]
            assert isinstance(mask, Tensor)
            batch_dict["texts"] = text
            df = DataFrame(  # allow insert NumPy arrays without wrapping in []
                {
                    "token_ids": Series(dtype="object"),
                    "mask": Series(dtype="object"),
                    "text": Series(dtype="str"),
                },
                index=[i],
            )
            df.loc[i, "token_ids"] = bd["input_ids"].ravel().numpy()
            df.loc[i, "mask"] = bd["attention_mask"].ravel().numpy()
            df.loc[i, "text"] = text
            all_tokenizations.append(df)

    return pd.concat(all_tokenizations, axis=0)


def get_nlp_embeddings(
    ds: NLPDataset,
    tokenizer: XLMRobertaTokenizerFast,
    model: XLMRobertaModel,
    batch_size: int = 32,
    load_limit: Optional[int] = None,
    num_texts: Optional[int] = None,
    runtime: Optional[RuntimePolicy] = None,
) -> DataFrame:
    device = _embedding_device(runtime)
    model.to(device)
    model.eval()
    X = ds.X(limit=load_limit)
    y = ds.y(limit=load_limit)

    # Each input text should start with "query: " or "passage: ", even for
    # non-English texts. For tasks other than retrieval, you can simply use
    # the "query: " prefix. See also the discussion for this model on HF, but
    # the TL;DR is basically that it doesn't matter too much which you prepend,
    # but empirically, using "query: " seems to give better performance overall
    all_texts = X.apply(lambda text: f"query: {text}").tolist()
    if num_texts is not None:
        all_texts = all_texts[:num_texts]
        y = y.iloc[:num_texts]

    # Tokenize the input texts
    # batch_dict is:
    # { input_ids: Tensor[B, N], attention_mask: Tensor[B, N] }
    #
    # B = batch size, i.e. number of samples
    # N = max number of tokens in longest sequence of tokens in batch
    #
    # input_ids has the tokens as integers (ones seem to be padding), i.e. is
    # the tokenization of the input. This of course will only be loosely
    # correlated with word count or etc.
    #
    # attention_mask is a boolean where mask[i, :] is such that for input i,
    # mask[i] is all ones up to number of non-masking tokens in input_ids
    all_embeddings = []
    with torch.no_grad():
        text_batches = [*batched(all_texts, batch_size)]
        for texts in tqdm(
            text_batches,
            total=len(text_batches),
            desc="Embedding batches",
            disable=len(text_batches) < 100,
        ):
            batch_dict = tokenizer(
                texts,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            batch_dict = batch_dict.to(device)
            mask = batch_dict["attention_mask"]
            assert isinstance(mask, Tensor)

            outputs = model(**batch_dict)
            embeddings = avg_pool(outputs.last_hidden_state, attention_mask=mask)
            # embeddings = F.normalize(embeddings, p=2, dim=1)  # shape [B, 1024]
            all_embeddings.append(embeddings.detach().cpu())

    embeddings = torch.cat(all_embeddings)
    N, p = embeddings.shape
    cols = [f"embed{i:04d}" for i in range(p)]
    df_embed = DataFrame(data=embeddings.numpy(), columns=cols, index=y.index)
    df = pd.concat([df_embed, y], axis=1)
    df.rename(columns={y.name: "target"}, inplace=True)
    return df


def get_vision_embeddings(
    ds: VisionDataset,
    processor: SiglipProcessor,
    model: SiglipModel,
    batch_size: int = 2,
    load_limit: Optional[int] = None,
    num_imgs: Optional[int] = None,
    runtime: Optional[RuntimePolicy] = None,
) -> DataFrame:
    device = _embedding_device(runtime)
    model.to(device)
    model.eval()
    X, y = ds.X(limit=load_limit), ds.y(limit=load_limit)

    all_imgs = X.tolist()
    if num_imgs is not None:
        all_imgs = all_imgs[:num_imgs]
        y = y.iloc[:num_imgs]

    all_embeddings = []
    with torch.no_grad():
        img_batches = [*batched(all_imgs, batch_size)]
        for img in tqdm(
            img_batches,
            total=len(img_batches),
            desc="Embedding images",
            disable=len(img_batches) < 20,
        ):
            processed: BatchFeature = processor(
                text=None,  # type: ignore
                images=img,
                # padding below is what is recommended at:
                # https://huggingface.co/docs/transformers/en/model_doc/siglip
                padding="max_length",
                # truncation=True,  # also leave at defauly, use later if needed
                # max_length=1024,  # leave at default, only add this if needed
                return_tensors="pt",  # type: ignore
            )
            processed = processed.to(device)
            outputs = model.vision_model(
                **processed, output_hidden_states=False, output_attentions=False
            )
            embeddings = outputs.pooler_output

            # embeddings = avg_pool(embeddings2)
            # embeddings = F.normalize(embeddings, p=2, dim=1)  # shape [B, 1024]
            all_embeddings.append(embeddings.detach().cpu())

    embeddings = torch.cat(all_embeddings)
    N, p = embeddings.shape
    cols = [f"embed{i:04d}" for i in range(p)]
    df_embed = DataFrame(data=embeddings.numpy(), columns=cols, index=y.index)
    df = pd.concat([df_embed, y], axis=1)
    df.rename(columns={y.name: "target"}, inplace=True)

    # Avoid the following nonsense:
    #
    # TypeError: Feature names are only supported if all input features have
    # string names, but your input has ['NoneType', 'str'] as feature name /
    # column name types. If you want feature names to be stored and
    # validated, you must convert them all to strings, by using X.columns =
    # X.columns.astype(str) for example. Otherwise you can remove feature /
    # column names from your input data, or convert them all to a non-string
    # data type.
    df.columns = df.columns.astype(str)
    return df


@overload
def get_embeddings(
    ds: VisionDataset,
    processor: SiglipProcessor,
    model: SiglipModel,
    batch_size: Optional[int] = 2,
    load_limit: Optional[int] = None,
    max_samples: Optional[int] = None,
    runtime: Optional[RuntimePolicy] = None,
) -> DataFrame: ...


@overload
def get_embeddings(
    ds: NLPDataset,
    processor: XLMRobertaTokenizerFast,
    model: XLMRobertaModel,
    batch_size: Optional[int] = 2,
    load_limit: Optional[int] = None,
    max_samples: Optional[int] = None,
    runtime: Optional[RuntimePolicy] = None,
) -> DataFrame: ...


def get_embeddings(
    ds: Union[VisionDataset, NLPDataset],
    processor: Union[XLMRobertaTokenizerFast, SiglipProcessor],
    model: Union[XLMRobertaModel, SiglipModel],
    batch_size: Optional[int] = 2,
    load_limit: Optional[int] = None,
    max_samples: Optional[int] = None,
    runtime: Optional[RuntimePolicy] = None,
) -> DataFrame:
    batch_size = batch_size or 2
    if isinstance(ds, VisionDataset):
        return get_vision_embeddings(
            ds=ds,
            processor=cast(SiglipProcessor, processor),
            model=cast(SiglipModel, model),
            batch_size=batch_size,
            load_limit=load_limit,
            num_imgs=max_samples,
            runtime=runtime,
        )
    elif isinstance(ds, NLPDataset):
        return get_nlp_embeddings(
            ds=ds,
            tokenizer=cast(XLMRobertaTokenizerFast, processor),
            model=cast(XLMRobertaModel, model),
            batch_size=batch_size,
            load_limit=load_limit,
            num_texts=max_samples,
            runtime=runtime,
        )
    else:
        raise ValueError(f"Unrecognized dataset type: {ds}")
