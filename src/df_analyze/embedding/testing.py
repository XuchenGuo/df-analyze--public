from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent  # isort: skip
# ROOT2 = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# sys.path.append(str(ROOT2))  # isort: skip
# fmt: on

import os
import sys
import traceback
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import (
    Optional,
    cast,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pandas import DataFrame, Series
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.model_selection import StratifiedShuffleSplit
from torch import Tensor
from tqdm import tqdm
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.siglip.modeling_siglip import SiglipModel
from transformers.models.siglip.processing_siglip import SiglipProcessor
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaModel
from transformers.models.xlm_roberta.tokenization_xlm_roberta_fast import (
    XLMRobertaTokenizerFast,
)

from df_analyze.embedding.dataset_files import (
    CLS_DATAFILES,
    REG_DATAFILES,
    VISION_CLS,
)
from df_analyze.embedding.datasets import NLPDataset, VisionDataset
from df_analyze.embedding.download import (
    load_nlp_intfloat_ml_model_offline,
    load_siglip_offline,
)
from df_analyze.embedding.loading import _load_datafile
from df_analyze.embedding.utils import batched, get_n_test_samples, get_reg_stratify

MACOS_NLP_RUNTIMES = ROOT / "nlp_embed_runtimes.parquet"
MACOS_VISION_RUNTIMES = ROOT / "vision_embed_runtimes.parquet"
NIAGARA_NLP_RUNTIMES = ROOT / "nlp_embed_runtimes_niagara.parquet"
NIAGARA_VISION_RUNTIMES = ROOT / "vision_embed_runtimes_niagara.parquet"


def parquet_file(path: Path) -> Path:
    if path.suffix == ".parquet":
        return path
    out = Path(str(path).replace(path.suffix, ".parquet"))
    return out


class VisionTestingDataset:
    def __init__(
        self,
        name: str,
        root: Path,
        is_cls: bool,
    ) -> None:
        self.name = name
        self.root = root
        self.is_cls = self.is_classification = is_cls
        self._df = None

    def __str__(self) -> str:
        cls = f"{self.__class__.__name__}"
        return f"{cls}('{self.name}')"

    __repr__ = __str__

    @property
    def X(self) -> Series:
        df = self.load_raw()
        return df["image"]

    @property
    def y(self) -> Series:
        df = self.load_raw()
        if "Places_in_Japan" in self.root.name:
            raise RuntimeError(f"This dataset: {self.name}@{self.root} has no labels.")
        return df["label"]

    def load_raw(self) -> DataFrame:
        # https://github.com/python-pillow/Pillow/issues/4987#issuecomment-710994934
        #
        # "
        # To protect against potential DOS attacks caused by “decompression
        # bombs” (i.e. malicious files which decompress into a huge amount of
        # data and are designed to crash or cause disruption by using up a lot of
        # memory), Pillow will issue a DecompressionBombWarning if the number of
        # pixels in an image is over a certain limit, PIL.Image.MAX_IMAGE_PIXELS.
        #
        # This threshold can be changed by setting PIL.Image.MAX_IMAGE_PIXELS. It
        # can be disabled by setting Image.MAX_IMAGE_PIXELS = None.
        #
        # If desired, the warning can be turned into an error with
        # warnings.simplefilter('error', Image.DecompressionBombWarning) or
        # suppressed entirely with warnings.simplefilter('ignore',
        # Image.DecompressionBombWarning). See also the logging documentation to
        # have warnings output to the logging facility instead of stderr.
        #
        # If the number of pixels is greater than twice
        # PIL.Image.MAX_IMAGE_PIXELS, then a DecompressionBombError will be
        # raised instead. So:
        #
        #   from PIL import Image
        #   Image.MAX_IMAGE_PIXELS = None   # disables the warning
        #   Image.open(...)   # whatever operation you now run should work
        # "

        Image.MAX_IMAGE_PIXELS = None  # disables the warning

        if self._df is not None:
            return self._df
        pq = self.root / "all.parquet"
        # PyArrow engine needed to properly decode bytes column
        df = pd.read_parquet(pq, engine="pyarrow")
        im = df["image"].apply(
            lambda b: Image.open(BytesIO(b)).convert("RGB")
        )  # type: ignore
        self._df = pd.concat([df["label"], im], axis=1) if "label" in df.columns else im
        return self._df

    def to_embedding_dataset(self) -> VisionDataset:
        pq = self.root / "all.parquet"
        return VisionDataset(datapath=pq, name=self.name)

    @classmethod
    def get_all_cls(cls) -> list[VisionTestingDataset]:
        datas = []
        roots = sorted(VISION_CLS.glob("*"))
        for root in roots:
            if "LaTeX" in root.name:
                continue  # BW, 2-dimensional images
            if "GenAI-Bench" in root.name:
                continue  # takes too long for a quick test
            if "Places_in_Japan" in root.name:
                continue  # missing labels
            # if "brain-mri" in root.name:
            #     continue  # too many dims
            data = cls(name=root.name, root=root, is_cls=True)
            datas.append(data)
        return datas


class NLPTestingDataset:
    def __init__(
        self,
        name: str,
        root: Path,
        datafiles: dict[str, Path | list[str] | None],
        is_cls: bool,
    ) -> None:
        self.name = name
        self.root = root
        self.is_cls = self.is_classification = is_cls
        self.datafiles: dict[str, Path | list[str] | None] = deepcopy(datafiles)
        self.datafiles.pop("root")
        self.labels = self.datafiles.pop("labels", None)
        for subset, path in self.datafiles.items():  # make paths rel to root
            if path is not None and isinstance(path, Path):
                self.datafiles[subset] = self.root / path
        targets = self.datafiles.pop("targetcols")
        assert isinstance(targets, list)
        if len(targets) < 1:
            raise ValueError(f"Must specify targets for data: {name} at {root}")
        self.targets: list[str] = targets
        self.textcol = cast(str, self.datafiles.pop("textcol"))
        self.dropcols = cast(list[str], self.datafiles.pop("dropcols"))
        self.namecols = cast(list[str], self.datafiles.pop("labelnamecols"))
        self._df = None

    @property
    def X(self) -> Series:
        df = self.load_raw()
        return df[self.textcol]

    @property
    def y(self) -> Series:
        df = self.load_raw()
        return df[self.targets[0]]

    def load_raw(self) -> DataFrame:
        if self._df is not None:
            return self._df

        all_out = self.datafiles["all"]
        assert all_out is not None, f"Impossible: {self.name}"
        assert isinstance(all_out, Path), f"Impossible: {self.name}"
        pq = parquet_file(all_out)

        if pq.exists():
            df = _load_datafile(pq)  # type: ignore
            if df is None:
                raise ValueError("Impossible!")
            self.rename_cols(df)
            self.drop_corrupt(df)
            # df.to_parquet(pq)  # REMOVE
            self._df = df
            return df

        # no pq, must save
        if all_out.exists():  # just load this file instead
            df = _load_datafile(all_out)  # type: ignore
            if df is None:
                raise ValueError("Impossible!")
            self.rename_cols(df)
            self.drop_corrupt(df)

            df.to_parquet(pq)
            print(f"Saved concatenated data to {pq}")
            self._df = df
            return df

        # If a simple "all" file is not provided, vertically concat all others
        datafiles = deepcopy(self.datafiles)
        datafiles.pop("all")
        dfs = [_load_datafile(file) for file in datafiles.values()]  # type: ignore
        # TODO: check for duplicates
        df = pd.concat(dfs, axis=0, ignore_index=True)
        self.rename_cols(df)
        self.drop_corrupt(df)
        self._df = df

        df.to_parquet(pq)
        print(f"Saved concatenated data to {pq}")
        return df

    def load(self) -> DataFrame:
        raw = self.load_raw()
        if len(self.targets) == 1:
            targetcol = self.targets[0]
        else:  # for now, just use first target
            targetcol = self.targets[0]

        textcol = self.textcol
        return raw.loc[:, [textcol, targetcol]].copy()

    def rename_cols(self, df: DataFrame) -> None:
        textcol = self.textcol
        targetcol = self.targets[0]
        target = "label" if self.is_cls else "target"
        df.rename(columns={textcol: "text", targetcol: target}, inplace=True)
        self.textcol = "text"
        self.targets[0] = target

    def drop_corrupt(self, df: DataFrame) -> None:
        ix = df["text"].apply(lambda x: not isinstance(x, str))
        ix = np.where(ix)[0]
        if len(ix) > 10:
            raise ValueError(
                "WARNING! Appear to be dropping a very large amount of data! "
                "Check to make sure you aren't over-writing existing data with "
                "an empty DataFrame."
            )
        df.drop(index=ix, inplace=True)  # type: ignore

    def to_embedding_dataset(self) -> NLPDataset:
        all_out = self.datafiles["all"]
        assert all_out is not None, f"Impossible: {self.name}"
        assert isinstance(all_out, Path), f"Impossible: {self.name}"
        pq = parquet_file(all_out)
        if not pq.exists():
            self.load()  # this will generate the .parquet file
        return NLPDataset(datapath=pq, name=self.name)

    def has_source_data(self) -> bool:
        for path in self.datafiles.values():
            if not isinstance(path, Path):
                continue
            if path.is_file() or parquet_file(path).is_file():
                return True
        return False

    @staticmethod
    def get_all_cls() -> list[NLPTestingDataset]:
        datas = []
        for dsname, info in tqdm(
            CLS_DATAFILES.items(), total=len(CLS_DATAFILES), disable=True
        ):
            data = NLPTestingDataset(
                name=dsname, root=info["root"], datafiles=info, is_cls=True
            )
            if data.has_source_data():
                datas.append(data)
        return datas

    @staticmethod
    def get_all_reg() -> list[NLPTestingDataset]:
        datas = []
        for dsname, info in tqdm(
            REG_DATAFILES.items(), total=len(REG_DATAFILES), disable=True
        ):
            data = NLPTestingDataset(
                name=dsname, root=info["root"], datafiles=info, is_cls=False
            )
            if data.has_source_data():
                datas.append(data)
        return datas

    @staticmethod
    def get_all() -> list[NLPTestingDataset]:
        return NLPTestingDataset.get_all_cls() + NLPTestingDataset.get_all_reg()


def test_load_all_and_preview_sizes() -> None:
    dses = NLPTestingDataset.get_all()
    datasets = []
    all_stats = []
    for ds in tqdm(dses):
        df = ds.load()
        dsname = ds.name
        try:
            lengths = ds.X.str.split(" ").apply(lambda s: len(s))
            stats = (
                lengths.astype(float)
                .describe(percentiles=[0.1, 0.25, 0.9, 0.95, 0.99])
                .to_frame()
                .T.rename(columns={"50%": "med"})
                .loc[:, ["min", "mean", "med", "90%", "95%", "99%", "max"]]
            )
            stats.insert(0, "ds", dsname)
            all_stats.append(stats)
            print(stats)
        except TypeError as e:
            print(e)

        print("=" * 81)
        print(f"{dsname}: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(df.head())
        datasets.append((dsname, df))
    datasets = sorted(datasets, key=lambda pair: len(pair[1]))
    stats = pd.concat(all_stats, axis=0, ignore_index=True)
    for dsname, df in datasets:
        print(f"{dsname}: {df.shape}")
    pd.options.display.max_rows = 100
    print(stats)


def avg_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def get_nlp_embeddings(
    ds: NLPTestingDataset,
    tokenizer: XLMRobertaTokenizerFast,
    model: XLMRobertaModel,
    batch_size: int = 32,
    max_texts: int = 1024,
) -> tuple[DataFrame, Series, DataFrame, float, float]:
    X, y = ds.X.reset_index(drop=True), ds.y.reset_index(drop=True)

    strat = y if ds.is_cls else get_reg_stratify(y)
    ss = StratifiedShuffleSplit(n_splits=1, train_size=max_texts)
    ix_train = next(ss.split(y, strat))[0]  # type: ignore
    X_tr, y_tr = X.iloc[ix_train], y.iloc[ix_train]
    strat = strat.iloc[ix_train]

    # Each input text should start with "query: " or "passage: ", even for
    # non-English texts. For tasks other than retrieval, you can simply use
    # the "query: " prefix. See also the discussion for this model on HF, but
    # the TL;DR is basically that it doesn't matter too much which you prepend,
    # but empirically, using "query: " seems to give better performance overall
    all_texts = X_tr.apply(lambda text: f"query: {text}").tolist()
    lengths = Series([len(s) for s in all_texts])
    stats = (
        lengths.astype(float)
        .describe(percentiles=[0.9])
        .to_frame()
        .T.rename(columns={"50%": "med"})
        .loc[:, ["min", "mean", "med", "90%", "max"]]
    )

    # Tokenize the input texts
    # batch_dict is:
    # {
    #     input_ids: Tensor[B, N],
    #     attention_mask: Tensor[B, N],
    # }
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
        start = perf_counter()
        for texts in tqdm(
            text_batches, total=len(text_batches), desc="Embedding batches"
        ):
            batch_dict = tokenizer(
                texts,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            mask = batch_dict["attention_mask"]
            assert isinstance(mask, Tensor)

            outputs = model(**batch_dict)
            embeddings = avg_pool(outputs.last_hidden_state, attention_mask=mask)
            # embeddings = F.normalize(embeddings, p=2, dim=1)  # shape [B, 1024]
            all_embeddings.append(embeddings)
        elapsed_s = perf_counter() - start
    samples_per_s = max_texts / elapsed_s

    start = perf_counter()
    embeddings = torch.cat(all_embeddings)
    cols = [f"embed{i+1:4d}" for i in range(embeddings.shape[1])]
    df_embed = DataFrame(data=embeddings.numpy(), columns=cols)
    df_text = X_tr.to_frame().rename(columns=lambda s: "text")
    df_targ = y_tr.to_frame().rename(columns=lambda s: "target")
    df_embed.index = df_text.index.copy()
    df = pd.concat([df_text, df_targ, df_embed], axis=1, ignore_index=False)
    elapsed_s = perf_counter() - start
    postproc_samples_per_s = max_texts / elapsed_s
    return df, strat, stats, samples_per_s, postproc_samples_per_s


def check_ds_vision_padding(
    ds: VisionTestingDataset,
    processor: SiglipProcessor,
    model: SiglipModel,
    batch_size: int = 32,
    max_imgs: int = 1024,
) -> None:
    print(f"Testing {ds.name}...")
    X, y = ds.X, ds.y

    strat = y if ds.is_cls else get_reg_stratify(y)
    ss = StratifiedShuffleSplit(n_splits=1, train_size=max_imgs)
    ix_train = next(ss.split(y, strat))[0]  # type: ignore
    X_tr = X.iloc[ix_train]
    strat = strat.iloc[ix_train]
    all_imgs = X_tr.tolist()

    all_embeddings = []
    with torch.no_grad():
        img_batches = [*batched(all_imgs, batch_size)]
        img = img_batches[0]
        processed: BatchFeature = processor(
            text=None,  # type: ignore
            images=img,
            # padding="longest",
            padding=True,
            return_tensors="pt",  # type: ignore
        )
        outputs = model.vision_model(
            **processed, output_hidden_states=False, output_attentions=False
        )
        embeddings = outputs.pooler_output

        # embeddings = avg_pool(embeddings2)
        # embeddings = F.normalize(embeddings, p=2, dim=1)  # shape [B, 1024]
        all_embeddings.append(embeddings)


def get_vision_embeddings(
    ds: VisionTestingDataset,
    processor: SiglipProcessor,
    model: SiglipModel,
    batch_size: int = 32,
    max_imgs: int = 1024,
) -> tuple[DataFrame, Series, DataFrame, float, float]:
    X, y = ds.X.reset_index(drop=True), ds.y.reset_index(drop=True)

    strat = y if ds.is_cls else get_reg_stratify(y)
    ss = StratifiedShuffleSplit(n_splits=1, train_size=max_imgs)
    ix_train = next(ss.split(y, strat))[0]  # type: ignore
    X_tr, y_tr = X.iloc[ix_train], y.iloc[ix_train]
    strat = strat.iloc[ix_train]

    all_imgs = X_tr.tolist()
    s1 = Series([s.size[0] for s in all_imgs], name="h")
    s2 = Series([s.size[1] for s in all_imgs], name="w")
    shapes = pd.concat([s1, s2], axis=1)
    stats = (
        shapes.astype(float)
        .describe(percentiles=[0.9])
        .T.rename(columns={"50%": "med"})
        .loc[:, ["min", "mean", "med", "90%", "max"]]
    )

    all_embeddings = []
    with torch.no_grad():
        img_batches = [*batched(all_imgs, batch_size)]
        start = perf_counter()
        for img in tqdm(img_batches, total=len(img_batches), desc="Embedding batches"):
            processed: BatchFeature = processor(
                text=None,  # type: ignore
                images=img,
                padding="max_length",
                # truncation=True,
                # max_length=1024,
                return_tensors="pt",  # type: ignore
            )
            outputs = model.vision_model(
                **processed, output_hidden_states=False, output_attentions=False
            )
            embeddings = outputs.pooler_output

            # embeddings = avg_pool(embeddings2)
            # embeddings = F.normalize(embeddings, p=2, dim=1)  # shape [B, 1024]
            all_embeddings.append(embeddings)
        elapsed_s = perf_counter() - start
    samples_per_s = max_imgs / elapsed_s

    start = perf_counter()
    embeddings = torch.cat(all_embeddings)
    cols = [f"embed{i+1:4d}" for i in range(embeddings.shape[1])]
    df_embed = DataFrame(data=embeddings.numpy(), columns=cols)
    df_img = X_tr.to_frame().rename(columns=lambda s: "img")
    df_targ = y_tr.to_frame().rename(columns=lambda s: "target")
    df_embed.index = df_img.index.copy()
    df = pd.concat([df_img, df_targ, df_embed], axis=1, ignore_index=False)
    elapsed_s = perf_counter() - start
    postproc_samples_per_s = max_imgs / elapsed_s
    return df, strat, stats, samples_per_s, postproc_samples_per_s


def estimate_nlp_embedding_times() -> None:
    ON_CLUSTER = os.environ.get("CC_CLUSTER") is not None
    N = 256 if ON_CLUSTER else 32
    BATCHES = [8, 16, 40] if ON_CLUSTER else [2, 4, 8]
    OUT = NIAGARA_NLP_RUNTIMES if ON_CLUSTER else MACOS_NLP_RUNTIMES
    # on Macbook, batch=2 seems fastest (4 very close, 1 by far too slow)

    model, tokenizer = load_nlp_intfloat_ml_model_offline()
    dses = NLPTestingDataset.get_all_cls()
    results = []

    for ds in dses:
        print("=" * 81)
        print(ds.name)
        N_samples = len(ds.y)
        for B in BATCHES:
            try:
                start = perf_counter()
                embeds, strat, stats, proc_speed, postproc_speed = get_nlp_embeddings(
                    ds=ds, tokenizer=tokenizer, model=model, batch_size=B, max_texts=N
                )
                total = round(perf_counter() - start, 3)
                proc_speed = round(proc_speed, 3)
                embed_time = round(N / proc_speed, 1)
                post = f"{round(postproc_speed / 1000, 0)}K"
            except Exception as e:
                print(e)
                continue

            print(
                f"At batch = {B}, total time = {total}; embed time = {embed_time}; embedding speed = {proc_speed} samples/s; postproc speed = {post} samples/s"
            )
            row = DataFrame(
                {
                    "ds": ds.name,
                    "n_samp": N_samples,
                    "batch": B,
                    "total_s": total,
                    "est_h": ((total / N) * N_samples) / 3600,
                    "embed_s": embed_time,
                    "embed_speed": proc_speed,
                },
                index=[0],
            )
            row = pd.concat([row, stats], axis=1, ignore_index=False)
            results.append(row)
            print(row)

    runtimes = pd.concat(results, axis=0, ignore_index=True)
    pd.options.display.max_rows = 500
    runtimes.to_parquet(OUT)
    print(runtimes)
    print(
        runtimes.groupby(["ds", "n_samp"])
        .apply(lambda grp: grp.nlargest(1, "total_s"), include_groups=False)
        .droplevel(2)
        .sort_values(by="embed_speed")
    )
    print(f"Saved runtimes to {OUT}")


def get_optimal_nlp_batch(ds: NLPTestingDataset) -> int:
    ON_CLUSTER = os.environ.get("CC_CLUSTER") is not None
    OUT = NIAGARA_NLP_RUNTIMES if ON_CLUSTER else MACOS_NLP_RUNTIMES
    DEFAULT = 8 if ON_CLUSTER else 2
    try:
        if not OUT.exists():
            return DEFAULT
        runtimes = pd.read_parquet(OUT)
        runtimes = (
            runtimes.groupby(["ds", "n_samp"])
            .apply(lambda grp: grp.nsmallest(1, "total_s"), include_groups=False)
            .droplevel(2)
            .reset_index()
        )
        if ds.name not in runtimes["ds"].values:
            return DEFAULT
        return runtimes[runtimes["ds"] == ds.name]["batch"].item()
    except Exception as e:
        traceback.print_exc()
        print(f"Got error: {e}")
        return DEFAULT


def get_optimal_vision_batch(ds: VisionTestingDataset) -> int:
    ON_CLUSTER = os.environ.get("CC_CLUSTER") is not None
    OUT = NIAGARA_VISION_RUNTIMES if ON_CLUSTER else MACOS_VISION_RUNTIMES
    DEFAULT = 8 if ON_CLUSTER else 2
    try:
        if not OUT.exists():
            return DEFAULT
        runtimes = pd.read_parquet(OUT)
        runtimes = (
            runtimes.groupby(["ds", "n_samp"])
            .apply(lambda grp: grp.nsmallest(1, "total_s"), include_groups=False)
            .droplevel(2)
            .reset_index()
        )
        if ds.name not in runtimes["ds"].values:
            return DEFAULT
        return runtimes[runtimes["ds"] == ds.name]["batch"].item()
    except Exception as e:
        traceback.print_exc()
        print(f"Got error: {e}")
        return DEFAULT


def estimate_vision_embedding_times(n_samples: Optional[int] = None) -> None:
    ON_CLUSTER = os.environ.get("CC_CLUSTER") is not None
    N = get_n_test_samples(n_samples)
    BATCHES = [8, 16, 40] if ON_CLUSTER else [2, 4, 8]
    OUT = NIAGARA_VISION_RUNTIMES if ON_CLUSTER else MACOS_VISION_RUNTIMES
    # on Macbook, batch=2 seems fastest (4 very close, 1 by far too slow)

    model, processor = load_siglip_offline()
    dses = VisionTestingDataset.get_all_cls()

    results = []

    for ds in dses:
        if "Places_in_Japan" in ds.name:  # no targets to stratify on
            continue
        if "rare-species" != ds.name:
            continue
        print("=" * 81)
        print(ds.name)
        N_samples = len(ds.y)
        for B in BATCHES:
            try:
                start = perf_counter()
                embeds, strat, stats, proc_speed, postproc_speed = get_vision_embeddings(
                    ds=ds, processor=processor, model=model, batch_size=B, max_imgs=N
                )
                total = round(perf_counter() - start, 3)
                proc_speed = round(proc_speed, 3)
                embed_time = round(N / proc_speed, 1)
                post = f"{round(postproc_speed / 1000, 0)}K"
            except Exception as e:
                traceback.print_exc()
                print(e)
                continue

            print(
                f"At batch = {B}, total time = {total}; embed time = {embed_time}; embedding speed = {proc_speed} samples/s; postproc speed = {post} samples/s"
            )
            wide_stats = pd.concat(
                [
                    stats.loc["h"]  # type: ignore
                    .to_frame()
                    .T.rename(columns=lambda col: f"{col}_h")
                    .reset_index(drop=True),
                    stats.loc["w"]  # type: ignore
                    .to_frame()
                    .T.rename(columns=lambda col: f"{col}_w")
                    .reset_index(drop=True),
                ],
                axis=1,
            )
            wide_stats = wide_stats.loc[  # type: ignore
                :,
                [
                    "min_h",
                    "min_w",
                    "mean_h",
                    "mean_w",
                    "med_h",
                    "med_w",
                    "90%_h",
                    "90%_w",
                    "max_h",
                    "max_w",
                ],
            ]
            row = DataFrame(
                {
                    "ds": ds.name,
                    "n_samp": N_samples,
                    "batch": B,
                    "total_s": total,
                    "est_h": ((total / N) * N_samples) / 3600,
                    "embed_s": embed_time,
                    "embed_speed": proc_speed,
                },
                index=[0],
            )
            row = pd.concat([row, wide_stats], axis=1, ignore_index=False)
            results.append(row)
            print(row)

    runtimes = pd.concat(results, axis=0, ignore_index=True)
    pd.options.display.max_rows = 500
    runtimes.to_parquet(OUT)
    print(runtimes)
    print(
        runtimes.groupby(["ds", "n_samp"])
        .apply(lambda grp: grp.nlargest(1, "total_s"), include_groups=False)
        .droplevel(2)
        .sort_values(by="embed_speed")
    )
    print(f"Saved runtimes to {OUT}")


def cluster_nlp_sanity_check(n_samples: Optional[int] = None) -> None:
    ON_CLUSTER = os.environ.get("CC_CLUSTER") is not None
    if n_samples is None:
        N = 256 if ON_CLUSTER else 128
    else:
        N = get_n_test_samples(n_samples)
    # BATCHES = [8, 16, 40] if ON_CLUSTER else [2, 4, 8]
    # on Macbook, batch=2 seems fastest (4 very close, 1 by far too slow)
    # OUT = NIAGARA_NLP_RUNTIMES if ON_CLUSTER else MACOS_NLP_RUNTIMES

    model, tokenizer = load_nlp_intfloat_ml_model_offline()
    dses = NLPTestingDataset.get_all_cls()
    if len(dses) < 1:
        raise FileNotFoundError("Could not find any NLP datasets.")
    results = []

    for ds in dses:
        if ds.name == "go_emotions":
            continue  # multilabel
        print("=" * 81)
        print(ds.name)
        batch = get_optimal_nlp_batch(ds)
        try:
            embeds, strat = get_nlp_embeddings(
                ds=ds, tokenizer=tokenizer, model=model, batch_size=batch, max_texts=N
            )[:2]
            X = embeds.drop(columns=["text", "target"])
            dfs = []
            for metricname, metric in dict(
                cosine=cosine_similarity,
                euclid=lambda x: 10 / (1 + euclidean_distances(x)),  # 10 since usu. small
            ).items():
                sims = metric(X.values)
                sims = DataFrame(data=sims, index=X.index, columns=X.index)
                for clust in sorted(strat.unique()):
                    ix = strat == clust
                    df = sims.loc[ix, ix]
                    df = (
                        df.where(np.triu(np.ones(df.shape), k=1).astype(bool))
                        .stack()
                        .reset_index()
                    )
                    df.columns = ["x1", "x2", "sim"]
                    within = df["sim"].mean()
                    between = sims.loc[ix, ~ix].values.mean()
                    row = DataFrame(
                        {
                            "cls": clust,
                            "within": within,
                            "between": between,
                            "OK": within > between,
                            "metric": metricname,
                        },
                        index=[0],
                    )
                    dfs.append(row)
            df = pd.concat(dfs, axis=0, ignore_index=True)
            df["ds"] = ds.name
            results.append(df)
            print(df)

        except Exception as e:
            raise RuntimeError(
                f"Could not run NLP clustering check for {ds.name}: {e}"
            ) from e


def vision_padding_check() -> None:
    model, processor = load_siglip_offline()
    dses = VisionTestingDataset.get_all_cls()

    if len(dses) < 1:
        raise FileNotFoundError("Could not find any vision datasets.")

    for ds in tqdm(dses, desc="Testing dataset padding"):
        try:
            check_ds_vision_padding(
                ds=ds, processor=processor, model=model, batch_size=2, max_imgs=16
            )
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(
                f"Could not check image padding for {ds.name}: {e}"
            ) from e


def cluster_vision_sanity_check(n_samples: Optional[int] = None) -> None:
    ON_CLUSTER = os.environ.get("CC_CLUSTER") is not None
    if n_samples is None:
        N = 256 if ON_CLUSTER else 16
    else:
        N = get_n_test_samples(n_samples)

    model, processor = load_siglip_offline()
    dses = VisionTestingDataset.get_all_cls()
    if len(dses) < 1:
        raise FileNotFoundError("Could not find any vision datasets.")
    results = []

    for ds in dses:
        print("=" * 81)
        print(ds.name)
        batch = get_optimal_vision_batch(ds)
        try:
            embeds, strat = get_vision_embeddings(
                ds=ds, processor=processor, model=model, batch_size=batch, max_imgs=N
            )[:2]
            X = embeds.drop(columns=["img", "target"])
            dfs = []
            for metricname, metric in dict(
                cosine=cosine_similarity, euclid=lambda x: euclidean_distances(x)
            ).items():
                sims = metric(X.values)
                if metricname == "euclid":
                    sims = 1 - sims / sims.max()
                sims = DataFrame(data=sims, index=X.index, columns=X.index)
                for clust in sorted(strat.unique()):
                    ix = strat == clust
                    df = sims.loc[ix, ix]
                    stacked = df.where(
                        np.triu(np.ones(df.shape), k=1).astype(bool)
                    ).stack()
                    try:
                        df = stacked.reset_index()
                    except Exception as e:
                        traceback.print_exc()
                        raise RuntimeError(
                            f"Could not assemble similarities for {ds.name}: {e}"
                        ) from e

                    df.columns = ["x1", "x2", "sim"]
                    within = df["sim"].mean()
                    between = sims.loc[ix, ~ix].values.mean()
                    row = DataFrame(
                        {
                            "cls": clust,
                            "within": within,
                            "between": between,
                            "OK": within > between,
                            "metric": metricname,
                        },
                        index=[0],
                    )
                    dfs.append(row)
            df = pd.concat(dfs, axis=0, ignore_index=True)
            df["ds"] = ds.name
            results.append(df)
            print(df)

        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(
                f"Could not run vision clustering check for {ds.name}: {e}"
            ) from e


def detect_sus_images(ds: VisionTestingDataset) -> None:
    """
    Surprisingly, not about image onctent, but size, since some multi-panel
    animated comics are included in the deepghs/nsfw_detect, and these images
    are large enough to trigger PIL DecompressionBombWarning:

    > `DecompressionBombWarning: Image size (115600000 pixels) exceeds limit
    > of 89478485 pixels, could be decompression bomb DOS attack.`


    """
    # https://github.com/python-pillow/Pillow/issues/4987#issuecomment-710994934
    #
    # "
    # To protect against potential DOS attacks caused by “decompression
    # bombs” (i.e. malicious files which decompress into a huge amount of
    # data and are designed to crash or cause disruption by using up a lot of
    # memory), Pillow will issue a DecompressionBombWarning if the number of
    # pixels in an image is over a certain limit, PIL.Image.MAX_IMAGE_PIXELS.
    #
    # This threshold can be changed by setting PIL.Image.MAX_IMAGE_PIXELS. It
    # can be disabled by setting Image.MAX_IMAGE_PIXELS = None.
    #
    # If desired, the warning can be turned into an error with
    # warnings.simplefilter('error', Image.DecompressionBombWarning) or
    # suppressed entirely with warnings.simplefilter('ignore',
    # Image.DecompressionBombWarning). See also the logging documentation to
    # have warnings output to the logging facility instead of stderr.
    #
    # If the number of pixels is greater than twice
    # PIL.Image.MAX_IMAGE_PIXELS, then a DecompressionBombError will be
    # raised instead. So:
    #
    #   from PIL import Image
    #   Image.MAX_IMAGE_PIXELS = None   # disables the warning
    #   Image.open(...)   # whatever operation you now run should work
    # "

    Image.MAX_IMAGE_PIXELS = None  # disables the warning
    df = ds.load_raw()
    shapes = df["image"].apply(lambda i: i.size)
    shapes = pd.concat(
        [shapes.apply(lambda s: s[0]), shapes.apply(lambda s: s[1])],
        axis=1,
        keys=["x", "y"],
    ).max(axis=1)
    shapes.index = df.index
    ix_large = (-shapes).argsort()
    large = df.iloc[ix_large[:20]]
    for img in large["image"]:
        plt.imshow(img)
        plt.show()
    print(shapes)
    raise


if __name__ == "__main__":
    # test_load_all_and_preview_sizes()
    # sys.exit()

    for ds in NLPTestingDataset.get_all():
        ds.load()
        # ds.to_embedding_dataset()

    # download_nlp_intfloat_ml_model()
    # sys.exit()
    # load_nlp_intfloat_ml_model_offline()
    # estimate_nlp_embedding_times()
    # cluster_nlp_sanity_check()

    # model, tokenizer = load_nlp_intfloat_ml_model_offline()
    # download_siglip_model()

    # SUS ds is nsfw_detect
    # model, processor = load_siglip_offline()
    # dses = VisionDataset.get_all_cls()
    # estimate_vision_embedding_times()
    # cluster_vision_sanity_check()
