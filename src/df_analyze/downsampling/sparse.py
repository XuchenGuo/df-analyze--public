from __future__ import annotations

import bz2
import gzip
import lzma
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, BinaryIO, Sequence
from warnings import warn

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import Series
from scipy.sparse import csr_matrix, vstack
from sklearn.datasets import load_svmlight_files
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from df_analyze._constants import DOWNSAMPLE_MAX_DENSE_BYTES
from df_analyze.downsampling.base import resolve_n_features
from df_analyze.downsampling.containers import FeatureDownsampleResult
from df_analyze.downsampling.methods import (
    resolve_feature_downsample_method,
    select_indexed_columns,
)
from df_analyze.downsampling.screening import (
    resolve_screening_split,
)
from df_analyze.enumerables import ValidationMethod
from df_analyze.preprocessing.cleaning import normalize_continuous
from df_analyze.preprocessing.prepare import PreparationInfo, PreparedData


@dataclass
class SparseInput:
    matrices: list[csr_matrix]
    targets: list[NDArray[np.float64]]
    index_base: int


def _open_sparse(path: Path) -> BinaryIO:
    name = path.name.lower()
    if name.endswith(".gz"):
        return gzip.open(path, "rb")
    if name.endswith(".bz2"):
        return bz2.open(path, "rb")
    if name.endswith(".xz"):
        return lzma.open(path, "rb")
    return path.open("rb")


def _has_zero_index(path: Path) -> bool:
    with _open_sparse(path) as source:
        data_rows = 0
        for raw in source:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line or line.startswith("#"):
                continue
            for token in line.split()[1:]:
                token = token.split("#", 1)[0]
                if ":" not in token:
                    continue
                index = token.split(":", 1)[0]
                if index == "0":
                    return True
            data_rows += 1
        if data_rows > 0:
            return False
    raise ValueError(f"SVMlight input {path} does not contain any data rows.")


def _detect_index_base(path: Path) -> int:
    if _has_zero_index(path):
        return 0
    warn(
        "SVMlight index-base detection is ambiguous because feature index 0 was "
        "not found. Assuming one-based input; use --svmlight-index-base zero if "
        "feature 0 is absent from a zero-based file.",
        stacklevel=2,
    )
    return 1


def load_sparse_input(paths: Sequence[Path], index_base: str = "auto") -> SparseInput:
    if len(paths) == 0:
        raise ValueError("Expected at least one SVMlight input path.")
    requested = str(index_base).lower()
    if requested not in {"auto", "zero", "one"}:
        raise ValueError("SVMlight index base must be auto, zero, or one.")
    if requested == "auto":
        zero_indices = [_has_zero_index(path) for path in paths]
        if any(zero_indices):
            resolved = 0
            if not all(zero_indices):
                warn(
                    "Feature index 0 was found in only some SVMlight inputs. "
                    "Treating all inputs as zero-based; confirm that every file uses "
                    "the same index base.",
                    stacklevel=2,
                )
        else:
            warn(
                "SVMlight index-base detection is ambiguous because feature index 0 "
                "was not found. Assuming one-based input; use "
                "--svmlight-index-base zero if feature 0 is absent from a zero-based "
                "file.",
                stacklevel=2,
            )
            resolved = 1
    else:
        resolved = int(requested == "one")
    zero_based = resolved == 0
    with ExitStack() as stack:
        sources = [stack.enter_context(_open_sparse(path)) for path in paths]
        loaded = load_svmlight_files(sources, zero_based=zero_based)
    matrices = [loaded[idx].tocsr() for idx in range(0, len(loaded), 2)]
    targets = [np.asarray(loaded[idx + 1]) for idx in range(0, len(loaded), 2)]
    return SparseInput(matrices=matrices, targets=targets, index_base=resolved)


def _encode_target(
    values: NDArray[np.float64],
    is_classification: bool,
    name: str,
    encoder: LabelEncoder | None = None,
) -> tuple[Series, dict[int, str] | None, LabelEncoder | None]:
    if not is_classification:
        return Series(values.astype(float), name=name), None, None
    if encoder is None:
        encoder = LabelEncoder().fit(values)
    encoded = encoder.transform(values)
    labels = {idx: str(value) for idx, value in enumerate(encoder.classes_)}
    return Series(encoded, name=name), labels, encoder


def _materialize(
    X: csr_matrix,
    rows: NDArray[np.int_],
    selected: NDArray[np.int_],
    columns: Sequence[str],
) -> pd.DataFrame:
    values = X[rows][:, selected].toarray()
    return pd.DataFrame(values, columns=list(columns))


def sparse_prepared_splits(options: Any) -> list[
    tuple[PreparedData, PreparedData, FeatureDownsampleResult]
]:
    paths = [options.datapath, *options.test_paths]
    loaded = load_sparse_input(paths, options.svmlight_index_base)
    for path, matrix, target in zip(paths, loaded.matrices, loaded.targets):
        if not np.isfinite(matrix.data).all():
            raise ValueError(
                f"SVMlight predictors in {path} contain NaN or infinite values."
            )
        if not np.isfinite(target).all():
            raise ValueError(
                f"SVMlight target in {path} contains NaN or infinite values."
            )
    split_sources: list[
        tuple[csr_matrix, NDArray[np.float64], csr_matrix, NDArray[np.float64], Path]
    ] = []
    if len(loaded.matrices) == 1:
        matrix = loaded.matrices[0]
        target = loaded.targets[0]
        all_rows = np.arange(matrix.shape[0], dtype=int)
        stratify = target if options.is_classification else None
        train_rows, test_rows = train_test_split(
            all_rows,
            test_size=options.test_val_size,
            random_state=options.seed,
            stratify=stratify,
        )
        split_sources.append(
            (
                matrix[np.sort(train_rows)].tocsr(),
                target[np.sort(train_rows)],
                matrix[np.sort(test_rows)].tocsr(),
                target[np.sort(test_rows)],
                paths[0],
            )
        )
    else:
        method = getattr(options, "tests_method", ValidationMethod.List)
        if method is ValidationMethod.List:
            for idx in range(1, len(loaded.matrices)):
                split_sources.append(
                    (
                        loaded.matrices[0],
                        loaded.targets[0],
                        loaded.matrices[idx],
                        loaded.targets[idx],
                        paths[idx],
                    )
                )
        elif method is ValidationMethod.LODO:
            for train_idx in range(len(loaded.matrices)):
                test_matrices = [
                    matrix
                    for idx, matrix in enumerate(loaded.matrices)
                    if idx != train_idx
                ]
                test_targets = [
                    target
                    for idx, target in enumerate(loaded.targets)
                    if idx != train_idx
                ]
                split_sources.append(
                    (
                        loaded.matrices[train_idx],
                        loaded.targets[train_idx],
                        vstack(test_matrices, format="csr"),
                        np.concatenate(test_targets),
                        paths[train_idx],
                    )
                )
        else:
            raise ValueError(f"Invalid external validation method: {method}")

    outputs = []
    selection_cache: dict[
        int, tuple[NDArray[np.int_], FeatureDownsampleResult]
    ] = {}
    for train_source, train_target, test_source, test_target, test_path in split_sources:
        if np.unique(train_target).size <= 1:
            raise ValueError(f"Target variable {options.target} is constant.")
        y_train, labels, target_encoder = _encode_target(
            train_target, options.is_classification, options.target
        )
        try:
            y_test = _encode_target(
                test_target,
                options.is_classification,
                options.target,
                target_encoder,
            )[0]
        except ValueError as error:
            raise ValueError(
                f"SVMlight target in {test_path} contains labels that are not "
                "present in its training partition."
            ) from error
        train_rows = np.arange(train_source.shape[0], dtype=int)
        test_rows = np.arange(test_source.shape[0], dtype=int)
        n_select = resolve_n_features(options.n_feat_downsample, train_source.shape[1])
        resolved = resolve_feature_downsample_method(
            options.feat_downsample,
            len(train_rows),
            train_source.shape[1],
            n_select,
            options.is_classification,
            y_train,
        )
        effective, screening, tuning, fallback_note = resolve_screening_split(
            options.feat_downsample,
            resolved,
            y_train,
            options.downsample_screening_fraction,
            options.is_classification,
        )
        feature_names = [
            f"feature_{idx + loaded.index_base}"
            for idx in range(train_source.shape[1])
        ]
        cache_key = id(train_source)
        cached = selection_cache.get(cache_key)
        if cached is None:
            selected, result = select_indexed_columns(
                train_source,
                train_rows,
                y_train,
                options.is_classification,
                effective,
                options.n_feat_downsample,
                options.downsample_chunk_size,
                screening,
                feature_names,
                options.downsample_variance_threshold,
                options.downsample_save_scores,
                sparse_input=True,
                input_format="svmlight",
            )
            result.requested_method = options.feat_downsample.value
            if fallback_note is not None:
                result.notes.append(fallback_note)
            result.tuning_rows = tuning.tolist()
            result.tuning_samples = len(tuning)
            selection_cache[cache_key] = (selected.copy(), deepcopy(result))
        else:
            selected, result = cached[0].copy(), deepcopy(cached[1])
            result.fit_seconds = 0.0
            result.notes.append("Reused feature scores from the same training partition.")
        dense_bytes = (len(train_rows) + len(test_rows)) * len(selected) * 8
        if dense_bytes > DOWNSAMPLE_MAX_DENSE_BYTES:
            gib = dense_bytes / 1024**3
            raise MemoryError(
                f"Selected SVMlight train/test matrices require about {gib:.2f} GiB "
                "when dense. Reduce --n-feat-downsample."
            )
        started = perf_counter()
        selected_names = [feature_names[idx] for idx in selected]
        X_train = _materialize(train_source, train_rows, selected, selected_names)
        X_test = _materialize(test_source, test_rows, selected, selected_names)
        combined = pd.concat([X_train, X_test], axis=0, ignore_index=True)
        normalized = normalize_continuous(
            combined,
            robust=True,
            fit_indices=np.arange(len(X_train), dtype=int),
        )
        X_train = normalized.iloc[: len(X_train)].reset_index(drop=True)
        X_test = normalized.iloc[len(X_train) :].reset_index(drop=True)
        result.transform_seconds = perf_counter() - started
        info = PreparationInfo(
            original_shape=(
                train_source.shape[0] + test_source.shape[0],
                train_source.shape[1],
            ),
            final_shape=X_train.shape,
            n_samples_dropped_via_target_NaNs=0,
            n_cont_indicator_added=0,
            target_info=None,
            runtimes={
                "feature downsampling": result.fit_seconds + result.transform_seconds
            },
            is_classification=options.is_classification,
        )
        train = PreparedData(
            X_train,
            y_train,
            None,
            options.is_classification,
            X_cont=X_train,
            labels=labels,
            info=info,
            phase="train",
        )
        test_info = deepcopy(info)
        test_info.final_shape = X_test.shape
        test = PreparedData(
            X_test,
            y_test,
            None,
            options.is_classification,
            X_cont=X_test,
            labels=labels,
            info=test_info,
            phase="test",
        )
        outputs.append((train, test, result))
    return outputs
