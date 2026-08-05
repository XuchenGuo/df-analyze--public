from __future__ import annotations

import bz2
import gzip
import lzma
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, BinaryIO, Sequence, overload
from warnings import warn

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame, Series
from scipy.sparse import csr_matrix
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler

from df_analyze._constants import (
    DOWNSAMPLE_DENSE_PEAK_FACTOR,
    DOWNSAMPLE_MAX_DENSE_BYTES,
)
from df_analyze.downsampling.base import resolve_n_features
from df_analyze.downsampling.containers import FeatureDownsampleResult
from df_analyze.downsampling.methods import (
    resolve_feature_downsample_method,
    select_indexed_columns,
)
from df_analyze.downsampling.screening import (
    resolve_screening_split,
)
from df_analyze.enumerables import FeatureDownsampleMethod, ValidationMethod
from df_analyze.preprocessing.cleaning import sanitize_names
from df_analyze.preprocessing.inspection.containers import ColumnType, InspectionInfo
from df_analyze.preprocessing.inspection.inference import Inference, InferredKind
from df_analyze.preprocessing.inspection.inspection import inspect_data, unify_nans
from df_analyze.preprocessing.prepare import (
    PreparationInfo,
    PreparedData,
    prepare_data,
    raw_train_test_indices,
)


@dataclass
class SparseInput:
    matrices: list[csr_matrix]
    targets: list[NDArray[np.float64]]
    index_base: int


@dataclass(frozen=True)
class _SparsePart:
    target: NDArray[np.float64]
    rows: NDArray[np.int_]
    path: Path
    metadata: DataFrame | None = None


@dataclass(frozen=True)
class _SparseSplit:
    train: _SparsePart
    tests: list[_SparsePart]


class _IndexedFeatureNames(Sequence[str]):
    """Lazy SVMlight feature names with list-compatible indexing."""

    def __init__(self, size: int, index_base: int):
        self.size = int(size)
        self.index_base = int(index_base)

    def __len__(self) -> int:
        return self.size

    @overload
    def __getitem__(self, index: int) -> str: ...

    @overload
    def __getitem__(self, index: slice) -> list[str]: ...

    def __getitem__(self, index: int | slice) -> str | list[str]:
        if isinstance(index, slice):
            return [
                f"feature_{idx + self.index_base}"
                for idx in range(*index.indices(self.size))
            ]
        resolved = index + self.size if index < 0 else index
        if resolved < 0 or resolved >= self.size:
            raise IndexError(index)
        return f"feature_{resolved + self.index_base}"


class _MappedFeatureNames(_IndexedFeatureNames):
    def __init__(
        self,
        size: int,
        index_base: int,
        mapped_names: NDArray[np.object_],
    ):
        super().__init__(size, index_base)
        self.mapped_names = mapped_names

    def __getitem__(self, index: int | slice) -> str | list[str]:
        if isinstance(index, slice):
            return [self[idx] for idx in range(*index.indices(self.size))]
        resolved = index + self.size if index < 0 else index
        if resolved < 0 or resolved >= self.size:
            raise IndexError(index)
        mapped = self.mapped_names[resolved]
        if mapped is not None:
            return str(mapped)
        return super().__getitem__(resolved)


def _read_sidecar(path: Path, separator: str = ",") -> DataFrame:
    name = path.name.lower()
    if name.endswith((".parquet", ".pq")):
        return pd.read_parquet(path)
    if name.endswith((".tsv", ".tsv.gz", ".tsv.bz2", ".tsv.xz")):
        return pd.read_csv(path, sep="\t")
    if name.endswith((".csv", ".csv.gz", ".csv.bz2", ".csv.xz")):
        return pd.read_csv(path, sep=separator)
    if name.endswith((".json", ".json.gz", ".json.bz2", ".json.xz")):
        return pd.read_json(path)
    raise ValueError(
        f"Unsupported sidecar format for {path}; use CSV, TSV, JSON, or Parquet."
    )


def _feature_names_and_protected(
    path: Path | None,
    n_features: int,
    index_base: int,
    requested_names: Sequence[str],
    separator: str,
) -> tuple[Sequence[str], list[int]]:
    requested = set(map(str, requested_names))
    if path is None:
        indices: list[int] = []
        for name in requested:
            prefix = "feature_"
            if not name.startswith(prefix):
                raise KeyError(
                    f"Protected SVMlight feature {name!r} requires a feature map."
                )
            try:
                idx = int(name[len(prefix) :]) - index_base
            except ValueError as error:
                raise KeyError(
                    f"Protected SVMlight feature {name!r} requires a feature map."
                ) from error
            if idx < 0 or idx >= n_features:
                raise KeyError(f"Protected SVMlight feature {name!r} was not found.")
            indices.append(idx)
        return _IndexedFeatureNames(n_features, index_base), sorted(set(indices))

    mapping = _read_sidecar(path, separator)
    required = {"feature_index", "feature_name"}
    missing = sorted(required - set(mapping.columns))
    if missing:
        raise ValueError(f"Feature map {path} is missing columns: {missing}")
    numeric = pd.to_numeric(mapping["feature_index"], errors="raise")
    if not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError("Feature-map indices must be integers.")
    internal = numeric.to_numpy(dtype=int) - index_base
    if np.any((internal < 0) | (internal >= n_features)):
        raise ValueError("Feature-map indices fall outside the SVMlight matrix.")
    if len(np.unique(internal)) != len(internal):
        raise ValueError("Feature-map indices must be unique.")
    text_names = mapping["feature_name"].astype(str)
    if text_names.duplicated().any():
        raise ValueError("Feature-map names must be unique.")

    mapped = np.full(n_features, None, dtype=object)
    mapped[internal] = text_names.to_numpy(dtype=object)
    resolved_requested = internal[text_names.isin(requested).to_numpy()].tolist()
    unresolved = requested - set(text_names[text_names.isin(requested)])
    for name in list(unresolved):
        if not name.startswith("feature_"):
            continue
        try:
            idx = int(name.removeprefix("feature_")) - index_base
        except ValueError:
            continue
        if 0 <= idx < n_features and mapped[idx] is None:
            resolved_requested.append(idx)
            unresolved.remove(name)
    if unresolved:
        raise KeyError(
            f"Protected SVMlight features were not found: {sorted(unresolved)}"
        )

    protected = set(resolved_requested)
    if "protected" in mapping.columns:
        flag = (
            mapping["protected"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"1", "true", "yes", "y"})
            .to_numpy()
        )
        protected.update(internal[flag].tolist())
    return (
        _MappedFeatureNames(n_features, index_base, mapped),
        sorted(protected),
    )


def _validated_metadata(
    metadata: DataFrame,
    target: NDArray[np.float64],
    target_name: str,
    is_classification: bool,
    path: Path,
) -> DataFrame:
    frame = metadata.copy()
    frame.columns = frame.columns.astype(str)
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"Clinical sidecar {path} has duplicate columns: {duplicates}")
    if len(frame) != len(target):
        raise ValueError(
            f"Clinical sidecar {path} has {len(frame):,} rows but its SVMlight "
            f"matrix has {len(target):,}; sidecars must be exactly row-aligned."
        )
    if target_name not in frame.columns:
        frame[target_name] = target
        return frame

    observed = unify_nans(frame[[target_name]].copy())[target_name]
    if observed.isna().any():
        raise ValueError(
            f"Clinical sidecar target {target_name!r} contains missing values."
        )
    if is_classification:
        cross = pd.crosstab(
            observed.astype(str),
            Series(target).astype(str),
            dropna=False,
        )
        metadata_to_sparse = (cross > 0).sum(axis=1)
        sparse_to_metadata = (cross > 0).sum(axis=0)
        if not ((metadata_to_sparse == 1).all() and (sparse_to_metadata == 1).all()):
            raise ValueError(
                f"Clinical target {target_name!r} in {path} is not in one-to-one "
                "label correspondence with the SVMlight target. Check row order."
            )
    else:
        numeric = pd.to_numeric(observed, errors="raise").to_numpy(dtype=float)
        if not np.allclose(numeric, target.astype(float), equal_nan=False):
            raise ValueError(
                f"Clinical target {target_name!r} in {path} does not match the "
                "SVMlight regression target. Check row order."
            )
    return frame


def _force_imaging_continuous(
    inspection: Any,
    feature_names: Sequence[str],
) -> None:
    """Keep selected numeric imaging values out of categorical inference."""
    for name in feature_names:
        for inferred in (
            inspection.ords,
            inspection.cats,
            inspection.binaries,
        ):
            inferred.infos.pop(name, None)
        inspection.conts.infos[name] = Inference(
            InferredKind.CertainCont,
            "Selected SVMlight imaging feature is numeric.",
        )
    inspection.conts = InspectionInfo(ColumnType.Continuous, dict(inspection.conts.infos))


def _open_sparse(path: Path) -> BinaryIO:
    name = path.name.lower()
    if name.endswith(".gz"):
        return gzip.open(path, "rb")
    if name.endswith(".bz2"):
        return bz2.open(path, "rb")
    if name.endswith(".xz"):
        return lzma.open(path, "rb")
    return path.open("rb")


def _embedded_sample_ids(path: Path, column: str) -> list[str]:
    """Read row IDs from SVMlight comments without loading predictor values."""
    identifiers: list[str] = []
    with _open_sparse(path) as source:
        for raw in source:
            line = raw.decode("utf-8", errors="replace").strip()
            data, marker, comment = line.partition("#")
            if not data.strip():
                continue
            if not marker or not comment.strip():
                raise ValueError(
                    f"SVMlight row IDs were requested, but a data row in {path} "
                    "has no comment."
                )
            text = comment.strip()
            if "=" in text:
                key, value = text.split("=", 1)
                if key.strip() != column:
                    raise ValueError(
                        f"Expected SVMlight comment '# {column}=value' in {path}, "
                        f"but found key {key.strip()!r}."
                    )
                text = value.strip()
            if not text:
                raise ValueError(f"SVMlight input {path} contains an empty row ID.")
            identifiers.append(text)
    return identifiers


def _sparse_file_stats(
    path: Path,
) -> tuple[bool, int, NDArray[np.float64]]:
    """Return index information and labels without loading the sparse matrix."""
    with _open_sparse(path) as source:
        data_rows = 0
        has_zero = False
        max_index = -1
        targets: list[float] = []
        for raw in source:
            line = raw.decode("utf-8", errors="replace").split("#", 1)[0].strip()
            if not line:
                continue
            tokens = line.split()
            try:
                targets.append(float(tokens[0]))
            except (IndexError, ValueError) as error:
                raise ValueError(
                    f"SVMlight input {path} contains a non-numeric target."
                ) from error
            for token in tokens[1:]:
                if ":" not in token:
                    continue
                index = token.split(":", 1)[0]
                if index == "qid":
                    continue
                try:
                    numeric_index = int(index)
                except ValueError:
                    # Let scikit-learn report the full syntax error while parsing.
                    continue
                has_zero |= numeric_index == 0
                max_index = max(max_index, numeric_index)
            data_rows += 1
    if data_rows == 0:
        raise ValueError(f"SVMlight input {path} does not contain any data rows.")
    if max_index < 0:
        raise ValueError(f"SVMlight input {path} does not contain any features.")
    return has_zero, max_index, np.asarray(targets, dtype=np.float64)


def _has_zero_index(path: Path) -> bool:
    return _sparse_file_stats(path)[0]


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


def _sparse_layout(
    paths: Sequence[Path],
    index_base: str,
) -> tuple[int, int, list[NDArray[np.float64]]]:
    if len(paths) == 0:
        raise ValueError("Expected at least one SVMlight input path.")
    requested = str(index_base).lower()
    if requested not in {"auto", "zero", "one"}:
        raise ValueError("SVMlight index base must be auto, zero, or one.")
    stats = [_sparse_file_stats(path) for path in paths]
    zero_indices = [item[0] for item in stats]
    if requested == "auto":
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
    max_source_index = max(item[1] for item in stats)
    n_features = max_source_index + 1 if zero_based else max_source_index
    return resolved, n_features, [item[2] for item in stats]


def _load_sparse_matrix(
    path: Path,
    n_features: int,
    index_base: int,
) -> csr_matrix:
    with _open_sparse(path) as source:
        matrix, _ = load_svmlight_file(
            source,
            n_features=n_features,
            zero_based=index_base == 0,
        )
    matrix = matrix.tocsr()
    if not np.isfinite(matrix.data).all():
        raise ValueError(f"SVMlight predictors in {path} contain NaN or infinite values.")
    return matrix


def load_sparse_input(paths: Sequence[Path], index_base: str = "auto") -> SparseInput:
    resolved, n_features, targets = _sparse_layout(paths, index_base)
    matrices: list[csr_matrix] = []
    # Parse files one at a time to lower peak memory, using one shared feature
    # dimension so their column indices still match.
    for path in paths:
        matrices.append(_load_sparse_matrix(path, n_features, resolved))
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
) -> NDArray[np.float64]:
    # Select the tiny retained column block first.  Row-first slicing would
    # temporarily copy all million source columns for a train/test subset.
    return np.asarray(X[:, selected][rows].toarray(), dtype=np.float64)


def _prepare_hybrid_split(
    split: _SparseSplit,
    train_values: NDArray[np.float64],
    test_values: NDArray[np.float64],
    selected_names: list[str],
    options: Any,
) -> tuple[PreparedData, PreparedData]:
    if split.train.metadata is None or any(part.metadata is None for part in split.tests):
        raise RuntimeError("Incomplete clinical metadata for a sparse split.")
    train_metadata = (
        split.train.metadata.iloc[split.train.rows].copy().reset_index(drop=True)
    )
    test_metadata = pd.concat(
        [
            part.metadata.iloc[part.rows].copy()
            for part in split.tests
            if part.metadata is not None
        ],
        ignore_index=True,
    )
    metadata_features = set(train_metadata.columns) - {options.target}
    collisions = sorted(metadata_features.intersection(selected_names))
    if collisions:
        raise ValueError(
            "Clinical and selected imaging feature names overlap: "
            f"{collisions[:20]}. Rename them in the feature map or sidecar."
        )
    if list(train_metadata.columns) != list(test_metadata.columns):
        raise ValueError(
            "Clinical sidecars must have identical columns after name sanitization."
        )

    train_imaging = DataFrame(train_values, columns=selected_names)
    test_imaging = DataFrame(test_values, columns=selected_names)
    frame = pd.concat(
        [
            pd.concat([train_metadata, train_imaging], axis=1),
            pd.concat([test_metadata, test_imaging], axis=1),
        ],
        ignore_index=True,
    )
    n_train = len(train_metadata)
    local_train = np.arange(n_train, dtype=int)
    local_test = np.arange(n_train, len(frame), dtype=int)
    inspection_frame = frame.iloc[local_train].reset_index(drop=True)
    _, inspection = inspect_data(
        inspection_frame,
        options.target,
        options.grouper,
        options.categoricals,
        options.ordinals,
        options.drops,
        _warn=True,
    )
    _force_imaging_continuous(inspection, selected_names)
    prepared = prepare_data(
        frame.drop(columns=options.drops, errors="ignore"),
        options.target,
        options.grouper,
        inspection,
        options.is_classification,
        local_train,
        [local_test],
        ValidationMethod.List,
    )
    return next(
        prepared.get_splits(
            test_size=options.test_val_size,
            seed=options.seed,
        )
    )


def sparse_prepared_splits(
    options: Any,
) -> list[tuple[PreparedData, PreparedData, FeatureDownsampleResult]]:
    # Supply defaults for older programmatic callers with fewer option fields.
    for name, default in (
        ("grouper", None),
        ("categoricals", []),
        ("ordinals", []),
        ("drops", []),
        ("separator", ","),
    ):
        if not hasattr(options, name):
            setattr(options, name, default)
    paths = [options.datapath, *options.test_paths]
    scan_started = perf_counter()
    index_base, n_features, sparse_targets = _sparse_layout(
        paths, options.svmlight_index_base
    )
    input_scan_seconds = perf_counter() - scan_started
    for path, target in zip(paths, sparse_targets):
        if not np.isfinite(target).all():
            raise ValueError(
                f"SVMlight target in {path} contains NaN or infinite values."
            )
    metadata_paths = list(getattr(options, "svmlight_metadata", []))
    metadata_frames: list[DataFrame | None]
    if metadata_paths:
        if len(metadata_paths) != len(paths):
            raise ValueError(
                "--svmlight-metadata requires exactly one sidecar per SVMlight "
                f"input; received {len(metadata_paths)} for {len(paths)} inputs."
            )
        metadata_frames = []
        first_columns: list[str] | None = None
        first_renames = None
        sample_id_column = getattr(options, "svmlight_sample_id_column", None)
        for metadata_path, target, sparse_path in zip(
            metadata_paths, sparse_targets, paths
        ):
            raw_metadata = _read_sidecar(
                metadata_path, getattr(options, "separator", ",")
            )
            if sample_id_column is not None:
                if sample_id_column not in raw_metadata.columns:
                    raise KeyError(
                        f"Sample-ID column {sample_id_column!r} was not found in "
                        f"clinical sidecar {metadata_path}."
                    )
                embedded_ids = _embedded_sample_ids(sparse_path, sample_id_column)
                clinical_ids = unify_nans(raw_metadata[[sample_id_column]].copy())[
                    sample_id_column
                ]
                if clinical_ids.isna().any():
                    raise ValueError(
                        f"Clinical sample-ID column {sample_id_column!r} contains "
                        f"missing values in {metadata_path}."
                    )
                if embedded_ids != clinical_ids.astype(str).tolist():
                    raise ValueError(
                        f"Clinical sample IDs in {metadata_path} do not match "
                        f"SVMlight row comments in {sparse_path}. Check row order."
                    )
            validated = _validated_metadata(
                raw_metadata,
                target,
                options.target,
                options.is_classification,
                metadata_path,
            )
            sanitized, renames = sanitize_names(validated, options.target)
            columns = sanitized.columns.astype(str).tolist()
            if first_columns is None:
                first_columns = columns
                first_renames = renames
            elif columns != first_columns:
                raise ValueError(
                    "Clinical sidecars must have identical columns in identical "
                    "order after sanitization."
                )
            metadata_frames.append(sanitized)
        if options.is_classification:
            all_metadata_targets = pd.concat(
                [
                    frame[options.target].astype(str)
                    for frame in metadata_frames
                    if frame is not None
                ],
                ignore_index=True,
            )
            all_sparse_targets = pd.concat(
                [Series(target).astype(str) for target in sparse_targets],
                ignore_index=True,
            )
            global_cross = pd.crosstab(
                all_metadata_targets,
                all_sparse_targets,
                dropna=False,
            )
            if not (
                ((global_cross > 0).sum(axis=1) == 1).all()
                and ((global_cross > 0).sum(axis=0) == 1).all()
            ):
                raise ValueError(
                    "Clinical-to-SVMlight class correspondence changes between "
                    "sidecars. Use one consistent SVMlight label encoding."
                )
        if first_renames is not None:
            if options.grouper is not None:
                options.grouper = first_renames.rename_columns([options.grouper])[0]
            if sample_id_column is not None:
                sample_id_column = first_renames.rename_columns([sample_id_column])[0]
                options.svmlight_sample_id_column = sample_id_column
            options.categoricals = first_renames.rename_columns(options.categoricals)
            options.ordinals = first_renames.rename_columns(options.ordinals)
            options.drops = first_renames.rename_columns(options.drops)
            if (
                sample_id_column is not None
                and sample_id_column != options.grouper
                and sample_id_column not in options.drops
            ):
                options.drops.append(sample_id_column)
    else:
        metadata_frames = [None] * len(paths)

    feature_names, protected_indices = _feature_names_and_protected(
        getattr(options, "svmlight_feature_map", None),
        n_features,
        index_base,
        getattr(options, "downsample_protected_features", []),
        getattr(options, "separator", ","),
    )
    split_sources: list[_SparseSplit] = []
    if len(paths) == 1:
        target = sparse_targets[0]
        all_rows = np.arange(len(target), dtype=int)
        if metadata_frames[0] is not None:
            train_rows, test_rows, _ = raw_train_test_indices(
                metadata_frames[0],
                options.target,
                options.grouper,
                options.is_classification,
                options.test_val_size,
                options.seed,
            )
        else:
            stratify = target if options.is_classification else None
            train_rows, test_rows = train_test_split(
                all_rows,
                test_size=options.test_val_size,
                random_state=options.seed,
                stratify=stratify,
            )
        split_sources.append(
            _SparseSplit(
                train=_SparsePart(
                    target,
                    np.sort(train_rows),
                    paths[0],
                    metadata_frames[0],
                ),
                tests=[
                    _SparsePart(
                        target,
                        np.sort(test_rows),
                        paths[0],
                        metadata_frames[0],
                    )
                ],
            )
        )
    else:
        method = getattr(options, "tests_method", ValidationMethod.List)
        if method is ValidationMethod.List:
            for idx in range(1, len(paths)):
                split_sources.append(
                    _SparseSplit(
                        train=_SparsePart(
                            sparse_targets[0],
                            np.arange(len(sparse_targets[0]), dtype=int),
                            paths[0],
                            metadata_frames[0],
                        ),
                        tests=[
                            _SparsePart(
                                sparse_targets[idx],
                                np.arange(len(sparse_targets[idx]), dtype=int),
                                paths[idx],
                                metadata_frames[idx],
                            )
                        ],
                    )
                )
        elif method is ValidationMethod.LODO:
            for train_idx in range(len(paths)):
                split_sources.append(
                    _SparseSplit(
                        train=_SparsePart(
                            sparse_targets[train_idx],
                            np.arange(len(sparse_targets[train_idx]), dtype=int),
                            paths[train_idx],
                            metadata_frames[train_idx],
                        ),
                        tests=[
                            _SparsePart(
                                sparse_targets[idx],
                                np.arange(len(sparse_targets[idx]), dtype=int),
                                paths[idx],
                                metadata_frames[idx],
                            )
                            for idx in range(len(paths))
                            if idx != train_idx
                        ],
                    )
                )
        else:
            raise ValueError(f"Invalid external validation method: {method}")

    outputs = []
    selection_cache: dict[
        tuple[str, bytes], tuple[NDArray[np.int_], FeatureDownsampleResult]
    ] = {}
    cached_train_path: Path | None = None
    cached_train_matrix: csr_matrix | None = None
    for split_idx, split in enumerate(split_sources):
        train_load_seconds = 0.0
        if cached_train_path != split.train.path:
            cached_train_matrix = None
            cached_train_path = split.train.path
            load_started = perf_counter()
            cached_train_matrix = _load_sparse_matrix(
                split.train.path, n_features, index_base
            )
            train_load_seconds = perf_counter() - load_started
        if cached_train_matrix is None:
            raise RuntimeError("Sparse training matrix cache was not initialized.")
        train_source = cached_train_matrix
        if train_source.shape[0] != len(split.train.target):
            raise RuntimeError(
                f"SVMlight row count changed while loading {split.train.path}."
            )
        train_rows = split.train.rows
        train_target = split.train.target[train_rows]
        test_target = np.concatenate([part.target[part.rows] for part in split.tests])
        test_description = ", ".join(str(part.path) for part in split.tests)
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
                f"SVMlight target in {test_description} contains labels that are not "
                "present in its training partition."
            ) from error
        n_select = resolve_n_features(options.n_feat_downsample, train_source.shape[1])
        resolved = resolve_feature_downsample_method(
            options.feat_downsample,
            len(train_rows),
            train_source.shape[1],
            n_select,
            options.is_classification,
            y_train,
        )
        groups_train = None
        if options.grouper is not None:
            if split.train.metadata is None:
                raise ValueError(
                    "A grouping column for SVMlight input requires clinical metadata."
                )
            if options.grouper not in split.train.metadata.columns:
                raise KeyError(
                    f"Grouping column not found in clinical sidecar: {options.grouper}"
                )
            group_frame = unify_nans(
                split.train.metadata[[options.grouper]].iloc[train_rows].copy()
            )
            groups_train = group_frame[options.grouper].reset_index(drop=True)
        effective, screening, tuning, fallback_note = resolve_screening_split(
            options.feat_downsample,
            resolved,
            y_train,
            options.downsample_screening_fraction,
            options.is_classification,
            groups_train,
            seed=options.seed,
        )
        cache_key = (str(split.train.path), train_rows.tobytes())
        cached = selection_cache.get(cache_key)
        if cached is None:
            selection_request = (
                options.feat_downsample if fallback_note is None else effective
            )
            try:
                selected, result = select_indexed_columns(
                    train_source,
                    train_rows,
                    y_train,
                    options.is_classification,
                    selection_request,
                    options.n_feat_downsample,
                    options.downsample_chunk_size,
                    screening,
                    feature_names,
                    options.downsample_variance_threshold,
                    options.downsample_save_scores,
                    sparse_input=True,
                    input_format="svmlight",
                    seed=options.seed,
                    protected_indices=protected_indices,
                )
            except ValueError as error:
                if (
                    options.feat_downsample is not FeatureDownsampleMethod.Auto
                    or screening is None
                    or "usable downsampling scores" not in str(error)
                ):
                    raise
                fallback_note = (
                    f"Auto feature downsampling fell back from {effective.value} to "
                    "normalized-variance because supervised scoring produced no "
                    f"usable feature scores: {error}"
                )
                warn(fallback_note, stacklevel=2)
                screening = None
                tuning = np.arange(len(y_train), dtype=int)
                selected, result = select_indexed_columns(
                    train_source,
                    train_rows,
                    y_train,
                    options.is_classification,
                    FeatureDownsampleMethod.NormalizedVariance,
                    options.n_feat_downsample,
                    options.downsample_chunk_size,
                    None,
                    feature_names,
                    options.downsample_variance_threshold,
                    options.downsample_save_scores,
                    sparse_input=True,
                    input_format="svmlight",
                    seed=options.seed,
                    protected_indices=protected_indices,
                )
            result.requested_method = options.feat_downsample.value
            result.source_index_base = index_base
            if (
                options.feat_downsample is FeatureDownsampleMethod.Auto
                and result.auto_reason is None
            ):
                result.auto_reason = fallback_note
            if fallback_note is not None:
                result.notes.append(fallback_note)
            result.tuning_rows = tuning.tolist()
            result.tuning_samples = len(tuning)
            selection_cache[cache_key] = (
                selected.copy(),
                result.clone_for_reuse(),
            )
        else:
            selected, result = (
                cached[0].copy(),
                cached[1].clone_for_reuse(),
            )
            result.fit_seconds = 0.0
            result.notes.append("Reused feature scores from the same training partition.")
        result.input_scan_seconds = input_scan_seconds if split_idx == 0 else 0.0
        if split_idx > 0:
            result.notes.append(
                "Reused the input layout/target scan measured on the first split."
            )
        result.train_load_seconds = train_load_seconds
        n_test_rows = sum(len(part.rows) for part in split.tests)
        dense_bytes = (len(train_rows) + n_test_rows) * len(selected) * 8
        peak_bytes = dense_bytes * DOWNSAMPLE_DENSE_PEAK_FACTOR
        if peak_bytes > DOWNSAMPLE_MAX_DENSE_BYTES:
            gib = peak_bytes / 1024**3
            raise MemoryError(
                "Selected SVMlight train/test matrices are estimated to require "
                f"about {gib:.2f} GiB of peak dense working memory. Reduce "
                "--n-feat-downsample."
            )
        started = perf_counter()
        selected_names = [feature_names[idx] for idx in selected]
        train_values = _materialize(train_source, train_rows, selected)
        test_load_seconds = 0.0
        if len(split.tests) == 1:
            part = split.tests[0]
            if part.path == split.train.path:
                test_source = train_source
            else:
                load_started = perf_counter()
                test_source = _load_sparse_matrix(part.path, n_features, index_base)
                test_load_seconds += perf_counter() - load_started
            if test_source.shape[0] != len(part.target):
                raise RuntimeError(
                    f"SVMlight row count changed while loading {part.path}."
                )
            test_values = _materialize(test_source, part.rows, selected)
            del test_source
        else:
            test_values = np.empty((n_test_rows, len(selected)), dtype=np.float64)
            start = 0
            for part in split.tests:
                stop = start + len(part.rows)
                load_started = perf_counter()
                test_source = _load_sparse_matrix(part.path, n_features, index_base)
                test_load_seconds += perf_counter() - load_started
                if test_source.shape[0] != len(part.target):
                    raise RuntimeError(
                        f"SVMlight row count changed while loading {part.path}."
                    )
                test_values[start:stop] = _materialize(test_source, part.rows, selected)
                del test_source
                start = stop
        scaler = MaxAbsScaler(copy=False)
        train_values = scaler.fit_transform(train_values)
        test_values = scaler.transform(test_values)
        if result.resolved_method == FeatureDownsampleMethod.None_.value:
            result.notes.append(
                "The requested feature limit did not reduce the SVMlight matrix; "
                "all source columns were materialized densely."
            )
        if split.train.metadata is not None:
            train, test = _prepare_hybrid_split(
                split,
                train_values,
                test_values,
                selected_names,
                options,
            )
            excluded_metadata = {
                options.target,
                *options.drops,
            }
            if options.grouper is not None:
                excluded_metadata.add(options.grouper)
            n_metadata_predictors = len(
                set(split.train.metadata.columns) - excluded_metadata
            )
            result.notes.append(
                "Appended and normally preprocessed "
                f"{max(0, n_metadata_predictors):,} clinical metadata columns "
                "after imaging-feature downsampling."
            )
            original_shape = (
                len(train_rows) + n_test_rows,
                train_source.shape[1] + max(0, n_metadata_predictors),
            )
            if train.info is not None:
                train.info.original_shape = original_shape
            if test.info is not None:
                test.info.original_shape = original_shape
        else:
            X_train = pd.DataFrame(train_values, columns=selected_names)
            X_test = pd.DataFrame(test_values, columns=selected_names)
            info = PreparationInfo(
                original_shape=(
                    len(train_rows) + n_test_rows,
                    train_source.shape[1],
                ),
                final_shape=X_train.shape,
                n_samples_dropped_via_target_NaNs=0,
                n_cont_indicator_added=0,
                target_info=None,
                runtimes={},
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
        source_feature_count = train_source.shape[1] + max(
            0, n_metadata_predictors if split.train.metadata is not None else 0
        )
        source_train_shape = (len(train_rows), source_feature_count)
        source_test_shapes = [
            (len(part.rows), source_feature_count) for part in split.tests
        ]
        for prepared_split in (train, test):
            if prepared_split.info is not None:
                prepared_split.info.source_train_shape = source_train_shape
                prepared_split.info.source_test_shapes = source_test_shapes.copy()
                prepared_split.info.final_train_shape = train.X.shape
                prepared_split.info.final_test_shape = test.X.shape
        result.test_load_seconds = test_load_seconds
        result.transform_seconds = max(0.0, perf_counter() - started - test_load_seconds)
        for prepared_split in (train, test):
            if prepared_split.info is not None:
                prepared_split.info.runtimes["feature downsampling"] = (
                    result.total_seconds
                )
        outputs.append((train, test, result))
    return outputs
