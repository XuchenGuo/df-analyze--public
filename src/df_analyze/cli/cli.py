from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


"""
File for defining all options passed to `df-analyze.py`.
"""
import os
import secrets
import sys
import traceback
from argparse import ArgumentParser, Namespace, RawTextHelpFormatter
from copy import deepcopy
from enum import Enum
from pathlib import Path
from random import choice, randint, uniform
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
)
from warnings import warn

import jsonpickle
import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame

from df_analyze._constants import (
    DOWNSAMPLE_CHUNK_SIZE_DEFAULT,
    FULL_RESULTS,
    N_FEAT_DOWNSAMPLE_DEFAULT,
    N_WRAPPER_DEFAULT,
    P_FILTER_CAT_DEFAULT,
    P_FILTER_CONT_DEFAULT,
    P_FILTER_TOTAL_DEFAULT,
    SEED,
    SENTINEL,
    VERSION,
)
from df_analyze.analysis.univariate.associate import (
    CatClsStats,
    CatRegStats,
    ContClsStats,
    ContRegStats,
)
from df_analyze.cli.parsing import (
    column_parser,
    int_or_percent_parser,
    resolved_path,
    resolved_path_list,
    seed_parser,
    separator,
)
from df_analyze.cli.text import (
    ADAPTIVE_ERROR_HELP,
    AER_ADAPTIVE_BINNING_HELP,
    AER_ALPHA_HELP,
    AER_BINS_HELP,
    AER_CONFIDENCE_METRIC_HELP,
    AER_ENS_ALPHA_HELP,
    AER_ENS_BETA_HELP,
    AER_ENS_LAMBDA_HELP,
    AER_ENS_TAU0_HELP,
    AER_ENS_TAU_HIGH_HELP,
    AER_ENS_TAU_LOW_HELP,
    AER_ENS_TOP_N_HELP,
    AER_ENS_TRIM_Q_HELP,
    AER_ENSEMBLE_HELP,
    AER_ENSEMBLE_STRATEGIES_HELP,
    AER_MIN_BIN_COUNT_HELP,
    AER_MONOTONIC_HELP,
    AER_NMIN_HELP,
    AER_OOF_FOLDS_HELP,
    AER_PRIOR_STRENGTH_HELP,
    AER_SMOOTH_HELP,
    AER_TARGET_ERROR_HELP,
    AER_TOP_K_HELP,
    ASSOC_SELECT_CAT_CLS_STATS,
    ASSOC_SELECT_CAT_REG_STATS,
    ASSOC_SELECT_CONT_CLS_STATS,
    ASSOC_SELECT_CONT_REG_STATS,
    CATEGORICAL_HELP_STR,
    CLS_HELP_STR,
    CLS_TUNE_METRIC,
    DEVICE_HELP,
    DEVICE_INSTALL_HELP,
    DF_HELP_STR,
    DF_TEST_SETS_METHOD_HELP_STR,
    DF_TESTS_HELP_STR,
    DF_TRAIN_HELP_STR,
    DOWNSAMPLE_SCREENING_HELP,
    DROP_HELP_STR,
    EC_EMPTY_UNIONS_HELP,
    EC_EPSILON_HELP,
    EC_FOLDS_HELP,
    EC_METHODS_HELP,
    EC_MODEL_SEED_MODE_HELP,
    EC_RECURRENCE_THRESHOLD_HELP,
    EC_REPETITIONS_HELP,
    EC_SAVE_PREDICTIONS_HELP,
    EMBED_SELECT_MODEL_HELP,
    ERROR_CONSISTENCY_HELP,
    EXPLODE_HELP,
    FEAT_DOWNSAMPLE_HELP,
    FEAT_SELECT_HELP,
    FILTER_METHOD_HELP,
    GROUP_HELP_STR,
    HTUNE_TRIALS_HELP,
    LARGE_FEATURE_MODE_HELP,
    MODE_HELP_STR,
    MT_AGG_STRATEGY_HELP_STR,
    MT_TOP_K_HELP_STR,
    N_FEAT_CAT_FILTER_HELP,
    N_FEAT_CONT_FILTER_HELP,
    N_FEAT_DOWNSAMPLE_HELP,
    N_FEAT_TOTAL_FILTER_HELP,
    N_FEAT_WRAPPER_HELP,
    NAN_HELP,
    NO_PRED_HELP,
    NORM_HELP,
    ORDINAL_HELP_STR,
    OUTDIR_HELP,
    PRED_SELECT_CLS_SCORE,
    PRED_SELECT_REG_SCORE,
    REDUNDANT_CORR_THRESHOLD,
    REDUNDANT_SELECTION,
    REDUNDANT_THRESHOLD,
    REG_HELP_STR,
    REG_TUNE_METRIC,
    SEED_HELP_STR,
    SEP_HELP_STR,
    SHEET_HELP_STR,
    TABPFN_VERSION_HELP,
    TARGET_HELP_STR,
    TARGETS_HELP_STR,
    TEST_VALSIZES_HELP,
    USAGE_EXAMPLES,
    USAGE_STRING,
    VERBOSITY_HELP,
    VERSION_HELP,
    WRAP_SELECT_HELP,
    WRAP_SELECT_MODEL_HELP,
)
from df_analyze.enumerables import (
    ClassifierScorer,
    ClsScore,
    DfAnalyzeClassifier,
    DfAnalyzeRegressor,
    EmbedSelectionModel,
    FeatureDownsampleMethod,
    FeatureSelection,
    FilterSelection,
    NanHandling,
    Normalization,
    RegressorScorer,
    RegScore,
    SeedKind,
    TabPFNVersion,
    ValidationMethod,
    WrapperSelection,
    WrapperSelectionModel,
)
from df_analyze.loading import load_spreadsheet
from df_analyze.runtime.hardware import DeviceIntent, RuntimePolicy, get_runtime
from df_analyze.runtime.install import DeviceInstall
from df_analyze.saving import add_fold_idx

if TYPE_CHECKING:
    from df_analyze.models.base import DfAnalyzeModel
    from df_analyze.testing.datasets import TestDataset
from df_analyze.saving import ProgramDirs, get_hash
from df_analyze.utils import Debug

Size = Union[float, int]


class ArgumentError(Exception):
    pass


class Verbosity(Enum):
    """
    Properties
    ----------
    ERROR
        Only log errors.

    INFO
        Log results of each full hyperparameter tuning and other interim progress bars.

    DEBUG
        Maximum level of logging.
    """

    ERROR = 0
    INFO = 1
    DEBUG = 2


class ProgramOptions(Debug):
    """Just a container for handling CLI options and default logic (while also
    providing better typing than just using the `Namespace` from the
    `ArgumentParser`).

    Notes
    -----
    For `joblib.Memory` to cache properly, we need all arguments to be
    hashable. This means immutable (among other things) so we use `Tuple` types
    for arguments or options where there are multiple steps to go through, e.g.
    feature selection.
    """

    def __init__(
        self,
        datapath: Optional[Path],
        test_paths: list[Path],
        tests_method: ValidationMethod,
        target: str,
        grouper: Optional[str],
        categoricals: list[str],
        ordinals: list[str],
        drops: list[str],
        nan_handling: NanHandling,
        norm: Normalization,
        # feat_clean: Tuple[FeatureCleaning, ...],
        feat_select: Tuple[FeatureSelection, ...],
        embed_select: Optional[tuple[EmbedSelectionModel, ...]],
        wrapper_select: Optional[WrapperSelection],
        wrapper_model: WrapperSelectionModel,
        # n_feat: int,
        n_filter_cont: Optional[Union[int, float]],
        n_filter_cat: Optional[Union[int, float]],
        n_feat_filter: Union[int, float],
        n_feat_wrapper: Union[int, float, None],
        filter_assoc_cont_cls: ContClsStats,
        filter_assoc_cat_cls: CatClsStats,
        filter_assoc_cont_reg: ContRegStats,
        filter_assoc_cat_reg: CatRegStats,
        filter_pred_cls_score: ClsScore,
        filter_pred_reg_score: RegScore,
        redundant_selection: bool,
        redundant_threshold: float,
        redundant_corr_threshold: float,
        is_classification: bool,
        classifiers: Tuple[DfAnalyzeClassifier, ...],
        regressors: Tuple[DfAnalyzeRegressor, ...],
        # htune: bool,
        # htune_val: ValMethod,
        # htune_val_size: Size,
        htune_trials: int,
        htune_cls_metric: ClassifierScorer,
        htune_reg_metric: RegressorScorer,
        # test_val: ValMethod,
        test_val_size: Size,
        # mc_repeats: int,
        outdir: Path,
        is_spreadsheet: bool,
        separator: str,
        seed: Union[int, SeedKind],
        verbosity: Verbosity,
        no_warn_explosion: bool,
        no_preds: bool,
        feat_downsample: Union[
            str, FeatureDownsampleMethod
        ] = FeatureDownsampleMethod.None_,
        n_feat_downsample: Union[int, float] = N_FEAT_DOWNSAMPLE_DEFAULT,
        downsample_variance_threshold: Optional[float] = None,
        downsample_save_scores: bool = False,
        downsample_chunk_size: int = DOWNSAMPLE_CHUNK_SIZE_DEFAULT,
        downsample_screening_fraction: float = 0.25,
        large_feature_mode: bool = False,
        assume_numeric_features: bool = False,
        skip_full_prepared_save_before_downsample: bool = False,
        input_format: str = "auto",
        svmlight_index_base: str = "auto",
        device: Union[str, DeviceIntent] = DeviceIntent.Auto,
        device_install: Union[str, DeviceInstall, None] = None,
        targets: Optional[list[str]] = None,
        mt_agg_strategy: str = "borda",
        mt_top_k: Optional[int] = None,
        error_consistency: bool = False,
        ec_folds: int = 5,
        ec_repetitions: int = 5,
        ec_model_seed_mode: str = "vary",
        ec_methods: Optional[tuple[str, ...]] = None,
        ec_save_predictions: bool = False,
        ec_empty_unions: str = "warn",
        ec_epsilon: float = 0.0,
        ec_recurrence_threshold: float = 0.5,
        adaptive_error: bool = False,
        aer_oof_folds: int = 5,
        aer_bins: int = 20,
        aer_target_error: float = 0.05,
        aer_alpha: float = 0.05,
        aer_min_bin_count: int = 10,
        aer_prior_strength: float = 2.0,
        aer_smooth: bool = True,
        aer_monotonic: bool = False,
        aer_adaptive_binning: bool = False,
        aer_confidence_metric: str = "auto",
        aer_nmin: int = 1,
        aer_top_k: int = 0,
        aer_ensemble: bool = False,
        aer_ensemble_strategies: Optional[tuple[str, ...]] = None,
        aer_ens_top_n: int = 3,
        aer_ens_beta: float = 5.0,
        aer_ens_tau0: float = 0.3,
        aer_ens_lambda: float = 0.5,
        aer_ens_alpha: float = 0.7,
        aer_ens_trim_q: float = 0.6,
        aer_ens_tau_low: float = 0.15,
        aer_ens_tau_high: float = 0.35,
        tabpfn_version: Union[str, TabPFNVersion] = TabPFNVersion.V3,
    ) -> None:
        # memoization-related
        # other
        self.version = VERSION
        if isinstance(seed, int):
            self.seed = seed
        elif seed in [SeedKind.Default, None]:
            self.seed = SEED
        elif seed is SeedKind.Random:
            self.seed = secrets.randbelow(2**16 - 1)
        else:
            raise ValueError(f"Invalid seed type: {type(seed)}: {seed}")
        self.cli_argv = list(sys.argv)
        self.cli_args = " ".join(sys.argv)
        self.datapath: Optional[Path] = self.validate_datapath(datapath)
        self.test_paths: list[Path] = [self.validate_test_path(p) for p in test_paths]
        self.tests_method = (
            tests_method
            if isinstance(tests_method, ValidationMethod)
            else ValidationMethod.from_arg(tests_method)
        )
        self.targets: list[str] = targets if targets is not None else [target]
        self.targets = [str(t).strip() for t in self.targets if str(t).strip() != ""]
        if len(self.targets) == 0:
            raise ValueError("Expected at least one target column name.")
        if len(set(self.targets)) != len(self.targets):
            raise ValueError("Target column names must be unique.")
        self.target: str = self.targets[0]
        self.grouper: Optional[str] = grouper
        self.categoricals: list[str] = categoricals
        self.ordinals: list[str] = ordinals
        self.drops: list[str] = drops
        self.nan_handling: NanHandling = nan_handling
        self.norm: Normalization = norm
        # self.feat_clean: Tuple[FeatureCleaning, ...] = tuple(sorted(set(feat_clean)))
        self.feat_select: Tuple[FeatureSelection, ...] = tuple(sorted(set(feat_select)))
        self.embed_select: Optional[tuple[EmbedSelectionModel, ...]] = embed_select
        self.wrapper_select: Optional[WrapperSelection] = wrapper_select
        self.wrapper_model: WrapperSelectionModel = wrapper_model
        # self.n_feat: int = n_feat
        self.n_filter_cont: Optional[Union[int, float]] = n_filter_cont
        self.n_filter_cat: Optional[Union[int, float]] = n_filter_cat
        self.n_feat_filter: Union[int, float] = n_feat_filter
        self.n_feat_wrapper: Union[int, float, None] = n_feat_wrapper
        self.filter_assoc_cont_cls: ContClsStats = filter_assoc_cont_cls
        self.filter_assoc_cat_cls: CatClsStats = filter_assoc_cat_cls
        self.filter_assoc_cont_reg: ContRegStats = filter_assoc_cont_reg
        self.filter_assoc_cat_reg: CatRegStats = filter_assoc_cat_reg
        self.filter_pred_cls_score: ClsScore = filter_pred_cls_score
        self.filter_pred_reg_score: RegScore = filter_pred_reg_score
        self.redundant_selection: bool = redundant_selection
        self.redundant_threshold: float = abs(redundant_threshold)
        # TODO: only select all features in redundant stepwise if their
        # absolute Pearson correlation with the highest-scoring feature is
        # above this threshold
        self.redundant_corr_threshold: float = abs(redundant_corr_threshold)
        if mt_agg_strategy not in {"borda", "freq"}:
            raise ValueError("Multi-target aggregation must be 'borda' or 'freq'.")
        self.mt_agg_strategy: str = mt_agg_strategy
        self.mt_top_k: Optional[int] = int(mt_top_k) if mt_top_k is not None else None
        if self.mt_top_k is not None and self.mt_top_k <= 0:
            self.mt_top_k = None
        self.error_consistency: bool = bool(error_consistency)
        self.ec_folds: int = int(ec_folds)
        self.ec_repetitions: int = int(ec_repetitions)
        self.ec_model_seed_mode: str = str(ec_model_seed_mode).lower()
        if self.ec_model_seed_mode not in {"vary", "fixed"}:
            raise ValueError("EC model seed mode must be 'vary' or 'fixed'.")
        self.ec_methods: Optional[tuple[str, ...]] = (
            None if ec_methods is None or len(ec_methods) == 0 else tuple(ec_methods)
        )
        self.ec_save_predictions: bool = bool(ec_save_predictions)
        self.ec_empty_unions: str = str(ec_empty_unions)
        self.ec_epsilon: float = float(ec_epsilon)
        self.ec_recurrence_threshold: float = float(ec_recurrence_threshold)
        self.is_classification: bool = is_classification
        self.classifiers: Tuple[DfAnalyzeClassifier, ...] = tuple(
            sorted(set(classifiers))
        )
        self.regressors: Tuple[DfAnalyzeRegressor, ...] = tuple(sorted(set(regressors)))
        self.tabpfn_version = TabPFNVersion.from_arg(tabpfn_version)
        self._validate_optional_model_dependencies()
        # self.htune: bool = htune
        # self.htune_val: ValMethod = htune_val
        # self.htune_val_size: Size = htune_val_size
        self.htune_trials: int = htune_trials
        self.htune_cls_metric: ClassifierScorer = htune_cls_metric
        self.htune_reg_metric: RegressorScorer = htune_reg_metric
        # self.test_val: ValMethod = test_val
        self.test_val_size: Size = test_val_size
        # self.mc_repeats: int = mc_repeats
        # TODO: fix below
        self.outdir: Optional[Path] = self.get_outdir(outdir, self.datapath)
        self.is_spreadsheet: bool = is_spreadsheet
        self.separator: str = separator
        self.verbosity: Verbosity = verbosity
        self.no_warn_explosion: bool = no_warn_explosion
        self.no_preds: bool = no_preds
        self.feat_downsample = FeatureDownsampleMethod.from_arg(feat_downsample)
        self.n_feat_downsample = n_feat_downsample
        self.downsample_variance_threshold = downsample_variance_threshold
        self.downsample_save_scores = bool(downsample_save_scores)
        self.downsample_chunk_size = int(downsample_chunk_size)
        if self.downsample_chunk_size <= 0:
            raise ValueError("Downsample chunk size must be positive.")
        self.downsample_screening_fraction = float(downsample_screening_fraction)
        if not 0.0 < self.downsample_screening_fraction < 1.0:
            raise ValueError("Downsample screening fraction must be in (0, 1).")
        self.large_feature_mode = bool(large_feature_mode)
        self.assume_numeric_features = bool(assume_numeric_features)
        self.skip_full_prepared_save_before_downsample = bool(
            skip_full_prepared_save_before_downsample
        )
        self.input_format = str(input_format).lower()
        self.svmlight_index_base = str(svmlight_index_base).lower()
        self.device: DeviceIntent = DeviceIntent.from_arg(device)
        self.device_install: DeviceInstall = DeviceInstall.from_arg(
            device_install, self.device.value
        )
        self.adaptive_error: bool = adaptive_error
        self.aer_oof_folds: int = aer_oof_folds
        self.aer_bins: int = aer_bins
        self.aer_target_error: float = aer_target_error
        self.aer_alpha: float = aer_alpha
        self.aer_min_bin_count: int = aer_min_bin_count
        self.aer_prior_strength: float = aer_prior_strength
        self.aer_smooth: bool = aer_smooth
        self.aer_monotonic: bool = aer_monotonic
        self.aer_adaptive_binning: bool = aer_adaptive_binning
        self.aer_confidence_metric: str = aer_confidence_metric
        self.aer_nmin: int = aer_nmin
        self.aer_top_k: int = aer_top_k
        self.aer_ensemble: bool = aer_ensemble
        self.aer_ensemble_strategies: Optional[tuple[str, ...]] = (
            tuple(aer_ensemble_strategies)
            if aer_ensemble_strategies is not None
            else None
        )
        self.aer_ens_top_n: int = aer_ens_top_n
        self.aer_ens_beta: float = aer_ens_beta
        self.aer_ens_tau0: float = aer_ens_tau0
        self.aer_ens_lambda: float = aer_ens_lambda
        self.aer_ens_alpha: float = aer_ens_alpha
        self.aer_ens_trim_q: float = aer_ens_trim_q
        self.aer_ens_tau_low: float = aer_ens_tau_low
        self.aer_ens_tau_high: float = aer_ens_tau_high

        self.program_dirs: ProgramDirs = ProgramDirs.new(self.outdir, self.hash())

        # cleanup
        if DfAnalyzeClassifier.Dummy not in self.classifiers:
            self.classifiers = (DfAnalyzeClassifier.Dummy, *self.classifiers)
        if DfAnalyzeRegressor.Dummy not in self.regressors:
            self.regressors = (DfAnalyzeRegressor.Dummy, *self.regressors)

        is_cls = self.is_classification
        self.comparables = {
            "model": self.classifiers if is_cls else self.regressors,
            "selection": self.feat_select,
            "norm": self.norm,
        }

        self.spam_warnings()

    @property
    def models(self) -> list[Type[DfAnalyzeModel]]:
        sources = self.classifiers if self.is_classification else self.regressors
        clses = []
        for source in sources:
            if source.value != DfAnalyzeClassifier.TabPFN.value:
                clses.append(source.get_model())
                continue
            from df_analyze.models.tabpfn import TABPFN_CLASSIFIERS, TABPFN_REGRESSORS

            versions = TABPFN_CLASSIFIERS if self.is_classification else TABPFN_REGRESSORS
            clses.append(versions[self.tabpfn_version.value])
        return clses

    @property
    def runtime(self) -> RuntimePolicy:
        intent = getattr(self, "device", DeviceIntent.Auto)
        policy = getattr(self, "_runtime_policy", None)
        if policy is None or policy.intent is not intent:
            policy = get_runtime(intent)
            self._runtime_policy = policy
        workload = getattr(self, "_runtime_workload", None)
        if workload is None:
            return policy
        return policy.with_workload(*workload)

    def set_runtime_workload(self, n_samples: int, n_features: int) -> None:
        self._runtime_workload = (int(n_samples), int(n_features))

    def _validate_optional_model_dependencies(self) -> None:
        from df_analyze.enumerables import (
            is_catboost_available,
            is_kan_available,
            is_tabpfn_available,
            is_xgboost_available,
        )

        optional_models = (
            (
                "CatBoost",
                "catboost",
                DfAnalyzeClassifier.CatBoost,
                DfAnalyzeRegressor.CatBoost,
                is_catboost_available,
            ),
            (
                "XGBoost",
                "xgboost",
                DfAnalyzeClassifier.XGBoost,
                DfAnalyzeRegressor.XGBoost,
                is_xgboost_available,
            ),
            (
                "TabPFN",
                "tabpfn",
                DfAnalyzeClassifier.TabPFN,
                DfAnalyzeRegressor.TabPFN,
                is_tabpfn_available,
            ),
            (
                "KAN",
                "pykan",
                DfAnalyzeClassifier.KAN,
                DfAnalyzeRegressor.KAN,
                is_kan_available,
            ),
        )

        for display, package, classifier, regressor, available in optional_models:
            requested_cls = classifier in self.classifiers
            requested_reg = regressor in self.regressors
            if not (requested_cls or requested_reg) or available():
                continue
            self.classifiers = tuple(
                model for model in self.classifiers if model is not classifier
            )
            self.regressors = tuple(
                model for model in self.regressors if model is not regressor
            )
            selected_args = []
            if requested_cls:
                selected_args.append(f"--classifiers {classifier.value}")
            if requested_reg:
                selected_args.append(f"--regressors {regressor.value}")
            warn(
                f"{display} was requested but `{package}` is not installed. "
                f"Continuing after removing it from {', '.join(selected_args)}. "
                f"Install it with `pip install {package}` to enable it."
            )

        if self.is_classification and len(self.classifiers) == 0:
            raise ArgumentError(
                "No available classifiers remain after dependency checks. Install "
                "the missing optional package(s) named in the preceding warning(s), "
                "or choose a different value for `--classifiers`."
            )
        if not self.is_classification and len(self.regressors) == 0:
            raise ArgumentError(
                "No available regressors remain after dependency checks. Install "
                "the missing optional package(s) named in the preceding warning(s), "
                "or choose a different value for `--regressors`."
            )

    @staticmethod
    def random(
        ds: Optional[TestDataset], outdir: Optional[Path] = None, multitest: bool = False
    ) -> ProgramOptions:
        if ds is None:
            is_cls = np.random.randint(0, 2, dtype=bool)
        else:
            n_samples, n_feats = ds.shape
            is_cls = ds.is_classification
        # feat_clean = FeatureCleaning.random_n()
        seed = SeedKind.random_seed()
        if isinstance(seed, str):
            seed = SeedKind(seed)
        feat_select = FeatureSelection.random_n()
        wrap_select = WrapperSelection.random()
        wrap_model = WrapperSelectionModel.random()
        embed_select = (EmbedSelectionModel.random_none(),)
        # n_feat = n_feats
        n_feat_filter = uniform(0.5, 0.95)
        n_feat_wrapper = choice([uniform(0.2, 0.5), None])
        n_filter_cont = uniform(0.1, 0.25)
        n_filter_cat = n_feat_filter - n_filter_cont
        filter_assoc_cont_cls = ContClsStats.random()
        filter_assoc_cat_cls = CatClsStats.random()
        filter_assoc_cont_reg = ContRegStats.random()
        filter_assoc_cat_reg = CatRegStats.random()
        filter_pred_cls_score = ClsScore.random()
        filter_pred_reg_score = RegScore.random()
        redundant = choice([True, False])
        redundant_threshold = uniform(0, 0.2)
        redundant_corr_threshold = uniform(0.8, 0.95)
        is_classification = is_cls
        classifiers = tuple(
            model
            for model in DfAnalyzeClassifier.random_n()
            if model is not DfAnalyzeClassifier.TabPFN
        )
        regressors = tuple(
            model
            for model in DfAnalyzeRegressor.random_n()
            if model is not DfAnalyzeRegressor.TabPFN
        )
        # htune: bool = choice([True, False])
        # htune_val: ValMethod = "kfold"
        # htune_val_size: Size = 5
        htune_trials: int = randint(20, 100)
        htune_cls_metric = ClassifierScorer.random()
        htune_reg_metric = RegressorScorer.random()
        # test_val: ValMethod = "kfold"
        test_val_size: Size = 0.4
        # mc_repeats: int = 0
        sub = "classification" if is_cls else "regression"
        dsname = ds.dsname if ds is not None else "random_data"
        outdir = outdir or ((FULL_RESULTS / sub) / dsname)
        datapath = ds.datapath if ds is not None else None
        is_spreadsheet: bool = False
        separator: str = ","
        verbosity: Verbosity = Verbosity.ERROR
        no_warn_explosion: bool = False
        return ProgramOptions(
            datapath=datapath,
            test_paths=[],  # TODO: think of something better here...
            tests_method=ValidationMethod.List,  # TODO: work with above
            target="target",
            grouper=None,
            categoricals=ds.categoricals if ds is not None else [],
            ordinals=[],
            drops=[],
            nan_handling=NanHandling.random(),
            norm=Normalization.random(),
            # feat_clean=feat_clean,
            feat_select=feat_select,
            embed_select=embed_select,  # type: ignore
            wrapper_select=wrap_select,
            wrapper_model=wrap_model,
            # n_feat=n_feat,
            n_filter_cat=n_filter_cat,
            n_filter_cont=n_filter_cont,
            n_feat_filter=n_feat_filter,
            n_feat_wrapper=n_feat_wrapper,
            filter_assoc_cont_cls=filter_assoc_cont_cls,
            filter_assoc_cat_cls=filter_assoc_cat_cls,
            filter_assoc_cont_reg=filter_assoc_cont_reg,
            filter_assoc_cat_reg=filter_assoc_cat_reg,
            filter_pred_cls_score=filter_pred_cls_score,
            filter_pred_reg_score=filter_pred_reg_score,
            redundant_selection=redundant,
            redundant_threshold=redundant_threshold,
            redundant_corr_threshold=redundant_corr_threshold,
            is_classification=is_classification,
            classifiers=classifiers,
            regressors=regressors,
            # htune=htune,
            # htune_val=htune_val,
            # htune_val_size=htune_val_size,
            htune_trials=htune_trials,
            htune_cls_metric=htune_cls_metric,
            htune_reg_metric=htune_reg_metric,
            # test_val=test_val,
            test_val_size=test_val_size,
            seed=seed,
            outdir=outdir,
            is_spreadsheet=is_spreadsheet,
            separator=separator,
            verbosity=verbosity,
            no_warn_explosion=no_warn_explosion,
            no_preds=False,
        )

    def spam_warnings(self) -> None:
        if self.verbosity is Verbosity.ERROR:
            return  # don't warn user

        if self.htune_trials < 100:
            warn(
                "Without pruning, Optuna generally only shows clear superiority\n"
                "to random search at roughly 50-100 trials. See e.g.\n"
                "    Akiba et al. (2019)\n"
                "    Optuna: A Next-generation Hyperparameter Optimization Framework \n"
                "    https://arxiv.org/pdf/1907.10902.pdf\n"
                "For deep learners, e.g. if using `mlp` as either a classifer\n"
                "or regressor, experience suggests more like 100-200 trials (with\n"
                "pruning) are needed when exploring new architectures. For the\n"
                "current MLP architecture, probably 100 trials is sufficient.\n"
            )

        if ("step-up" in self.feat_select) or ("step-down" in self.feat_select):
            warn(
                "Step-up and step-down feature selection can have very high time-complexity.\n"
                "It is strongly recommended to run these selection procedures in isolation,\n"
                "and not in the same process as all other feature selection procedures.\n"
                "See also the relevant notes on runtime complexity of these techniques:\n"
                "https://scikit-learn.org/stable/modules/feature_selection.html#sequential-feature-selection"
            )
        if "step-down" in self.feat_select:
            warn(
                "Step-down feature selection in particular will usually be intractable\n"
                "even on small (100 features, 1000 samples) datasets and when selecting\n"
                "a much smaller number of features (10-20), unless using a very fast\n"
                "estimator (linear regression, logistic regression, maybe svm)."
            )
        print("To silence these warnings, use `--verbosity=0`.")

    @staticmethod
    def validate_datapath(df_path: Optional[Path]) -> Optional[Path]:
        if df_path is None:
            return None
        datapath = resolved_path(df_path)
        if not datapath.exists():
            raise FileNotFoundError(f"The specified file {datapath} does not exist.")
        if not datapath.is_file():
            raise FileNotFoundError(f"{datapath} is not a file.")
        return Path(datapath).resolve()

    @staticmethod
    def validate_test_path(path: Path) -> Path:
        datapath = resolved_path(path)
        if not datapath.exists():
            raise FileNotFoundError(f"The specified file {datapath} does not exist.")
        if not datapath.is_file():
            raise FileNotFoundError(f"{datapath} is not a file.")
        return Path(datapath).resolve()

    @staticmethod
    def get_outdir(outdir: Optional[Path], datapath: Optional[Path]) -> Optional[Path]:
        if outdir is None or datapath is None:
            return None
        name = datapath.stem
        outdir = outdir / name
        try:
            os.makedirs(outdir, exist_ok=True)
            return outdir
        except OSError as e:
            print(f"Got error: {e}", file=sys.stderr)
            traceback.print_exc()
            return None

    def hash(self) -> str:
        return get_hash(
            self.__dict__,
            ignores=[
                "cleaning_options",
                "selection_options",
                "comparables",
                "program_dirs",
            ],
        )

    def to_json(self) -> None:
        try:
            path = self.program_dirs.options
            if path is None:
                return
            path.write_text(str(jsonpickle.encode(self, unpicklable=True, indent=4)))
        except Exception as e:
            print(f"Got error saving options: {e}")
            traceback.print_exc()

    @staticmethod
    def from_json(root: Path) -> ProgramOptions:
        def is_jsonpickle_dict(value: Any) -> bool:
            return isinstance(value, dict) and "__objclass__" in value

        def to_enum(d: dict) -> Enum:
            enum = d["__objclass__"]
            return enum(d["_value_"])

        def is_path(d: dict) -> bool:
            return isinstance(d, dict) and "_raw_paths" in d

        def to_path(d: dict) -> Enum:
            return Path(d["_raw_paths"][0])

        options = root / "options.json"
        obj = jsonpickle.decode(options.read_text())
        if isinstance(obj, ProgramOptions):
            obj = vars(obj)
        obj = cast(dict, obj)
        obj.pop("version", None)
        obj.pop("cli_args", None)
        obj.pop("comparables", None)
        obj.pop("program_dirs", None)

        program_dirs = ProgramDirs.new(
            root=options.parent.parent, hsh=options.parent.name
        )

        for key, d in obj.items():
            if is_jsonpickle_dict(d):
                obj[key] = to_enum(d)
            elif is_path(d):
                obj[key] = to_path(d)
            elif isinstance(d, list) and len(d) > 0 and is_jsonpickle_dict(d[0]):
                for i, item in enumerate(d):
                    d[i] = to_enum(item)

        tuples = ["feat_select", "embed_select"]
        for key in tuples:
            if key in obj:
                obj[key] = tuple(obj[key])

        opts = ProgramOptions(**obj)
        opts.program_dirs = program_dirs
        return opts

    @staticmethod
    def from_jsonfile(file: Path) -> ProgramOptions:
        obj = jsonpickle.decode(file.read_text())
        opts = cast(ProgramOptions, obj)
        if not hasattr(opts, "targets"):
            opts.targets = [opts.target]
        if not hasattr(opts, "mt_agg_strategy"):
            opts.mt_agg_strategy = "borda"
        if not hasattr(opts, "mt_top_k"):
            opts.mt_top_k = None
        if not hasattr(opts, "device"):
            opts.device = DeviceIntent.Auto
        if not hasattr(opts, "device_install"):
            opts.device_install = DeviceInstall.Never
        if not hasattr(opts, "cli_argv"):
            opts.cli_argv = list(sys.argv)
        if not hasattr(opts, "tabpfn_version"):
            opts.tabpfn_version = TabPFNVersion.V3
        if not hasattr(opts, "error_consistency"):
            opts.error_consistency = False
        if not hasattr(opts, "ec_folds"):
            opts.ec_folds = 5
        if not hasattr(opts, "ec_repetitions"):
            opts.ec_repetitions = 5
        if not hasattr(opts, "ec_model_seed_mode"):
            opts.ec_model_seed_mode = "vary"
        if not hasattr(opts, "ec_methods"):
            opts.ec_methods = None
        if not hasattr(opts, "ec_save_predictions"):
            opts.ec_save_predictions = False
        if not hasattr(opts, "ec_empty_unions"):
            opts.ec_empty_unions = "warn"
        if not hasattr(opts, "ec_epsilon"):
            opts.ec_epsilon = 0.0
        if not hasattr(opts, "ec_recurrence_threshold"):
            opts.ec_recurrence_threshold = 0.5
        defaults = {
            "feat_downsample": FeatureDownsampleMethod.None_,
            "n_feat_downsample": N_FEAT_DOWNSAMPLE_DEFAULT,
            "downsample_variance_threshold": None,
            "downsample_save_scores": False,
            "downsample_chunk_size": DOWNSAMPLE_CHUNK_SIZE_DEFAULT,
            "downsample_screening_fraction": 0.25,
            "large_feature_mode": False,
            "assume_numeric_features": False,
            "skip_full_prepared_save_before_downsample": False,
            "input_format": "auto",
            "svmlight_index_base": "auto",
        }
        for name, value in defaults.items():
            if not hasattr(opts, name):
                setattr(opts, name, value)
        return opts

    def uses_svmlight_input(self) -> bool:
        if self.input_format == "svmlight":
            return True
        if self.input_format == "table" or self.datapath is None:
            return False
        name = self.datapath.name.lower()
        suffixes = (".svm", ".svmlight", ".libsvm", ".binary")
        compression = ("", ".gz", ".bz2", ".xz")
        return any(name.endswith(suffix + comp) for suffix in suffixes for comp in compression)

    def _load_df(self, path: Path) -> DataFrame:
        if path is None:
            raise RuntimeError("Expected `pathlib.Path`, but got `None`")

        if path.name.endswith("parquet"):
            return pd.read_parquet(path)
        if path.name.endswith("csv"):
            return pd.read_csv(path, sep=self.separator)
        if path.name.endswith("json"):
            return pd.read_json(path)
        raise ValueError(f"Unrecognized filetype: '{path.suffix}'")

    def load_df(self) -> DataFrame:
        if self.datapath is None:
            raise RuntimeError("Cannot load data as `self.datapath` is None")
        if self.is_spreadsheet:
            return load_spreadsheet(self.datapath, self.separator)[0]

        return self._load_df(self.datapath)

    def load_test_dfs(self) -> list[DataFrame]:
        return [self._load_df(p) for p in self.test_paths]

    def merged_df(
        self, primary: Optional[DataFrame] = None
    ) -> Optional[tuple[DataFrame, ndarray, list[ndarray]]]:
        if len(self.test_paths) == 0:
            return None

        method = self.tests_method
        if method not in ValidationMethod:
            raise RuntimeError(
                f"Unrecognized argument to --df-tests-method: {method} (This error should be impossible)"
            )
        df = self.load_df() if primary is None else primary
        dfs = self.load_test_dfs()

        # track indices for re-splitting later, i.e. so we can recover the original
        # dfs by df_merged.iloc[ix].reset_index(drop=True)
        ix_train = np.arange(df.shape[0])
        ix_tests = []
        lengths = [df.shape[0] for df in dfs]
        last = ix_train[-1]
        for length in lengths:
            ix_tests.append(np.arange(last + 1, last + 1 + length))
            last = ix_tests[-1][-1]

        ix_all = np.concatenate([ix_train, *ix_tests])
        if not (np.diff(ix_all) == 1).all():
            raise ValueError("Gap or repetition in indices")
        if len(ix_all) != (len(df) + sum(lengths)):
            raise ValueError("Merged df indices do not match number of samples.")

        # ix_tests = [np.arange(df.shape[0]) for df in dfs]
        # for i, ix_test in enumerate(ix_tests):
        #     if i == 0:
        #         ix_test += ix_train[-1] + 1
        #     else:
        #         ix_prev = ix_tests[i - 1]
        #         ix_test += ix_prev[-1] + 1

        train_cols = sorted(df.columns)
        for i, df_test in enumerate(dfs):
            cols = sorted(df_test.columns)
            if cols != train_cols:
                test_path = self.test_paths[i]
                raise ValueError(
                    f"The column names in file: {test_path} ({cols}) do not match the "
                    f"column names of the training dataframe: {self.datapath} ({train_cols})"
                )

        df_all = pd.concat([df, *dfs], axis=0, ignore_index=True, join="inner")
        # Check again concat didn't mess anything up
        all_cols = sorted(df_all.columns)
        if all_cols != train_cols:
            raise RuntimeError(
                "Unexpected error. Concatenating test dataframes resulted in "
                f"columns: {all_cols}, which are inconsistent with the columns "
                f"in the training data: {train_cols}"
            )

        return df_all, ix_train, ix_tests


def parse_and_merge_args(parser: ArgumentParser, args: Optional[str] = None) -> Namespace:
    # CLI args supersede when non default and also specified in sheet
    # see https://stackoverflow.com/a/76230387 for a similar problem
    cli_parser = deepcopy(parser)
    sheet_parser = deepcopy(parser)
    sentinel_parser = deepcopy(parser)

    parse_input = None if args is None else args.split()
    cli_args, unknown_cli_args = cli_parser.parse_known_args(parse_input)
    if unknown_cli_args:
        cli_parser.error(f"unrecognized arguments: {' '.join(unknown_cli_args)}")

    sentinels = {key: SENTINEL for key in cli_args.__dict__}
    sentinel_parser.set_defaults(**sentinels)

    if cli_args.version:
        print(f"df-analyze {VERSION}")
        exit(0)

    # validate core data input args
    if cli_args.spreadsheet is None and cli_args.df is None:
        if cli_args.df_train is None:
            raise ValueError(
                "Must specify one of either `--spreadsheet [file]` or `--df [file]` or "
                "`--df-train`."
            )
        else:
            if len(cli_args.df_tests) == 0:
                raise ValueError(
                    "If using `--df-train` argument, must pass at least one test "
                    "dataset file to `--df-tests`."
                )

    sentinel_cli_args = sentinel_parser.parse_known_args(parse_input)[0]
    explicit_cli_args = {
        key: val for key, val in sentinel_cli_args.__dict__.items() if val is not SENTINEL
    }

    spreadsheet = cli_args.spreadsheet
    if spreadsheet is not None:
        if cli_args.df_train is not None:
            raise ValueError(
                "Specifying training and test sets is incompatible with `--spreadsheet` "
                "argument. Use either (1) `--df`, (2) `--spreadsheet`, or (3) both "
                "`--df-train` and `--df-test` arguments. "
            )
        options = load_spreadsheet(spreadsheet)[1]
    else:
        options = ""

    sheet_args = sheet_parser.parse_known_args(options.split())[0].__dict__
    # sentinel_sheet_args = sentinel_parser.parse_args(options.split())
    # explicit_sheet_args = Namespace(
    #     **{key: val for key, val in sentinel_sheet_args.__dict__.items() if val is not SENTINEL}
    # )

    cli_args = Namespace(**{**sheet_args, **explicit_cli_args})
    return cli_args


def make_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="df-analyze",
        usage=USAGE_STRING,
        formatter_class=RawTextHelpFormatter,
        epilog=USAGE_EXAMPLES,
    )
    parser.add_argument(
        "--spreadsheet",
        type=resolved_path,
        required=False,
        default=None,
        help=SHEET_HELP_STR,
    )
    parser.add_argument(
        "--df",
        action="store",
        type=resolved_path,
        required=False,
        default=None,
        help=DF_HELP_STR,
    )
    parser.add_argument(
        "--df-train",
        action="store",
        type=resolved_path,
        required=False,
        default=None,
        help=DF_TRAIN_HELP_STR,
    )
    parser.add_argument(
        "--df-tests",
        action="store",
        type=resolved_path_list,
        required=False,
        default=[],
        help=DF_TESTS_HELP_STR,
    )
    parser.add_argument(
        "--df-tests-method",
        action="store",
        type=ValidationMethod.parse,
        choices=ValidationMethod.choices(),
        required=False,
        default=ValidationMethod.List,
        help=DF_TEST_SETS_METHOD_HELP_STR,
    )
    parser.add_argument(
        "--separator",
        type=separator,
        required=False,
        default=",",
        help=SEP_HELP_STR,
    )
    parser.add_argument(
        "--target",
        action="store",
        nargs="+",  # allow spaces: https://stackoverflow.com/a/26990349,
        type=str,
        default=["target"],
        help=TARGET_HELP_STR,
    )
    parser.add_argument(
        "--targets",
        action="store",
        type=column_parser,
        default=[],
        help=TARGETS_HELP_STR,
    )
    parser.add_argument(
        "--grouper",
        action="store",
        nargs="+",  # allow spaces: https://stackoverflow.com/a/26990349,
        type=str,
        default=None,
        help=GROUP_HELP_STR,
    )
    parser.add_argument(
        "--categoricals",
        # nargs="+",
        action="store",
        type=column_parser,
        default=[],
        help=CATEGORICAL_HELP_STR,
    )
    parser.add_argument(
        "--ordinals",
        # nargs="+",
        action="store",
        type=column_parser,
        default=[],
        help=ORDINAL_HELP_STR,
    )
    parser.add_argument(
        "--drops",
        # nargs="+",
        action="store",
        type=column_parser,
        default=[],
        help=DROP_HELP_STR,
    )
    parser.add_argument(
        "--mode",
        action="store",
        choices=["classify", "regress"],
        default="classify",
        help=MODE_HELP_STR,
    )
    parser.add_argument(
        "--device",
        type=DeviceIntent.parse,
        choices=DeviceIntent.choices(),
        default=DeviceIntent.Auto.value,
        help=DEVICE_HELP,
    )
    parser.add_argument(
        "--device-install",
        type=DeviceInstall.parse,
        choices=DeviceInstall.choices(),
        default=None,
        help=DEVICE_INSTALL_HELP,
    )
    parser.add_argument(
        "--tabpfn-version",
        type=TabPFNVersion.parse,
        choices=TabPFNVersion.choices(),
        metavar="{v3,v2.6,v2.5}",
        default=TabPFNVersion.V3.value,
        help=TABPFN_VERSION_HELP,
    )
    # NOTE: `nargs="+"` allows repeats, must be removed after
    parser.add_argument(
        "--classifiers",
        nargs="+",
        # type=DfAnalyzeClassifier.parse,
        type=str,
        choices=DfAnalyzeClassifier.choices(),
        default=DfAnalyzeClassifier.defaults(),
        metavar="",
        help=CLS_HELP_STR,
    )
    parser.add_argument(
        "--regressors",
        nargs="+",
        type=str,
        choices=DfAnalyzeRegressor.choices(),
        default=DfAnalyzeRegressor.defaults(),
        help=REG_HELP_STR,
    )
    parser.add_argument(
        "--feat-select",
        nargs="+",
        type=FeatureSelection.parseN,  # applied to each spaced element in arg
        choices=FeatureSelection.choicesN(),
        default=(FeatureSelection.Filter,),
        metavar="",  # silences ugly options spam related to enums
        help=FEAT_SELECT_HELP,
    )
    parser.add_argument(
        "--feat-downsample",
        type=FeatureDownsampleMethod.parse,
        choices=FeatureDownsampleMethod.choices(),
        default=FeatureDownsampleMethod.None_.value,
        metavar="",
        help=FEAT_DOWNSAMPLE_HELP,
    )
    parser.add_argument(
        "--n-feat-downsample",
        type=int_or_percent_parser(N_FEAT_DOWNSAMPLE_DEFAULT),
        default=N_FEAT_DOWNSAMPLE_DEFAULT,
        help=N_FEAT_DOWNSAMPLE_HELP,
    )
    parser.add_argument(
        "--downsample-variance-threshold",
        type=float,
        default=None,
        help="Minimum sample variance retained by variance downsampling.",
    )
    parser.add_argument(
        "--downsample-save-scores",
        action="store_true",
        help="Save scores for every feature instead of only the leading scores.",
    )
    parser.add_argument(
        "--downsample-chunk-size",
        type=int,
        default=DOWNSAMPLE_CHUNK_SIZE_DEFAULT,
        help="Maximum number of source features scored in one chunk.",
    )
    parser.add_argument(
        "--downsample-screening-fraction",
        type=float,
        default=0.25,
        help=DOWNSAMPLE_SCREENING_HELP,
    )
    parser.add_argument(
        "--large-feature-mode",
        action="store_true",
        help=LARGE_FEATURE_MODE_HELP,
    )
    parser.add_argument(
        "--assume-numeric-features",
        action="store_true",
        help=(
            "Skip the pandas dtype check in large-feature table mode; all predictor "
            "values must still convert to finite numbers."
        ),
    )
    parser.add_argument(
        "--skip-full-prepared-save-before-downsample",
        action="store_true",
        help="Do not save the full prepared matrix before downsampling.",
    )
    parser.add_argument(
        "--input-format",
        choices=("auto", "table", "svmlight"),
        default="auto",
        help="Input format; auto recognizes common SVMlight file suffixes.",
    )
    parser.add_argument(
        "--svmlight-index-base",
        choices=("auto", "zero", "one"),
        default="auto",
        help="Feature index base used to name SVMlight columns.",
    )
    # parser.add_argument(
    #     "--model-select",
    #     nargs="+",
    #     type=ModelFeatureSelection.parseN,  # applied to each spaced element in arg
    #     choices=ModelFeatureSelection.choicesN(),
    #     default=(ModelFeatureSelection.Embedded,),
    #     metavar="",  # silences ugly options spam related to enums
    #     help=MODEL_SELECT_HELP,
    # )
    parser.add_argument(
        "--embed-select",
        nargs="+",
        # type=enum_or_none_parser(EmbedSelectionModel),
        type=EmbedSelectionModel.parseN,
        # choices=[m.value for m in EmbedSelectionModel]
        # + ["none"]
        # + [EmbedSelectionModel.LGBM, EmbedSelectionModel.Linear, None],
        choices=EmbedSelectionModel.choicesN(),
        default=(EmbedSelectionModel.Linear,),
        help=EMBED_SELECT_MODEL_HELP,
        metavar="",
    )
    parser.add_argument(
        "--wrapper-select",
        type=WrapperSelection.parseN,
        choices=WrapperSelection.choicesN(),
        default=None,
        metavar="",
        help=WRAP_SELECT_HELP,
    )
    parser.add_argument(
        "--wrapper-model",
        type=WrapperSelectionModel.parseN,
        choices=WrapperSelectionModel.choicesN(),
        default=WrapperSelectionModel.Linear,
        metavar="",
        help=WRAP_SELECT_MODEL_HELP,
    )
    # parser.add_argument(  # TODO UNIMPLEMENTED!
    #     "--n-selection-tune-rounds",
    #     type=int,
    #     default=0,
    #     help=SELECT_TUNE_ROUNDS_HELP,
    # )
    # parser.add_argument(
    #     "--feat-clean",
    #     nargs="+",
    #     type=str,
    #     choices=FEATURE_CLEANINGS,
    #     default=["constant"],
    #     help=FEAT_CLEAN_HELP,
    # )
    parser.add_argument(
        "--norm",
        type=Normalization.parse,
        choices=Normalization.choices(),
        default=Normalization.Robust.value,
        help=NORM_HELP,
    )
    parser.add_argument(
        "--nan",
        type=NanHandling.parse,
        choices=NanHandling.choices(),
        default=NanHandling.Mean.value,
        help=NAN_HELP,
    )
    # parser.add_argument(
    #     "--n-feat-embed",
    #     type=int_or_percent_or_none_parser(default=None),
    #     default=None,
    #     help=N_FEAT_EMBED_HELP,
    # )
    parser.add_argument(
        "--n-feat-filter",
        type=int_or_percent_parser(default=P_FILTER_TOTAL_DEFAULT),
        default=P_FILTER_TOTAL_DEFAULT,
        help=N_FEAT_TOTAL_FILTER_HELP,
    )
    parser.add_argument(
        "--n-feat-wrapper",
        type=int_or_percent_parser(default=N_WRAPPER_DEFAULT),
        default=N_WRAPPER_DEFAULT,
        help=N_FEAT_WRAPPER_HELP,
    )
    parser.add_argument(
        "--n-filter-cont",
        type=int_or_percent_parser(default=P_FILTER_CONT_DEFAULT),
        default=None,
        help=N_FEAT_CONT_FILTER_HELP,
    )
    parser.add_argument(
        "--n-filter-cat",
        type=int_or_percent_parser(default=P_FILTER_CAT_DEFAULT),
        default=None,
        help=N_FEAT_CAT_FILTER_HELP,
    )
    parser.add_argument(
        "--filter-method",
        type=FilterSelection.parse,
        choices=FilterSelection.choices(),
        default=FilterSelection.Association,
        metavar="",
        help=FILTER_METHOD_HELP,
    )
    parser.add_argument(
        "--filter-assoc-cont-classify",
        choices=ContClsStats.choices(),
        type=ContClsStats.parse,
        default=ContClsStats.default(),
        metavar="",
        help=ASSOC_SELECT_CONT_CLS_STATS,
    )
    parser.add_argument(
        "--filter-assoc-cat-classify",
        choices=CatClsStats.choices(),
        type=CatClsStats.parse,
        default=CatClsStats.default(),
        metavar="",
        help=ASSOC_SELECT_CAT_CLS_STATS,
    )
    parser.add_argument(
        "--filter-assoc-cont-regress",
        choices=ContRegStats.choices(),
        type=ContRegStats.parse,
        default=ContRegStats.default(),
        metavar="",
        help=ASSOC_SELECT_CONT_REG_STATS,
    )
    parser.add_argument(
        "--filter-assoc-cat-regress",
        choices=CatRegStats.choices(),
        type=CatRegStats.parse,
        default=CatRegStats.default(),
        metavar="",
        help=ASSOC_SELECT_CAT_REG_STATS,
    )
    parser.add_argument(
        "--filter-pred-regress",
        choices=RegScore.choices(),
        type=RegScore.parse,
        default=RegScore.default(),
        help=PRED_SELECT_REG_SCORE,
    )
    parser.add_argument(
        "--filter-pred-classify",
        choices=ClsScore.choices(),
        type=ClsScore.parse,
        default=ClsScore.default(),
        help=PRED_SELECT_CLS_SCORE,
    )
    parser.add_argument(
        "--redundant-wrapper-selection",
        action="store_true",
        help=REDUNDANT_SELECTION,
    )
    parser.add_argument(
        "--redundant-threshold",
        type=lambda x: abs(float(x)),
        default=0.005,
        help=REDUNDANT_THRESHOLD,
    )
    parser.add_argument(
        "--redundant-corr-threshold",
        type=lambda x: abs(float(x)),
        default=0.8,
        help=REDUNDANT_CORR_THRESHOLD,
    )
    # parser.add_argument(
    #     "--htune",
    #     action="store_true",
    #     help=HTUNE_HELP,
    # )
    # parser.add_argument(
    #     "--htune-val",
    #     type=str,
    #     choices=HTUNE_VAL_METHODS,
    #     default=3,
    #     help=HTUNEVAL_HELP_STR,
    # )
    # parser.add_argument(
    #     "--htune-val-size",
    #     type=cv_size,
    #     default=3,
    #     help=HTUNE_VALSIZE_HELP,
    # )
    parser.add_argument(
        "--htune-trials",
        type=int,
        default=100,
        help=HTUNE_TRIALS_HELP,
    )
    parser.add_argument(
        "--htune-cls-metric",
        choices=ClassifierScorer.choices(),
        type=ClassifierScorer.parse,
        default=ClassifierScorer.default(),
        help=CLS_TUNE_METRIC,
    )
    parser.add_argument(
        "--htune-reg-metric",
        choices=RegressorScorer.choices(),
        type=RegressorScorer.parse,
        default=RegressorScorer.default(),
        help=REG_TUNE_METRIC,
    )
    # parser.add_argument(
    #     "--mc-repeats",
    #     type=int,
    #     default=10,
    #     help=MC_REPEATS_HELP,
    # )
    # parser.add_argument(
    #     "--test-val",
    #     type=str,
    #     choices=HTUNE_VAL_METHODS,
    #     default="kfold",
    #     help=TEST_VAL_HELP,
    # )
    parser.add_argument(
        "--test-val-size",
        type=int_or_percent_parser(default=0.4),
        default=0.4,
        help=TEST_VALSIZES_HELP,
    )
    parser.add_argument(
        "--error-consistency",
        "--ec",
        dest="error_consistency",
        action="store_true",
        default=False,
        help=ERROR_CONSISTENCY_HELP,
    )
    parser.add_argument(
        "--ec-folds",
        type=int,
        default=5,
        help=EC_FOLDS_HELP,
    )
    parser.add_argument(
        "--ec-repetitions",
        type=int,
        default=5,
        help=EC_REPETITIONS_HELP,
    )
    parser.add_argument(
        "--ec-model-seed-mode",
        choices=["vary", "fixed"],
        default="vary",
        help=EC_MODEL_SEED_MODE_HELP,
    )
    parser.add_argument(
        "--ec-methods",
        nargs="+",
        default=None,
        help=EC_METHODS_HELP,
    )
    parser.add_argument(
        "--ec-save-predictions",
        action="store_true",
        default=False,
        help=EC_SAVE_PREDICTIONS_HELP,
    )
    parser.add_argument(
        "--ec-empty-unions",
        choices=["0", "1", "nan", "drop", "error", "warn"],
        default="warn",
        help=EC_EMPTY_UNIONS_HELP,
    )
    parser.add_argument(
        "--ec-epsilon",
        type=float,
        default=0.0,
        help=EC_EPSILON_HELP,
    )
    parser.add_argument(
        "--ec-recurrence-threshold",
        type=float,
        default=0.5,
        help=EC_RECURRENCE_THRESHOLD_HELP,
    )
    parser.add_argument(
        "--mt-agg-strategy",
        type=str,
        choices=["borda", "freq"],
        default="borda",
        help=MT_AGG_STRATEGY_HELP_STR,
    )
    parser.add_argument(
        "--mt-top-k",
        type=int,
        default=None,
        help=MT_TOP_K_HELP_STR,
    )
    parser.add_argument(
        "--adaptive-error",
        action="store_true",
        default=False,
        help=ADAPTIVE_ERROR_HELP,
    )
    parser.add_argument(
        "--aer-oof-folds",
        type=int,
        default=5,
        help=AER_OOF_FOLDS_HELP,
    )
    parser.add_argument(
        "--aer-bins",
        type=int,
        default=20,
        help=AER_BINS_HELP,
    )
    parser.add_argument(
        "--aer-target-error",
        type=float,
        default=0.05,
        help=AER_TARGET_ERROR_HELP,
    )
    parser.add_argument(
        "--aer-alpha",
        type=float,
        default=0.05,
        help=AER_ALPHA_HELP,
    )
    parser.add_argument(
        "--aer-min-bin-count",
        type=int,
        default=10,
        help=AER_MIN_BIN_COUNT_HELP,
    )
    parser.add_argument(
        "--aer-prior-strength",
        type=float,
        default=2.0,
        help=AER_PRIOR_STRENGTH_HELP,
    )
    parser.add_argument(
        "--no-aer-smooth",
        dest="aer_smooth",
        action="store_false",
        default=True,
        help=AER_SMOOTH_HELP,
    )
    parser.add_argument(
        "--aer-monotonic",
        action="store_true",
        default=False,
        help=AER_MONOTONIC_HELP,
    )
    parser.add_argument(
        "--aer-adaptive-binning",
        action="store_true",
        default=False,
        help=AER_ADAPTIVE_BINNING_HELP,
    )
    parser.add_argument(
        "--aer-confidence-metric",
        type=str,
        default="auto",
        help=AER_CONFIDENCE_METRIC_HELP,
    )
    parser.add_argument(
        "--aer-nmin",
        type=int,
        default=1,
        help=AER_NMIN_HELP,
    )
    parser.add_argument(
        "--aer-top-k",
        type=int,
        default=0,
        help=AER_TOP_K_HELP,
    )
    parser.add_argument(
        "--aer-ensemble",
        action="store_true",
        default=False,
        help=AER_ENSEMBLE_HELP,
    )
    parser.add_argument(
        "--aer-ensemble-strategies",
        nargs="+",
        type=str,
        default=None,
        help=AER_ENSEMBLE_STRATEGIES_HELP,
    )
    parser.add_argument(
        "--aer-ens-top-n",
        type=int,
        default=3,
        help=AER_ENS_TOP_N_HELP,
    )
    parser.add_argument(
        "--aer-ens-beta",
        type=float,
        default=5.0,
        help=AER_ENS_BETA_HELP,
    )
    parser.add_argument(
        "--aer-ens-tau0",
        type=float,
        default=0.3,
        help=AER_ENS_TAU0_HELP,
    )
    parser.add_argument(
        "--aer-ens-lambda",
        type=float,
        default=0.5,
        help=AER_ENS_LAMBDA_HELP,
    )
    parser.add_argument(
        "--aer-ens-alpha",
        type=float,
        default=0.7,
        help=AER_ENS_ALPHA_HELP,
    )
    parser.add_argument(
        "--aer-ens-trim-q",
        type=float,
        default=0.6,
        help=AER_ENS_TRIM_Q_HELP,
    )
    parser.add_argument(
        "--aer-ens-tau-low",
        type=float,
        default=0.15,
        help=AER_ENS_TAU_LOW_HELP,
    )
    parser.add_argument(
        "--aer-ens-tau-high",
        type=float,
        default=0.35,
        help=AER_ENS_TAU_HIGH_HELP,
    )
    parser.add_argument(
        "--outdir",
        type=resolved_path,
        required=False,
        default=None,
        help=OUTDIR_HELP,
    )
    parser.add_argument(
        "--no-preds",
        action="store_true",
        help=NO_PRED_HELP,
    )
    parser.add_argument(
        "--verbosity",
        type=lambda a: Verbosity(int(a)),
        default=Verbosity(1),
        help=VERBOSITY_HELP,
    )
    parser.add_argument(
        "--no-warn-explosion",
        action="store_true",
        help=EXPLODE_HELP,
    )
    parser.add_argument(
        "--seed",
        type=seed_parser(SeedKind),
        default=SeedKind.default(),
        help=SEED_HELP_STR,
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help=VERSION_HELP,
    )
    return parser


class ArgKind(Enum):
    Choice = "choice"
    ChoiceN = "choice-n"
    Flag = "flag"
    IntOrPercent = "int-percent"
    Int = "int"
    Path = "path"
    Separator = "seperator"
    String = "string"
    StringList = "string-list"


class RandKind(Enum):
    ChooseN = "choose-n"
    ChooseNoneOrN = "choose-none-or-n"
    ChooseOne = "one"
    Columns = "columns"
    Custom = "custom"
    Flag = "flag"
    Int = "int"
    IntOrPercent = "int-or-percent"
    Path = "path"
    Target = "target"


ArgsDict = dict[str, tuple[RandKind, Optional[list[str]]]]


def get_parser_dict() -> ArgsDict:
    """
    Returns an ArgsDict, which is a dictionary where each key is the name (with
    dashes) of a CLI argument (e.g. one key is `--df-train`), and where each
    value is a tuple[RandKind, Optional[list[Choices]]].
    """
    args_dict: ArgsDict = {
        "--df": (RandKind.Path, None),
        "--df-train": (RandKind.Path, None),
        "--df-tests": (RandKind.Custom, None),  # TODO: Handle!
        "--df-tests-method": (RandKind.ChooseN, ValidationMethod.choices()),
        "--spreadsheet": (RandKind.Path, None),
        "--separator": (RandKind.Custom, None),
        "--seed": (RandKind.Custom, None),
        "--target": (RandKind.Target, None),
        # TODO do this properly for --grouper
        "--grouper": (RandKind.Custom, None),
        "--categoricals": (RandKind.Columns, None),
        "--ordinals": (RandKind.Columns, None),
        "--drops": (RandKind.Columns, None),
        "--mode": (RandKind.ChooseOne, ["classify", "regress"]),
        "--device": (RandKind.ChooseOne, DeviceIntent.choices()),
        "--device-install": (RandKind.ChooseOne, DeviceInstall.choices()),
        "--tabpfn-version": (RandKind.ChooseOne, TabPFNVersion.choices()),
        "--classifiers": (
            RandKind.ChooseN,
            DfAnalyzeClassifier.choices(),
        ),
        "--regressors": (
            RandKind.ChooseN,
            DfAnalyzeRegressor.choices(),
        ),
        "--feat-select": (RandKind.ChooseNoneOrN, FeatureSelection.choices()),
        "--embed-select": (RandKind.ChooseNoneOrN, EmbedSelectionModel.choices()),
        "--wrapper-select": (RandKind.ChooseNoneOrN, WrapperSelection.choices()),
        "--wrapper-model": (RandKind.ChooseOne, WrapperSelectionModel.choices()),
        "--norm": (RandKind.ChooseOne, Normalization.choices()),
        "--nan": (RandKind.ChooseOne, NanHandling.choices()),
        "--n-feat-filter": (RandKind.IntOrPercent, None),
        "--n-feat-wrapper": (RandKind.IntOrPercent, None),
        "--n-filter-cont": (RandKind.IntOrPercent, None),
        "--n-filter-cat": (RandKind.IntOrPercent, None),
        "--filter-method": (RandKind.ChooseOne, FilterSelection.choices()),
        "--filter-assoc-cont-classify": (RandKind.ChooseOne, ContClsStats.choices()),
        "--filter-assoc-cat-classify": (RandKind.ChooseOne, CatClsStats.choices()),
        "--filter-assoc-cont-regress": (RandKind.ChooseOne, ContRegStats.choices()),
        "--filter-assoc-cat-regress": (RandKind.ChooseOne, CatRegStats.choices()),
        "--filter-pred-regress": (RandKind.ChooseOne, RegScore.choices()),
        "--filter-pred-classify": (RandKind.ChooseOne, ClsScore.choices()),
        "--redundant-wrapper-selection": (RandKind.Flag, None),
        "--redundant-threshold": (RandKind.Custom, None),
        "--redundant-corr-threshold": (RandKind.Custom, None),
        "--htune-trials": (RandKind.ChooseOne, ["5", "10", "20"]),
        "--htune-cls-metric": (RandKind.ChooseOne, ClassifierScorer.choices()),
        "--htune-reg-metric": (RandKind.ChooseOne, RegressorScorer.choices()),
        "--test-val-size": (RandKind.IntOrPercent, None),
        "--outdir": (RandKind.Path, None),
        "--verbosity": (RandKind.ChooseOne, ["0", "1", "2"]),
        "--no-warn-explosion": (RandKind.Flag, None),
    }
    return args_dict


def randip(
    imin: int, imax: int, pmin: float = 0.1, pmax: float = 0.5
) -> Union[float, int]:
    use_ints = bool(np.random.binomial(n=1, p=0.5, size=1).item())
    if use_ints:
        return np.random.randint(imin, imax)
    return np.random.uniform(pmin, pmax)


def random_cli_args(
    ds: TestDataset,
    tempdir: Path,
    spreadsheet: bool = True,
    multitest: bool = False,
    random_target: bool = True,
    n_tests_min: int = 1,
    n_tests_max: int = 4,
) -> tuple[list[str], Path, list[Path]]:
    """Generate random valid CLI args and data paths for testing.

    Returns
    -------
    arg_options: list[str]
        List of CLI args with valid random arguments

    datapath: Path
        Path to which temporary training data will be written if multitest=True,
        otherwise, path to which full temporary data will be written.

    testpaths: list[Path]
        Paths to which temporary testing datasets will be written if
        multitest=True, otherwise, empty list.
    """
    if multitest and spreadsheet:
        raise RuntimeError(
            "`--spreadsheet` option is incompatible with multiple test set inputs."
        )
    # if (not multitest) and (not spreadsheet):
    #     raise RuntimeError("Either test multiple test sets or spreadsheets. ")

    # choose random sizes for number of features selected, test set, etc
    n, p = ds.shape[0], ds.shape[1] - 1
    p_max = min(15, p)
    p_min = 1
    pp_max = p_max / p
    pp_min = 1 / p

    n_max = n // 2
    n_min = min(100, n)
    tmin = 0.1
    tmax = 0.5

    int_or_percents = {
        "--n-feat-filter": randip(p_min, p_max, pp_min, pp_max),
        "--n-feat-wrapper": randip(p_min, p_max, pp_min, pp_max),
        "--n-filter-cont": randip(p_min, p_max, pp_min, pp_max),
        "--n-filter-cat": randip(p_min, p_max, pp_min, pp_max),
        "--test-val-size": randip(n_min, n_max, tmin, tmax),
    }

    cols = set(ds.load().columns.to_list())
    if random_target:
        target = choice(list(cols))
    else:
        target = "target"
    cols.remove(target)
    # if " " in target:
    #     target = f"'{target}'"

    n_drop = np.random.randint(0, 3)
    drops = np.random.choice(list(cols), n_drop).tolist()
    cols.difference_update(drops)
    # drops = " ".join([f"'{d}'" if " " in d else d for d in drops])
    # drops = f"'{' '.join(drops)}'"
    drops = " ".join([f'"{x}"' if " " in x else x for x in drops])

    n_cats = np.random.randint(len(cols) // 2)
    cats = np.random.choice(list(cols), n_cats).tolist()
    cols.difference_update(cats)
    # cats = " ".join([f"'{c}'" if " " in c else c for c in cats])
    # cats = f"'{' '.join(cats)}'"
    cats = " ".join([f'"{x}"' if " " in x else x for x in cats])

    n_ords = np.random.randint(len(cols) // 2)
    ords = np.random.choice(list(cols), n_ords).tolist()
    cols.difference_update(ords)
    # ords = " ".join([f"'{o}'" if " " in o else o for o in ords])
    # ords = f"'{' '.join(ords)}'"
    ords = " ".join([f'"{x}"' if " " in x else x for x in ords])

    # there is no reason ever to be returning the path in `ds.datapath`` since
    # the caller of this function already has access to `ds
    spreadsheet_tempfile = tempdir / f"{ds.dsname}.csv"
    tempfile = tempdir / f"{ds.dsname}.parquet"
    # TODO: handle --df-train and --df-tests cases
    testfiles: list[Path] = []

    args_dict = get_parser_dict()
    if multitest:
        args_dict.pop("--df", None)
        args_dict.pop("--spreadsheet", None)

        n_test = np.random.randint(n_tests_min, n_tests_max)
        fold_idxs = np.arange(n_test).tolist()
        testfiles = [add_fold_idx(tempfile, fold_idx) for fold_idx in fold_idxs]
        for file in testfiles:
            file.touch()
    else:
        args_dict.pop("--df-train", None)
        args_dict.pop("--df-tests", None)
        args_dict.pop("--df-tests-method", None)
        if not spreadsheet:
            args_dict.pop("--spreadsheet", None)

    arg_options = []
    for argstr, (kind, choices) in args_dict.items():
        # handle ds-related randoms first
        if kind is RandKind.Path:
            if argstr == "--df":
                if spreadsheet:
                    arg_options.append(f"{argstr} {spreadsheet_tempfile}")
                else:
                    # raise RuntimeError(
                    #     "Impossible! Can't have --df argument, since must be doing multitest"
                    # )
                    arg_options.append(f"{argstr} {tempfile}")
            elif argstr == "--df-train":
                if spreadsheet:
                    raise ValueError(
                        "Cannot use `--spreadsheet` option with `--df-train` option"
                    )
                arg_options.append(f"{argstr} {ds.datapath}")
            elif argstr == "--spreadsheet":
                if spreadsheet:
                    arg_options.append(f"{argstr} {spreadsheet_tempfile}")
                else:
                    raise ValueError("Impossible!")
            elif argstr == "--outdir":
                arg_options.append(f"{argstr} {tempdir}")
            else:
                raise KeyError(
                    f"Unhandled argument: {argstr}, kind={kind}, choices={choices}"
                )
        elif (kind is RandKind.Custom) and (argstr == "--separator"):
            continue
        elif (kind is RandKind.Custom) and (argstr == "--seed"):
            seed = SeedKind.random_seed()
            arg_options.append(f"{argstr} {seed}")
        elif (kind is RandKind.Custom) and (argstr == "--grouper"):
            continue  # TODO: actually implement something here
        elif (kind is RandKind.Custom) and (argstr == "--redundant-threshold"):
            # just pick something that could work (though not well) for all metrics
            score = uniform(0, 0.1)
            arg_options.append(f"{argstr} {score}")
        elif (kind is RandKind.Custom) and (argstr == "--redundant-corr-threshold"):
            score = uniform(0.0, 1.0)
            arg_options.append(f"{argstr} {score}")
        elif (kind is RandKind.Custom) and (argstr == "--df-tests"):
            if not multitest:
                pass
            else:
                paths_str = ",".join(map(str, testfiles))
                arg_options.append(f"{argstr} {paths_str}")
        elif kind is RandKind.IntOrPercent:
            numeric = round(int_or_percents[argstr], 2)
            arg_options.append(f"{argstr} {numeric}")
        elif kind is RandKind.Target:
            arg_options.append(f"{argstr} {target}")
        elif kind is RandKind.Columns:
            if argstr == "--categoricals":
                if n_cats > 0:
                    arg_options.append(f"{argstr} {cats}")
            elif argstr == "--ordinals":
                if n_ords > 0:
                    arg_options.append(f"{argstr} {ords}")
            elif argstr == "--drops":
                if n_drop > 0:
                    arg_options.append(f"{argstr} {drops}")
            else:
                raise KeyError(
                    f"Unhandled argument: {argstr}, kind={kind}, choices={choices}"
                )
        elif kind is RandKind.Flag:
            do_flag = bool(np.random.binomial(n=1, p=0.5, size=1).item())
            if do_flag:
                arg_options.append(f"{argstr}")
        elif kind in [RandKind.ChooseN, RandKind.ChooseNoneOrN, RandKind.ChooseOne]:
            if choices is None:
                raise ValueError("RandKind cannot be a choice with `choices=None`")
            if kind is RandKind.ChooseOne:
                if argstr == "--mode" and not random_target:
                    chosen = "classify" if ds.is_classification else "regress"
                else:
                    chosen = choice(choices)
                arg_options.append(f"{argstr} {chosen}")
            elif kind is RandKind.ChooseN:
                n_choose = np.random.randint(1, len(choices))
                chosen = " ".join(np.random.choice(choices, n_choose).tolist())
            elif kind is RandKind.ChooseNoneOrN:
                is_none = bool(np.random.binomial(n=1, p=0.5, size=1).item())
                if is_none:
                    arg_options.append(f"{argstr} none")
                else:
                    n_choose = np.random.randint(1, len(choices))
                    chosen = " ".join(np.random.choice(choices, n_choose).tolist())
                    arg_options.append(f"{argstr} {chosen}")
        else:
            raise KeyError(
                f"Unhandled argument: {argstr}, kind={kind}, choices={choices}"
            )

    if spreadsheet:
        return arg_options, spreadsheet_tempfile, testfiles
    else:
        return arg_options, tempfile, testfiles


def get_options(args: Optional[str] = None) -> ProgramOptions:
    """parse command line arguments"""
    # parser = ArgumentParser(description=DESC)
    parser = make_parser()
    cli_args = parse_and_merge_args(parser, args)

    if cli_args.ec_folds < 2:
        raise ArgumentError("Argument `--ec-folds` must be at least 2.")
    if cli_args.ec_repetitions < 1:
        raise ArgumentError("Argument `--ec-repetitions` must be at least 1.")
    if not np.isfinite(cli_args.ec_epsilon) or cli_args.ec_epsilon < 0:
        raise ArgumentError("Argument `--ec-epsilon` must be a finite value >= 0.")
    if not np.isfinite(cli_args.ec_recurrence_threshold) or not (
        0.0 <= cli_args.ec_recurrence_threshold <= 1.0
    ):
        raise ArgumentError(
            "Argument `--ec-recurrence-threshold` must be between 0 and 1."
        )
    if cli_args.ec_methods:
        from df_analyze.analysis.error_consistency.regression import (
            normalize_regression_method,
        )

        normalized = []
        for method in cli_args.ec_methods:
            if str(method).strip().lower() == "all":
                continue
            try:
                clean = normalize_regression_method(method)
            except ValueError as error:
                raise ArgumentError(str(error)) from error
            if clean not in normalized:
                normalized.append(clean)
        cli_args.ec_methods = tuple(normalized) if normalized else None

    mode = str(cli_args.mode).lower()
    is_cls = True if "class" in mode else False

    cats = set(cli_args.categoricals)
    ords = set(cli_args.ordinals)

    for cat in cats:
        if cat in ords:
            warn(
                f"Feature '{cat}' is present in both `--categoricals` and "
                f"`--ordinals` options. Feature '{cat}' will be treated as "
                "ordinal to reduce compute times"
            )
            ords.remove(cat)

    classifiers = set(DfAnalyzeClassifier.from_args(cli_args.classifiers))
    regressors = set(DfAnalyzeRegressor.from_args(cli_args.regressors))
    if DfAnalyzeClassifier.SVM in classifiers:
        warn(
            "Found `svm` classifier as a specified model. SVMs are at the "
            "moment too computationally-expensive on most datasets. This "
            "model has been disabled and performance metrics for this model "
            "will not appear in final results."
        )

        classifiers.discard(DfAnalyzeClassifier.SVM)
    if DfAnalyzeRegressor.SVM in regressors:
        warn(
            "Found `svm` regressor as a specified model. SVMs are at the "
            "moment too computationally-expensive on most datasets. This "
            "model has been disabled and performance metrics for this model "
            "will not appear in final results."
        )

        regressors.discard(DfAnalyzeRegressor.SVM)

    if DfAnalyzeClassifier.Dummy not in classifiers:
        classifiers.add(DfAnalyzeClassifier.Dummy)
    if DfAnalyzeRegressor.Dummy not in regressors:
        regressors.add(DfAnalyzeRegressor.Dummy)

    classifiers = tuple(sorted(classifiers))
    regressors = tuple(sorted(regressors))

    # https://stackoverflow.com/a/26990349,
    grouper = " ".join(cli_args.grouper) if cli_args.grouper is not None else None

    if cli_args.df is not None and cli_args.df_train is not None:
        raise ArgumentError("Arguments `--df` and `--df-train` are mutually exclusive.")

    if cli_args.df is None:
        if cli_args.df_train is not None:
            datapath = cli_args.df_train
        else:
            datapath = cli_args.spreadsheet
    else:
        datapath = cli_args.df

    parsed_target = " ".join(
        cli_args.target
    ).strip()  # https://stackoverflow.com/a/26990349,
    parsed_targets_raw = (
        list(cli_args.targets) if len(cli_args.targets) > 0 else [parsed_target]
    )
    parsed_targets = [str(t).strip() for t in parsed_targets_raw if str(t).strip() != ""]
    if len(parsed_targets) == 0:
        raise ArgumentError(
            "Argument `--targets` must include at least one non-empty value."
        )

    input_name = "" if datapath is None else datapath.name.lower()
    sparse_suffixes = (".svm", ".svmlight", ".libsvm", ".binary")
    compression_suffixes = ("", ".gz", ".bz2", ".xz")
    uses_svmlight = cli_args.input_format == "svmlight" or (
        cli_args.input_format == "auto"
        and any(
            input_name.endswith(suffix + compression)
            for suffix in sparse_suffixes
            for compression in compression_suffixes
        )
    )
    if uses_svmlight:
        if cli_args.large_feature_mode:
            raise ArgumentError(
                "Arguments `--large-feature-mode` and SVMlight input are mutually "
                "exclusive."
            )
        if len(parsed_targets) > 1:
            raise ArgumentError("SVMlight input supports exactly one target.")
        unsupported = []
        if grouper is not None:
            unsupported.append("--grouper")
        if cats:
            unsupported.append("--categoricals")
        if ords:
            unsupported.append("--ordinals")
        if cli_args.drops:
            unsupported.append("--drops")
        if unsupported:
            raise ArgumentError(
                "SVMlight input does not support table column arguments: "
                + ", ".join(unsupported)
                + "."
            )
        downsample_method = FeatureDownsampleMethod.from_arg(cli_args.feat_downsample)
        if downsample_method is FeatureDownsampleMethod.None_:
            raise ArgumentError("SVMlight input requires `--feat-downsample`.")
        indexed_methods = {
            FeatureDownsampleMethod.Auto,
            FeatureDownsampleMethod.Random,
            FeatureDownsampleMethod.Variance,
            FeatureDownsampleMethod.FTest,
            FeatureDownsampleMethod.RankEnsemble,
            FeatureDownsampleMethod.SelectorEnsemble,
            FeatureDownsampleMethod.StableRank,
        }
        if downsample_method not in indexed_methods:
            allowed = ", ".join(sorted(method.value for method in indexed_methods))
            raise ArgumentError(
                "SVMlight input requires an indexed feature downsampling method. "
                f"Choose one of: {allowed}."
            )

    return ProgramOptions(
        datapath=datapath,
        test_paths=cli_args.df_tests,
        tests_method=cli_args.df_tests_method,
        target=parsed_targets[0],
        targets=parsed_targets,
        grouper=grouper,
        categoricals=sorted(cats),
        ordinals=sorted(ords),
        drops=cli_args.drops,
        norm=Normalization.from_arg(cli_args.norm),
        nan_handling=NanHandling.from_arg(cli_args.nan),
        feat_select=FeatureSelection.from_args(cli_args.feat_select),
        embed_select=EmbedSelectionModel.from_args(cli_args.embed_select),
        wrapper_select=WrapperSelection.from_argN(cli_args.wrapper_select),
        wrapper_model=WrapperSelectionModel.from_arg(cli_args.wrapper_model),
        # n_feat=cli_args.n_feat,
        n_feat_wrapper=cli_args.n_feat_wrapper,
        n_feat_filter=cli_args.n_feat_filter,
        n_filter_cont=cli_args.n_filter_cont,
        n_filter_cat=cli_args.n_filter_cat,
        filter_assoc_cont_cls=ContClsStats.from_arg(cli_args.filter_assoc_cont_classify),
        filter_assoc_cat_cls=CatClsStats.from_arg(cli_args.filter_assoc_cat_classify),
        filter_assoc_cont_reg=ContRegStats.from_arg(cli_args.filter_assoc_cont_regress),
        filter_assoc_cat_reg=CatRegStats.from_arg(cli_args.filter_assoc_cat_regress),
        filter_pred_cls_score=ClsScore.from_arg(cli_args.filter_pred_classify),
        filter_pred_reg_score=RegScore.from_arg(cli_args.filter_pred_regress),
        redundant_selection=cli_args.redundant_wrapper_selection,
        redundant_threshold=cli_args.redundant_threshold,
        redundant_corr_threshold=cli_args.redundant_corr_threshold,
        mt_agg_strategy=cli_args.mt_agg_strategy,
        mt_top_k=cli_args.mt_top_k,
        error_consistency=cli_args.error_consistency,
        ec_folds=cli_args.ec_folds,
        ec_repetitions=cli_args.ec_repetitions,
        ec_model_seed_mode=cli_args.ec_model_seed_mode,
        ec_methods=cli_args.ec_methods,
        ec_save_predictions=cli_args.ec_save_predictions,
        ec_empty_unions=cli_args.ec_empty_unions,
        ec_epsilon=cli_args.ec_epsilon,
        ec_recurrence_threshold=cli_args.ec_recurrence_threshold,
        is_classification=is_cls,
        classifiers=classifiers,
        regressors=regressors,
        tabpfn_version=cli_args.tabpfn_version,
        # htune=cli_args.htune,
        # htune_val=cli_args.htune_val,
        # htune_val_size=cli_args.htune_val_size,
        htune_trials=cli_args.htune_trials,
        htune_cls_metric=ClassifierScorer.from_arg(cli_args.htune_cls_metric),
        htune_reg_metric=RegressorScorer.from_arg(cli_args.htune_reg_metric),
        # test_val=cli_args.test_val,
        test_val_size=cli_args.test_val_size,
        # mc_repeats=cli_args.mc_repeats,
        outdir=cli_args.outdir,
        seed=cli_args.seed,
        is_spreadsheet=cli_args.spreadsheet is not None,
        separator=cli_args.separator,
        verbosity=cli_args.verbosity,
        no_warn_explosion=cli_args.no_warn_explosion,
        no_preds=cli_args.no_preds,
        feat_downsample=cli_args.feat_downsample,
        n_feat_downsample=cli_args.n_feat_downsample,
        downsample_variance_threshold=cli_args.downsample_variance_threshold,
        downsample_save_scores=cli_args.downsample_save_scores,
        downsample_chunk_size=cli_args.downsample_chunk_size,
        downsample_screening_fraction=cli_args.downsample_screening_fraction,
        large_feature_mode=cli_args.large_feature_mode,
        assume_numeric_features=cli_args.assume_numeric_features,
        skip_full_prepared_save_before_downsample=(
            cli_args.skip_full_prepared_save_before_downsample
        ),
        input_format=cli_args.input_format,
        svmlight_index_base=cli_args.svmlight_index_base,
        device=cli_args.device,
        device_install=cli_args.device_install,
        adaptive_error=cli_args.adaptive_error,
        aer_oof_folds=cli_args.aer_oof_folds,
        aer_bins=cli_args.aer_bins,
        aer_target_error=cli_args.aer_target_error,
        aer_alpha=cli_args.aer_alpha,
        aer_min_bin_count=cli_args.aer_min_bin_count,
        aer_prior_strength=cli_args.aer_prior_strength,
        aer_smooth=cli_args.aer_smooth,
        aer_monotonic=cli_args.aer_monotonic,
        aer_adaptive_binning=cli_args.aer_adaptive_binning,
        aer_confidence_metric=cli_args.aer_confidence_metric,
        aer_nmin=cli_args.aer_nmin,
        aer_top_k=cli_args.aer_top_k,
        aer_ensemble=cli_args.aer_ensemble,
        aer_ensemble_strategies=(
            tuple(cli_args.aer_ensemble_strategies)
            if cli_args.aer_ensemble_strategies is not None
            else None
        ),
        aer_ens_top_n=cli_args.aer_ens_top_n,
        aer_ens_beta=cli_args.aer_ens_beta,
        aer_ens_tau0=cli_args.aer_ens_tau0,
        aer_ens_lambda=cli_args.aer_ens_lambda,
        aer_ens_alpha=cli_args.aer_ens_alpha,
        aer_ens_trim_q=cli_args.aer_ens_trim_q,
        aer_ens_tau_low=cli_args.aer_ens_tau_low,
        aer_ens_tau_high=cli_args.aer_ens_tau_high,
    )


if __name__ == "__main__":
    parser = make_parser()
    args = get_parser_dict()
    for arg, info in args.items():
        print(f"{arg}: {info}")
