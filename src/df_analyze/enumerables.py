# pyright: reportCallIssue=false
from __future__ import annotations

import secrets
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum, EnumMeta
from importlib.util import find_spec
from math import isnan
from random import choice, randint
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
    no_type_check,
)
from warnings import warn

import numpy as np
from numpy import ndarray
from pandas import Series
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    explained_variance_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

from df_analyze._constants import SEED
from df_analyze.scoring import npv, ppv, robust_auroc_score, sensitivity, specificity

if TYPE_CHECKING:
    from df_analyze.models.base import DfAnalyzeModel

T = TypeVar("T", bound="RandEnum")


def is_catboost_available() -> bool:
    return find_spec("catboost") is not None


def is_xgboost_available() -> bool:
    return find_spec("xgboost") is not None


def is_tabpfn_available() -> bool:
    return find_spec("tabpfn") is not None


def is_kan_available() -> bool:
    return find_spec("kan") is not None


class RandEnum(Generic[T]):
    @classmethod
    def random(cls: Type[T]) -> T:
        if not isinstance(cls, EnumMeta):
            raise ValueError("Undefined")
        return choice([*cls])  # type: ignore

    @classmethod
    def random_n(cls: Type[T]) -> tuple[T, ...]:
        n = randint(1, len(cls) - 1)  # type: ignore
        if not isinstance(cls, EnumMeta):
            raise ValueError("Undefined")
        return tuple(np.random.choice([*cls], size=n, replace=False).tolist())  # type: ignore

    @classmethod
    def random_none(cls: Type[T]) -> Optional[T]:
        if not isinstance(cls, EnumMeta):
            raise ValueError("Undefined")
        return choice([*cls, None])  # type: ignore

    @classmethod
    def choices(cls: Type[T]) -> list[str]:
        return [x.value for x in cls]  # type: ignore

    @classmethod
    def choicesN(cls) -> list[Optional[str]]:
        # info = " | ".join([e.value for e in cls])  # type: ignore
        # return f"< {info} | None >"
        return [None, *[x for x in cls]]  # type: ignore

    @classmethod
    def parse(cls, s: str) -> str:
        try:
            return cls(s).value  # type: ignore
        except Exception:
            return cls(s.lower()).value  # type: ignore

    @classmethod
    def parseN(cls, s: str) -> Optional[RandEnum]:
        if s.lower() in ["none", ""]:
            return None
        return cls(s.lower())  # pyright: ignore[reportCallIssue]

    @classmethod
    def from_arg(cls: Type[T], arg: str) -> T:
        return cls(arg)  # pyright: ignore[reportCallIssue]

    @classmethod
    def from_argN(cls: Type[T], arg: str) -> Optional[T]:
        if "none" in str(arg).lower():
            return None
        return cls(arg)  # pyright: ignore[reportCallIssue]

    @classmethod
    def from_args(cls: Type[T], args: Sequence[str]) -> tuple[T, ...]:
        if args is None:
            return tuple()
        if isinstance(args, list) or isinstance(args, tuple):
            if all(isinstance(arg, cls) for arg in args):
                return tuple([cls(arg) for arg in args])  # pyright: ignore[reportCallIssue]

            clean = tuple([s for s in args if "none" not in str(s).lower()])
            if len(clean) == 0:
                return tuple()
            return tuple([cls(s) for s in clean])
        return (cls(args),)  # pyright: ignore[reportCallIssue]

    @no_type_check
    def __lt__(self: T, other: Type[T]) -> bool:
        cls = self.__class__
        if not isinstance(other, cls):
            raise ValueError(f"Can only compare {cls} to other objects of type {cls}")
        return self.name < other.name  # type: ignore

    def __gt__(self, other):
        return not (self < other)


class Scorer:
    def tuning_score(
        self, y_true: Union[Series, ndarray], y_pred: Union[Series, ndarray]
    ) -> float: ...

    def higher_is_better(self) -> bool: ...

    @staticmethod
    def get_scores(
        y_true: Series, y_pred: Series, y_prob: Optional[ndarray]
    ) -> dict[str, float]: ...

    @staticmethod
    def null_scores() -> dict[str, float]: ...

    @staticmethod
    def default() -> Scorer: ...

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return False
        return self is other


class DfAnalyzeClassifier(RandEnum, Enum):
    CatBoost = "catboost"
    XGBoost = "xgb"
    TabPFN = "tabpfn"
    DecisionTree = "dtree"
    ExtraTrees = "et"
    KNN = "knn"
    LGBM = "lgbm"
    RF = "rf"
    LR = "lr"
    SGD = "sgd"
    MLP = "mlp"
    KAN = "kan"
    SVM = "svm"
    GANDALF = "gandalf"
    Dummy = "dummy"

    def get_model(self) -> Type[DfAnalyzeModel]:
        if self is DfAnalyzeClassifier.CatBoost:
            from df_analyze.models.catboost import CatBoostClassifier

            return CatBoostClassifier
        if self is DfAnalyzeClassifier.XGBoost:
            from df_analyze.models.xgboost import XGBoostClassifier

            return XGBoostClassifier
        if self is DfAnalyzeClassifier.TabPFN:
            from df_analyze.models.tabpfn import TabPFNClassifierV3

            return TabPFNClassifierV3
        from df_analyze.models.dummy import DummyClassifier
        from df_analyze.models.gandalf import GandalfEstimator
        from df_analyze.models.kan import KANEstimator
        from df_analyze.models.knn import KNNClassifier
        from df_analyze.models.lgbm import (
            LightGBMClassifier,
            LightGBMRFClassifier,
        )
        from df_analyze.models.linear import LRClassifier, SGDClassifier
        from df_analyze.models.mlp import MLPEstimator
        from df_analyze.models.svm import SVMClassifier
        from df_analyze.models.trees import (
            DecisionTreeClassifier,
            ExtraTreesClassifier,
        )

        return {
            DfAnalyzeClassifier.DecisionTree: DecisionTreeClassifier,
            DfAnalyzeClassifier.ExtraTrees: ExtraTreesClassifier,
            DfAnalyzeClassifier.KNN: KNNClassifier,
            DfAnalyzeClassifier.LGBM: LightGBMClassifier,
            DfAnalyzeClassifier.RF: LightGBMRFClassifier,
            DfAnalyzeClassifier.LR: LRClassifier,
            DfAnalyzeClassifier.SGD: SGDClassifier,
            DfAnalyzeClassifier.MLP: MLPEstimator,
            DfAnalyzeClassifier.KAN: KANEstimator,
            DfAnalyzeClassifier.SVM: SVMClassifier,
            DfAnalyzeClassifier.GANDALF: GandalfEstimator,
            DfAnalyzeClassifier.Dummy: DummyClassifier,
        }[self]

    @staticmethod
    def defaults() -> tuple[str, ...]:
        return (
            DfAnalyzeClassifier.Dummy.value,
            DfAnalyzeClassifier.KNN.value,
            DfAnalyzeClassifier.LGBM.value,
            DfAnalyzeClassifier.SGD.value,
            DfAnalyzeClassifier.LR.value,
        )


class DfAnalyzeRegressor(RandEnum, Enum):
    CatBoost = "catboost"
    XGBoost = "xgb"
    TabPFN = "tabpfn"
    DecisionTree = "dtree"
    ExtraTrees = "et"
    KNN = "knn"
    LGBM = "lgbm"
    RF = "rf"
    ElasticNet = "elastic"
    SGD = "sgd"
    MLP = "mlp"
    KAN = "kan"
    SVM = "svm"
    GANDALF = "gandalf"
    Dummy = "dummy"

    def get_model(self) -> Type[DfAnalyzeModel]:
        if self is DfAnalyzeRegressor.CatBoost:
            from df_analyze.models.catboost import CatBoostRegressor

            return CatBoostRegressor
        if self is DfAnalyzeRegressor.XGBoost:
            from df_analyze.models.xgboost import XGBoostRegressor

            return XGBoostRegressor
        if self is DfAnalyzeRegressor.TabPFN:
            from df_analyze.models.tabpfn import TabPFNRegressorV3

            return TabPFNRegressorV3
        from df_analyze.models.dummy import DummyRegressor
        from df_analyze.models.gandalf import GandalfEstimator
        from df_analyze.models.kan import KANEstimator
        from df_analyze.models.knn import KNNRegressor
        from df_analyze.models.lgbm import (
            LightGBMRegressor,
            LightGBMRFRegressor,
        )
        from df_analyze.models.linear import ElasticNetRegressor, SGDRegressor
        from df_analyze.models.mlp import MLPEstimator
        from df_analyze.models.svm import SVMRegressor
        from df_analyze.models.trees import DecisionTreeRegressor, ExtraTreesRegressor

        return {
            DfAnalyzeRegressor.DecisionTree: DecisionTreeRegressor,
            DfAnalyzeRegressor.ExtraTrees: ExtraTreesRegressor,
            DfAnalyzeRegressor.KNN: KNNRegressor,
            DfAnalyzeRegressor.LGBM: LightGBMRegressor,
            DfAnalyzeRegressor.RF: LightGBMRFRegressor,
            DfAnalyzeRegressor.ElasticNet: ElasticNetRegressor,
            DfAnalyzeRegressor.SGD: SGDRegressor,
            DfAnalyzeRegressor.MLP: MLPEstimator,
            DfAnalyzeRegressor.KAN: KANEstimator,
            DfAnalyzeRegressor.SVM: SVMRegressor,
            DfAnalyzeRegressor.GANDALF: GandalfEstimator,
            DfAnalyzeRegressor.Dummy: DummyRegressor,
        }[self]

    @staticmethod
    def defaults() -> tuple[str, ...]:
        return (
            DfAnalyzeRegressor.Dummy.value,
            DfAnalyzeRegressor.KNN.value,
            DfAnalyzeRegressor.LGBM.value,
            DfAnalyzeRegressor.SGD.value,
            DfAnalyzeRegressor.ElasticNet.value,
        )


@dataclass
class ClassifierScorer(Scorer, RandEnum, Enum):
    Accuracy = "acc"
    AUROC = "auroc"
    Sensitivity = "sens"
    Specificity = "spec"
    PPV = "ppv"
    NPV = "npv"
    F1 = "f1"
    BalancedAccuracy = "bal-acc"

    @staticmethod
    @abstractmethod
    def default() -> ClassifierScorer:
        return ClassifierScorer.Accuracy

    def tuning_score(
        self, y_true: Union[Series, ndarray], y_pred: Union[Series, ndarray]
    ) -> float:
        item = self.value
        if self is ClassifierScorer.AUROC:
            warn(
                "AUROC cannot be used for tuning as it requires probabilities, "
                "which requires classifier calibration. This is too expensive "
                "to perform on each tuning trial. Defaulting to balanced "
                "accuracy instead. "
            )
            item = ClassifierScorer.BalancedAccuracy.value
        if np.asarray(np.isnan(y_true)).any():
            raise ValueError("Impossible! NaNs in y_true.")
        if np.asarray(np.isnan(y_pred)).any():
            raise ValueError("NaNs in y_pred")

        raws = {
            ClassifierScorer.Accuracy.value: accuracy_score,
            ClassifierScorer.Sensitivity.value: sensitivity,
            ClassifierScorer.Specificity.value: specificity,
            ClassifierScorer.PPV.value: ppv,
            ClassifierScorer.NPV.value: npv,
            ClassifierScorer.F1.value: f1_score,
            ClassifierScorer.BalancedAccuracy.value: balanced_accuracy_score,
        }
        scorer = raws[item]
        kwargs = dict(average="macro") if self is ClassifierScorer.F1 else {}
        score = float(scorer(y_true, y_pred, **kwargs))
        if self in (ClassifierScorer.PPV, ClassifierScorer.NPV) and not np.isfinite(
            score
        ):
            return 0.0
        return score

    def higher_is_better(self) -> bool:
        return True

    @staticmethod
    def get_scores(
        y_true: Union[Series, ndarray],
        y_pred: Union[Series, ndarray],
        y_prob: Optional[ndarray],
    ) -> dict[str, float]:
        if y_prob is not None and y_prob.shape[1] == 2:
            y_prob = y_prob[:, 1].reshape(-1, 1)

        if np.asarray(np.isnan(y_true)).any():
            raise ValueError("Impossible! NaNs in y_true.")
        if np.asarray(np.isnan(y_pred)).any():
            raise ValueError("NaNs in y_pred")

        auroc = np.nan if y_prob is None else robust_auroc_score(y_true, y_prob)
        raws = {
            ClassifierScorer.Accuracy.value: accuracy_score(y_true, y_pred),
            ClassifierScorer.AUROC.value: auroc,
            ClassifierScorer.Sensitivity.value: sensitivity(y_true, y_pred),
            ClassifierScorer.Specificity.value: specificity(y_true, y_pred),
            ClassifierScorer.PPV.value: ppv(y_true, y_pred),
            ClassifierScorer.NPV.value: npv(y_true, y_pred),
            ClassifierScorer.F1.value: f1_score(y_true, y_pred, average="macro"),
            ClassifierScorer.BalancedAccuracy.value: balanced_accuracy_score(
                y_true, y_pred
            ),
        }
        return {raw: float(value) for raw, value in raws.items()}

    @staticmethod
    def null_scores() -> dict[str, float]:
        raws = {
            ClassifierScorer.Accuracy.value: np.nan,
            ClassifierScorer.AUROC.value: np.nan,
            ClassifierScorer.Sensitivity.value: np.nan,
            ClassifierScorer.Specificity.value: np.nan,
            ClassifierScorer.PPV.value: np.nan,
            ClassifierScorer.NPV.value: np.nan,
            ClassifierScorer.F1.value: np.nan,
            ClassifierScorer.BalancedAccuracy.value: np.nan,
        }
        return {raw: float(value) for raw, value in raws.items()}


@dataclass
class RegressorScorer(Scorer, RandEnum, Enum):
    MAE = "mae"
    MSqE = "msqe"
    MdAE = "mdae"
    R2 = "r2"
    VarExp = "var-exp"

    @staticmethod
    @abstractmethod
    def default() -> RegressorScorer:
        return RegressorScorer.MAE

    def tuning_score(
        self, y_true: Union[Series, ndarray], y_pred: Union[Series, ndarray]
    ) -> float:
        raws = {
            RegressorScorer.MAE.value: mean_absolute_error,
            RegressorScorer.MSqE.value: mean_squared_error,
            RegressorScorer.MdAE.value: median_absolute_error,
            RegressorScorer.R2.value: r2_score,
            RegressorScorer.VarExp.value: explained_variance_score,
        }
        scorer = raws[self.value]
        return scorer(y_true, y_pred)

    def higher_is_better(self) -> bool:
        return {
            RegressorScorer.MAE.value: False,
            RegressorScorer.MSqE.value: False,
            RegressorScorer.MdAE.value: False,
            RegressorScorer.R2.value: True,
            RegressorScorer.VarExp.value: True,
        }[self.value]

    @staticmethod
    def get_scores(
        y_true: Union[Series, ndarray],
        y_pred: Union[Series, ndarray],
        y_prob: Optional[ndarray] = None,
    ) -> dict[str, float]:
        raws = {
            RegressorScorer.MAE.value: mean_absolute_error(y_true, y_pred),
            RegressorScorer.MSqE.value: mean_squared_error(y_true, y_pred),
            RegressorScorer.MdAE.value: median_absolute_error(y_true, y_pred),
            RegressorScorer.R2.value: r2_score(y_true, y_pred),
            RegressorScorer.VarExp.value: explained_variance_score(y_true, y_pred),
        }
        return {raw: float(value) for raw, value in raws.items()}

    @staticmethod
    def null_scores() -> dict[str, float]:
        raws = {
            RegressorScorer.MAE.value: np.nan,
            RegressorScorer.MSqE.value: np.nan,
            RegressorScorer.MdAE.value: np.nan,
            RegressorScorer.R2.value: np.nan,
            RegressorScorer.VarExp.value: np.nan,
        }
        return {raw: float(value) for raw, value in raws.items()}


class EmbedSelectionModel(RandEnum, Enum):
    LGBM = "lgbm"
    Linear = "linear"


class ValidationMethod(RandEnum, Enum):
    LODO = "lodo"
    List = "list"


class NanHandling(RandEnum, Enum):
    Drop = "drop"
    Mean = "mean"
    Median = "median"
    Impute = "impute"


class Normalization(RandEnum, Enum):
    MinMax = "minmax"
    Robust = "robust"


class EstimationMode(RandEnum, Enum):
    Classify = "classify"
    Regress = "regress"


class FeatureCleaning(RandEnum, Enum):
    Correlated = "correlated"
    LowInfo = "lowinfo"


class FeatureSelection(RandEnum, Enum):
    Filter = "filter"
    Embedded = "embed"
    Wrapper = "wrap"


class FeatureDownsampleMethod(RandEnum, Enum):
    None_ = "none"
    NormalizedVariance = "normalized-variance"
    FTest = "f-test"


class ModelFeatureSelection(RandEnum, Enum):
    Embedded = "embed"
    Wrapper = "wrap"
    NoSelection = "none"


class WrapperSelection(RandEnum, Enum):
    StepUp = "step-up"
    StepDown = "step-down"

    # Genetic = "genetic"
    # ParticleSwarm = "swarm"
    def direction(self) -> Literal["forward", "backward"]:
        return "forward" if self is WrapperSelection.StepUp else "backward"


class WrapperSelectionModel(RandEnum, Enum):
    Linear = "linear"
    LGBM = "lgbm"
    KNN = "knn"


class DimensionReduction(RandEnum, Enum):
    PCA = "pca"
    kPCA = "kpca"


class FilterSelection(RandEnum, Enum):
    Relief = "relief"
    Association = "assoc"
    Prediction = "pred"


class EmbeddedSelection(RandEnum, Enum):
    LightGBM = "lgbm"
    LASSO = "lasso"


class RegScore(RandEnum, Enum):
    MAE = "mae"
    MSqE = "msqe"
    MdAE = "mdae"
    R2 = "r2"
    VarExp = "var-exp"

    @staticmethod
    def default() -> RegScore:
        return RegScore.MAE

    def minimum(self) -> float:
        return {
            RegScore.MAE: 0.0,
            RegScore.MSqE: 0.0,
            RegScore.MdAE: 0.0,
            RegScore.R2: -np.inf,
            RegScore.VarExp: 0.0,
        }[self]

    def higher_is_better(self) -> bool:
        return self in [RegScore.R2, RegScore.VarExp]

    def longname(self) -> str:
        return {
            RegScore.MAE: "Mean Absolute Error",
            RegScore.MSqE: "Mean Squared Error",
            RegScore.MdAE: "Median Absolute Error",
            RegScore.R2: "R-Squared",
            RegScore.VarExp: "Percent Variance Explained",
        }[self]


class ClsScore(RandEnum, Enum):
    Accuracy = "acc"
    AUROC = "auroc"
    Sensitivity = "sens"
    Specificity = "spec"

    @staticmethod
    def default() -> ClsScore:
        return ClsScore.Accuracy

    def higher_is_better(self) -> bool:
        return True

    def minimum(self) -> float:
        return {
            ClsScore.Accuracy: 0.0,
            ClsScore.AUROC: 0.5,
            ClsScore.Sensitivity: 0.0,
            ClsScore.Specificity: 0.0,
        }[self]

    def longname(self) -> str:
        return {
            ClsScore.Accuracy: "Accuracy",
            ClsScore.AUROC: "Area Under the ROC Curve",
            ClsScore.Sensitivity: "Sensitivity",
            ClsScore.Specificity: "Specificity",
        }[self]


class EstimatorKind(RandEnum, Enum):
    Linear = "lin"
    SVM = "svm"
    KNN = "knn"
    RF = "rf"
    LGBM = "lgbm"
    MLP = "mlp"

    def regressor(self) -> Any:
        """Return the regressor form"""

    def classifier(self) -> Any:
        """Return the classifier form"""


class CVSplit(Enum):
    KFold3 = "3-fold"
    KFold5 = "5-fold"
    KFold10 = "10-fold"
    KFold20 = "20-fold"
    Holdout = "holdout"

    def to_string(self, value: float) -> str:
        if self is not CVSplit.Holdout:
            return self.value
        return f"{value * 100}%-holdout"

    @staticmethod
    def from_str(s: str) -> CVSplit:
        try:
            cv = float(s)
        except Exception as e:
            raise ValueError(
                "Could not convert a `... -size` argument (e.g. --htune-val-size) value to float"
            ) from e
        # validate
        if isnan(cv):
            raise ValueError("NaN is not a valid size")
        if cv <= 0:
            raise ValueError(
                "`... -size` arguments (e.g. --htune-val-size) must be positive"
            )
        if cv == 1:
            raise ValueError(
                "'1' is not a valid value for `... -size` arguments (e.g. --htune-val-size)."
            )

        if 0 < cv < 1:
            return CVSplit.Holdout

        if (cv > 1) and not cv.is_integer():
            raise ValueError(
                "`... -size` arguments (e.g. --htune-val-size) greater than 1, as it specifies the `k` in k-fold"
            )

        if cv != round(cv):
            raise ValueError(
                "`--htune-val-size` must be an integer if greater than 1, as it specifies the `k` in k-fold"
            )
        if cv > 10:
            warn(
                "`--htune-val-size` greater than 10 is not recommended.",
                category=UserWarning,
            )
        if cv > 1:
            return CVSplit(cv)

        return CVSplit(s)


class SeedKind(RandEnum, Enum):
    Default = "default"
    Random = "random"

    @classmethod
    def random_seed(cls: Type[SeedKind]) -> Union[str, int]:
        is_seeded = np.random.randint(0, 2, dtype=bool)
        if is_seeded:
            return secrets.randbelow(2**16 - 1)
        return choice([*cls]).value

    @staticmethod
    def default() -> int:
        return SEED
