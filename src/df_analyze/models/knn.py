from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import platform
from typing import Any, Mapping, Optional, Type, Union
from warnings import warn

import numpy as np
import optuna
from optuna import Study, Trial
from pandas import DataFrame, Series
from sklearn.metrics import accuracy_score, r2_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from df_analyze.enumerables import Scorer
from df_analyze.models.base import DfAnalyzeModel
from df_analyze.runtime.hardware import DeviceIntent, RuntimeComponent, RuntimePolicy


def _float_array(X: Any) -> np.ndarray:
    values = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
    values = np.asarray(values, dtype=np.float32)
    return values.reshape(-1, 1) if values.ndim == 1 else values


def _target_array(y: Any) -> np.ndarray:
    values = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
    return values.reshape(-1, 1) if values.ndim == 1 else np.asarray(values)


def _cuda_memory_error(error: BaseException) -> bool:
    try:
        import torch

        if isinstance(error, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    message = str(error).lower()
    return "cuda" in message and (
        "out of memory" in message or "status_alloc_failed" in message
    )


class _TorchKNN:
    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = "uniform",
        metric: str = "l2",
        *,
        p: float = 2,
        algorithm: str = "auto",
        leaf_size: int = 30,
        metric_params: Optional[Mapping[str, Any]] = None,
        device: str = "cuda",
        batch_size: int = 4096,
        runtime: Optional[RuntimePolicy] = None,
        **unsupported: Any,
    ) -> None:
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise TypeError(f"Unsupported CUDA KNN arguments: {names}")
        if metric_params:
            raise TypeError("CUDA KNN does not support metric_params.")
        self.n_neighbors = int(n_neighbors)
        self.weights = str(weights)
        self.metric = str(metric).lower()
        self.p = float(p)
        self.algorithm = str(algorithm)
        self.leaf_size = int(leaf_size)
        self.metric_params = None if metric_params is None else dict(metric_params)
        self.device = device
        self.batch_size = max(1, int(batch_size))
        self.runtime = runtime
        self._X = None
        self._X_norm = None
        self._X_centered_norm = None
        self._X_cpu: Optional[np.ndarray] = None
        self._y_cpu: Optional[np.ndarray] = None
        self._cpu_model = None

    @staticmethod
    def _torch():
        import torch

        return torch

    @staticmethod
    def _normalize(x):
        torch = _TorchKNN._torch()
        norm = torch.linalg.vector_norm(x, dim=1, keepdim=True)
        return x / torch.clamp(norm, min=1e-12)

    @staticmethod
    def _correlation_normalize(x):
        """Preserve scipy/sklearn's NaN correlation for constant rows."""
        torch = _TorchKNN._torch()
        norm = torch.linalg.vector_norm(x, dim=1, keepdim=True)
        return x / norm

    def _fit_features(self, X: Any) -> None:
        self._X_cpu = _float_array(X)
        if self._X_cpu.shape[0] == 0:
            raise ValueError("Found array with 0 sample(s); at least 1 is required.")
        self.n_features_in_ = int(self._X_cpu.shape[1])
        self.n_samples_fit_ = int(self._X_cpu.shape[0])
        if self._y_cpu is None or self._y_cpu.shape[0] != self.n_samples_fit_:
            raise ValueError(
                "Found input variables with inconsistent numbers of samples."
            )
        try:
            torch = self._torch()
            self._X = torch.as_tensor(
                self._X_cpu, dtype=torch.float32, device=self.device
            )
        except Exception as error:
            if not _cuda_memory_error(error):
                raise
            self._activate_cpu("training data did not fit in GPU memory")

    def _new_cpu_model(self):
        raise NotImplementedError

    def _activate_cpu(self, reason: str) -> None:
        if self._cpu_model is not None:
            return
        if self._X_cpu is None or self._y_cpu is None:
            raise RuntimeError("KNN training data is unavailable for CPU fallback.")
        if self.runtime is not None and self.runtime.intent in {
            DeviceIntent.Auto,
            DeviceIntent.CUDA,
        }:
            raise RuntimeError(
                f"CUDA KNN {reason}. The complete model task must be retried on CPU."
            )
        warn(f"CUDA KNN {reason}; continuing with sklearn KNN on CPU.")
        if self.runtime is not None:
            self.runtime.record_cpu_fallback(
                RuntimeComponent.KNN, "cuda_out_of_memory"
            )
        self._cpu_model = self._new_cpu_model()
        target = self._y_cpu.ravel() if self._y_cpu.shape[1] == 1 else self._y_cpu
        self._cpu_model.fit(self._X_cpu, target)
        self._X = None
        self._X_norm = None
        self._X_centered_norm = None
        try:
            self._torch().cuda.empty_cache()
        except Exception:
            pass

    def _train_for_metric(self):
        if self._X is None:
            raise RuntimeError("Need to fit KNN before prediction.")
        if self.metric == "cosine":
            if self._X_norm is None:
                self._X_norm = self._normalize(self._X)
            return self._X_norm
        if self.metric == "correlation":
            if self._X_centered_norm is None:
                centered = self._X - self._X.mean(dim=1, keepdim=True)
                self._X_centered_norm = self._correlation_normalize(centered)
            return self._X_centered_norm
        return self._X

    def _distances(self, query):
        torch = self._torch()
        train = self._train_for_metric()
        if self.metric in {"l2", "euclidean"}:
            return torch.cdist(query, train, p=2)
        if self.metric == "minkowski":
            return torch.cdist(query, train, p=self.p)
        if self.metric in {"l1", "manhattan", "cityblock"}:
            return torch.cdist(query, train, p=1)
        if self.metric == "cosine":
            return 1.0 - self._normalize(query) @ train.T
        if self.metric == "correlation":
            centered = query - query.mean(dim=1, keepdim=True)
            return 1.0 - self._correlation_normalize(centered) @ train.T
        raise ValueError(f"Unsupported CUDA KNN metric: {self.metric}")

    def _validated_neighbor_count(self) -> int:
        if self._X is None:
            raise RuntimeError("Need to fit KNN before prediction.")
        if self.n_neighbors <= 0:
            raise ValueError(f"Expected n_neighbors > 0. Got {self.n_neighbors}.")
        n_samples_fit = int(self._X.shape[0])
        if self.n_neighbors > n_samples_fit:
            raise ValueError(
                "Expected n_neighbors <= n_samples_fit, but "
                f"n_neighbors = {self.n_neighbors}, "
                f"n_samples_fit = {n_samples_fit}."
            )
        return self.n_neighbors

    def _safe_batch_size(self) -> int:
        if self._X is None:
            return self.batch_size
        try:
            free_bytes, _ = self._torch().cuda.mem_get_info()
        except Exception:
            return self.batch_size
        bytes_per_query = max(1, int(self._X.shape[0]) * 16)
        return max(1, min(self.batch_size, int(free_bytes * 0.25 // bytes_per_query)))

    def _kneighbors_torch(self, X: Any):
        if self._X is None:
            raise RuntimeError("Need to fit KNN before prediction.")
        torch = self._torch()
        query = _float_array(X)
        if query.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {query.shape[1]} features, but this KNN is expecting "
                f"{self.n_features_in_} features as input."
            )
        k = self._validated_neighbor_count()
        distances = []
        indices = []
        batch_size = self._safe_batch_size()
        with torch.no_grad():
            start = 0
            while start < int(query.shape[0]):
                current_batch_size = min(batch_size, int(query.shape[0]) - start)
                while True:
                    try:
                        current = torch.as_tensor(
                            query[start : start + current_batch_size],
                            dtype=torch.float32,
                            device=self.device,
                        )
                        dist = self._distances(current)
                        values, index = torch.topk(
                            dist, k=k, dim=1, largest=False, sorted=True
                        )
                        distances.append(values)
                        indices.append(index)
                        del dist
                        start += current_batch_size
                        break
                    except Exception as error:
                        if not _cuda_memory_error(error) or current_batch_size <= 1:
                            raise
                        current_batch_size = max(1, current_batch_size // 2)
                        batch_size = min(batch_size, current_batch_size)
                        self.batch_size = min(self.batch_size, current_batch_size)
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
        return torch.cat(distances), torch.cat(indices)

    def kneighbors(
        self, X: Any, n_neighbors: Optional[int] = None, return_distance: bool = True
    ):
        if self._cpu_model is not None:
            return self._cpu_model.kneighbors(X, n_neighbors, return_distance)
        old_neighbors = self.n_neighbors
        if n_neighbors is not None:
            self.n_neighbors = int(n_neighbors)
        try:
            try:
                distances, indices = self._kneighbors_torch(X)
            except Exception as error:
                if not _cuda_memory_error(error):
                    raise
                self._activate_cpu("prediction exceeded available GPU memory")
                return self._cpu_model.kneighbors(X, n_neighbors, return_distance)
        finally:
            self.n_neighbors = old_neighbors
        index_array = indices.detach().cpu().numpy()
        if not return_distance:
            return index_array
        return distances.detach().cpu().numpy(), index_array

    def _weights(self, distances):
        torch = self._torch()
        if self.weights == "uniform":
            return torch.ones_like(distances)
        if self.weights != "distance":
            raise ValueError(f"Unsupported KNN weights: {self.weights}")
        zero = distances <= 1e-12
        inverse = 1.0 / torch.clamp(distances, min=1e-12)
        return torch.where(
            zero.any(dim=1, keepdim=True), zero.to(distances.dtype), inverse
        )


class TorchKNNClassifier(_TorchKNN):
    def fit(self, X: Any, y: Any) -> TorchKNNClassifier:
        self._y_cpu = _target_array(y)
        self._single_output = self._y_cpu.shape[1] == 1
        self._classes = []
        codes = []
        for column in range(self._y_cpu.shape[1]):
            classes, encoded = np.unique(self._y_cpu[:, column], return_inverse=True)
            self._classes.append(classes)
            codes.append(encoded)
        self.classes_ = self._classes[0] if self._single_output else self._classes
        self._fit_features(X)
        if self._cpu_model is None:
            try:
                self._codes = self._torch().as_tensor(
                    np.column_stack(codes), dtype=self._torch().long, device=self.device
                )
            except Exception as error:
                if not _cuda_memory_error(error):
                    raise
                self._activate_cpu("targets did not fit in GPU memory")
        return self

    def _new_cpu_model(self):
        return KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            metric=self.metric,
            p=self.p,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric_params=self.metric_params,
        )

    def _vote_arrays(self, X: Any) -> list[np.ndarray]:
        distances, indices = self._kneighbors_torch(X)
        weights = self._weights(distances)
        votes_out = []
        torch = self._torch()
        with torch.no_grad():
            for target, classes in enumerate(self._classes):
                codes = self._codes[:, target][indices]
                votes = torch.zeros(
                    (indices.shape[0], len(classes)),
                    dtype=torch.float32,
                    device=self.device,
                )
                votes.scatter_add_(1, codes, weights)
                votes_out.append(votes.detach().cpu().numpy())
        return votes_out

    def _votes(self, X: Any) -> list[np.ndarray]:
        try:
            return self._vote_arrays(X)
        except Exception as error:
            if not _cuda_memory_error(error):
                raise
            self._activate_cpu("prediction exceeded available GPU memory")
            probabilities = self._cpu_model.predict_proba(X)
            return [probabilities] if isinstance(probabilities, np.ndarray) else probabilities

    def predict(self, X: Any) -> np.ndarray:
        if self._cpu_model is not None:
            return np.asarray(self._cpu_model.predict(X))
        columns = [classes[np.argmax(votes, axis=1)] for votes, classes in zip(self._votes(X), self._classes)]
        output = np.column_stack(columns)
        return output.ravel() if self._single_output else output

    def predict_proba(self, X: Any):
        if self._cpu_model is not None:
            return self._cpu_model.predict_proba(X)
        probabilities = []
        for votes in self._votes(X):
            total = votes.sum(axis=1, keepdims=True)
            probabilities.append(
                np.divide(votes, total, out=np.zeros_like(votes), where=total > 0)
            )
        return probabilities[0] if self._single_output else probabilities

    def score(self, X: Any, y: Any) -> float:
        return float(accuracy_score(np.asarray(y), self.predict(X)))


class TorchKNNRegressor(_TorchKNN):
    def fit(self, X: Any, y: Any) -> TorchKNNRegressor:
        self._y_cpu = _target_array(y).astype(np.float32)
        self._single_output = self._y_cpu.shape[1] == 1
        self._fit_features(X)
        if self._cpu_model is None:
            try:
                self._y = self._torch().as_tensor(
                    self._y_cpu, dtype=self._torch().float32, device=self.device
                )
            except Exception as error:
                if not _cuda_memory_error(error):
                    raise
                self._activate_cpu("targets did not fit in GPU memory")
        return self

    def _new_cpu_model(self):
        return KNeighborsRegressor(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            metric=self.metric,
            p=self.p,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric_params=self.metric_params,
        )

    def predict(self, X: Any) -> np.ndarray:
        if self._cpu_model is not None:
            return np.asarray(self._cpu_model.predict(X))
        try:
            distances, indices = self._kneighbors_torch(X)
            weights = self._weights(distances)
            values = self._y[indices]
            if self.weights == "uniform":
                prediction = values.mean(dim=1)
            else:
                total = weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
                prediction = (values * weights.unsqueeze(-1)).sum(dim=1) / total
            output = prediction.detach().cpu().numpy()
            return output.ravel() if self._single_output else output
        except Exception as error:
            if not _cuda_memory_error(error):
                raise
            self._activate_cpu("prediction exceeded available GPU memory")
            return np.asarray(self._cpu_model.predict(X))

    def score(self, X: Any, y: Any) -> float:
        return float(r2_score(y, self.predict(X)))


class KNNEstimator(DfAnalyzeModel):
    runtime_component = RuntimeComponent.KNN
    shortname = "knn"
    longname = "K-Neighbours Estimator"
    timeout_s = 30 * 60  # 30 minutes is enough given the grid

    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        n_jobs = 1 if platform.system().lower() == "darwin" else -1
        self.is_classifier = False
        self.needs_calibration = False
        self.target_cols: list[str] = []
        self.fixed_args = dict(n_jobs=n_jobs)
        self.model_cls: Type[Any] = type(None)
        self.grid = {
            "n_neighbors": [1, 5, 10, 25, 50],
            "weights": ["uniform", "distance"],
            # Keep the original public grid unchanged. The CUDA implementation
            # can still accept L1 when it is explicitly supplied.
            "metric": ["cosine", "l2", "correlation"],
        }

    def model_cls_args(self, full_args: dict[str, Any]) -> tuple[type, dict[str, Any]]:
        if self.runtime.device_for(RuntimeComponent.KNN) == "cuda":
            args = dict(full_args)
            args.pop("n_jobs", None)
            args["device"] = "cuda"
            args["runtime"] = self.runtime
            model_cls = TorchKNNClassifier if self.is_classifier else TorchKNNRegressor
            return model_cls, args
        return self.model_cls, full_args

    def _cleanup_after_fold(self) -> None:
        # Fold-local tensors are released when the estimator goes out of scope.
        # Emptying PyTorch's allocator and forcing a full Python GC after every
        # KNN fold defeats allocator reuse and is substantially slower than the
        # distance calculation on small and medium datasets.
        return

    def optuna_args(self, trial: Trial) -> dict[str, str | float | int]:
        return dict(
            n_neighbors=trial.suggest_int("n_neighbors", 1, 50, step=1),
            weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
            metric=trial.suggest_categorical(
                "metric", ["cosine", "l1", "l2", "correlation"]
            ),
        )

    def htune_optuna(
        self,
        X_train: DataFrame,
        y_train: Union[Series, DataFrame],
        g_train: Optional[Series],
        metric: Scorer,
        n_trials: int = 100,
        n_jobs: int = -1,
        verbosity: int = optuna.logging.ERROR,
    ) -> Study:
        if self.grid is None:
            raise ValueError("Impossible!")
        n_hp = 1
        for opts in self.grid.values():
            n_hp *= len(opts)
        cpu_jobs = -1 if self.fixed_args["n_jobs"] == 1 else 1
        n_jobs = self.runtime.tuning_jobs(RuntimeComponent.KNN, cpu_jobs)
        return super().htune_optuna(
            X_train,
            y_train,
            g_train,
            metric=metric,
            n_trials=n_hp,
            n_jobs=n_jobs,
            verbosity=verbosity,
        )

    def _target_cols_for_output(self, n_targets: int) -> list[str]:
        if len(self.target_cols) == n_targets:
            return self.target_cols
        return [f"target_{i}" for i in range(n_targets)]

    def refit_tuned(
        self,
        X: DataFrame,
        y: Union[Series, DataFrame],
        g: Optional[Series] = None,
        tuned_args: Optional[Mapping] = None,
    ) -> None:
        if isinstance(y, DataFrame):
            self.target_cols = [str(col) for col in y.columns]
        else:
            name = y.name if y.name is not None else "target"
            self.target_cols = [str(name)]
        super().refit_tuned(X=X, y=y, g=g, tuned_args=tuned_args)

    def tuned_predict(self, X: DataFrame) -> Union[Series, DataFrame, np.ndarray]:
        preds = super().tuned_predict(X)
        if (
            isinstance(preds, np.ndarray)
            and preds.ndim == 2
            and preds.shape[1] > 1
        ):
            cols = self._target_cols_for_output(preds.shape[1])
            return DataFrame(preds, index=X.index, columns=cols)
        return preds


class KNNClassifier(KNNEstimator):
    shortname = "knn"
    longname = "K-Neighbours Classifier"
    timeout_s = 30 * 60

    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.is_classifier = True
        self.model_cls = KNeighborsClassifier

    def predict_proba(
        self, X: DataFrame
    ) -> Union[np.ndarray, dict[str, np.ndarray]]:
        probs = super().predict_proba(X)
        if isinstance(probs, (list, tuple)):
            if len(probs) == 1:
                return np.asarray(probs[0])
            cols = self._target_cols_for_output(len(probs))
            return {col: np.asarray(arr) for col, arr in zip(cols, probs)}
        if (
            isinstance(probs, np.ndarray)
            and probs.ndim == 3
            and probs.shape[1] > 1
        ):
            cols = self._target_cols_for_output(probs.shape[1])
            return {col: np.asarray(probs[:, i, :]) for i, col in enumerate(cols)}
        return probs


class KNNRegressor(KNNEstimator):
    shortname = "knn"
    longname = "K-Neighbours Regressor"
    timeout_s = 30 * 60

    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.is_classifier = False
        self.model_cls = KNeighborsRegressor
        self.shortname = "knn"
        self.longname = "K-Neighbours Regressor"
