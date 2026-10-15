"""Causal rolling classifiers adapted from the original MIT-licensed V3 research.

Original code copyright (c) 2026 Ariel Tolome; notice retained in
research/ORIGINAL_CODE_LICENSE.txt. Modifications for TabPFN Options Lab:
V3.5-only loading, UTC provenance, explicit device selection, and an
empirical-rate (not trading) baseline. No checkpoint is distributed; only
an existing, separately authorized local checkpoint may be loaded.
"""
from bisect import insort_right
from datetime import datetime, timezone
from importlib.metadata import version
from math import isclose, isfinite
import os
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo


FEATURES = ("iv_rank_30d", "vrp_spread", "vix_slope", "underlying_rsi",
            "net_credit_yield", "short_delta")
_NEW_YORK = ZoneInfo("America/New_York")


def _number(value):
    if isinstance(value, (str, bytes, bool)):
        raise ValueError("Expected a finite number")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Expected a finite number") from exc
    if not isfinite(value):
        raise ValueError("Expected a finite number")
    return value


def _features(values):
    values = tuple(_number(value) for value in values)
    if len(values) != len(FEATURES):
        raise ValueError("Expected six features in FEATURES order")
    rank, _, slope, rsi, credit_yield, delta = values
    if not (0 <= rank <= 100 and slope > 0 and 0 <= rsi <= 100
            and 0 < credit_yield < 1 and -1 <= delta < 0):
        raise ValueError("Features outside their economic bounds")
    return values


def _probability(value):
    value = _number(value)
    if not 0 <= value <= 1:
        raise ValueError("Probability must be in [0, 1]")
    return value


def probability_quantity(p):
    """Illustrative 0/1/2-contract tiers, not orders or fractional Kelly sizing."""
    p = _probability(p)
    return 2 if p >= 0.90 else 1 if p >= 0.82 else 0


def _timestamp(value):
    """Compare UTC instants; reject ambiguous/nonexistent naive New York times."""
    if isinstance(value, str):
        if len(value) < 16 or value[10:11] not in ("T", " "):
            raise ValueError("A timestamp must include its date and time")
        value = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    if not isinstance(value, datetime):
        raise ValueError("Expected a datetime or ISO timestamp")
    if value.tzinfo is None:
        local = value.replace(tzinfo=_NEW_YORK)
        if (local.utcoffset() != local.replace(fold=1).utcoffset()
                or local.astimezone(timezone.utc).astimezone(_NEW_YORK).replace(tzinfo=None) != value):
            raise ValueError("Ambiguous or nonexistent NY timestamp; provide a UTC offset")
        value = local
    return value.astimezone(timezone.utc)


def _utc_iso(value):
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _chronological_split(rows):
    """Last 20% by entry time validate; training labels must predate that cohort."""
    boundary = len(rows) - (len(rows) + 4) // 5
    validation = rows[boundary:]
    if not validation:
        return [], []
    first_entry = validation[0]["entry_time"]
    training = [row for row in rows[:boundary] if row["label_time"] < first_entry]
    return training, validation


def _class1_probabilities(classes, probabilities):
    """Look up label 1 explicitly; never assume predict_proba column order."""
    classes = tuple(classes)
    if len(classes) != 2 or set(classes) != {0, 1}:
        raise ValueError("Classifier must expose exactly classes 0 and 1")
    column = classes.index(1)
    result = []
    for row in probabilities:
        row = tuple(_probability(value) for value in row)
        if len(row) != 2 or not isclose(sum(row), 1, rel_tol=0, abs_tol=1e-6):
            raise ValueError("Invalid binary probability distribution")
        result.append(row[column])
    return result


class RollingTradeClassifier:
    """Resolve labels before selecting an entry-chronological rolling window.

    Stored rows are copied and features frozen. A fitted model is reused only
    while eligible training rows are identical. Failures are never substituted
    with another model or a fabricated prediction.
    """

    def __init__(self, mode, rows=(), min_samples=500, max_samples=5000, device="cpu"):
        if mode not in ("empirical", "xgboost", "tabpfn"):
            raise ValueError("Mode must be empirical, xgboost, or tabpfn")
        if (type(min_samples) is not int or type(max_samples) is not int
                or not 2 <= min_samples <= max_samples <= 5000):
            raise ValueError("Require 2 <= min_samples <= max_samples <= 5000")
        if device not in ("cpu", "cuda", "auto"):
            raise ValueError("Device must be cpu, cuda, or auto")
        self.mode = mode
        self.device = device
        self.min_samples = min_samples
        self.max_samples = max_samples
        self._rows = []
        self._fitted_rows = None
        self._model = None
        self._model_metadata = {}
        self.metadata = {}
        for row in rows:
            self.add_trade(row)

    def add_trade(self, row):
        if not isinstance(row, dict) or not {"x", "y", "entry_time", "label_time"} <= row.keys():
            raise ValueError("Trade row needs x, y, entry_time, and label_time")
        if type(row["y"]) is not int or row["y"] not in (0, 1):
            raise ValueError("Only uncensored integer labels 0 and 1 may train a model")
        entry, label = _timestamp(row["entry_time"]), _timestamp(row["label_time"])
        if entry >= label:
            raise ValueError("Trade entry must strictly precede its realized label")
        stored = {"x": _features(row["x"]), "y": row["y"],
                  "entry_time": entry, "label_time": label}
        insort_right(self._rows, stored, key=lambda item: (item["entry_time"], item["label_time"]))

    def _training_rows(self, as_of):
        cutoff = _timestamp(as_of)
        return [row for row in self._rows if row["label_time"] < cutoff][-self.max_samples:]

    def predict(self, features, as_of):
        started = perf_counter()
        features = _features(features)
        rows = self._training_rows(as_of)
        self.metadata = {
            "as_of": _utc_iso(_timestamp(as_of)),
            "n_train": len(rows),
            "latest_label_time": _utc_iso(max(row["label_time"] for row in rows)) if rows else None,
            "first_training_entry_time": _utc_iso(rows[0]["entry_time"]) if rows else None,
            "model": {"empirical": "historical_target_success_rate",
                      "xgboost": "XGBClassifier", "tabpfn": "TabPFNClassifier"}[self.mode],
            "model_version": "v3.5" if self.mode == "tabpfn" else None,
            "version": None, "hyperparameters": {}, "inference_seconds": 0.0,
            "fit_seconds": 0.0, "total_seconds": 0.0, "refit": False,
            "status": "insufficient_history",
        }
        if len(rows) < self.min_samples or {row["y"] for row in rows} != {0, 1}:
            self.metadata["total_seconds"] = perf_counter() - started
            return None
        if self.mode == "empirical":
            inference_started = perf_counter()
            p = sum(row["y"] for row in rows) / len(rows)
            self.metadata.update(version="stdlib", status="ready",
                                 inference_seconds=perf_counter() - inference_started)
        else:
            if self.mode == "xgboost":
                training, validation = _chronological_split(rows)
                self.metadata["tuning"] = {
                    "n_train": len(training), "n_validation": len(validation),
                    "n_purged": len(rows) - len(training) - len(validation),
                    "first_validation_entry_time": _utc_iso(validation[0]["entry_time"]),
                    "latest_training_label_time": _utc_iso(max(r["label_time"] for r in training)) if training else None,
                }
                if {row["y"] for row in training} != {0, 1}:
                    self.metadata.update(status="insufficient_purged_training",
                                         total_seconds=perf_counter() - started)
                    return None
            if rows != self._fitted_rows:
                fit_started = perf_counter()
                model, details = (self._fit_tabpfn(rows) if self.mode == "tabpfn"
                                  else self._fit_xgboost(rows, training, validation))
                self._model, self._model_metadata, self._fitted_rows = model, details, rows
                self.metadata.update(fit_seconds=perf_counter() - fit_started, refit=True)
            self.metadata.update(self._model_metadata)
            import numpy as np
            inference_started = perf_counter()
            probabilities = self._model.predict_proba(np.asarray([features], dtype=float))
            values = _class1_probabilities(self._model.classes_, probabilities)
            if len(values) != 1:
                raise ValueError("Expected one prediction for one feature vector")
            p = values[0]
            self.metadata.update(inference_seconds=perf_counter() - inference_started, status="ready")
        self.metadata["total_seconds"] = perf_counter() - started
        return _probability(p)

    def _fit_tabpfn(self, rows):
        os.environ["TABPFN_NO_BROWSER"] = "1"
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion
        import numpy as np

        parameters = {"device": self.device, "random_state": 42, "n_preprocessing_jobs": 1}
        checkpoint = os.environ.get("TABPFN_MODEL_PATH")
        if checkpoint:
            parameters["model_path"] = str(Path(checkpoint).expanduser())
        model = TabPFNClassifier.create_default_for_version(ModelVersion.V3_5, **parameters)
        checkpoint_path = Path(model.model_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                "An authorized TabPFN-3.5 checkpoint is required before running. "
                "Obtain it under PriorLabs' terms and cache it via the package, or set "
                "TABPFN_MODEL_PATH to its local file. This program does not accept "
                "licenses or download weights."
            )
        model.fit(np.asarray([r["x"] for r in rows], dtype=float), np.asarray([r["y"] for r in rows]))
        architectures = [config.name for config in model.configs_]
        if not architectures or any(name != "TabPFN-v3.5" for name in architectures):
            raise ValueError("Requested TabPFN-3.5 but loaded a different architecture")
        parameters.update(checkpoint_filename=checkpoint_path.name, n_estimators=model.n_estimators,
                          resolved_n_estimators=len(model.ensemble_configs_), fit_mode=model.fit_mode)
        parameters.pop("model_path", None)
        return model, {"version": version("tabpfn"), "hyperparameters": parameters,
                       "loaded_architectures": architectures,
                       "resolved_devices": [str(device) for device in model.devices_]}

    def _fit_xgboost(self, rows, training, validation):
        from xgboost import XGBClassifier
        import numpy as np

        x_train = np.asarray([r["x"] for r in training], dtype=float)
        y_train = np.asarray([r["y"] for r in training])
        x_validation = np.asarray([r["x"] for r in validation], dtype=float)
        y_validation = [r["y"] for r in validation]
        fixed = {"n_estimators": 100, "random_state": 42, "n_jobs": 1,
                 "objective": "binary:logistic", "eval_metric": "logloss",
                 "tree_method": "hist", "device": "cpu"}
        best_score, best_parameters = float("inf"), None
        for depth in (2, 3):
            for rate in (0.03, 0.1):
                parameters = dict(fixed, max_depth=depth, learning_rate=rate)
                candidate = XGBClassifier(**parameters)
                candidate.fit(x_train, y_train)
                probabilities = _class1_probabilities(candidate.classes_, candidate.predict_proba(x_validation))
                if len(probabilities) != len(y_validation):
                    raise ValueError("Validation prediction count does not match its labels")
                score = sum((p - y) ** 2 for p, y in zip(probabilities, y_validation)) / len(y_validation)
                if score < best_score:
                    best_score, best_parameters = score, parameters
        model = XGBClassifier(**best_parameters)
        model.fit(np.asarray([r["x"] for r in rows], dtype=float), np.asarray([r["y"] for r in rows]))
        return model, {"version": version("xgboost"), "hyperparameters": best_parameters,
                       "validation_brier": best_score}
