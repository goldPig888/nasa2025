"""Lightweight modeling utilities for city air-quality time series.

The goal is to bridge the historical datasets we assemble (OpenAQ + MERRA-2)
with simple, explainable models that can produce near-term forecasts and
categorical risk summaries. We keep the implementation dependency-free beyond
NumPy/Pandas so it works inside the existing Streamlit environment.
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .data_cache import (
    ModelCacheManifest,
    dataframe_signature,
    latest_file,
    safe_city_name,
)

POLLUTANT_COLUMNS = ["pm25", "pm10", "o3", "no2", "so2", "co"]

# AQI breakpoints (EPA style) for translating concentrations to categories.
# Values are (conc_low, conc_high, aqi_low, aqi_high).
AQI_BREAKPOINTS = {
    "pm25": [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 500.4, 301, 500),
    ],
    "pm10": [
        (0, 54, 0, 50),
        (55, 154, 51, 100),
        (155, 254, 101, 150),
        (255, 354, 151, 200),
        (355, 424, 201, 300),
        (425, 604, 301, 500),
    ],
    "o3": [
        (0.000, 0.054, 0, 50),
        (0.055, 0.070, 51, 100),
        (0.071, 0.085, 101, 150),
        (0.086, 0.105, 151, 200),
        (0.106, 0.200, 201, 300),
    ],
    "no2": [
        (0.0, 53.0, 0, 50),
        (54.0, 100.0, 51, 100),
        (101.0, 360.0, 101, 150),
        (361.0, 649.0, 151, 200),
        (650.0, 1249.0, 201, 300),
        (1250.0, 2049.0, 301, 500),
    ],
    "so2": [
        (0.0, 35.0, 0, 50),
        (36.0, 75.0, 51, 100),
        (76.0, 185.0, 101, 150),
        (186.0, 304.0, 151, 200),
        (305.0, 604.0, 201, 300),
        (605.0, 1004.0, 301, 500),
    ],
    "co": [
        (0.0, 4.4, 0, 50),
        (4.5, 9.4, 51, 100),
        (9.5, 12.4, 101, 150),
        (12.5, 15.4, 151, 200),
        (15.5, 30.4, 201, 300),
        (30.5, 40.4, 301, 500),
    ],
}

AQI_LABELS = [
    (0, 50, "Good"),
    (51, 100, "Moderate"),
    (101, 150, "Unhealthy for Sensitive Groups"),
    (151, 200, "Unhealthy"),
    (201, 300, "Very Unhealthy"),
    (301, 500, "Hazardous"),
]


def compute_aqi(pollutant: str, value: Optional[float]) -> Optional[float]:
    """Return AQI index for a pollutant concentration using linear interpolation."""

    if value is None or np.isnan(value):
        return None
    if pollutant not in AQI_BREAKPOINTS:
        return None

    for conc_low, conc_high, aqi_low, aqi_high in AQI_BREAKPOINTS[pollutant]:
        if conc_low <= value <= conc_high:
            return ((aqi_high - aqi_low) / (conc_high - conc_low)) * (value - conc_low) + aqi_low
    return None


def aqi_category(aqi_value: Optional[float]) -> Optional[str]:
    if aqi_value is None:
        return None
    for low, high, label in AQI_LABELS:
        if low <= aqi_value <= high:
            return label
    return "Hazardous" if aqi_value > 500 else None


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out[out["date"].notna()].sort_values("date")
    return out.reset_index(drop=True)


def aggregate_city_timeseries(df: pd.DataFrame, how: str = "median") -> pd.DataFrame:
    """Collapse station-level rows to a single daily observation per city."""

    if df.empty:
        return df.copy()
    df = ensure_datetime_index(df)
    numeric_cols = [c for c in df.columns if c not in {"date", "station", "data_source"}]
    grouped = df.groupby("date")[numeric_cols].agg(how)
    return grouped.reset_index()


def _generate_features(
    df: pd.DataFrame,
    pollutant: str,
    lags: Sequence[int],
    windows: Sequence[int],
    include_other: bool,
) -> pd.DataFrame:
    working = ensure_datetime_index(df)
    if pollutant not in working.columns:
        raise ValueError(f"'{pollutant}' not found in dataframe columns")

    working = working.copy()
    target_series = pd.to_numeric(working[pollutant], errors="coerce")

    for lag in lags:
        working[f"{pollutant}_lag_{lag}"] = target_series.shift(lag)

    for window in windows:
        shifted = target_series.shift(1)
        working[f"{pollutant}_rollmean_{window}"] = shifted.rolling(window).mean()
        working[f"{pollutant}_rollstd_{window}"] = shifted.rolling(window).std()

    if include_other:
        for col in POLLUTANT_COLUMNS:
            if col == pollutant or col not in working.columns:
                continue
            working[f"{col}_lag_1"] = pd.to_numeric(working[col], errors="coerce").shift(1)

    dow = working["date"].dt.dayofweek
    doy = working["date"].dt.dayofyear
    month = working["date"].dt.month

    working["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    working["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    working["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    working["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    working["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    working["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    working["time_trend"] = (working["date"] - working["date"].min()).dt.days

    return working


def _prepare_training_matrix(
    df: pd.DataFrame,
    pollutant: str,
    horizon: int,
    lags: Sequence[int],
    windows: Sequence[int],
    include_other: bool,
    top_k: Optional[int],
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    features_df = _generate_features(df, pollutant, lags, windows, include_other)

    candidate_cols = [
        c
        for c in features_df.columns
        if c not in {"date", "station", "data_source", pollutant}
        and c not in POLLUTANT_COLUMNS
    ]
    if not candidate_cols:
        raise ValueError("No candidate features available for AQI classification")

    target = pd.to_numeric(features_df[pollutant], errors="coerce").shift(-horizon)
    features_df["target__"] = target

    prepared = features_df.dropna(subset=candidate_cols + ["target__"])
    if prepared.empty:
        raise ValueError("Not enough rows after feature generation to train model")

    X = prepared[candidate_cols]
    y = prepared["target__"]

    if top_k is not None and top_k < len(candidate_cols):
        scored = []
        y_values = y.values
        for col in candidate_cols:
            x = X[col].values
            finite = np.isfinite(x)
            if finite.sum() < 2:
                continue
            x = x[finite]
            y_subset = y_values[finite]
            if np.allclose(x, x[0]):
                continue
            corr = np.corrcoef(x, y_subset)[0, 1]
            if np.isnan(corr):
                continue
            scored.append((abs(corr), col))
        scored.sort(reverse=True)
        selected = [col for _, col in scored[:top_k]]
        if not selected:
            selected = candidate_cols[: top_k or len(candidate_cols)]
        X = X[selected]
    else:
        selected = candidate_cols

    return X, y, selected


@dataclass
class RegressionModel:
    pollutant: str
    horizon: int
    feature_names: List[str]
    mean_: np.ndarray
    scale_: np.ndarray
    coef_: np.ndarray
    intercept_: float
    alpha: float
    lags: Tuple[int, ...]
    windows: Tuple[int, ...]
    include_other: bool

    def _standardize(self, features: pd.DataFrame) -> np.ndarray:
        X = features[self.feature_names].values.astype(float)
        X = (X - self.mean_) / self.scale_
        return X

    def predict_from_features(self, features: pd.DataFrame) -> np.ndarray:
        X = self._standardize(features)
        return self.intercept_ + X @ self.coef_

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        generated = _generate_features(df, self.pollutant, self.lags, self.windows, self.include_other)
        return generated[self.feature_names]

    def predict_one(self, df: pd.DataFrame) -> Optional[float]:
        generated = _generate_features(df, self.pollutant, self.lags, self.windows, self.include_other)
        row = generated.dropna(subset=self.feature_names).tail(1)
        if row.empty:
            return None
        return float(self.predict_from_features(row)[0])


@dataclass
class PersistenceModel:
    """Simple rolling-mean forecaster used when regression cannot be trained."""

    pollutant: str
    horizon: int
    window: int = 3

    def predict_one(self, df: pd.DataFrame) -> Optional[float]:
        if self.pollutant not in df.columns:
            return None
        prepared = ensure_datetime_index(df)
        series = pd.to_numeric(prepared[self.pollutant], errors="coerce").dropna()
        if series.empty:
            return None
        span = max(1, min(self.window, len(series)))
        return float(series.tail(span).mean())


@dataclass
class ModelReport:
    pollutant: str
    horizon: int
    samples: int
    rmse: float
    mae: float
    r2: float
    feature_weights: Dict[str, float]


def _ridge_fit(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale == 0] = 1.0
    Xs = (X - mean) / scale
    ones = np.ones((Xs.shape[0], 1))
    Xa = np.hstack([ones, Xs])

    XtX = Xa.T @ Xa
    regularizer = np.eye(XtX.shape[0]) * alpha
    regularizer[0, 0] = 0.0
    XtX_reg = XtX + regularizer
    Xty = Xa.T @ y

    try:
        beta = np.linalg.solve(XtX_reg, Xty)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(XtX_reg) @ Xty

    intercept = beta[0]
    coef = beta[1:]
    return (mean, scale, intercept, coef)


def _finalize_regression_model(
    pollutant: str,
    horizon: int,
    X_df: pd.DataFrame,
    y_series: pd.Series,
    feature_names: List[str],
    *,
    alpha: float,
    lags: Sequence[int],
    windows: Sequence[int],
    include_other: bool,
) -> Tuple[RegressionModel, ModelReport]:
    if X_df.empty:
        raise ValueError("No rows available for regression training")

    X = X_df.values.astype(float)
    y = y_series.values.astype(float)

    if len(X) == 0 or len(y) == 0:
        raise ValueError("Insufficient data to train regression model")

    mean, scale, intercept, coef = _ridge_fit(X, y, alpha)

    model = RegressionModel(
        pollutant=pollutant,
        horizon=horizon,
        feature_names=feature_names,
        mean_=mean,
        scale_=scale,
        coef_=coef,
        intercept_=intercept,
        alpha=alpha,
        lags=tuple(lags),
        windows=tuple(windows),
        include_other=include_other,
    )

    preds = model.predict_from_features(X_df)
    residuals = y - preds
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))
    ss_tot = float(np.var(y) * len(y))
    ss_res = float(np.sum(residuals ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0

    weights = {fname: float(abs(weight)) for fname, weight in zip(feature_names, coef)}

    report = ModelReport(
        pollutant=pollutant,
        horizon=horizon,
        samples=len(y),
        rmse=rmse,
        mae=mae,
        r2=r2,
        feature_weights=weights,
    )

    return model, report


def _train_persistence_model(
    df: pd.DataFrame,
    pollutant: str,
    horizon: int,
    *,
    window: int = 3,
    reason: Optional[Exception] = None,
) -> Tuple[PersistenceModel, ModelReport]:
    prepared = ensure_datetime_index(df)
    if pollutant not in prepared.columns:
        raise ValueError(f"Column '{pollutant}' not present for fallback model") from reason

    series = pd.to_numeric(prepared[pollutant], errors="coerce").dropna()
    if series.empty:
        raise ValueError("No clean values available for fallback model") from reason

    window = max(1, min(window, len(series)))
    if reason is not None:
        print(f"[models] Falling back to persistence model for {pollutant}: {reason}")

    model = PersistenceModel(pollutant=pollutant, horizon=horizon, window=window)
    report = ModelReport(
        pollutant=pollutant,
        horizon=horizon,
        samples=int(len(series)),
        rmse=float("nan"),
        mae=float("nan"),
        r2=float("nan"),
        feature_weights={"rolling_mean_window": float(window)},
    )

    return model, report


def train_pollutant_model(
    df: pd.DataFrame,
    pollutant: str = "pm25",
    horizon: int = 1,
    lags: Sequence[int] = (1, 2, 3, 7, 14),
    windows: Sequence[int] = (3, 7, 14),
    include_other: bool = True,
    alpha: float = 1e-2,
    top_k: Optional[int] = 12,
) -> Tuple[Union[RegressionModel, PersistenceModel], ModelReport]:
    """Train a small ridge-regression forecaster for a pollutant.

    Falls back to a rolling-mean persistence model when there isn't enough
    clean history to support the richer feature engineering pipeline.
    """

    attempt_settings = [
        {
            "lags": lags,
            "windows": windows,
            "include_other": include_other,
            "top_k": top_k,
        },
        {
            "lags": (1, 2, 3),
            "windows": (3,),
            "include_other": False,
            "top_k": None,
        },
        {
            "lags": (1,),
            "windows": (),
            "include_other": False,
            "top_k": None,
        },
    ]

    last_exc: Optional[Exception] = None

    for setting in attempt_settings:
        try:
            X_df, y_series, selected = _prepare_training_matrix(
                df,
                pollutant,
                horizon,
                setting["lags"],
                setting["windows"],
                setting["include_other"],
                setting["top_k"],
            )
            return _finalize_regression_model(
                pollutant,
                horizon,
                X_df,
                y_series,
                selected,
                alpha=alpha,
                lags=setting["lags"],
                windows=setting["windows"],
                include_other=setting["include_other"],
            )
        except ValueError as exc:
            last_exc = exc
            continue

    return _train_persistence_model(
        df,
        pollutant,
        horizon,
        window=3,
        reason=last_exc,
    )


def forecast_future(
    history: pd.DataFrame,
    model: Union[RegressionModel, PersistenceModel],
    steps: int = 3,
) -> pd.DataFrame:
    """Iteratively forecast future pollutant values using the trained model."""

    if steps < 1:
        raise ValueError("steps must be >= 1")

    horizon_days = model.horizon
    history = ensure_datetime_index(history)
    working = history.copy()
    outputs: List[Dict[str, object]] = []

    for step in range(1, steps + 1):
        prediction = model.predict_one(working)
        if prediction is None:
            break
        last_date = working["date"].max()
        target_date = last_date + pd.Timedelta(days=horizon_days)
        aqi_value = compute_aqi(model.pollutant, prediction)
        outputs.append(
            {
                "date": target_date,
                "predicted_value": prediction,
                "aqi": aqi_value,
                "category": aqi_category(aqi_value),
                "step": step,
            }
        )

        new_row = {col: np.nan for col in working.columns}
        new_row["date"] = target_date
        if model.pollutant in new_row:
            new_row[model.pollutant] = prediction
        else:
            new_row[model.pollutant] = prediction
        working = pd.concat([working, pd.DataFrame([new_row])], ignore_index=True)

    return pd.DataFrame(outputs)


def analyze_city_data(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Produce simple descriptive stats and correlations for a city dataset."""

    if df.empty:
        return {}

    df = ensure_datetime_index(df)
    numeric_cols = [c for c in df.columns if c not in {"date", "station", "data_source"}]
    numeric = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    summary = numeric.describe().T
    latest = numeric.iloc[-1]
    corr = numeric.corr()

    trends = []
    days = (df["date"] - df["date"].min()).dt.days.values
    for col in numeric_cols:
        series = numeric[col].values
        mask = np.isfinite(series)
        if mask.sum() < 3:
            slope = np.nan
        else:
            slope, _ = np.polyfit(days[mask], series[mask], 1)
        trends.append({"pollutant": col, "daily_slope": slope})
    trend_df = pd.DataFrame(trends)

    latest_aqi = []
    for col in numeric_cols:
        value = latest.get(col)
        aqi_value = compute_aqi(col, value)
        latest_aqi.append(
            {
                "pollutant": col,
                "value": value,
                "aqi": aqi_value,
                "category": aqi_category(aqi_value),
            }
        )

    return {
        "summary": summary,
        "correlation": corr,
        "trend": trend_df,
        "latest_aqi": pd.DataFrame(latest_aqi),
    }


def train_city_models(
    df: pd.DataFrame,
    pollutants: Optional[Iterable[str]] = None,
    horizon: int = 1,
    **kwargs,
) -> Dict[str, Tuple[RegressionModel, ModelReport]]:
    """Train models for multiple pollutants and return a mapping."""

    if pollutants is None:
        pollutants = [col for col in POLLUTANT_COLUMNS if col in df.columns]

    outputs: Dict[str, Tuple[RegressionModel, ModelReport]] = {}
    for pollutant in pollutants:
        try:
            model, report = train_pollutant_model(df, pollutant=pollutant, horizon=horizon, **kwargs)
            outputs[pollutant] = (model, report)
        except ValueError:
            continue
    return outputs


# ---------------------------------------------------------------------------
# Automated city-level training pipeline
# ---------------------------------------------------------------------------
def build_training_frame(
    city: str,
    openaq_df: Optional[pd.DataFrame] = None,
    merra_df: Optional[pd.DataFrame] = None,
    data_dir: str = "data",
) -> pd.DataFrame:
    safe_city = safe_city_name(city)

    if openaq_df is None:
        openaq_path = latest_file(os.path.join(data_dir, f"{safe_city}_openaq_daily_*.csv"))
        if not openaq_path:
            raise FileNotFoundError(f"No OpenAQ daily dataset found for {city} in {data_dir}.")
        openaq_df = pd.read_csv(openaq_path)

    openaq = openaq_df.copy()
    if "date" not in openaq.columns:
        raise ValueError("OpenAQ dataframe must include a 'date' column.")
    openaq["date"] = pd.to_datetime(openaq["date"], errors="coerce")
    openaq = openaq[openaq["date"].notna()].sort_values("date")

    merra = None
    if merra_df is not None:
        merra = merra_df.copy()
    else:
        merra_path = latest_file(os.path.join(data_dir, f"{safe_city}_merra_daily_*.csv"))
        if merra_path:
            merra = pd.read_csv(merra_path)

    if merra is not None and not merra.empty and "date" in merra.columns:
        merra["date"] = pd.to_datetime(merra["date"], errors="coerce")
        merra = merra[merra["date"].notna()].sort_values("date")
        merra_filtered = merra.drop(columns=[col for col in ["city", "lat", "lon"] if col in merra.columns])
        merged = pd.merge(openaq, merra_filtered, on="date", how="left", suffixes=("", "_merra"))
    else:
        merged = openaq

    for col in merged.columns:
        if col == "date":
            continue
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    merged = merged.sort_values("date").reset_index(drop=True)
    return merged


def _synthetic_oversample(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    if len(X) == 0:
        return X, y

    rng = np.random.default_rng(random_state)
    unique, counts = np.unique(y, return_counts=True)
    max_count = counts.max()

    X_aug = [X]
    y_aug = [y]

    for cls, count in zip(unique, counts):
        if count == 0:
            continue
        if count >= max_count:
            continue
        idx = np.where(y == cls)[0]
        need = max_count - count
        if len(idx) == 0:
            continue
        samples = []
        labels = []
        for _ in range(need):
            i, j = rng.choice(idx, size=2, replace=True)
            lam = rng.random()
            synth = X[i] + lam * (X[j] - X[i])
            samples.append(synth)
            labels.append(cls)
        if samples:
            X_aug.append(np.vstack(samples))
            y_aug.append(np.array(labels, dtype=y.dtype))

    X_bal = np.vstack(X_aug)
    y_bal = np.concatenate(y_aug)
    return X_bal, y_bal


class KNNClassifier:
    def __init__(self, k: int = 5):
        self.k = k
        self._X = None
        self._y = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        if len(X) == 0:
            raise ValueError("Cannot fit classifier with no samples")
        self._X = X.astype(float)
        self._y = y.astype(int)
        self.k = max(1, min(self.k, len(self._X)))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._X is None:
            raise ValueError("Classifier has not been fitted")
        preds = []
        for row in X.astype(float):
            distances = np.linalg.norm(self._X - row, axis=1)
            idx = np.argsort(distances)[: self.k]
            votes, counts = np.unique(self._y[idx], return_counts=True)
            preds.append(votes[np.argmax(counts)])
        return np.array(preds, dtype=int)


@dataclass
class ClassificationModel:
    pollutant: str
    horizon: int
    feature_names: List[str]
    mean_: np.ndarray
    scale_: np.ndarray
    classifier: KNNClassifier
    class_labels: List[str]
    lags: Tuple[int, ...]
    windows: Tuple[int, ...]
    include_other: bool

    def _standardize(self, features: pd.DataFrame) -> np.ndarray:
        X = features[self.feature_names].values.astype(float)
        return (X - self.mean_) / self.scale_

    def predict_from_features(self, features: pd.DataFrame) -> str:
        X = self._standardize(features)
        pred_idx = self.classifier.predict(X)[0]
        return self.class_labels[pred_idx]

    def predict_one(self, df: pd.DataFrame) -> Optional[str]:
        generated = _generate_features(df, self.pollutant, self.lags, self.windows, self.include_other)
        row = generated.dropna(subset=self.feature_names).tail(1)
        if row.empty:
            return None
        return self.predict_from_features(row)


def train_aqi_classifier(
    df: pd.DataFrame,
    pollutant: str = "pm25",
    horizon: int = 1,
    lags: Sequence[int] = (1, 2, 3, 7, 14),
    windows: Sequence[int] = (3, 7, 14),
    include_other: bool = True,
    top_k: Optional[int] = 15,
    k_neighbors: int = 5,
    random_state: int = 42,
):
    if pollutant not in df.columns:
        raise ValueError(f"Column '{pollutant}' not present for classification training")

    features_df = _generate_features(df, pollutant, lags, windows, include_other)

    target_series = pd.to_numeric(features_df[pollutant], errors="coerce").shift(-horizon)
    aqi_values = target_series.apply(lambda val: compute_aqi(pollutant, val))
    categories = aqi_values.apply(aqi_category)

    candidate_cols = [
        c
        for c in features_df.columns
        if c not in {"date", "station", "data_source", pollutant}
        and c not in POLLUTANT_COLUMNS
    ]

    features_df["target_class"] = categories
    prepared = features_df.dropna(subset=candidate_cols + ["target_class"])
    if prepared.empty or prepared["target_class"].nunique() < 2:
        raise ValueError("Insufficient data variety to train classification model")

    X = prepared[candidate_cols].values.astype(float)
    y_labels = prepared["target_class"].astype(str).values

    class_names = sorted(prepared["target_class"].dropna().unique())
    label_to_idx = {label: idx for idx, label in enumerate(class_names)}
    y = np.array([label_to_idx[label] for label in y_labels], dtype=int)

    split_idx = max(1, int(len(X) * 0.8))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    mean = X_train.mean(axis=0)
    scale = X_train.std(axis=0)
    scale[scale == 0] = 1.0
    X_train_std = (X_train - mean) / scale
    X_test_std = (X_test - mean) / scale if len(X_test) else np.empty((0, X.shape[1]))

    X_bal, y_bal = _synthetic_oversample(X_train_std, y_train, random_state=random_state)

    classifier = KNNClassifier(k=k_neighbors)
    classifier.fit(X_bal, y_bal)

    train_preds = classifier.predict(X_train_std)
    train_accuracy = float((train_preds == y_train).mean()) if len(y_train) else float("nan")
    if len(X_test_std):
        test_preds = classifier.predict(X_test_std)
        test_accuracy = float((test_preds == y_test).mean())
    else:
        test_accuracy = float("nan")

    model = ClassificationModel(
        pollutant=pollutant,
        horizon=horizon,
        feature_names=candidate_cols,
        mean_=mean,
        scale_=scale,
        classifier=classifier,
        class_labels=class_names,
        lags=tuple(lags),
        windows=tuple(windows),
        include_other=include_other,
    )

    report = {
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "class_distribution": {label: int((y == label_to_idx[label]).sum()) for label in class_names},
    }

    return model, report


def train_city_models_auto(
    city: str,
    openaq_df: Optional[pd.DataFrame] = None,
    merra_df: Optional[pd.DataFrame] = None,
    pollutants: Optional[Iterable[str]] = None,
    horizon: int = 1,
    forecast_steps: int = 3,
    data_dir: str = "data",
    *,
    use_cache: bool = True,
    force_retrain: bool = False,
) -> Dict[str, object]:
    history = build_training_frame(city, openaq_df=openaq_df, merra_df=merra_df, data_dir=data_dir)
    history = history.sort_values("date").reset_index(drop=True)

    history_signature = dataframe_signature(history)
    safe_city = safe_city_name(city)
    cache_root = Path(data_dir) / "models" / safe_city
    cache_root.mkdir(parents=True, exist_ok=True)

    bundle_name = f"bundle_h{horizon}_s{forecast_steps}.pkl"
    bundle_path = cache_root / bundle_name
    manifest_path = cache_root / "manifest.json"

    manifest_entries: List[ModelCacheManifest] = []
    if manifest_path.exists():
        try:
            with manifest_path.open("r", encoding="utf-8") as fh:
                raw_manifest = json.load(fh)
            if isinstance(raw_manifest, list):
                manifest_entries = [ModelCacheManifest.from_dict(item) for item in raw_manifest]
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            manifest_entries = []

    cache_entry = next(
        (
            entry
            for entry in manifest_entries
            if entry.city == safe_city and entry.horizon == horizon and entry.forecast_steps == forecast_steps
        ),
        None,
    )

    if use_cache and not force_retrain and cache_entry:
        cache_file = Path(cache_entry.path)
        if cache_entry.history_signature == history_signature and cache_file.exists():
            try:
                with cache_file.open("rb") as fh:
                    cached_payload = pickle.load(fh)
                print(f"[cache] Loaded cached model bundle for {city} → {cache_file}")
                cached_payload["history"] = history
                return cached_payload
            except (pickle.PickleError, ValueError, OSError):
                print(f"[cache] Failed to load cached bundle at {cache_file}; retraining.")
        elif cache_entry.history_signature != history_signature:
            print("[cache] Cached model signature mismatch; retraining with fresh data.")
        elif not cache_file.exists():
            print(f"[cache] Cached bundle missing at {cache_file}; retraining.")

    print(f"[models] Training city models for {city} (horizon={horizon}, steps={forecast_steps})")

    available_pollutants = [col for col in POLLUTANT_COLUMNS if col in history.columns]
    if pollutants is None:
        pollutants = available_pollutants
    else:
        pollutants = [p for p in pollutants if p in available_pollutants]

    regression_results = {}
    forecast_results = {}

    for pollutant in pollutants:
        try:
            model, report = train_pollutant_model(
                history,
                pollutant=pollutant,
                horizon=horizon,
            )
            forecast = forecast_future(history, model, steps=forecast_steps)
            regression_results[pollutant] = {"model": model, "report": report}
            forecast_results[pollutant] = forecast
        except ValueError:
            continue

    classification_result = None
    classification_report = None
    classification_prediction = None
    chosen_pollutant = None
    if "pm25" in history.columns:
        chosen_pollutant = "pm25"
    elif available_pollutants:
        chosen_pollutant = available_pollutants[0]

    if chosen_pollutant is not None:
        try:
            clf_model, clf_report = train_aqi_classifier(
                history,
                pollutant=chosen_pollutant,
                horizon=horizon,
            )
            classification_result = clf_model
            classification_report = clf_report
            prediction = clf_model.predict_one(history)
            classification_prediction = {
                "pollutant": chosen_pollutant,
                "predicted_category": prediction,
                "classes": clf_model.class_labels,
            }
        except ValueError:
            pass

    regression_metrics_rows = []
    for pollutant, payload in regression_results.items():
        rep: ModelReport = payload["report"]
        regression_metrics_rows.append(
            {
                "pollutant": pollutant,
                "samples": rep.samples,
                "rmse": rep.rmse,
                "mae": rep.mae,
                "r2": rep.r2,
            }
        )
    metrics_df = pd.DataFrame(regression_metrics_rows)

    result = {
        "history": history,
        "regression": regression_results,
        "forecasts": forecast_results,
        "metrics": metrics_df,
        "classification_model": classification_result,
        "classification_report": classification_report,
        "classification_prediction": classification_prediction,
    }

    if use_cache:
        bundle_payload = {key: value for key, value in result.items() if key != "history"}
        try:
            with bundle_path.open("wb") as fh:
                pickle.dump(bundle_payload, fh)
            print(f"[cache] Cached model bundle → {bundle_path}")

            new_entry = ModelCacheManifest(
                city=safe_city,
                horizon=horizon,
                forecast_steps=forecast_steps,
                history_signature=history_signature,
                path=str(bundle_path),
            )
            manifest_entries = [
                entry
                for entry in manifest_entries
                if not (
                    entry.city == safe_city
                    and entry.horizon == horizon
                    and entry.forecast_steps == forecast_steps
                )
            ]
            manifest_entries.append(new_entry)
            with manifest_path.open("w", encoding="utf-8") as fh:
                json.dump([entry.to_dict() for entry in manifest_entries], fh, indent=2)
        except OSError as exc:
            print(f"[cache] Failed to write cache files: {exc}")

    return result
