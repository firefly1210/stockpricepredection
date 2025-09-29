from __future__ import annotations
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
from typing import List, Any, Tuple
from pathlib import Path
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Optional models
try:
    import xgboost as xgb  # type: ignore
except Exception:  # pragma: no cover
    xgb = None  # type: ignore

try:
    import lightgbm as lgb  # type: ignore
except Exception:  # pragma: no cover
    lgb = None  # type: ignore

from .features import add_technical_indicators


def prepare_dataset(prices: pd.DataFrame):
    """
    Build a supervised learning dataset from OHLCV prices.
    Target is next-day Close (Close_FWD_1).

    Returns:
        X (np.ndarray), y (np.ndarray), feature_names (List[str]), target_name (str), df_supervised (pd.DataFrame)
    """
    df = add_technical_indicators(prices)

    # Define target as next-day Close
    df["Close_FWD_1"] = df["Close"].shift(-1)

    # Remove rows with NaNs from indicators/lags
    df = df.dropna().copy()

    # Feature columns (exclude target)
    exclude_cols = {"Close_FWD_1"}
    feature_names = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_names].values
    y = df["Close_FWD_1"].values

    return X, y, feature_names, "Close_FWD_1", df


def train_test_split_time(X: np.ndarray, y: np.ndarray, index: pd.Index, test_fraction: float = 0.2):
    n = len(X)
    n_test = max(1, int(n * test_fraction))
    split = n - n_test
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    idx_train, idx_test = index[:split], index[split:]
    return X_train, X_test, y_train, y_test, idx_train, idx_test


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray, n_estimators: int = 400) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
        max_features="sqrt",
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42,
) -> Any:
    if xgb is None:
        raise ImportError("xgboost is not installed. Install it with: pip install xgboost")
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        n_jobs=-1,
        objective="reg:squarederror",
        tree_method="hist",
    )
    model.fit(X_train, y_train)
    return model


def train_lightgbm_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 600,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    feature_fraction: float = 0.8,
    random_state: int = 42,
) -> Any:
    if lgb is None:
        raise ImportError("lightgbm is not installed. Install it with: pip install lightgbm")
    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        feature_fraction=feature_fraction,
        random_state=random_state,
        n_jobs=-1,
        objective="regression",
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        if not np.isfinite(mape):
            mape = np.nan
    r2 = r2_score(y_true, y_pred)
    return {"mae": float(mae), "mape": float(mape), "r2": float(r2)}


def _features_for_next_day(df_prices: pd.DataFrame, feature_names: List[str]) -> pd.Series:
    """
    Build the feature row needed to predict the next day, using the latest available data.
    """
    df_feat = add_technical_indicators(df_prices)
    df_feat = df_feat.dropna()
    if df_feat.empty:
        raise ValueError("Not enough data after feature engineering.")
    last_row = df_feat.iloc[-1]
    # Ensure all feature names exist (some may be missing if insufficient history)
    missing = [f for f in feature_names if f not in last_row.index]
    if missing:
        raise ValueError(f"Missing features due to short history: {missing[:5]}…")
    return last_row[feature_names]


def forecast_next_n_days(prices: pd.DataFrame, model: Any, feature_names: List[str], steps: int = 5) -> pd.DataFrame:
    """
    Recursive multi-step forecast using the trained model.
    Appends predicted Close into a working copy and recomputes features each step.
    """
    work = prices.copy()
    forecasts = []

    # Start from last business day in index
    last_date = work.index[-1]

    for _ in range(steps):
        # Build features from the latest data
        feats = _features_for_next_day(work, feature_names)
        yhat = float(model.predict([feats.values])[0])

        # Next business day
        next_day = last_date + BDay(1)
        last_vol = float(work["Volume"].iloc[-1]) if "Volume" in work.columns else 0.0

        # Append a synthetic next day row (OHLC approximated by close)
        new_row = pd.DataFrame({
            "Open": yhat,
            "High": yhat,
            "Low": yhat,
            "Close": yhat,
            "Volume": last_vol,
        }, index=pd.DatetimeIndex([next_day]))
        work = pd.concat([work, new_row])
        last_date = next_day
        forecasts.append({"Date": next_day, "PredictedClose": yhat})

    fcst_df = pd.DataFrame(forecasts).set_index("Date")
    return fcst_df


# Persistence helpers

def save_model(model: Any, feature_names: List[str], model_name: str, dir_path: str = "models") -> str:
    Path(dir_path).mkdir(exist_ok=True)
    payload = {"model": model, "feature_names": feature_names}
    path = Path(dir_path) / f"{model_name}.joblib"
    joblib.dump(payload, path)
    return str(path)


def load_model(path: str) -> Tuple[Any, List[str]]:
    payload = joblib.load(path)
    return payload["model"], list(payload["feature_names"]) 
