import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import yfinance as yf
from datetime import date
from io import BytesIO
from pathlib import Path

from ml.model import (
    prepare_dataset,
    train_test_split_time,
    train_random_forest,
    train_xgboost_regressor,
    train_lightgbm_regressor,
    evaluate_model,
    forecast_next_n_days,
    save_model,
    load_model,
)

st.set_page_config(page_title="Stock Price Prediction", layout="wide")

@st.cache_data(ttl=3600)
def load_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError("No data downloaded. Check the ticker and date range.")
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df

st.title("📈 Stock Price Prediction App")

with st.sidebar:
    st.header("Settings")

    # Mode: Train new vs Load existing
    mode = st.radio("Mode", ["Train new model", "Load existing model"], index=0)

    ticker = st.text_input("Ticker", value="AAPL")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date", value=pd.Timestamp("2015-01-01").date())
    with col2:
        end_date = st.date_input("End date", value=date.today())

    test_frac = st.slider("Test fraction", min_value=0.05, max_value=0.5, value=0.2, step=0.05)
    horizon = st.slider("Forecast horizon (days)", min_value=1, max_value=30, value=7)

    model_type = None
    rf_params = {}
    xgb_params = {}
    lgb_params = {}

    if mode == "Train new model":
        model_type = st.selectbox("Model", ["RandomForest", "XGBoost", "LightGBM"], index=0)
        if model_type == "RandomForest":
            rf_params["n_estimators"] = st.slider("Trees (RandomForest)", min_value=100, max_value=1500, value=400, step=100)
        elif model_type == "XGBoost":
            xgb_params["n_estimators"] = st.slider("Estimators (XGBoost)", min_value=100, max_value=2000, value=600, step=100)
            xgb_params["learning_rate"] = st.slider("Learning rate", min_value=0.01, max_value=0.3, value=0.05, step=0.01)
            xgb_params["max_depth"] = st.slider("Max depth", min_value=2, max_value=12, value=6, step=1)
        elif model_type == "LightGBM":
            lgb_params["n_estimators"] = st.slider("Estimators (LightGBM)", min_value=100, max_value=3000, value=800, step=100)
            lgb_params["learning_rate"] = st.slider("Learning rate ", min_value=0.01, max_value=0.3, value=0.05, step=0.01)
            lgb_params["num_leaves"] = st.slider("Num leaves", min_value=7, max_value=255, value=31, step=2)

        save_chk = st.checkbox("Save trained model", value=True)
        default_model_name = f"{ticker}_{model_type}_{str(end_date).replace('-', '')}"
        model_name = st.text_input("Model name", value=default_model_name)
    else:
        # Load existing
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        files = sorted(models_dir.glob("*.joblib"))
        if files:
            chosen = st.selectbox("Choose model file", files, format_func=lambda p: p.name)
        else:
            st.warning("No models found in ./models. Train and save a model first.")
            chosen = None

    go_btn = st.button("Run", type="primary")

status = st.empty()

try:
    if go_btn:
        status.info("Downloading prices…")
        prices = load_prices(ticker, str(start_date), str(end_date))

        status.info("Preparing dataset…")
        X_all, y_all, feature_names_all, target_name, df_supervised = prepare_dataset(prices)

        # Default: use full features unless loading model with its own feature order
        feature_names_used = feature_names_all

        model = None
        trained = False

        if mode == "Train new model":
            X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split_time(
                X_all, y_all, df_supervised.index, test_fraction=test_frac
            )

            status.info("Training model…")
            if model_type == "RandomForest":
                model = train_random_forest(X_train, y_train, n_estimators=rf_params["n_estimators"]) 
            elif model_type == "XGBoost":
                model = train_xgboost_regressor(
                    X_train,
                    y_train,
                    **xgb_params,
                )
            elif model_type == "LightGBM":
                model = train_lightgbm_regressor(
                    X_train,
                    y_train,
                    **lgb_params,
                )
            trained = True

            # Evaluate on test
            status.info("Evaluating…")
            y_pred_test = model.predict(X_test)
            metrics = evaluate_model(y_test, y_pred_test)

            # Build evaluation dataframe
            df_eval = pd.DataFrame({
                "Actual": y_test,
                "Predicted": y_pred_test,
            }, index=idx_test)

            # Save model if requested
            if save_chk:
                status.info("Saving model…")
                path_saved = save_model(model, feature_names_used, model_name)
                st.toast(f"Saved: {path_saved}")
        else:
            if chosen is None:
                st.stop()
            status.info("Loading model…")
            model, feature_names_used = load_model(str(chosen))

            # Rebuild X and y using the loaded feature order to evaluate
            X_all_loaded = df_supervised[feature_names_used].values
            X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split_time(
                X_all_loaded, y_all, df_supervised.index, test_fraction=test_frac
            )
            status.info("Evaluating…")
            y_pred_test = model.predict(X_test)
            metrics = evaluate_model(y_test, y_pred_test)
            df_eval = pd.DataFrame({"Actual": y_test, "Predicted": y_pred_test}, index=idx_test)

        status.info("Forecasting…")
        fcst = forecast_next_n_days(prices, model, feature_names_used, steps=horizon)

        status.empty()

        # Layout
        left, right = st.columns([2, 1])
        with right:
            st.subheader("Test Metrics")
            st.metric("MAE", f"{metrics['mae']:.2f}")
            st.metric("MAPE", f"{metrics['mape']:.2f}%")
            st.metric("R²", f"{metrics['r2']:.3f}")

            st.subheader("Forecast (next days)")
            st.dataframe(fcst.tail(horizon))

            # Download forecast CSV
            csv_bytes = fcst.to_csv().encode("utf-8")
            fname = f"{ticker}_forecast.csv" if mode == "Load existing model" else f"{ticker}_{model_type}_forecast.csv"
            st.download_button("Download forecast CSV", data=csv_bytes, file_name=fname, mime="text/csv")

        with left:
            st.subheader(f"{ticker} Close Price: Actual, Predicted (test), and Forecast")
            fig = go.Figure()
            # Full actual close
            fig.add_trace(go.Scatter(x=prices.index, y=prices["Close"], mode="lines", name="Actual Close"))
            # Predicted on test
            fig.add_trace(go.Scatter(x=df_eval.index, y=df_eval["Predicted"], mode="lines", name="Predicted (test)", line=dict(dash="dot")))
            # Forecast
            fig.add_trace(go.Scatter(x=fcst.index, y=fcst["PredictedClose"], mode="lines+markers", name="Forecast", line=dict(dash="dash")))
            fig.update_layout(height=600, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)

        st.success("Done")

    else:
        st.info("Select a mode, set parameters, and click 'Run'.")
except ImportError as e:
    status.empty()
    st.error(str(e))
except Exception as e:
    status.empty()
    st.error(f"Error: {e}")
