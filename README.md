# stockpricepredection
helps to predict stock prices
# Stock Price Prediction (NSE/BSE) — Training and App

This repository provides:
- Feature engineering utilities for OHLCV data (ml/features.py)
- Model training utilities (ml/model.py)
- Single-ticker training script (train.py)
- Batch training script for many Indian tickers (train_batch.py)
- Interactive Streamlit app (app.py)

The models learn to predict the next trading day Close price (target: Close_FWD_1) using technical indicators, lags, and calendar features.


## Prerequisites
- Windows with PowerShell (pwsh)
- Python 3.9+ (3.10+ recommended)
- Internet access (yfinance pulls data from Yahoo Finance)

Optional accelerators/packages:
- XGBoost (optional): `pip install xgboost`
- LightGBM (optional): `pip install lightgbm`

Note: Prebuilt wheels for xgboost/lightgbm are typically available on Windows. If you encounter build errors, ensure you are on a recent Python version and up-to-date pip.


## Quick Start (Windows PowerShell)
1) Create and activate a virtual environment
```pwsh path=null start=null
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Upgrade pip and install required packages
```pwsh path=null start=null
python -m pip install --upgrade pip
pip install numpy pandas yfinance scikit-learn joblib plotly streamlit
# Optional models
pip install xgboost lightgbm
```

3) Verify the repository structure (key files)
```pwsh path=null start=null
tree /F
```
Look for these files:
- train.py — single-ticker training
- train_batch.py — batch training across many tickers
- app.py — Streamlit app
- ml\features.py, ml\model.py — feature engineering and modeling


## Single-Ticker Training
Train and save models for a single instrument.

Examples:
- RandomForest only (default):
```pwsh path=null start=null
python .\train.py --ticker RELIANCE.NS --start 2015-01-01 --end today --output-dir .\models
```
- Train all supported models (RandomForest + XGBoost + LightGBM):
```pwsh path=null start=null
python .\train.py --ticker TCS.NS --start 2015-01-01 --end today --test-fraction 0.2 --all-models --output-dir .\models
```

Outputs:
- Models saved to .\models as <TICKER>_<MODEL>_<YYYYMMDD>.joblib
- Console prints evaluation metrics on the test split: MAE, MAPE, R²


## Batch Training for Indian Stocks
Use train_batch.py to train and save models for many tickers, with logging, resume support, and optional parallelism.

1) Prepare a tickers file (TXT or CSV)
- Plain text (one base symbol per line, no suffix):
```text path=null start=null
RELIANCE
TCS
HDFCBANK
INFY
```
- Or a CSV with a header column named `ticker`:
```csv path=null start=null
ticker
RELIANCE
TCS
HDFCBANK
INFY
```

Notes:
- The script will append an exchange suffix if the symbol does not already contain one.
- NSE suffix: .NS, BSE suffix: .BO

2) Run batch training (NSE example)
```pwsh path=null start=null
python .\train_batch.py ^
  --tickers-file .\tickers_nse.txt ^
  --suffix .NS ^
  --start 2015-01-01 ^
  --end today ^
  --workers 4 ^
  --sleep 0.3 ^
  --output-dir .\models\india ^
  --results-csv .\models\india\results.csv
```

3) Include all model types (if installed)
```pwsh path=null start=null
python .\train_batch.py --tickers-file .\tickers_nse.txt --suffix .NS --all-models --workers 4
```

4) Force retrain (ignore already-saved models)
```pwsh path=null start=null
python .\train_batch.py --tickers-file .\tickers_nse.txt --no-resume
```

Batch outputs:
- Saved models: .\models\india\<TICKER>_<MODEL>_<YYYYMMDD>.joblib
- Results summary CSV: .\models\india\results.csv with columns
  - ticker, model, status (ok/skipped/failed), mae, mape, r2, saved_path, error

Best practices:
- Use a modest number of workers (2–8) and a small --sleep to be gentle on yfinance.
- Use --no-resume only when you intentionally want to overwrite previous runs.


## Streamlit App
Train or evaluate via a UI and generate short-term forecasts.
```pwsh path=null start=null
streamlit run .\app.py
```
In the sidebar:
- Choose "Train new model" to train on-the-fly, or "Load existing model" to evaluate/forecast with a saved model
- Provide ticker (e.g., RELIANCE.NS), date range, test fraction, and model-specific parameters
- Forecast horizon determines how many next business days to predict

Outputs:
- Test metrics (MAE, MAPE, R²)
- Interactive chart of historical close, test predictions, and forecast
- Downloadable forecast CSV


## How It Works (under the hood)
- Data: yfinance downloads OHLCV; multi-index columns are normalized to single-level [Open, High, Low, Close, Volume]
- Features: ml/features.py adds moving averages (SMA/EMA), RSI, MACD, Bollinger Bands, volume averages, lags, and calendar features
- Target: next-day Close (Close_FWD_1)
- Split: chronological train/test by fraction
- Models: scikit-learn RandomForest; optional XGBoost/LightGBM
- Evaluation: MAE, MAPE, R² on the test partition
- Forecast: recursive next-day predictions using technical features recomputed with the appended predicted close


## Common Issues and Tips
- Invalid ticker or no data: Ensure you include the correct exchange suffix (.NS for NSE, .BO for BSE) or pass --suffix in batch mode
- yfinance rate limiting/intermittent failures: Reduce --workers, increase --sleep, or retry later
- XGBoost/LightGBM not installed: Either install them (optional) or omit them from --models
- Activation script blocked: If .venv\Scripts\Activate.ps1 is blocked by execution policy, run in the current session:
```pwsh path=null start=null
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```
- Reproducibility: The scripts set default random seeds for supported models, but online data updates can change results over time


## Reference CLI
Single ticker:
```pwsh path=null start=null
python .\train.py --ticker RELIANCE.NS --start 2015-01-01 --end today --all-models --output-dir .\models
```
Batch (NSE):
```pwsh path=null start=null
python .\train_batch.py --tickers-file .\tickers_nse.txt --suffix .NS --workers 4 --sleep 0.3 --results-csv .\models\india\results.csv --output-dir .\models\india
```
Streamlit app:
```pwsh path=null start=null
streamlit run .\app.py
```


## Disclaimer
This code is for educational and research purposes only. It does not constitute financial advice. Use at your own risk.

# stockpricepredection
helps to predict stock prices

