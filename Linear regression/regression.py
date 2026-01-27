"""
GOOG price forecasting demo using Linear Regression, forecasting the NEXT 30 TRADING DAYS.

Main changes vs the prior version:
- Forecast horizon is fixed to 30 trading days (not a % of dataset length).
- Forecast dates are generated using business days (Mon–Fri) via pandas.bdate_range,
  so we don't create weekend dates.

Notes:
- This is an educational baseline, not a production trading model.
- We use a Pipeline (StandardScaler + LinearRegression) to avoid scaling bugs.
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pickle


# -----------------------------
# 1) Download and prepare data
# -----------------------------
ticker = "GOOG"
df = yf.download(ticker)  # index is DatetimeIndex of trading days

# Keep only columns we need
df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

# Feature: intraday volatility as a percent of Close
df["HL_PCT"] = (df["High"] - df["Low"]) / df["Close"] * 100.0

# Feature: percent change from Open to Close
df["PCT_change"] = (df["Close"] - df["Open"]) / df["Open"] * 100.0

# Inputs used by the model
feature_cols = ["Close", "HL_PCT", "PCT_change", "Volume"]
df = df[feature_cols].copy()

# Fill missing values (rare, but safe)
df.fillna(-99999, inplace=True)


# -----------------------------------------------------
# 2) Forecast horizon = NEXT 30 TRADING DAYS (fixed)
# -----------------------------------------------------
DESIRED_FORECAST_DAYS = 30

# Use at most 10% of data, but never more than 30
forecast_out = min(DESIRED_FORECAST_DAYS, max(1, len(df) // 10))
print("forecast_out (trading days) =", forecast_out)


# ---------------------------------------
# 3) Create label = future Close price
# ---------------------------------------
# Shift Close upward by 30 rows:
# - For each row, label is the Close price 30 trading days later.
df["label"] = df["Close"].shift(-forecast_out)

# IMPORTANT:
# The last 30 rows now have label=NaN (no future Close available).
# We want those last 30 rows for prediction (X_lately),
# but we must exclude them from training.


# ---------------------------------------
# 4) Build X, y, and X_lately correctly
# ---------------------------------------

X = np.array(df.drop(["label"], axis=1))          # features
# X_lately: the most recent 30 rows -> used to predict next 30 trading days
X_lately = X[-forecast_out:]

# Training set: everything except the last 30 rows
X = X[:-forecast_out]

df.dropna(inplace=True)  # drop rows with NaN labels

y = np.array(df["label"])                           # labels (has NaNs at the end)

# Sanity check: y should contain no NaNs now
assert not np.isnan(y).any(), "Training labels contain NaNs. Check the shifting logic."

print("Train samples:", len(X), "Prediction samples:", len(X_lately))


# ---------------------------------------
# 5) Train/test split
# ---------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# ---------------------------------------
# 6) Create a consistent ML pipeline
# ---------------------------------------
# Pipeline ensures:
# - scaler is fit ONLY on training data
# - same scaling is applied to test and future predictions
model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("reg", LinearRegression(n_jobs=-1)),
    ]
)

model_path = "linearregression_goog_30bd.pickle"

# Load existing model if present, else train and save
if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print("Loaded model from:", model_path)
else:
    model.fit(X_train, y_train)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print("Trained + saved model to:", model_path)

# Evaluate with R^2 score
r2 = model.score(X_test, y_test)
print("R^2 accuracy:", r2)


# ---------------------------------------
# 7) Predict the next 30 trading days
# ---------------------------------------
forecast_set = model.predict(X_lately)


# ---------------------------------------------------
# 8) Append forecast rows using TRADING DAY dates
# ---------------------------------------------------
df["Forecast"] = np.nan

# Last date in the downloaded dataset (already a trading day)
last_date = df.index[-1]

# Create the next 30 business days (Mon–Fri).
# We start from the next business day after last_date.
future_dates = pd.bdate_range(start=last_date + pd.offsets.BDay(1), periods=forecast_out)

# Add forecast values into new rows indexed by future business-day dates
for date, pred in zip(future_dates, forecast_set):
    # Make feature columns and label NaN for future rows (unknown),
    # but set Forecast to the prediction.
    df.loc[date, feature_cols] = np.nan
    df.loc[date, "label"] = np.nan
    df.loc[date, "Forecast"] = pred

print(df.tail(10))


# ---------------------------------------
# 9) Plot historical Close + forecast
# ---------------------------------------
plt.figure(figsize=(12, 6))
df["Close"].plot()
df["Forecast"].plot()
plt.legend()
plt.xlabel("Date")
plt.ylabel("Price")
plt.title(f"{ticker} Close vs Forecast (Next {forecast_out} Trading Days)")
plt.show()

