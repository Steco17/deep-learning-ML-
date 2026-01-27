# GOOG Stock Price Forecast (Next 30 Trading Days)

This project downloads historical Google (GOOG) stock data, builds a few basic features, trains (or loads) a Linear Regression model, and forecasts the next **30 trading days**. It then plots the historical close price alongside the forecast.

> Educational project: this is a simple baseline model meant for learning, not a trading strategy.

---

## What the script does (high level)

1. **Downloads stock data** for `GOOG` using `yfinance`
2. **Selects core columns**: Open, High, Low, Close, Volume
3. **Creates features**:
   - `HL_PCT`: daily volatility percentage  
     \[
     HL\_PCT = \frac{High - Low}{Close} \times 100
     \]
   - `PCT_change`: daily return percentage  
     \[
     PCT\_change = \frac{Close - Open}{Open} \times 100
     \]
4. **Creates labels** (supervised learning target):
   - The label is the **Close price 30 trading days in the future**
   - Done using a shift: `label = Close.shift(-30)`
5. **Splits the dataset**:
   - `X_lately`: the last 30 rows (used for forecasting)
   - `X, y`: all earlier rows (used for training/testing)
6. **Trains or loads a model**:
   - Uses a Scikit-learn `Pipeline`:
     - `StandardScaler` (feature scaling)
     - `LinearRegression`
   - Saves/loads the model with `pickle`
7. **Forecasts the next 30 trading days**
8. **Appends predicted values** into the DataFrame using business-day dates (`bdate_range`)
9. **Plots** `Close` vs `Forecast`

---

## Why a Pipeline is used (important)

A common mistake is scaling the features and then accidentally training the model on unscaled data (or using a different scaling for prediction).  
This project avoids that by using a `Pipeline`, so the **same scaler fitted on training data** is always used consistently for:

- training
- testing
- future forecasting

---


---

## Requirements

- Python 3.9+ recommended

Main libraries:
- `yfinance`
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`

---

## Installation

Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate

### Run the script
```bash
python forecast_goog.py
```

## Common error and fix: “train set will be empty”
```bash
ValueError: With n_samples=0 ... the resulting train set will be empty
```
It means your dataset is too small relative to the forecast horizon.

Example:

If you only have 20 rows but try to forecast 30 days ahead, training data becomes empty.

Fix: download more history or reduce forecast_out.

Recommended safety check in code:
```bash
if len(df) <= forecast_out + 5:
    raise ValueError("Not enough data to train and forecast.")
```