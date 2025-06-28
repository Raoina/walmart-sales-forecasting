
# 🛒 Walmart Time Series Forecasting - Dept 26

This project addresses a real-world **time series forecasting** task using Walmart's historical sales data. We focused specifically on **Department 26**, applying various statistical and machine learning models to predict **Weekly Sales**.

---

## 📁 Dataset Overview
- **Source**: Cleaned Walmart dataset
- **Target**: `Weekly_Sales`
- **Features Used**:
  - Temperature, Fuel_Price, CPI, Unemployment
  - IsHoliday flag
  - Time-based features: Week, Year, Month, Day
  - Lag features: Lag_1, Lag_2, Rolling_Mean_4

---

## 🔍 Time Series Analysis
- Visualized trends and seasonality
- Performed ACF & PACF analysis to assess autocorrelation and determine ARIMA lags
- Incorporated holiday effects and temporal variables

---

## 📉 Models Implemented

### 📊 Statistical Models:
- **Holt-Winters Exponential Smoothing**
- **Holt Linear Trend**
- **ARIMA**
- **SARIMA**
- **SARIMAX**

### 🤖 Machine Learning:
- **Random Forest Regressor** with engineered time series features
- **Recursive Multi-step Forecasting**

### 🔮 Deep Learning:
- **LSTM** model using previous sales window
- (Tuned and compared performance)

### 🧙 Facebook Prophet:
- Captured holiday effects and seasonality
- Fast and interpretable

### 💡 Innovation:
- **N-BEATS** deep learning model (for time series)
- Added to improve robustness and modernize approach

---

## 📈 Evaluation Metrics:
All models were evaluated using:
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **MAPE** (Optional in Prophet)

A comparison table summarizes the performance across all models.

---

## 🧪 Insights:
- Models with external regressors (SARIMAX, Prophet) performed well during seasonal shifts.
- Random Forest showed strong predictive power with engineered features.
- LSTM needed careful scaling and lag alignment.
- N-BEATS demonstrated potential on long-horizon predictions.

---

## 🖥 Deployment

A full **Streamlit App** was developed to:
- Upload or enter historical data
- Choose forecast horizon (1–20 weeks)
- Perform recursive forecasting using Random Forest
- Export results as CSV or image

### 🎯 Features:
- Upload CSV or input manually
- Smart feature generation (lags, rolling mean)
- Clean UI and downloadable outputs

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
streamlit run app_forecast_from_history.py
```

Or run on Google Colab using `ngrok` (instructions in notebook)

---

## 📎 Files

- `Final_Project.ipynb` → Main analysis notebook
- `app_forecast_from_history.py` → Final Streamlit app
- `random_forest_model.joblib` → Best trained model
- `scaler_X.joblib` + `scaler_y.joblib` → Required scalers for deployment

---


