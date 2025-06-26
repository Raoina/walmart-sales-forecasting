
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load model and scalers
model = joblib.load("random_forest_model.joblib")
scaler_X = joblib.load("scaler_X.joblib")
scaler_y = joblib.load("scaler_y.joblib")

st.title("🛒 Walmart Forecast App")

st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

def is_raw_data(df):
    return {'Dept', 'Weekly_Sales', 'Date'}.issubset(df.columns)

def is_ready_data(df):
    required_cols = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'IsHoliday',
                     'Week', 'Year', 'Month', 'Day', 'Lag_1', 'Lag_2', 'Rolling_Mean_4']
    return all(col in df.columns for col in required_cols)

def engineer_features(df):
    df = df[df['Dept'] == 26.0].copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values('Date')
    df['Week'] = df['Date'].dt.isocalendar().week
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Lag_1'] = df['Weekly_Sales'].shift(1)
    df['Lag_2'] = df['Weekly_Sales'].shift(2)
    df['Rolling_Mean_4'] = df['Weekly_Sales'].shift(1).rolling(window=4).mean()
    df.dropna(inplace=True)
    return df

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if is_raw_data(df):
        st.info("📊 Raw Walmart dataset detected. Applying feature engineering...")
        df = engineer_features(df)
        input_df = df[['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'IsHoliday',
                       'Week', 'Year', 'Month', 'Day', 'Lag_1', 'Lag_2', 'Rolling_Mean_4']]
    elif is_ready_data(df):
        st.success("✅ Input matches expected feature set.")
        input_df = df
    else:
        st.error("❌ Uploaded file must either be raw Walmart data or contain all engineered features.")
        st.stop()

    st.subheader("📋 Input Preview")
    st.write(input_df.head())

    # Predict
    X_scaled = scaler_X.transform(input_df)
    y_pred_scaled = model.predict(X_scaled).reshape(-1, 1)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    st.subheader("📈 Forecast Results")
    result_df = pd.DataFrame({
        "Week #": np.arange(1, len(y_pred) + 1),
        "Predicted Weekly Sales": y_pred.flatten()
    })
    st.write(result_df)

    # Plot
    st.subheader("📊 Forecast Plot")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(result_df["Week #"], result_df["Predicted Weekly Sales"], marker='o', label='Forecast')
    ax.set_title("Forecasted Weekly Sales")
    ax.set_xlabel("Week")
    ax.set_ylabel("Sales")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

else:
    st.info("👈 Please upload a CSV file to get started.")
