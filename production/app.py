
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from io import BytesIO

# Load model and scalers
model = joblib.load("random_forest_model.joblib")
scaler_X = joblib.load("scaler_X.joblib")
scaler_y = joblib.load("scaler_y.joblib")

st.set_page_config(page_title="Forecast Based on Historical Data", layout="centered")
st.title("📈 Sales Forecasting Based on Historical Data")
st.markdown("Upload or enter historical weekly data and choose the forecast horizon. The model will use past values to forecast future weekly sales recursively.")

# Choose data input method
input_mode = st.sidebar.radio("Choose Input Mode:", ["📤 Upload CSV", "✍️ Manual Input"])

# Columns expected in historical data
hist_columns = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'IsHoliday', 'Weekly_Sales',
                'Week', 'Year', 'Month', 'Day']

df_hist = None

if input_mode == "📤 Upload CSV":
    uploaded_file = st.file_uploader("Upload historical data (at least 5 rows)", type=["csv"])
    if uploaded_file:
        try:
            df_hist = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded successfully.")
        except Exception as e:
            st.error(f"❌ Failed to read file: {e}")

elif input_mode == "✍️ Manual Input":
    st.info("Please enter at least 5 historical data rows to start forecasting.")
    num_rows = st.number_input("Number of historical rows to input", min_value=5, max_value=20, value=5)
    df_hist = pd.DataFrame(columns=hist_columns)
    for i in range(int(num_rows)):
        with st.expander(f"🗂 Row {i+1}"):
            row = {}
            for col in hist_columns:
                dtype = float if col != "IsHoliday" else int
                row[col] = st.number_input(f"{col} (Row {i+1})", value=0.0 if dtype==float else 0)
            df_hist.loc[i] = row

if df_hist is not None and len(df_hist) >= 5:
    forecast_horizon = st.slider("🔮 Forecast how many future weeks?", min_value=1, max_value=20, value=12)

    # Step 1: Feature Engineering
    df = df_hist.copy()
    df = df.sort_values(["Year", "Week"]).reset_index(drop=True)
    df["Lag_1"] = df["Weekly_Sales"].shift(1)
    df["Lag_2"] = df["Weekly_Sales"].shift(2)
    df["Rolling_Mean_4"] = df["Weekly_Sales"].shift(1).rolling(4).mean()
    df.dropna(inplace=True)

    forecast_results = []
    last_known = df.iloc[-1:].copy()

    for step in range(forecast_horizon):
        row = last_known.copy()

        # Shift week/month manually
        row["Week"] = row["Week"] + 1
        if row["Week"].values[0] > 52:
            row["Week"] = 1
            row["Year"] += 1
        if row["Month"].values[0] < 12:
            row["Month"] += 1

        row["Day"] = (row["Day"] + 7) % 28 + 1

        # Recalculate rolling & lags
        row["Lag_1"] = last_known["Weekly_Sales"].values[0]
        row["Lag_2"] = last_known["Lag_1"].values[0]
        hist_for_roll = df["Weekly_Sales"].tolist() + [r["Weekly_Sales"] for r in forecast_results[-4:]]
        row["Rolling_Mean_4"] = np.mean(hist_for_roll[-4:])

        # Remove Weekly_Sales and scale
        X_input = row.drop(columns=["Weekly_Sales"])
        X_scaled = scaler_X.transform(X_input)

        # Predict
        pred_scaled = model.predict(X_scaled).reshape(-1, 1)
        pred = scaler_y.inverse_transform(pred_scaled)[0][0]

        # Add prediction to results
        row["Weekly_Sales"] = pred
        forecast_results.append(row.iloc[0])
        last_known = row.copy()

    forecast_df = pd.DataFrame(forecast_results).reset_index(drop=True)

    st.subheader("📊 Forecast Results")
    st.dataframe(forecast_df[["Year", "Week", "Weekly_Sales"]].style.format("{:.2f}"))

    # Download button for CSV
    csv_download = forecast_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Forecast CSV", csv_download, "forecast_output.csv", "text/csv")

    # Plot forecast
    st.subheader("📈 Forecast Plot")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(forecast_df["Week"], forecast_df["Weekly_Sales"], marker='o', label="Forecast")
    ax.set_xlabel("Week")
    ax.set_ylabel("Weekly Sales")
    ax.set_title("Predicted Weekly Sales for Future Weeks")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    # Download plot
    img_bytes = BytesIO()
    fig.savefig(img_bytes, format='png')
    st.download_button("📷 Download Plot as Image", img_bytes.getvalue(), "forecast_plot.png", "image/png")
else:
    st.info("Please upload or enter at least 5 rows of historical data.")
