import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load saved model and scalers
model = joblib.load('random_forest_model.joblib')
scaler_X = joblib.load('scaler_X.joblib')
scaler_y = joblib.load('scaler_y.joblib')

st.set_page_config(page_title="Dept 26 Sales Forecast", layout="wide")
st.title("📈 Walmart Dept 26 - Weekly Sales Forecast (Random Forest)")

st.sidebar.header("Upload Input CSV")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file with future features", type=["csv"])

if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file)

        # Feature check
        expected_cols = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'IsHoliday',
                         'Week', 'Year', 'Month', 'Day', 'Lag_1', 'Lag_2', 'Rolling_Mean_4']

        if not all(col in input_df.columns for col in expected_cols):
            st.error("Uploaded file is missing required features.")
        else:
            # Scale and predict
            X_input_scaled = scaler_X.transform(input_df[expected_cols])
            pred_scaled = model.predict(X_input_scaled).reshape(-1, 1)
            pred = scaler_y.inverse_transform(pred_scaled)

            input_df['Predicted_Weekly_Sales'] = pred

            # Display forecast
            st.subheader("🔮 Forecast Results")
            st.dataframe(input_df[['Year', 'Week', 'Predicted_Weekly_Sales']].style.format("{:.2f}"))

            # Plot
            st.subheader("📊 Forecast Plot")
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(input_df.index, pred, marker='o', label='Forecast')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Weekly Sales')
            ax.set_title('Forecasted Weekly Sales')
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Awaiting CSV upload with future features to forecast sales.")
