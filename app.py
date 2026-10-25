import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Set page configuration
st.set_page_config(
    page_title="Solar Power Forecasting",
    page_icon="☀️",
    layout="wide"
)

# Load trained model and scaler
@st.cache_resource
def load_assets():
    model = joblib.load('best_solar_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
    assets_loaded = True
except Exception as e:
    assets_loaded = False

# Header Section
st.title("☀️ Solar Power Generation Output Forecaster")
st.markdown("""
This interactive system predicts real-time **AC Power Output (in kW)** for a solar plant using ambient temperature, module temperature, and solar irradiation metrics.
""")

st.divider()

if not assets_loaded:
    st.error("⚠️ Model or Scaler file not found. Please place `best_solar_model.pkl` and `scaler.pkl` in the same directory.")
else:
    # Input Layout (2 Columns)
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("⚙️ Environmental Sensor Inputs")
        
        amb_temp = st.slider(
            "Ambient Temperature (°C)", 
            min_value=15.0, 
            max_value=45.0, 
            value=28.5, 
            step=0.1
        )
        
        mod_temp = st.slider(
            "Module Temperature (°C)", 
            min_value=15.0, 
            max_value=70.0, 
            value=35.0, 
            step=0.1
        )
        
        irradiation = st.slider(
            "Solar Irradiation (kW/m²)", 
            min_value=0.01, 
            max_value=1.50, 
            value=0.65, 
            step=0.01
        )

    with col2:
        st.subheader("📈 Predicted Energy Output")
        
        # Preprocess inputs using fitted scaler
        input_data = pd.DataFrame([[amb_temp, mod_temp, irradiation]], 
                                  columns=['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION'])
        input_scaled = scaler.transform(input_data)
        
        # Predict AC Power Output
        predicted_power = model.predict(input_scaled)[0]
        predicted_power = max(0, float(predicted_power))  # Ensure non-negative
        
        # Display Metric
        st.metric(
            label="Forecasted AC Power Output", 
            value=f"{predicted_power:.2f} kW"
        )
        
        # Capacity Gauge Level
        max_capacity = 1400.0  # Peak capacity benchmark for plant
        capacity_percentage = min(100.0, (predicted_power / max_capacity) * 100)
        
        st.write(f"**Plant Operating Capacity:** {capacity_percentage:.1f}%")
        st.progress(capacity_percentage / 100.0)
        
        if capacity_percentage > 75:
            st.success("🔥 High Generation State: Optimal solar irradiance conditions.")
        elif capacity_percentage > 35:
            st.info("🌤️ Moderate Generation State: Standard operating conditions.")
        else:
            st.warning("☁️ Low Generation State: Low irradiance or sub-optimal weather conditions.")

st.divider()
st.caption("Mini Project — Solar Power Output Forecasting Pipeline")