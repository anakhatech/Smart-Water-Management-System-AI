import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load AI Model
# -----------------------------
model = joblib.load("water_leakage_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Smart Water Management AI",
    page_icon="💧",
    layout="wide"
)

# -----------------------------
# App Title
# -----------------------------
st.title("💧 Smart Water Management System")
st.subheader("AI-Powered Water Leakage Prediction")

st.write(
    "Enter the sensor values below to predict whether water leakage is detected."
)

# -----------------------------
# User Inputs
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    pressure = st.number_input("Pressure", value=65.0)
    flow_rate = st.number_input("Flow Rate", value=70.0)
    temperature = st.number_input("Temperature", value=100.0)

with col2:
    vibration = st.number_input("Vibration", value=3.0)
    rpm = st.number_input("RPM", value=2000.0)
    operational_hours = st.number_input("Operational Hours", value=3000)

with col3:
    latitude = st.number_input("Latitude", value=25.2)
    longitude = st.number_input("Longitude", value=55.2)

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("🔍 Predict Water Leakage"):

    # Create empty input with all trained features
    input_data = pd.DataFrame(0, index=[0], columns=feature_columns)

    # Add numerical values
    input_data["Pressure"] = pressure
    input_data["Flow_Rate"] = flow_rate
    input_data["Temperature"] = temperature
    input_data["Vibration"] = vibration
    input_data["RPM"] = rpm
    input_data["Operational_Hours"] = operational_hours
    input_data["Latitude"] = latitude
    input_data["Longitude"] = longitude

    # AI Prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.divider()

    if prediction == 1:
        st.error("🚨 ALERT: Water Leakage Detected!")
        st.write(f"Leakage Probability: {probability * 100:.1f}%")
    else:
        st.success("✅ No Water Leakage Detected.")
        st.write(f"Leakage Probability: {probability * 100:.1f}%")