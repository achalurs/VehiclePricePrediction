# app.py
# Streamlit app for Vehicle Price Prediction (with History + CSV Download)
import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("model/vehicle_price_model.joblib")

st.set_page_config(page_title="Vehicle Price Prediction", page_icon="🚗", layout="centered")

st.title("🚗 Vehicle Price Prediction App")
st.write("Enter vehicle details below to predict its price.")

# --- Initialize session state for storing predictions ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- User Inputs ---
make = st.text_input("Make (e.g., Toyota, Ford, BMW)")
model_name = st.text_input("Model (e.g., Camry, F-150, X5)")
year = st.number_input("Year", min_value=1990, max_value=2025, value=2024)
mileage = st.number_input("Mileage (in miles)", min_value=0, value=10)
cylinders = st.number_input("Cylinders", min_value=2, max_value=12, value=4)
fuel = st.selectbox("Fuel Type", ["Gasoline", "Diesel", "Electric", "Hybrid", "Other"])
transmission = st.selectbox("Transmission", ["Automatic", "Manual", "Other"])
body = st.selectbox("Body Type", ["SUV", "Sedan", "Pickup Truck", "Hatchback", "Other"])
doors = st.selectbox("Doors", [2, 3, 4, 5])
drivetrain = st.selectbox("Drivetrain", [
    "Front-wheel Drive", "Rear-wheel Drive", "All-wheel Drive", "Four-wheel Drive", "Other"
])

# Derived feature
age = 2025 - year

# Prepare data for prediction
input_data = pd.DataFrame([{
    "make": make,
    "model": model_name,
    "year": year,
    "age": age,
    "mileage": mileage,
    "cylinders": cylinders,
    "fuel": fuel,
    "transmission": transmission,
    "body": body,
    "doors": doors,
    "drivetrain": drivetrain
}])

# --- Prediction Button ---
if st.button("🔮 Predict Price"):
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Estimated Vehicle Price: **${prediction:,.2f}**")

    # Add prediction to session state
    record = input_data.copy()
    record["Predicted Price ($)"] = round(prediction, 2)
    st.session_state.history.append(record)

# --- Display Prediction History ---
if st.session_state.history:
    st.markdown("### 📋 Previous Predictions")
    history_df = pd.concat(st.session_state.history, ignore_index=True)
    st.dataframe(history_df)

    # Download button
    csv = history_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Predictions as CSV",
        data=csv,
        file_name="vehicle_price_predictions.csv",
        mime="text/csv"
    )

    # Clear button
    if st.button("🧹 Clear History"):
        st.session_state.history = []
        st.success("Previous predictions cleared!")

st.markdown("---")
st.caption("Developed by Achal Urs S — Vehicle Price Prediction Project")