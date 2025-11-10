import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ------------------------------
# 🎨 Custom Background & Styles
# ------------------------------
st.markdown(
    """
    <style>
    /* App background gradient */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
        color: #333333;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Title style */
    .stTitle {
        color: #1f4e79;
        font-size: 38px;
        font-weight: bold;
    }

    /* Sidebar header style */
    .css-1d391kg h2 {
        color: #1f4e79;
    }

    /* Button hover effect */
    div.stButton > button:hover {
        background-color: #1f4e79;
        color: white;
    }

    /* Previous predictions table card */
    div[data-testid="stDataFrameContainer"] {
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 10px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------
# ✅ Load the trained model safely
# ------------------------------
model_path = os.path.join(os.path.dirname(__file__), "model", "vehicle_price_model.joblib")
model = joblib.load(model_path)
st.success("✅ Model Loaded Successfully – Ready to Predict!")

st.set_page_config(page_title="Vehicle Price Prediction", page_icon="🚗", layout="centered")

# ------------------------------
# 🎯 App Title and Description
# ------------------------------
st.title("🚗 Vehicle Price Prediction")
st.markdown("""
This app predicts **vehicle prices** based on their specifications using a trained Machine Learning model.  
Fill in the details below to estimate the price of your vehicle.
""")

# ------------------------------
# 🧾 Sidebar Information
# ------------------------------
st.sidebar.header("About")
st.sidebar.info(
    "Developed by **Achal Urs S**\n\n"
    "Built with Python, Scikit-learn, and Streamlit."
)

# ------------------------------
# 📋 User Input Section
# ------------------------------
st.header("Enter Vehicle Details")

make = st.text_input("🚘 Make (e.g., Toyota, Ford)")
model_name = st.text_input("🚙 Model (e.g., Corolla, Mustang)")
year = st.number_input("📅 Year", min_value=1990, max_value=2025, value=2018)
mileage = st.number_input("🛣 Mileage (in miles)", min_value=0, max_value=500000, value=50000)
fuel = st.selectbox("⛽ Fuel Type", ["Gasoline", "Diesel", "Electric", "Hybrid"])
transmission = st.selectbox("⚙️ Transmission", ["Automatic", "Manual"])
body = st.selectbox("🚗 Body Type", ["SUV", "Sedan", "Hatchback", "Truck", "Coupe", "Van"])
drivetrain = st.selectbox("🛞 Drivetrain", ["FWD", "RWD", "AWD", "4WD"])
cylinders = st.number_input("🔧 Cylinders", min_value=2, max_value=16, value=4)
doors = st.number_input("🚪 Doors", min_value=2, max_value=6, value=4)

# ------------------------------
# 📊 Predict Button
# ------------------------------
if st.button("🔮 Predict Vehicle Price"):
    # Create input DataFrame
    input_data = pd.DataFrame({
        "make": [make],
        "model": [model_name],
        "year": [year],
        "mileage": [mileage],
        "fuel": [fuel],
        "transmission": [transmission],
        "body": [body],
        "drivetrain": [drivetrain],
        "cylinders": [cylinders],
        "doors": [doors]
    })

    # Predict
    try:
        predicted_price = model.predict(input_data)[0]
        st.success(f"💰 Estimated Vehicle Price: **${predicted_price:,.2f}**")

        # Save prediction in session state
        if "history" not in st.session_state:
            st.session_state.history = []
        st.session_state.history.append({
            "Make": make,
            "Model": model_name,
            "Year": year,
            "Mileage": mileage,
            "Fuel": fuel,
            "Predicted Price ($)": round(predicted_price, 2)
        })

    except Exception as e:
        st.error("❌ Unable to predict. Please check your inputs.")
        st.error(str(e))

# ------------------------------
# 📈 Show Previous Predictions
# ------------------------------
st.subheader("📜 Previous Predictions")

if "history" in st.session_state and len(st.session_state.history) > 0:
    df_history = pd.DataFrame(st.session_state.history)
    st.dataframe(df_history)

    # Download as CSV
    csv = df_history.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download History as CSV", csv, "vehicle_predictions.csv", "text/csv")

    # Clear history
    if st.button("🧹 Clear History"):
        st.session_state.history = []
        st.experimental_rerun()
else:
    st.info("No previous predictions yet.")

# ------------------------------
# ✅ Footer
# ------------------------------
st.markdown("---")
st.caption("Developed by **Achal Urs S** | Vehicle Price Prediction ML App 🚗")
