🚗 Vehicle Price Prediction Using Machine Learning

### 👨‍💻 Author: Achal Urs S

---

## 🧩 Project Overview
This project predicts **vehicle prices** using **machine learning** based on specifications such as make, model, year, mileage, fuel type, and drivetrain.  
It includes a trained ML model and a user-friendly **Streamlit web app** that allows users to interactively estimate vehicle prices.

---

## 🎯 Objective
To develop a system that accurately predicts the **price of a vehicle** using regression algorithms trained on real-world vehicle data.

---

## 🧠 Dataset Information
**File:** `dataset.csv`  
**Total Entries:** 1,002  
**Columns:** 17  

| Feature | Description |
|----------|--------------|
| make | Manufacturer (e.g., Toyota, Ford, BMW) |
| model | Model name |
| year | Year of manufacture |
| price | Vehicle price (Target variable) |
| mileage | Vehicle mileage (in miles) |
| cylinders | Number of cylinders |
| fuel | Fuel type (Gasoline, Diesel, Electric) |
| transmission | Transmission type |
| body | Body style (SUV, Sedan, Pickup Truck, etc.) |
| drivetrain | Type of drivetrain (FWD, RWD, AWD, etc.) |

---

## ⚙️ Technologies Used
- **Python**
- **Pandas**, **NumPy** – Data preprocessing  
- **scikit-learn** – Model training and evaluation  
- **Joblib** – Model saving and loading  
- **Streamlit** – Web application interface  

---

## 🧩 Project Structure

VehiclePricePrediction/ │ ├── dataset.csv ├── vehicle_price_train.py      # Model training script ├── app.py                      # Streamlit web app ├── model/ │   └── vehicle_price_model.joblib ├── report.txt                  # Full detailed report └── README.md                   # GitHub documentation

---

## 🔁 Project Workflow Diagram

```text
                 ┌────────────────────────┐
                 │     Dataset (CSV)      │
                 │ Vehicle specs & prices │
                 └──────────┬─────────────┘
                            │
                            ▼
              ┌───────────────────────────────┐
              │   Data Preprocessing           │
              │ - Handle missing values        │
              │ - Encode categorical data      │
              │ - Scale numeric features       │
              └──────────┬─────────────────────┘
                            │
                            ▼
              ┌───────────────────────────────┐
              │   Feature Engineering          │
              │ - Create 'age' from 'year'     │
              │ - Select important attributes  │
              └──────────┬─────────────────────┘
                            │
                            ▼
              ┌───────────────────────────────┐
              │  Model Training (RandomForest) │
              │ - Fit on 80% of dataset        │
              │ - Evaluate on 20%              │
              └──────────┬─────────────────────┘
                            │
                            ▼
              ┌───────────────────────────────┐
              │   Save Trained Model (.joblib) │
              └──────────┬─────────────────────┘
                            │
                            ▼
              ┌───────────────────────────────┐
              │  Streamlit Web App (app.py)    │
              │ - User inputs car details      │
              │ - Predicts vehicle price       │
              │ - Shows & saves history        │
              │ - Allows CSV download          │
              └───────────────────────────────┘


---

🚀 How to Run

1️⃣ Install Dependencies

pip install pandas numpy scikit-learn streamlit joblib

2️⃣ Train the Model

python vehicle_price_train.py

This creates the trained model file:

model/vehicle_price_model.joblib

3️⃣ Launch the Web App

streamlit run app.py

Then open the link shown in the terminal (usually http://localhost:8501).


---

🌐 Streamlit App Features

✅ Input form for vehicle specifications
✅ Instant price prediction
✅ “Previous Predictions” table
✅ “⬇️ Download as CSV” button to export history
✅ “🧹 Clear History” button to reset session


---

📊 Model Performance

Algorithm: Random Forest Regressor

Metrics:

RMSE ≈ 2000–3000

MAE ≈ 1500–2500

R² ≈ 0.85+




---

🧠 Future Improvements

Add XGBoost / LightGBM for better accuracy

Include NLP from vehicle description

Deploy app online (Streamlit Cloud / Render / AWS)



---

🧾 Author

Name: Achal Urs S
Project: Vehicle Price Prediction Using Machine Learning
Developed with: Python, scikit-learn, Streamlit

---
