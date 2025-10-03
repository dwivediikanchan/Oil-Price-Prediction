import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer # Import SimpleImputer

st.set_page_config(page_title="Oil Price Prediction", page_icon="🛢️", layout="wide")

st.title("🛢 Oil Price Prediction")
st.markdown("### Predict crude oil prices based on demand, supply, and other features.")

# -----------------------------
# Load Dataset (use your dataset)
# -----------------------------
data = pd.read_csv("/content/Crude oil.csv")  # <-- replace with your file name if different
st.subheader("📂 Dataset Preview")
st.dataframe(data.head())

# -----------------------------
# Preprocessing
# -----------------------------
# Select features & target

# Convert Date column to datetime
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

# Sort by date (important for time series data)
data = data.sort_values('Date')

# Drop the Date column for regression (not numeric)
X = data.drop(columns=['Close/Last', 'Date'])  # assuming "Close/Last" is target column
y = data['Close/Last']

# Handle missing values using imputation
imputer = SimpleImputer(strategy='mean') # You can change the strategy (e.g., 'median', 'most_frequent')
X = imputer.fit_transform(X)
# Convert back to DataFrame to keep column names for Streamlit input
X = pd.DataFrame(X, columns=data.drop(columns=['Close/Last', 'Date']).columns)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate
print("R² Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))


# -----------------------------
# User Input Prediction
# -----------------------------
st.sidebar.header("🔧 Enter Input Features")

input_data = {}
for col in X.columns:
    input_data[col] = st.sidebar.number_input(f"{col}", float(X[col].min()), float(X[col].max()), float(X[col].mean()))

if st.sidebar.button("Predict Price"):
    features = np.array([list(input_data.values())])
    prediction = model.predict(features)[0]
    st.success(f"💰 Predicted Oil Price: **${prediction:.2f}**")

# -----------------------------
# Bulk Prediction via CSV Upload
# -----------------------------
uploaded = st.file_uploader("📤 Upload a CSV for Bulk Prediction", type=["csv"])
if uploaded is not None:
    new_data = pd.read_csv(uploaded)
    # Apply the same imputer to the new data
    new_data_imputed = imputer.transform(new_data)
    new_data_imputed = pd.DataFrame(new_data_imputed, columns=new_data.columns)
    preds = model.predict(new_data_imputed)
    new_data["Predicted Price"] = preds
    st.dataframe(new_data)
    st.download_button("⬇ Download Predictions", new_data.to_csv(index=False), "oil_price_predictions.csv")
