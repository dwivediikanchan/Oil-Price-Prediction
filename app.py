import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ---------------------------
# Load or Train Model
# ---------------------------


data = pd.read_csv("C:\Users\HP\Downloads\Crude oil.csv")   # <-- Replace with your dataset file
X = data[['Open', 'High', 'Low', 'Volume', 'MA7', 'MA30', 'Return1', 'Lag1', 'Lag7']]   # your input features
y = data['Close/Last']

model = LinearRegression()
model.fit(X, y)

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Oil Price Prediction", page_icon="🛢️", layout="wide")

st.title("🛢️ Oil Price Prediction App")
st.markdown("### Powered by Linear Regression")

col1, col2 = st.columns(2)

with col1:
    st.image("https://images.unsplash.com/photo-1581091215367-59abcb02c38e", caption="Global Oil Industry", use_container_width=True)

with col2:
    st.write("This app predicts **oil prices** based on key factors like demand, supply, and temperature.")
    st.info("Upload your dataset or enter inputs manually to predict oil price.")

# ---------------------------
# Sidebar for Inputs
# ---------------------------
st.sidebar.header("🔧 Input Features")

temp = st.sidebar.number_input("Temperature", value=30.0)
demand = st.sidebar.number_input("Global Demand Index", value=100.0)
supply = st.sidebar.number_input("Supply Index", value=50.0)

if st.sidebar.button("Predict Price"):
    features = np.array([[temp, demand, supply]])
    prediction = model.predict(features)[0]
    st.success(f"💰 Predicted Oil Price: **${prediction:.2f}**")

    # Visualization
    fig, ax = plt.subplots()
    ax.scatter(y, model.predict(X), alpha=0.5)
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    ax.set_title("Model Performance")
    st.pyplot(fig)

# ---------------------------
# Upload Data for Bulk Prediction
# ---------------------------
uploaded = st.file_uploader("📂 Upload a CSV file for bulk predictions", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    preds = model.predict(df)
    df["Predicted Price"] = preds
    st.dataframe(df)
    st.download_button("⬇️ Download Predictions", df.to_csv(index=False), "predictions.csv")

