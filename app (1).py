import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import time

# --- Page Config ---
st.set_page_config(
    page_title="🛢 Crude Oil Price Predictor",
    page_icon="⛽",
    layout="wide"
)

# --- Sidebar ---
st.sidebar.header("📂 Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Model Settings")
test_size = st.sidebar.slider("Test Data Size (%)", 10, 50, 20, 5)
run_model = st.sidebar.button("Run Prediction")

# --- Main Header ---
st.markdown("""
<h1 style='text-align: center; color: white;'>🛢 Crude Oil Price Predictor</h1>
<p style='text-align: center; font-size:18px; color: white;'>Predict oil prices based on demand, supply, and market features.</p>
""", unsafe_allow_html=True)

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # --- Tabs ---
    tabs = st.tabs(["📄 Data Overview", "⚙️ Model Training", "📊 Predictions"])

    # --- Tab 1: Data Overview ---
    with tabs[0]:
        st.subheader("Dataset Preview")
        st.dataframe(data.head())

        st.subheader("Dataset Statistics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", data.shape[0], "📈")
        col2.metric("Columns", data.shape[1], "🗂️")
        col3.metric("Missing Values", data.isnull().sum().sum(), "⚠️")

        st.markdown("### Column Types")
        st.write(data.dtypes)

    # --- Data Cleaning ---
    if 'Date' in data.columns:
        data['Date'] = data['Date'].astype(str).str[:10]
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        data['Date'] = data['Date'].map(pd.Timestamp.toordinal)

    # Encode categorical columns automatically
    for col in data.select_dtypes(include='object').columns:
        data[col] = pd.factorize(data[col])[0]

    # Fill missing values
    data.fillna(data.mean(), inplace=True)

    # --- Tab 2: Model Training ---
    with tabs[1]:
        st.subheader("⚡ Select Features and Target")
        all_columns = data.columns.tolist()
        target = st.selectbox("Select Target (Y)", all_columns, index=len(all_columns)-1)
        features = st.multiselect("Select Features (X)", [c for c in all_columns if c != target], default=[c for c in all_columns if c != target])

        if run_model and features and target:
            X = data[features]
            y = data[target]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

            # Progress bar while training
            st.info("Training model...")
            progress_text = st.empty()
            progress_bar = st.progress(0)
            for i in range(1, 101):
                time.sleep(0.01)
                progress_text.text(f"Training progress: {i}%")
                progress_bar.progress(i)

            # Fit model
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.success("✅ Model Trained Successfully!")
            st.markdown(f"### 🔹 Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
            st.markdown(f"### 🔹 R² Score: {r2_score(y_test, y_pred):.2f}")

    # --- Tab 3: Predictions ---
    with tabs[2]:
        if run_model and features and target:
            st.subheader("📊 Actual vs Predicted Prices")
            fig, ax = plt.subplots(figsize=(8,5))
            ax.scatter(y_test, y_pred, color='white', label='Predicted vs Actual')
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Fit')
            ax.set_xlabel("Actual Prices")
            ax.set_ylabel("Predicted Prices")
            ax.set_title("Actual vs Predicted Crude Oil Prices")
            ax.legend()
            st.pyplot(fig)

            st.subheader("Predictions Table")
            result_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
            st.dataframe(result_df.head(10))
else:
    st.warning("⚠️ Please upload a CSV file to get started.")
