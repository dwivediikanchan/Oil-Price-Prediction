import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer # Import SimpleImputer

st.set_page_config(page_title="Oil Price Prediction", page_icon="🛢️", layout="wide")

st.title("🛢 Oil Price Prediction using Linear Regression")
st.markdown("### Predict crude oil prices based on demand, supply, and other features.")

# -----------------------------
# Load Dataset (use your dataset)
# -----------------------------
data = pd.read_csv("Crude oil.csv")  # <-- replace with your file name if different
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


import streamlit as st
import matlplotlib as plt
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# --- Page Configuration ---
st.set_page_config(
    page_title="🛢 Oil Price Predictor",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Sidebar ---
st.sidebar.header("📂 Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Model Settings")
test_size = st.sidebar.slider("Test Data Size (%)", min_value=10, max_value=50, value=20, step=5)
run_model = st.sidebar.button("Run Prediction")

# --- Main Title ---
st.markdown("""
    <h1 style='text-align: center; color: #FF6F00;'>🛢 Crude Oil Price Prediction</h1>
    <p style='text-align: center; font-size:18px; color: gray;'>Predict oil prices using demand, supply, and market data.</p>
""", unsafe_allow_html=True)

# --- Load Data ---
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("✅ Dataset Loaded Successfully!")
    st.dataframe(data.head())

    # --- Feature Selection ---
    st.markdown("### ⚡ Select Features and Target")
    all_columns = data.columns.tolist()
    features = st.multiselect("Features (X)", all_columns, default=all_columns[:-1])
    target = st.selectbox("Target (Y)", all_columns, index=len(all_columns)-1)

    if run_model and features and target:
        X = data[features]
        y = data[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        st.markdown("### 📊 Predictions vs Actual")
        fig = px.scatter(x=y_test, y=y_pred, labels={'x':'Actual', 'y':'Predicted'}, title="Actual vs Predicted Oil Prices")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"### 🔹 Mean Squared Error: {mse:.2f}")

        # Optionally show predictions
        result_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
        st.dataframe(result_df.head())

else:
    st.warning("⚠️ Please upload a CSV file to get started.")



