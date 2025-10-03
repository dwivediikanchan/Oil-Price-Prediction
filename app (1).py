import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import time

# --- Page Config ---
st.set_page_config(
    page_title="🛢 Sidebar Custom Predictor",
    page_icon="⛽",
    layout="wide"
)

# --- Sidebar: File Upload & Settings ---
st.sidebar.header("📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Model Settings")
test_size = st.sidebar.slider("Test Data Size (%)", 10, 50, 20, 5)

# Initialize session state for model and features
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False
if "model" not in st.session_state:
    st.session_state.model = None
if "features" not in st.session_state:
    st.session_state.features = None
if "target" not in st.session_state:
    st.session_state.target = None

# --- Main Header ---
st.markdown("""
<h1 style='text-align: center; color: white;'>🛢 Oil Price Predictor</h1>
<p style='text-align: center; font-size:18px; color: gray;'>Predict oil prices with sidebar custom inputs and feature insights.</p>
""", unsafe_allow_html=True)

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # --- Data Cleaning ---
    if 'Date' in data.columns:
        data['Date'] = data['Date'].astype(str).str[:10]
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        data['Date'] = data['Date'].map(pd.Timestamp.toordinal)

    for col in data.select_dtypes(include='object').columns:
        data[col] = pd.factorize(data[col])[0]

    data.fillna(data.mean(), inplace=True)

    # --- Select Features & Target ---
    st.sidebar.subheader("⚡ Select Target and Features")
    all_columns = data.columns.tolist()
    st.session_state.target = st.sidebar.selectbox("Select Target (Y)", all_columns, index=len(all_columns)-1)
    st.session_state.features = st.sidebar.multiselect(
        "Select Features (X)", 
        [c for c in all_columns if c != st.session_state.target],
        default=[c for c in all_columns if c != st.session_state.target]
    )

    # --- Run Model ---
    if st.sidebar.button("Run Prediction"):
        if st.session_state.features and st.session_state.target:
            X = data[st.session_state.features]
            y = data[st.session_state.target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

            # Progress bar
            st.info("Training model...")
            progress_text = st.empty()
            progress_bar = st.progress(0)
            for i in range(1, 101):
                time.sleep(0.01)
                progress_text.text(f"Training progress: {i}%")
                progress_bar.progress(i)

            model = LinearRegression()
            model.fit(X_train, y_train)
            st.session_state.model = model
            st.session_state.model_trained = True
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.success("✅ Model Trained Successfully!")

    # --- Show Tabs Only After Training ---
    if st.session_state.model_trained:
        tabs = st.tabs(["📄 Data Overview", "📊 Predictions", "📊 Feature Importance"])

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

        # --- Tab 2: Predictions ---
        with tabs[1]:
            y_pred = st.session_state.model.predict(st.session_state.X_test)
            st.subheader("📊 Actual vs Predicted Prices")
            fig, ax = plt.subplots(figsize=(8,5))
            ax.scatter(st.session_state.y_test, y_pred, color='white', label='Predicted vs Actual')
            ax.plot([st.session_state.y_test.min(), st.session_state.y_test.max()],
                    [st.session_state.y_test.min(), st.session_state.y_test.max()],
                    'r--', label='Perfect Fit')
            ax.set_xlabel("Actual Prices")
            ax.set_ylabel("Predicted Prices")
            ax.set_title("Actual vs Predicted Crude Oil Prices")
            ax.legend()
            st.pyplot(fig)

            st.subheader("Predictions Table")
            result_df = pd.DataFrame({"Actual": st.session_state.y_test, "Predicted": y_pred})
            st.dataframe(result_df.head(10))

        # --- Tab 3: Feature Importance ---
        with tabs[2]:
            coef_df = pd.DataFrame({
                "Feature": st.session_state.features,
                "Coefficient": st.session_state.model.coef_
            })
            coef_df['Importance'] = np.abs(coef_df['Coefficient'])
            coef_df = coef_df.sort_values(by='Importance', ascending=False)

            st.subheader("📊 Feature Importance (Coefficient Magnitude)")
            fig, ax = plt.subplots(figsize=(8,5))
            sns.barplot(x='Importance', y='Feature', data=coef_df, palette="Oranges_r", ax=ax)
            ax.set_title("Feature Importance")
            st.pyplot(fig)

        # --- Sidebar Custom Prediction ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("📝 Predict Custom Price")
        custom_input_values = {}
        for feature in st.session_state.features:
            min_val = float(data[feature].min())
            max_val = float(data[feature].max())
            mean_val = float(data[feature].mean())
            custom_input_values[feature] = st.sidebar.number_input(
                f"{feature}", value=mean_val, min_value=min_val, max_value=max_val, key=f"sidebar_{feature}"
            )
        if st.sidebar.button("Predict Custom Price"):
            input_df = pd.DataFrame([custom_input_values])
            predicted_price = st.session_state.model.predict(input_df)[0]
            st.sidebar.success(f"🛢 Predicted Oil Price: **{predicted_price:.2f}**")

else:
    st.warning("⚠️ Please upload a CSV file to get started.")
