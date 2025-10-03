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
    page_title=" Oil Predictor",
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
<h1 style='text-align: center; color: white;'>🛢 Oil Price Predictor</h1>
<p style='text-align: center; font-size:18px; color: gray;'>Predict oil prices with feature importance insights and custom input predictions.</p>
""", unsafe_allow_html=True)

# --- Main App Logic ---
if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # --- Tabs ---
    tabs = st.tabs(["📄 Data Overview", "⚙️ Model Training", "📊 Predictions", "📝 Custom Predictions", "📊 Feature Importance"])

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

    for col in data.select_dtypes(include='object').columns:
        data[col] = pd.factorize(data[col])[0]

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

    # --- Tab 4: Custom Predictions ---
    with tabs[3]:
        if run_model and features and target:
            st.subheader("📝 Predict Using Custom Feature Values")
            n_inputs = st.number_input("How many predictions to make?", min_value=1, max_value=10, value=1, step=1)

            custom_inputs = []
            for i in range(n_inputs):
                st.markdown(f"### Input {i+1}")
                input_dict = {}
                for feature in features:
                    min_val = float(data[feature].min())
                    max_val = float(data[feature].max())
                    mean_val = float(data[feature].mean())
                    input_dict[feature] = st.number_input(
                        f"{feature} (Input {i+1})",
                        value=mean_val,
                        min_value=min_val,
                        max_value=max_val,
                        key=f"{feature}_{i}"
                    )
                custom_inputs.append(input_dict)

            if st.button("Predict Custom Prices"):
                input_df = pd.DataFrame(custom_inputs)
                predicted_prices = model.predict(input_df)
                st.success("🛢 Predicted Prices:")
                for i, price in enumerate(predicted_prices):
                    st.write(f"Prediction {i+1}: **{price:.2f}**")

    # --- Tab 5: Feature Importance ---
    with tabs[4]:
        if run_model and features and target:
            st.subheader("📊 Feature Importance (Coefficient Magnitude)")
            coef_df = pd.DataFrame({
                "Feature": features,
                "Coefficient": model.coef_
            })
            coef_df['Importance'] = np.abs(coef_df['Coefficient'])
            coef_df = coef_df.sort_values(by='Importance', ascending=False)

            fig, ax = plt.subplots(figsize=(8,5))
            sns.barplot(x='Importance', y='Feature', data=coef_df, palette="Oranges_r", ax=ax)
            ax.set_title("Feature Importance")
            st.pyplot(fig)

else:
    st.warning("⚠️ Please upload a CSV file to get started.")
