import pandas as pd
import numpy as np
import joblib
import streamlit as st

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Water Pollutant Predictor",
    page_icon="💧",
    layout="centered"
)

# ---------- BACKGROUND + CSS ----------
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://sensorex.com/wp-content/uploads/2021/09/splashing-165192_1280.jpg");
        background-size: cover;
        background-attachment: fixed;
    }
    .block-container {
        backdrop-filter: blur(8px);
        background-color: rgba(0,0,0);
        border-radius: 10px;
        padding: 0.5rem;
        margin-top: 1rem;
    }
    .stButton > button {
        background-color: #007acc;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.6em 1.2em;
    }
    .pollutant-box {
        padding: 0.5em 1em;
        background-color: #e3f2fd;
        border-radius: 8px;
        font-weight: bold;
        color: #000000;
        font-size: 1.1em;
        margin-bottom: 0.5em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- CONTAINER ----------
with st.container():
    st.markdown('<div class="block-container">', unsafe_allow_html=True)
    st.title("Welcome to the Aqua Monitor")
    st.markdown("### 💧 Water Pollutant Predictor")
    st.markdown("### Predict the quality of water based on **Year** and **Station ID**")
    st.markdown("This tool uses a machine learning model to forecast pollutants like **O₂, NO₃, SO₄, PO₄, CL**.")

    st.markdown("---")

    # ---------- INPUTS ----------
    col1, col2 = st.columns(2)
    with col1:
        year_input = st.number_input("📅 Enter Year", min_value=2000, max_value=2100, value=2004)

    with col2:
        station_id = st.text_input("🏢 Enter Station ID", value='1')

    # ---------- LOAD MODEL ----------
    model = joblib.load("pollution_model.pkl")
    model_cols = joblib.load("model_columns.pkl")

    # ---------- PREDICT ----------
    if st.button("🔮 Predict"):
        if not station_id:
            st.warning("⚠️ Please enter the Station ID")
        else:
            input_df = pd.DataFrame({'year': [year_input], 'id':[station_id]})
            input_encoded = pd.get_dummies(input_df, columns=['id'])

            for col in model_cols:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
            input_encoded = input_encoded[model_cols]

            predicted_pollutants = model.predict(input_encoded)[0]
            pollutants = ['O2', 'NO3', 'SO4', 'PO4', 'CL']

            st.markdown("### ✅ Predicted Pollutant Levels")
            for p, val in zip(pollutants, predicted_pollutants):
                st.markdown(f'<div class="pollutant-box">{p} : {val:.2f}</div>', unsafe_allow_html=True)

    # ---------- CONTACT & CREDITS ----------
    st.markdown("---")
    with st.expander("📬 Contact Developer"):
        st.markdown("**Name:** Sowjanya Thatavarthi  \n**Email:** sowjanyathatavarthi2601@gmail.com  \n**LinkedIn:** [linkedin.com/in/sowjanyathatavarthi](https://www.linkedin.com/in/sowjanya-thatavarthi-254a882bb)")
    st.markdown(" ")
    st.markdown("### ℹ️ About This Project")
    st.markdown("""
    **Water Quality Prediction – RMS**  
    This project uses machine learning (Random Forest Regressor + MultiOutputRegressor) to predict water pollution levels from real datasets. Developed under the **AICTE Virtual Internship**, supported by **Shell** and **Edunet Foundation**, it helps monitor water safety across various Indian stations.

    🛡️ Supported by **Ministry of Education Innovation Cell (MIC)**  
    """)

    st.markdown('</div>', unsafe_allow_html=True)