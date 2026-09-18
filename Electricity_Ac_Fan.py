import streamlit as st
import pandas as pd
import joblib

from pathlib import Path


st.set_page_config(
    page_title="Electric Bill Prediction",
    page_icon="⚡",
    layout="centered"
)


MODEL_PATH = (
    Path(__file__).resolve().parent
    / "polynomial_regression_ac_fan.pkl"
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()


st.title("⚡ Electric Bill Prediction")

st.write("Polynomial Regression Model")

st.write(
    "Predict the Electric Bill using AC Units and Fan Units"
)


st.divider()


ac_units = st.number_input(
    "Enter AC Units",
    min_value=0.0,
    value=10.0,
    step=1.0
)


fan_units = st.number_input(
    "Enter Fan Units",
    min_value=0.0,
    value=20.0,
    step=1.0
)


if st.button("Predict Electric Bill", type="primary"):

    new_data = pd.DataFrame({
        "AC_Units": [ac_units],
        "Fan_Units": [fan_units]
    })


    prediction = model.predict(new_data)[0]


    prediction = max(0, prediction)


    st.success(
        f"Predicted Electric Bill: ₹{prediction:,.2f}"
    )
