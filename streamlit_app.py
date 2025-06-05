import streamlit as st
import joblib
import numpy as np

model = joblib.load("elm_water_model.pkl")

st.title("Water Potability Prediction")

fields = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
inputs = []

for field in fields:
    val = st.number_input(f"Enter {field}", min_value=0.0)
    inputs.append(val)

if st.button("Predict"):
    result = model.predict([inputs])
    st.write("Prediction:", "Potable" if result[0] == 1 else "Not Potable")