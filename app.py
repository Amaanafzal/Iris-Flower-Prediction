
import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Iris Flower Prediction",
    page_icon="🌸"
)

# Load trained model
saved = joblib.load("iris_decision_tree.pkl")
model = saved["model"]
features = saved["features"]
class_names = saved["class_names"]

st.title("🌸 Iris Flower Prediction")
st.write("Decision Tree Machine Learning Model")

# User inputs
st.subheader("Enter Flower Measurements")

sepal_length = st.number_input(
    "Sepal Length (cm)", min_value=0.0, value=5.1
)

sepal_width = st.number_input(
    "Sepal Width (cm)", min_value=0.0, value=3.5
)

petal_length = st.number_input(
    "Petal Length (cm)", min_value=0.0, value=1.4
)

petal_width = st.number_input(
    "Petal Width (cm)", min_value=0.0, value=0.2
)

# Predict flower
if st.button("Predict Flower"):
    data = pd.DataFrame([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]], columns=features)

    prediction = model.predict(data)[0]
    flower = class_names[prediction]

    st.success(f"Predicted Flower: {flower.title()}")
