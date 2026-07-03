


import streamlit as st
import pandas as pd
import joblib

# Load trained pipeline
import os



model = joblib.load("complete_pipeline.pkl")


st.title("🚢 Titanic Survival Prediction App")

st.write("Enter passenger details to predict survival:")

# User Inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 25)
fare = st.number_input("Fare", min_value=0.0, value=30.0)
embarked = st.selectbox("Embarked", ["C", "Q", "S"])

# Convert input into DataFrame
input_data = pd.DataFrame({
    "Pclass": [pclass],
    "Sex": [sex],
    "Age": [age],
    "Fare": [fare],
    "Embarked": [embarked]
})

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ The passenger is likely to SURVIVE")
    else:
        st.error("❌ The passenger is NOT likely to survive")