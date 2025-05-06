import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title('Calories Burnt Prediction')

# Load the model
model = pickle.load(open('calories_burnt_model.pkl', 'rb'))  # Change path if needed

# Input fields
gender = st.selectbox('Gender', ['male', 'female'])
age = st.number_input('Age', min_value=10, max_value=80, value=30)
height = st.number_input('Height (cm)', min_value=100.0, max_value=220.0, value=170.0)
weight = st.number_input('Weight (kg)', min_value=40.0, max_value=150.0, value=70.0)
duration = st.number_input('Duration (minutes)', min_value=1.0, max_value=180.0, value=30.0)
heart_rate = st.number_input('Heart Rate (bpm)', min_value=60.0, max_value=180.0, value=100.0)
body_temp = st.number_input('Body Temperature (C)', min_value=36.0, max_value=42.0, value=37.0)

# Preprocess gender
gender = 1 if gender == 'male' else 0

# Prediction button
if st.button('Predict Calories'):
    input_data = np.array([[gender, age, height, weight, duration, heart_rate, body_temp]])
    prediction = model.predict(input_data)[0]
    st.success(f'Predicted Calories Burnt: {prediction:.2f}')