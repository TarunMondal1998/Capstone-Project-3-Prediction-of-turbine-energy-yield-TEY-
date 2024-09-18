# Importing required libraries
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pickle

# Load the saved model
model = load_model('ann_model.h5')

# Load the scaler
scaler = pickle.load(open('scalar.pkl', 'rb'))

# Add a logo at the top of the app
st.image('logo.png', use_column_width=False)

# Streamlit App Title
st.title("Turbine Efficiency Prediction")

st.header("By Tarun Mondal")

# Create input fields for user to input the 8 features
st.header("Enter the ambient and turbine parameters")

AT = st.number_input("Ambient temperature (AT) C", min_value=0.0, max_value=50.0, value=6.85, step=0.01)
AP = st.number_input("Ambient pressure (AP) mbar", min_value=900.0, max_value=1100.0, value=1007.9, step=0.1)
AH = st.number_input("Ambient humidity (AH) (%)", min_value=0.0, max_value=100.0, value=96.8, step=0.1)
AFDP = st.number_input("Air filter difference pressure (AFDP) mbar", min_value=0.0, max_value=10.0, value=3.5, step=0.01)
GTEP = st.number_input("Gas turbine exhaust pressure (GTEP) mbar", min_value=10.0, max_value=30.0, value=19.7, step=0.01)
TAT = st.number_input("Turbine after temperature (TAT) C", min_value=400.0, max_value=600.0, value=550.0, step=0.1)
CO = st.number_input("Carbon monoxide (CO) mg/m3", min_value=0.0, max_value=10.0, value=3.15, step=0.01)
NOX = st.number_input("Nitrogen oxides (NOx) mg/m3", min_value=0.0, max_value=100.0, value=82.7, step=0.1)

# Store the user input in a numpy array for prediction
input_data = np.array([[AT, AP, AH, AFDP, GTEP, TAT, CO, NOX]])

# Scale the input data
scaled_input = scaler.transform(input_data)

# Predict using the trained model
if st.button('Predict Turbine Energy Yield'):
    prediction = model.predict(scaled_input)
    st.success(f'Turbine energy yield (TEY) is: {np.exp(prediction[0][0]):.2f} MWH')