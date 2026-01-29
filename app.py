# app.py is entry point
import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import os

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'pythonproject-main', 'cybersecurity_model.joblib')
model = load(MODEL_PATH)

# Simple noise-based robustness check
def add_noise(x, noise_level=0.05):
    noise = np.random.normal(0, noise_level, x.shape)
    return x + noise

# Streamlit UI
st.header('Cybersecurity Threat Prediction')
sensor_data = st.slider('Sensor Data', 0.0, 100.0)
vehicle_speed = st.slider('Vehicle Speed (in km/h)', 0, 200)
network_traffic = st.slider('Network Traffic (in MB)', 0.0, 1000.0)
sensor_type = st.selectbox('Sensor Type', ['Type 1', 'Type 2', 'Type 3'])
sensor_status = st.selectbox('Sensor Status', ['Active', 'Inactive', 'Error'])
vehicle_model = st.selectbox('Vehicle Model', ['Model A', 'Model B', 'Model C'])
firmware_version = st.selectbox('Firmware Version', ['v1.0', 'v2.0', 'v3.0'])
geofencing_status = st.selectbox('Geofencing Status', ['Enabled', 'Disabled'])

if st.button("Predict Threat"):
    try:
        input_data = pd.DataFrame(
            [[sensor_data, vehicle_speed, network_traffic,
              sensor_type, sensor_status, vehicle_model,
              firmware_version, geofencing_status]],
            columns=[
                'Sensor_Data', 'Vehicle_Speed', 'Network_Traffic',
                'Sensor_Type', 'Sensor_Status', 'Vehicle_Model',
                'Firmware_Version', 'Geofencing_Status'
            ]
        )

        input_data_processed = preprocessor.transform(input_data) # raw data to machine format 
        input_data_processed = np.reshape(input_data_processed, (1, -1)) # to convert into tabular format 
        #samples =total rows,features= total comulns

        # Robustness test using noisy input
        noisy_input = add_noise(input_data_processed)

        prediction = model.predict(noisy_input)

        if prediction[0] == 1:
            st.markdown('### High Probability of Cybersecurity Threat')
        else:
            st.markdown('### Low Probability of Cybersecurity Threat')

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

#The model is already trained and validated on clean data. In app.py, Streamlit is used only for interaction, while adversarial examples
#are generated to test how robust the trained model is against manipulated or attacked inputs
#why we are predicting on the real world data this might be possible ki actual data thoda sa biased aaye due to attacker or hacker so we are calculationg for the nearby 
