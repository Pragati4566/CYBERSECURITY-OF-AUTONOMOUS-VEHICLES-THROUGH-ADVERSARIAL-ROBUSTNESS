#app.py is entry point
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from joblib import load #pretrained ml model saved in file is loaded

# Load the trained model and preprocessing pipeline
import os #to interacct wih the os
import tensorflow as tf

# Get current directory dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'pythonproject-main', 'cybersecurity_model.h5')
model = tf.keras.models.load_model(MODEL_PATH)



def generate_adversarial_examples(model, x, y_true, epsilon=0.1):
    x = tf.convert_to_tensor(x)
    y_true = tf.convert_to_tensor(y_true)
    
    # Ensure y_true is shaped correctly
    y_true = tf.reshape(y_true, (x.shape[0], 1))  # Batch dimension should match x
    
    with tf.GradientTape() as tape:
        tape.watch(x)
        predictions = model(x, training=False)
        loss = tf.keras.losses.binary_crossentropy(y_true, predictions)
        gradient = tape.gradient(loss, x)
        adversarial_example = x + epsilon * tf.sign(gradient)
        
    return adversarial_example

# Streamlit UI
st.header('Cybersecurity Threat Prediction')

# User inputs for the features
sensor_data = st.slider('Sensor Data', 0.0, 100.0)
vehicle_speed = st.slider('Vehicle Speed (in km/h)', 0, 200)
network_traffic = st.slider('Network Traffic (in MB)', 0.0, 1000.0)
sensor_type = st.selectbox('Sensor Type', ['Type 1', 'Type 2', 'Type 3'])
sensor_status = st.selectbox('Sensor Status', ['Active', 'Inactive', 'Error'])
vehicle_model = st.selectbox('Vehicle Model', ['Model A', 'Model B', 'Model C'])
firmware_version = st.selectbox('Firmware Version', ['v1.0', 'v2.0', 'v3.0'])
geofencing_status = st.selectbox('Geofencing Status', ['Enabled', 'Disabled'])

#now user press the button predict threat
if st.button("Predict Threat"):
    try:
        # Create a DataFrame for the input
        input_data = pd.DataFrame(
            [[sensor_data, vehicle_speed, network_traffic, sensor_type, sensor_status, vehicle_model, firmware_version, geofencing_status]],
            columns=['Sensor_Data', 'Vehicle_Speed', 'Network_Traffic', 'Sensor_Type', 'Sensor_Status', 'Vehicle_Model', 'Firmware_Version', 'Geofencing_Status']
        )
        
        # Preprocess the input data
        input_data_processed = preprocessor.transform(input_data) #converting the raw data in the numerical format
        
        # Ensure input_data_processed has shape (1, 16) for the model
        input_data_processed = np.reshape(input_data_processed, (1, -1))  # Reshape to (1, 16)
        
        # Assuming binary classification with a positive label
        y_true = np.array([1]) #assuming that this is attack
        
        # Generate adversarial example
        input_data_processed_adv = generate_adversarial_examples(model, input_data_processed, y_true)
        
        # Ensure the data is in batch format (reshape to (1, 16)) #this means converting to table format 
        input_data_processed_adv = np.reshape(input_data_processed_adv, (1, -1))
        
        # Make a prediction
        prediction = model.predict(input_data_processed_adv)
        
        # Display the result
        if prediction[0] > 0.5:
            st.markdown('### High Probability of Adversarial Attack')
        else:
            st.markdown('### Low Probability of Adversarial Attack')
    
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
#The model is already trained and validated on clean data. In app.py, Streamlit is used only for interaction, while adversarial examples
#are generated to test how robust the trained model is against manipulated or attacked inputs
#why we are predicting on the real world data this might be possible ki actual data thoda sa biased aaye due to attacker or hacker so we are calculationg for the nearby 
