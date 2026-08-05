import time
import numpy as np
import pandas as pd
from joblib import load
import os

# Load trained model and preprocessor
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'cybersecurity_random_forest_model.joblib')
PREPROCESSOR_PATH = os.path.join(BASE_DIR, 'preprocessing_pipeline.joblib')

model = load(MODEL_PATH)
preprocessor = load(PREPROCESSOR_PATH)

# Load real dataset from CSV
csv_path = os.path.join(BASE_DIR, 'cybersecurity_data_4000_rows.csv')
data = pd.read_csv(csv_path)

# Select feature columns (same as training)
features = [
    'Sensor_Data', 'Vehicle_Speed', 'Network_Traffic',
    'Sensor_Type', 'Sensor_Status', 'Vehicle_Model',
    'Firmware_Version', 'Geofencing_Status'
]

X_real = data[features]

# Preprocess the dataset
X_processed = preprocessor.transform(X_real)

# Number of iterations for timing
n_iterations = 100
times = []

# Measure prediction latency on full dataset
for _ in range(n_iterations):
    start_time = time.time()
    _ = model.predict(X_processed)
    end_time = time.time()

    response_time = end_time - start_time
    times.append(response_time)

# Calculate average response time
average_response_time = np.mean(times)

print(f"Average prediction time over {n_iterations} runs: {average_response_time:.6f} seconds")
