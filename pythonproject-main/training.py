from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from joblib import dump

# Load and preprocess data
csv_file_path = '/Users/apple/Library/Mobile Documents/com~apple~CloudDocs/cybersecurity_data_50_rows.csv'
data = pd.read_csv(csv_file_path)

# Clean numerical columns
data['Vehicle_Speed'] = data['Vehicle_Speed'].str.extract('(\d+\.?\d*)').astype(float)
data['Network_Traffic'] = data['Network_Traffic'].str.extract('(\d+\.?\d*)').astype(float)

# Define features and label
features = ['Sensor_Data', 'Vehicle_Speed', 'Network_Traffic', 'Sensor_Type', 'Sensor_Status', 
            'Vehicle_Model', 'Firmware_Version', 'Geofencing_Status']
label = 'Adversarial_Attack'

X = data[features]
y = data[label]

# Update feature types
categorical_features = ['Sensor_Type', 'Sensor_Status', 'Vehicle_Model', 'Firmware_Version', 'Geofencing_Status']
numerical_features = ['Sensor_Data', 'Vehicle_Speed', 'Network_Traffic']

# Preprocessing pipelines
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
X_processed = pipeline.fit_transform(X)

# Save preprocessing pipeline
dump(pipeline, 'preprocessing_pipeline.joblib')

# Convert labels to numeric
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_processed, y_encoded, test_size=0.2, random_state=42)


# Initialize Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42) #decision tree number=estimator

# Train model
model.fit(X_train, y_train)

# Evaluate model
accuracy = model.score(X_test, y_test)
print(f"Random Forest accuracy: {accuracy:.4f}")

# Save the trained model
dump(model, 'cybersecurity_random_forest_model.joblib')
print("Random Forest model saved as cybersecurity_random_forest_model.joblib")
