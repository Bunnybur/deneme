"""
Configuration file for Sensor Fault Detection System
Centralizes all file paths and parameters
"""
import os

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(DATA_DIR, 'models')
SRC_DIR = os.path.join(BASE_DIR, 'src')

# Data files
RAW_DATA_PATH = os.path.join(DATA_DIR, 'sensor-fault-detection.csv')
CLEANED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'sensor_data_cleaned.csv')
STANDARDIZED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'sensor_data_standardized.csv')

# Model files
MODEL_PATH = os.path.join(MODELS_DIR, 'isolation_forest_model.pkl')
SCALER_PATH = os.path.join(MODELS_DIR, 'standard_scaler.pkl')

# Model parameters
CONTAMINATION_RATE = 0.08  # Expected percentage of anomalies
RANDOM_STATE = 42  # For reproducibility
N_ESTIMATORS = 100  # Number of trees in Isolation Forest

# API configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
