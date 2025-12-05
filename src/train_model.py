"""
Model Training Script
=====================
Step 4 of the data processing pipeline.

Tasks:
1. Load standardized data
2. Train Isolation Forest model for anomaly detection
3. Evaluate model performance
4. Save model artifacts for API use

Author: Advanced Computer Programming Course
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (STANDARDIZED_DATA_PATH, MODEL_PATH, SCALER_PATH,
                        CONTAMINATION_RATE, RANDOM_STATE, N_ESTIMATORS)

def load_data():
    """Load standardized data for training."""
    print("="*70)
    print("STEP 4: MODEL TRAINING")
    print("="*70)
    
    print("\n[4.1] Loading standardized data...")
    try:
        df = pd.read_csv(STANDARDIZED_DATA_PATH)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        print(f"    ✓ Loaded {len(df):,} records")
        return df
    except FileNotFoundError:
        print(f"    ✗ Error: Standardized data not found")
        print("    → Run the data pipeline first")
        sys.exit(1)

def prepare_features(df):
    """Prepare features for model training."""
    print("\n[4.2] Preparing features...")
    
    # Use standardized values for training
    X = df[['Value_Standardized']].values
    
    print(f"    ✓ Feature shape: {X.shape}")
    print(f"    ✓ Features: Value_Standardized (z-scores)")
    print(f"    ✓ Mean: {X.mean():.6f}")
    print(f"    ✓ Std: {X.std():.6f}")
    
    return X, df

def train_isolation_forest(X):
    """Train Isolation Forest model."""
    print("\n[4.3] Training Isolation Forest...")
    print(f"    • Algorithm: Isolation Forest")
    print(f"    • Contamination: {CONTAMINATION_RATE} ({CONTAMINATION_RATE*100:.1f}%)")
    print(f"    • n_estimators: {N_ESTIMATORS}")
    print(f"    • Random state: {RANDOM_STATE}")
    
    model = IsolationForest(
        contamination=CONTAMINATION_RATE,
        random_state=RANDOM_STATE,
        n_estimators=N_ESTIMATORS,
        max_samples='auto',
        verbose=0
    )
    
    predictions = model.fit_predict(X)
    anomaly_scores = model.score_samples(X)
    
    print(f"    ✓ Model trained successfully")
    
    return model, predictions, anomaly_scores

def evaluate_model(df, predictions, anomaly_scores):
    """Evaluate model performance."""
    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70)
    
    n_normal = np.sum(predictions == 1)
    n_anomaly = np.sum(predictions == -1)
    total = len(predictions)
    
    print("\n[4.4] Prediction Summary:")
    print(f"    • Total Samples:  {total:,}")
    print(f"    • Normal (1):     {n_normal:,} ({n_normal/total*100:.2f}%)")
    print(f"    • Anomaly (-1):   {n_anomaly:,} ({n_anomaly/total*100:.2f}%)")
    
    # Add predictions to dataframe
    df_eval = df.copy()
    df_eval['Prediction'] = predictions
    df_eval['AnomalyScore'] = anomaly_scores
    
    # Check extreme fault detection
    print("\n[4.5] Anomaly Analysis:")
    anomalies = df_eval[df_eval['Prediction'] == -1]
    
    if len(anomalies) > 0:
        print(f"    • Anomaly Range:  {anomalies['Value'].min():.2f}°C - {anomalies['Value'].max():.2f}°C")
        print(f"    • Anomaly Mean:   {anomalies['Value'].mean():.2f}°C")
        print(f"    • Normal Mean:    {df_eval[df_eval['Prediction']==1]['Value'].mean():.2f}°C")
        
        extreme_caught = anomalies[anomalies['Value'] > 100]
        total_extreme = df_eval[df_eval['Value'] > 100]
        
        if len(total_extreme) > 0:
            print(f"\n    • Extreme Values (>100°C):")
            print(f"      Detected: {len(extreme_caught)}/{len(total_extreme)} ({len(extreme_caught)/len(total_extreme)*100:.1f}%)")
    
    # Top anomalies
    print("\n[4.6] Top 10 Most Anomalous Readings:")
    top_anomalies = df_eval.nsmallest(10, 'AnomalyScore')
    print("\n    Timestamp                    Value(°C)  Score    Status")
    print("    " + "-"*65)
    for idx, row in top_anomalies.iterrows():
        status = "FAULT" if row['Prediction'] == -1 else "Normal"
        print(f"    {row['Timestamp']}  {row['Value']:7.2f}   {row['AnomalyScore']:6.3f}   {status}")
    
    return df_eval

def save_model(model):
    """Save trained model."""
    print("\n" + "="*70)
    print("SAVING MODEL")
    print("="*70)
    
    print(f"\n[4.7] Saving model artifacts...")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    # Save model
    joblib.dump(model, MODEL_PATH)
    model_size = os.path.getsize(MODEL_PATH) / 1024  # KB
    print(f"    ✓ Model saved: {MODEL_PATH}")
    print(f"    ✓ File size: {model_size:.2f} KB")
    
    # Note about scaler
    print(f"\n    Note: Scaler already saved during standardization step")
    print(f"    ✓ Scaler location: {SCALER_PATH}")

def test_prediction_function(model):
    """Test the prediction function with sample values."""
    print("\n" + "="*70)
    print("TESTING PREDICTION FUNCTION")
    print("="*70)
    
    print("\n[4.8] Loading scaler for predictions...")
    try:
        scaler = joblib.load(SCALER_PATH)
        print(f"    ✓ Scaler loaded")
    except:
        print(f"    ✗ Could not load scaler")
        return
    
    def predict_fault(value):
        """Predict if a value is anomalous."""
        value_scaled = scaler.transform([[value]])
        prediction = model.predict(value_scaled)[0]
        return prediction
    
    print("\n    Test Predictions:")
    print("\n    Value(°C)  Prediction  Status")
    print("    " + "-"*40)
    
    test_values = [20.0, 25.5, 35.0, 60.0, 80.0, 120.0, 150.0]
    for val in test_values:
        pred = predict_fault(val)
        status = "⚠️ FAULT" if pred == -1 else "✓ Normal"
        print(f"    {val:7.1f}    {pred:4d}       {status}")

def main():
    """Main execution function."""
    # Load data
    df = load_data()
    
    # Prepare features
    X, df = prepare_features(df)
    
    # Train model
    model, predictions, anomaly_scores = train_isolation_forest(X)
    
    # Evaluate model
    df_eval = evaluate_model(df, predictions, anomaly_scores)
    
    # Save model
    save_model(model)
    
    # Test predictions
    test_prediction_function(model)
    
    print("\n" + "="*70)
    print("MODEL TRAINING COMPLETE")
    print("="*70)
    print("\n✓ Model trained and saved successfully")
    print("✓ Ready for deployment in API")
    print("\nNext Step:")
    print("  → Run 'python src/main.py' to start the FastAPI server")
    print("="*70)

if __name__ == "__main__":
    main()
