import pandas as pd
import numpy as np
import joblib
import os
import sys
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import STANDARDIZED_DATA_PATH, MODELS_DIR, RANDOM_STATE

def load_data_with_labels():
    print("\n" + "="*70)
    print("DATA LOADING")
    print("="*70)
    
    print("\n[1] Loading standardized data...")
    df = pd.read_csv(STANDARDIZED_DATA_PATH)
    print(f"    ✓ Loaded {len(df):,} records")
    
    print("\n[2] Generating labels (Temperature > 100°C = Fault)...")
    df['Label'] = (df['Value'] > 100).astype(int)
    
    n_normal = (df['Label'] == 0).sum()
    n_anomaly = (df['Label'] == 1).sum()
    
    print(f"    ✓ Labels generated:")
    print(f"      - Normal (0):  {n_normal:,} ({n_normal/len(df)*100:.2f}%)")
    print(f"      - Fault (1):   {n_anomaly:,} ({n_anomaly/len(df)*100:.2f}%)")
    
    return df

def train_isolation_forest(df):
    print("\n" + "="*70)
    print("ISOLATION FOREST TRAINING")
    print("="*70)
    
    print("\n[3] Preparing data...")
    X = df[['Value_Standardized']].values
    y_true = df['Label'].values
    
    contamination = (df['Label'] == 1).sum() / len(df)
    print(f"    ✓ Contamination rate: {contamination:.4f} ({contamination*100:.2f}%)")
    
    print("\n[4] Training Isolation Forest...")
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    model.fit(X)
    print("    ✓ Training complete")
    
    y_pred = model.predict(X)
    y_pred_binary = (y_pred == -1).astype(int)
    
    print("\n[5] Evaluation on full dataset...")
    accuracy = (y_pred_binary == y_true).sum() / len(y_true)
    print(f"    Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_binary, target_names=['Normal', 'Fault']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred_binary)
    print(f"                Predicted")
    print(f"                Normal  Fault")
    print(f"Actual Normal   {cm[0,0]:6d}  {cm[0,1]:7d}")
    print(f"       Fault    {cm[1,0]:6d}  {cm[1,1]:7d}")
    
    return model

def save_model(model):
    print("\n" + "="*70)
    print("SAVING MODEL")
    print("="*70)
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    filepath = os.path.join(MODELS_DIR, 'isolation_forest_model.pkl')
    joblib.dump(model, filepath)
    file_size = os.path.getsize(filepath) / 1024
    
    print(f"\n    ✓ Model saved:")
    print(f"      Path: {filepath}")
    print(f"      Size: {file_size:.2f} KB")

def main():
    df = load_data_with_labels()
    model = train_isolation_forest(df)
    save_model(model)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print("\n✓ Isolation Forest model trained successfully")
    print("✓ Model saved to:", MODELS_DIR)
    print("\nModel Type: Unsupervised Anomaly Detection")
    print("Output: -1 (anomaly), 1 (normal)")
    print("="*70)

if __name__ == "__main__":
    main()
