import pandas as pd
import numpy as np
import os
import sys
from tensorflow import keras
from keras import layers
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import STANDARDIZED_DATA_PATH, MODELS_DIR, RANDOM_STATE

np.random.seed(RANDOM_STATE)

def load_data():
    print("\n" + "="*70)
    print("DATA LOADING")
    print("="*70)
    
    print("\n[1] Loading standardized data...")
    df = pd.read_csv(STANDARDIZED_DATA_PATH)
    print(f"    ✓ Loaded {len(df):,} records")
    
    print("\n[2] Filtering normal data for training...")
    normal_data = df[df['Value'] <= 100].copy()
    print(f"    ✓ Normal data: {len(normal_data):,} records")
    
    return normal_data, df

def build_autoencoder(input_dim=1):
    print("\n" + "="*70)
    print("BUILDING AUTOENCODER")
    print("="*70)
    
    print("\n[3] Creating neural network architecture...")
    
    encoder = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(8, activation='relu'),
        layers.Dense(4, activation='relu'),
        layers.Dense(2, activation='relu')
    ], name='encoder')
    
    decoder = keras.Sequential([
        layers.Dense(4, activation='relu'),
        layers.Dense(8, activation='relu'),
        layers.Dense(input_dim, activation='linear')
    ], name='decoder')
    
    autoencoder = keras.Sequential([encoder, decoder], name='autoencoder')
    
    autoencoder.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    print("    ✓ Architecture:")
    print(f"      Input → 8 → 4 → 2 → 4 → 8 → Output")
    autoencoder.summary()
    
    return autoencoder

def train_autoencoder(model, normal_data):
    print("\n" + "="*70)
    print("TRAINING AUTOENCODER")
    print("="*70)
    
    print("\n[4] Preparing training data...")
    X_train = normal_data[['Value_Standardized']].values
    
    print(f"    ✓ Training samples: {len(X_train):,}")
    
    print("\n[5] Training...")
    history = model.fit(
        X_train, X_train,
        epochs=50,
        batch_size=256,
        validation_split=0.1,
        verbose=1,
        shuffle=True
    )
    
    print("\n    ✓ Training complete")
    final_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    print(f"    Final loss: {final_loss:.6f}")
    print(f"    Final val_loss: {final_val_loss:.6f}")
    
    return model, X_train

def calculate_threshold(model, X_train):
    print("\n[6] Calculating anomaly threshold...")
    
    reconstructions = model.predict(X_train, verbose=0)
    reconstruction_errors = np.mean(np.abs(X_train - reconstructions), axis=1)
    
    threshold = np.percentile(reconstruction_errors, 95)
    
    print(f"    ✓ Reconstruction error statistics:")
    print(f"      Mean: {np.mean(reconstruction_errors):.6f}")
    print(f"      Std:  {np.std(reconstruction_errors):.6f}")
    print(f"      95th percentile threshold: {threshold:.6f}")
    
    return threshold

def evaluate_autoencoder(model, df, threshold):
    print("\n[7] Evaluating on full dataset...")
    
    X_all = df[['Value_Standardized']].values
    y_true = (df['Value'] > 100).astype(int).values
    
    reconstructions = model.predict(X_all, verbose=0)
    reconstruction_errors = np.mean(np.abs(X_all - reconstructions), axis=1)
    
    y_pred = (reconstruction_errors > threshold).astype(int)
    
    accuracy = (y_pred == y_true).sum() / len(y_true)
    print(f"    Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    from sklearn.metrics import classification_report, confusion_matrix
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Fault']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(f"                Predicted")
    print(f"                Normal  Fault")
    print(f"Actual Normal   {cm[0,0]:6d}  {cm[0,1]:7d}")
    print(f"       Fault    {cm[1,0]:6d}  {cm[1,1]:7d}")

def save_model(model, threshold):
    print("\n" + "="*70)
    print("SAVING MODEL")
    print("="*70)
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    model_path = os.path.join(MODELS_DIR, 'autoencoder_model.keras')
    model.save(model_path)
    model_size = os.path.getsize(model_path) / 1024
    
    threshold_path = os.path.join(MODELS_DIR, 'autoencoder_threshold.pkl')
    joblib.dump(threshold, threshold_path)
    
    print(f"\n    ✓ Model saved:")
    print(f"      Path: {model_path}")
    print(f"      Size: {model_size:.2f} KB")
    print(f"    ✓ Threshold saved:")
    print(f"      Path: {threshold_path}")
    print(f"      Value: {threshold:.6f}")

def main():
    normal_data, full_df = load_data()
    model = build_autoencoder()
    model, X_train = train_autoencoder(model, normal_data)
    threshold = calculate_threshold(model, X_train)
    evaluate_autoencoder(model, full_df, threshold)
    save_model(model, threshold)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print("\n✓ Autoencoder model trained successfully")
    print("✓ Model saved to:", MODELS_DIR)
    print("\nModel Type: Deep Learning Anomaly Detection")
    print("Method: Reconstruction error threshold")
    print("="*70)

if __name__ == "__main__":
    main()
