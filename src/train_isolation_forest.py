"""
Isolation Forest Model Training & Evaluation
=============================================
Train an unsupervised anomaly detection model using Isolation Forest.

Tasks:
1. Load standardized data
2. Perform 80/20 train-test split
3. Train Isolation Forest model
4. Evaluate on test set
5. Analyze anomaly scores
6. Generate performance summary

Author: Advanced Computer Programming Course
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import sys
import os
import joblib

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import STANDARDIZED_DATA_PATH

def load_data():
    """Load the standardized sensor data."""
    print("="*70)
    print("ISOLATION FOREST - ANOMALY DETECTION MODEL")
    print("="*70)
    print("\n[1] Loading standardized data...")
    
    try:
        df = pd.read_csv(STANDARDIZED_DATA_PATH)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        print(f"    ✓ Loaded {len(df):,} records")
        print(f"    ✓ Features: {list(df.columns)}")
        return df
    except FileNotFoundError:
        print(f"    ✗ Error: Data not found at {STANDARDIZED_DATA_PATH}")
        sys.exit(1)

def prepare_data(df):
    """Prepare features and create ground truth labels."""
    print("\n[2] Preparing features and labels...")
    
    # Use standardized values as feature
    X = df[['Value_Standardized']].values
    
    # Create ground truth labels based on domain knowledge
    # Anomalies: values outside expected range (0-60°C) or statistical outliers
    Q1 = df['Value'].quantile(0.25)
    Q3 = df['Value'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Label: 1 = normal, -1 = anomaly (to match Isolation Forest output)
    y_true = np.ones(len(df))
    y_true[(df['Value'] < lower_bound) | (df['Value'] > upper_bound)] = -1
    
    anomaly_count = (y_true == -1).sum()
    normal_count = (y_true == 1).sum()
    
    print(f"    ✓ Feature shape: {X.shape}")
    print(f"    ✓ Ground truth labels created:")
    print(f"      - Normal readings:  {normal_count:,} ({normal_count/len(df)*100:.2f}%)")
    print(f"      - Anomalies (IQR):  {anomaly_count:,} ({anomaly_count/len(df)*100:.2f}%)")
    
    return X, y_true, df

def split_data(X, y_true, df, test_size=0.2, random_state=42):
    """Perform train-test split."""
    print(f"\n[3] Performing {int((1-test_size)*100)}/{int(test_size*100)} train-test split...")
    
    # Split while preserving temporal information
    indices = np.arange(len(X))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y_true, indices, test_size=test_size, random_state=random_state, stratify=y_true
    )
    
    print(f"    ✓ Training set:   {len(X_train):,} samples")
    print(f"      - Normal:       {(y_train == 1).sum():,}")
    print(f"      - Anomalies:    {(y_train == -1).sum():,}")
    print(f"    ✓ Test set:       {len(X_test):,} samples")
    print(f"      - Normal:       {(y_test == 1).sum():,}")
    print(f"      - Anomalies:    {(y_test == -1).sum():,}")
    
    return X_train, X_test, y_train, y_test, idx_train, idx_test

def train_model(X_train, contamination='auto'):
    """Train Isolation Forest model."""
    print(f"\n[4] Training Isolation Forest model...")
    print(f"    • Contamination: {contamination}")
    print(f"    • Estimators: 100")
    print(f"    • Random state: 42")
    
    # Initialize and train model
    model = IsolationForest(
        contamination=contamination,
        n_estimators=100,
        random_state=42,
        max_samples='auto',
        n_jobs=-1,
        verbose=0
    )
    
    print(f"    • Training on {len(X_train):,} samples...")
    model.fit(X_train)
    
    print(f"    ✓ Model trained successfully!")
    
    return model

def evaluate_model(model, X_test, y_test, df, idx_test):
    """Evaluate model performance on test set."""
    print(f"\n[5] Evaluating model on test set...")
    
    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Get anomaly scores (lower = more anomalous)
    anomaly_scores = model.score_samples(X_test)
    
    # Decision function (same as score_samples but with opposite sign)
    decision_scores = model.decision_function(X_test)
    
    print(f"    ✓ Predictions completed")
    print(f"      - Predicted Normal:    {(y_pred == 1).sum():,}")
    print(f"      - Predicted Anomalies: {(y_pred == -1).sum():,}")
    
    # Calculate metrics
    print(f"\n[6] Performance Metrics:")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=[1, -1])
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n    Confusion Matrix:")
    print(f"    ┌─────────────────┬──────────────────┐")
    print(f"    │                 │  Predicted       │")
    print(f"    │                 │ Normal │ Anomaly │")
    print(f"    ├─────────────────┼────────┼─────────┤")
    print(f"    │ Actual Normal   │ {tn:6d} │ {fp:7d} │")
    print(f"    │ Actual Anomaly  │ {fn:6d} │ {tp:7d} │")
    print(f"    └─────────────────┴────────┴─────────┘")
    
    # Calculate performance metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n    Performance Scores:")
    print(f"    • Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"    • Precision: {precision:.4f} (of predicted anomalies, {precision*100:.2f}% are true)")
    print(f"    • Recall:    {recall:.4f} (detected {recall*100:.2f}% of actual anomalies)")
    print(f"    • F1-Score:  {f1:.4f}")
    
    # Anomaly Score Analysis
    print(f"\n[7] Anomaly Score Distribution:")
    print(f"    • Mean score:   {anomaly_scores.mean():.4f}")
    print(f"    • Std dev:      {anomaly_scores.std():.4f}")
    print(f"    • Min (most anomalous): {anomaly_scores.min():.4f}")
    print(f"    • Max (most normal):    {anomaly_scores.max():.4f}")
    
    # Scores for normal vs anomalous
    normal_scores = anomaly_scores[y_test == 1]
    anomaly_scores_true = anomaly_scores[y_test == -1]
    
    print(f"\n    Score Statistics by True Label:")
    print(f"    • Normal readings:  {normal_scores.mean():.4f} ± {normal_scores.std():.4f}")
    print(f"    • True anomalies:   {anomaly_scores_true.mean():.4f} ± {anomaly_scores_true.std():.4f}")
    
    return y_pred, anomaly_scores, decision_scores, cm

def visualize_results(df, idx_test, y_test, y_pred, anomaly_scores):
    """Visualize model predictions and anomaly scores."""
    print(f"\n[8] Generating visualizations...")
    
    # Create test dataframe
    df_test = df.iloc[idx_test].copy()
    df_test['Predicted'] = y_pred
    df_test['Anomaly_Score'] = anomaly_scores
    df_test['True_Label'] = y_test
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Isolation Forest - Model Evaluation Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Confusion Matrix Heatmap
    ax1 = axes[0, 0]
    cm = confusion_matrix(y_test, y_pred, labels=[1, -1])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
    ax1.set_ylabel('True Label', fontsize=11)
    ax1.set_xlabel('Predicted Label', fontsize=11)
    ax1.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
    
    # Plot 2: Anomaly Score Distribution
    ax2 = axes[0, 1]
    normal_scores = anomaly_scores[y_test == 1]
    anomaly_scores_true = anomaly_scores[y_test == -1]
    
    ax2.hist(normal_scores, bins=50, alpha=0.6, color='green', label='True Normal', edgecolor='black')
    ax2.hist(anomaly_scores_true, bins=50, alpha=0.6, color='red', label='True Anomaly', edgecolor='black')
    ax2.axvline(normal_scores.mean(), color='green', linestyle='--', linewidth=2, label='Normal Mean')
    ax2.axvline(anomaly_scores_true.mean(), color='red', linestyle='--', linewidth=2, label='Anomaly Mean')
    ax2.set_xlabel('Anomaly Score (lower = more anomalous)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Anomaly Score Distribution by True Label', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Time Series with Predictions
    ax3 = axes[1, 0]
    df_test_sorted = df_test.sort_values('Timestamp')
    
    # Plot correct predictions
    correct_normal = df_test_sorted[(df_test_sorted['True_Label'] == 1) & (df_test_sorted['Predicted'] == 1)]
    correct_anomaly = df_test_sorted[(df_test_sorted['True_Label'] == -1) & (df_test_sorted['Predicted'] == -1)]
    
    ax3.scatter(correct_normal['Timestamp'], correct_normal['Value'], 
               color='green', s=10, alpha=0.4, label='True Positive (Normal)')
    ax3.scatter(correct_anomaly['Timestamp'], correct_anomaly['Value'], 
               color='red', s=30, alpha=0.7, label='True Positive (Anomaly)')
    
    # Plot incorrect predictions
    false_positive = df_test_sorted[(df_test_sorted['True_Label'] == 1) & (df_test_sorted['Predicted'] == -1)]
    false_negative = df_test_sorted[(df_test_sorted['True_Label'] == -1) & (df_test_sorted['Predicted'] == 1)]
    
    ax3.scatter(false_positive['Timestamp'], false_positive['Value'], 
               color='orange', s=30, alpha=0.7, marker='x', label='False Positive')
    ax3.scatter(false_negative['Timestamp'], false_negative['Value'], 
               color='purple', s=30, alpha=0.7, marker='x', label='False Negative')
    
    ax3.set_xlabel('Timestamp', fontsize=11)
    ax3.set_ylabel('Temperature (°C)', fontsize=11)
    ax3.set_title('Test Set Predictions Over Time', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Anomaly Score vs Temperature
    ax4 = axes[1, 1]
    
    normal_data = df_test[df_test['True_Label'] == 1]
    anomaly_data = df_test[df_test['True_Label'] == -1]
    
    ax4.scatter(normal_data['Value'], normal_data['Anomaly_Score'], 
               color='green', s=20, alpha=0.4, label='True Normal')
    ax4.scatter(anomaly_data['Value'], anomaly_data['Anomaly_Score'], 
               color='red', s=40, alpha=0.7, label='True Anomaly')
    
    ax4.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax4.set_xlabel('Temperature (°C)', fontsize=11)
    ax4.set_ylabel('Anomaly Score', fontsize=11)
    ax4.set_title('Anomaly Score vs Temperature Value', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    print(f"    ✓ Visualizations generated successfully")
    print(f"    ✓ Close the plot window to continue...")
    plt.show()

def save_model(model, output_path='data/isolation_forest_model.pkl'):
    """Save trained model to disk."""
    print(f"\n[9] Saving model...")
    try:
        joblib.dump(model, output_path)
        print(f"    ✓ Model saved to: {output_path}")
    except Exception as e:
        print(f"    ✗ Error saving model: {e}")

def generate_summary(cm, X_train, X_test, y_test, y_pred, anomaly_scores):
    """Generate final performance summary."""
    print("\n" + "="*70)
    print("MODEL TRAINING & EVALUATION SUMMARY")
    print("="*70)
    
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n📊 DATASET SPLIT:")
    print(f"   • Training samples:  {len(X_train):,} (80%)")
    print(f"   • Test samples:      {len(X_test):,} (20%)")
    print(f"   • Total samples:     {len(X_train) + len(X_test):,}")
    
    print(f"\n🤖 MODEL CONFIGURATION:")
    print(f"   • Algorithm:         Isolation Forest")
    print(f"   • Contamination:     auto")
    print(f"   • Estimators:        100 trees")
    print(f"   • Features:          1 (Standardized Temperature)")
    
    print(f"\n📈 TEST SET PERFORMANCE:")
    print(f"   • Accuracy:          {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   • Precision:         {precision:.4f}")
    print(f"   • Recall:            {recall:.4f}")
    print(f"   • F1-Score:          {f1:.4f}")
    
    print(f"\n🎯 PREDICTION BREAKDOWN:")
    print(f"   • True Positives:    {tp:,} (Anomalies correctly identified)")
    print(f"   • True Negatives:    {tn:,} (Normal correctly identified)")
    print(f"   • False Positives:   {fp:,} (Normal flagged as anomaly)")
    print(f"   • False Negatives:   {fn:,} (Anomalies missed)")
    
    print(f"\n🔍 ANOMALY SCORE INSIGHTS:")
    normal_scores = anomaly_scores[y_test == 1]
    anomaly_scores_true = anomaly_scores[y_test == -1]
    print(f"   • Normal readings:   {normal_scores.mean():.4f} ± {normal_scores.std():.4f}")
    print(f"   • True anomalies:    {anomaly_scores_true.mean():.4f} ± {anomaly_scores_true.std():.4f}")
    print(f"   • Score separation:  {abs(normal_scores.mean() - anomaly_scores_true.mean()):.4f}")
    
    print(f"\n✅ KEY FINDINGS:")
    if accuracy > 0.95:
        print(f"   • Model shows excellent performance (>95% accuracy)")
    elif accuracy > 0.85:
        print(f"   • Model shows good performance (>85% accuracy)")
    else:
        print(f"   • Model performance could be improved")
    
    if recall > 0.80:
        print(f"   • High recall: detecting {recall*100:.1f}% of anomalies")
    else:
        print(f"   • Moderate recall: some anomalies might be missed")
    
    if precision > 0.80:
        print(f"   • High precision: {precision*100:.1f}% of flagged anomalies are correct")
    else:
        print(f"   • Lower precision: some false alarms expected")
    
    print(f"\n💡 RECOMMENDATIONS:")
    if fp > tp:
        print(f"   • Consider adjusting contamination parameter to reduce false positives")
    if fn > 0:
        print(f"   • {fn} anomalies were missed - review threshold settings")
    print(f"   • Use anomaly scores for ranking alerts by severity")
    print(f"   • Monitor model performance over time for drift")
    
    print("\n" + "="*70)
    print("✓ Model training and evaluation complete!")
    print("="*70)

def main():
    """Main execution function."""
    # Load data
    df = load_data()
    
    # Prepare features and labels
    X, y_true, df = prepare_data(df)
    
    # Split data (80/20)
    X_train, X_test, y_train, y_test, idx_train, idx_test = split_data(X, y_true, df)
    
    # Train model
    model = train_model(X_train, contamination='auto')
    
    # Evaluate model
    y_pred, anomaly_scores, decision_scores, cm = evaluate_model(
        model, X_test, y_test, df, idx_test
    )
    
    # Visualize results
    visualize_results(df, idx_test, y_test, y_pred, anomaly_scores)
    
    # Save model
    save_model(model)
    
    # Generate summary
    generate_summary(cm, X_train, X_test, y_test, y_pred, anomaly_scores)

if __name__ == "__main__":
    main()
