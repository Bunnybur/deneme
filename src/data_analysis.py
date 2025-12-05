"""
Data Analysis and Visualization Script
=======================================
Step 3 of the data processing pipeline.

Tasks:
1. Load standardized data
2. Perform statistical analysis
3. Detect outliers using IQR method
4. Visualize time series with anomaly highlighting
5. Generate insights for model training

Author: Advanced Computer Programming Course
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import STANDARDIZED_DATA_PATH

def load_standardized_data():
    """Load the standardized sensor data."""
    print("="*70)
    print("STEP 3: DATA ANALYSIS & VISUALIZATION")
    print("="*70)
    
    print("\n[3.1] Loading standardized data...")
    try:
        df = pd.read_csv(STANDARDIZED_DATA_PATH)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        print(f"    ✓ Loaded {len(df):,} records")
        print(f"    ✓ Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
        return df
    except FileNotFoundError:
        print(f"    ✗ Error: Standardized data not found")
        print("    → Run the data pipeline first:")
        print("       1. python src/data_clean.py")
        print("       2. python src/data_standardization.py")
        sys.exit(1)

def statistical_analysis(df):
    """Perform comprehensive statistical analysis."""
    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS")
    print("="*70)
    
    # Original values
    values = df['Value']
    
    print("\n[3.2] Descriptive Statistics (Original Values):")
    print(f"    • Count:          {len(values):,}")
    print(f"    • Mean:           {values.mean():.2f}°C")
    print(f"    • Std Dev:        {values.std():.2f}°C")
    print(f"    • Min:            {values.min():.2f}°C")
    print(f"    • 25th Percentile: {values.quantile(0.25):.2f}°C")
    print(f"    • Median:         {values.median():.2f}°C")
    print(f"    • 75th Percentile: {values.quantile(0.75):.2f}°C")
    print(f"    • Max:            {values.max():.2f}°C")
    
    # IQR-based outlier detection
    print("\n[3.3] Outlier Detection (IQR Method):")
    Q1 = values.quantile(0.25)
    Q3 = values.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers_iqr = df[(values < lower_bound) | (values > upper_bound)]
    print(f"    • IQR:            {IQR:.2f}°C")
    print(f"    • Lower Bound:    {lower_bound:.2f}°C")
    print(f"    • Upper Bound:    {upper_bound:.2f}°C")
    print(f"    • IQR Outliers:   {len(outliers_iqr):,} ({len(outliers_iqr)/len(df)*100:.2f}%)")
    
    # Extreme value detection
    print("\n[3.4] Extreme Value Detection:")
    extreme_faults = df[df['Value'] > 100]
    print(f"    • Values >100°C:  {len(extreme_faults):,} ({len(extreme_faults)/len(df)*100:.2f}%)")
    
    if len(extreme_faults) > 0:
        print(f"    • Extreme Range:  {extreme_faults['Value'].min():.2f}°C - {extreme_faults['Value'].max():.2f}°C")
    
    # Domain analysis
    print("\n[3.5] Domain Analysis (Industrial HVAC):")
    normal_range = df[(df['Value'] >= 0) & (df['Value'] <= 60)]
    print(f"    • Expected Range: 0-60°C")
    print(f"    • Within Range:   {len(normal_range):,} ({len(normal_range)/len(df)*100:.2f}%)")
    print(f"    • Outside Range:  {len(df)-len(normal_range):,} ({(len(df)-len(normal_range))/len(df)*100:.2f}%)")
    
    return outliers_iqr, extreme_faults

def visualize_time_series(df, outliers_iqr, extreme_faults):
    """Create comprehensive time series visualizations."""
    print("\n" + "="*70)
    print("VISUALIZATION")
    print("="*70)
    print("\n[3.6] Generating time series plots...")
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle('PT100 Temperature Sensor - Comprehensive Analysis', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Complete time series with anomalies
    ax1 = axes[0]
    ax1.plot(df['Timestamp'], df['Value'], 
             linewidth=0.8, alpha=0.6, color='steelblue', label='All Readings')
    
    if len(extreme_faults) > 0:
        ax1.scatter(extreme_faults['Timestamp'], extreme_faults['Value'],
                   color='red', s=40, alpha=0.8, label='Extreme Faults (>100°C)', zorder=5)
    
    iqr_only = outliers_iqr[outliers_iqr['Value'] <= 100]
    if len(iqr_only) > 0:
        ax1.scatter(iqr_only['Timestamp'], iqr_only['Value'],
                   color='orange', s=25, alpha=0.7, label='IQR Outliers', zorder=4)
    
    ax1.set_ylabel('Temperature (°C)', fontsize=10)
    ax1.set_title('Complete Time Series with Anomaly Detection', fontsize=11, pad=10)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=100, color='red', linestyle='--', linewidth=1, alpha=0.4)
    
    # Plot 2: Normal operating range (zoomed)
    ax2 = axes[1]
    normal_data = df[df['Value'] <= 60]
    ax2.plot(normal_data['Timestamp'], normal_data['Value'],
             linewidth=0.8, color='green', alpha=0.7, label='Normal Range (0-60°C)')
    
    iqr_normal = outliers_iqr[outliers_iqr['Value'] <= 60]
    if len(iqr_normal) > 0:
        ax2.scatter(iqr_normal['Timestamp'], iqr_normal['Value'],
                   color='orange', s=30, alpha=0.8, label='Outliers', zorder=4)
    
    ax2.set_ylabel('Temperature (°C)', fontsize=10)
    ax2.set_title('Normal Operating Range (Zoomed View)', fontsize=11, pad=10)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 60])
    
    # Plot 3: Standardized values
    ax3 = axes[2]
    ax3.plot(df['Timestamp'], df['Value_Standardized'],
             linewidth=0.8, color='purple', alpha=0.7, label='Standardized Values')
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Mean (0)')
    ax3.axhline(y=3, color='red', linestyle=':', linewidth=1, alpha=0.5, label='+3σ')
    ax3.axhline(y=-3, color='red', linestyle=':', linewidth=1, alpha=0.5, label='-3σ')
    
    ax3.set_ylabel('Standardized Value (z-score)', fontsize=10)
    ax3.set_xlabel('Timestamp', fontsize=10)
    ax3.set_title('Standardized Values (After Scaling)', fontsize=11, pad=10)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    print("    ✓ Plots generated successfully")
    print("\n    Close the plot window to continue...")
    plt.show()

def main():
    """Main execution function."""
    # Load data
    df = load_standardized_data()
    
    # Statistical analysis
    outliers_iqr, extreme_faults = statistical_analysis(df)
    
    # Visualization
    visualize_time_series(df, outliers_iqr, extreme_faults)
    
    print("\n" + "="*70)
    print("DATA ANALYSIS COMPLETE")
    print("="*70)
    print(f"\n✓ Dataset analyzed: {len(df):,} records")
    print(f"✓ Outliers identified: {len(outliers_iqr):,}")
    print(f"✓ Extreme faults: {len(extreme_faults):,}")
    print("\nNext Step:")
    print("  → Run 'python src/train_model.py' to train the ML model")
    print("="*70)

if __name__ == "__main__":
    main()
