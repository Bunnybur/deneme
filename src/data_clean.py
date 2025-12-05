"""
Data Cleaning Script
====================
Step 1 of the data processing pipeline.

Tasks:
1. Load raw sensor data
2. Drop SensorId column (constant value of 1)
3. Identify and remove null values
4. Save cleaned data for next pipeline stage

Author: Advanced Computer Programming Course
"""

import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import RAW_DATA_PATH, CLEANED_DATA_PATH

def load_raw_data():
    """Load the raw sensor data from CSV file."""
    print("="*70)
    print("STEP 1: DATA CLEANING")
    print("="*70)
    
    print("\n[1.1] Loading raw data...")
    try:
        df = pd.read_csv(RAW_DATA_PATH, delimiter=';')
        print(f"    ✓ Loaded {len(df):,} records")
        print(f"    ✓ Columns: {list(df.columns)}")
        return df
    except FileNotFoundError:
        print(f"    ✗ Error: File not found at {RAW_DATA_PATH}")
        print("    → Please ensure sensor-fault-detection.csv is in the data/ folder")
        sys.exit(1)

def inspect_data(df):
    """Inspect data for quality issues."""
    print("\n[1.2] Data Quality Inspection...")
    
    # Check data types
    print("\n    Data Types:")
    for col in df.columns:
        print(f"      • {col}: {df[col].dtype}")
    
    # Check for null values
    print("\n    Null Values:")
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    
    if total_nulls == 0:
        print("      ✓ No null values found")
    else:
        for col in df.columns:
            if null_counts[col] > 0:
                pct = (null_counts[col] / len(df)) * 100
                print(f"      ⚠️  {col}: {null_counts[col]:,} nulls ({pct:.2f}%)")
    
    # Check SensorId uniqueness
    print("\n    SensorId Analysis:")
    unique_sensors = df['SensorId'].nunique()
    print(f"      • Unique SensorId values: {unique_sensors}")
    print(f"      • SensorId value(s): {df['SensorId'].unique()}")
    
    if unique_sensors == 1:
        print(f"      → SensorId is constant ({df['SensorId'].iloc[0]}), can be dropped")
    
    return null_counts

def drop_sensor_id(df):
    """Drop the SensorId column since it's always 1."""
    print("\n[1.3] Dropping SensorId column...")
    
    original_cols = list(df.columns)
    df_cleaned = df.drop('SensorId', axis=1)
    
    print(f"    ✓ Dropped SensorId column")
    print(f"    ✓ Remaining columns: {list(df_cleaned.columns)}")
    
    return df_cleaned

def handle_null_values(df):
    """Identify and remove rows with null values."""
    print("\n[1.4] Handling null values...")
    
    original_count = len(df)
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    
    if total_nulls == 0:
        print("    ✓ No null values to remove")
        return df
    
    # Drop rows with any null values
    df_cleaned = df.dropna()
    removed_count = original_count - len(df_cleaned)
    
    print(f"    ✓ Removed {removed_count:,} rows with null values")
    print(f"    ✓ Remaining records: {len(df_cleaned):,}")
    
    return df_cleaned

def save_cleaned_data(df):
    """Save the cleaned data to processed folder."""
    print("\n[1.5] Saving cleaned data...")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(CLEANED_DATA_PATH), exist_ok=True)
    
    # Save to CSV
    df.to_csv(CLEANED_DATA_PATH, index=False)
    
    file_size = os.path.getsize(CLEANED_DATA_PATH) / 1024  # KB
    print(f"    ✓ Saved to: {CLEANED_DATA_PATH}")
    print(f"    ✓ File size: {file_size:.2f} KB")
    print(f"    ✓ Records: {len(df):,}")
    print(f"    ✓ Columns: {list(df.columns)}")

def main():
    """Main execution function."""
    # Load raw data
    df = load_raw_data()
    
    # Inspect data quality
    null_counts = inspect_data(df)
    
    # Drop SensorId column
    df_cleaned = drop_sensor_id(df)
    
    # Handle null values
    df_cleaned = handle_null_values(df_cleaned)
    
    # Save cleaned data
    save_cleaned_data(df_cleaned)
    
    print("\n" + "="*70)
    print("DATA CLEANING COMPLETE")
    print("="*70)
    print(f"\n✓ Cleaned dataset ready: {len(df_cleaned):,} records")
    print(f"✓ Columns: {list(df_cleaned.columns)}")
    print("\nNext Step:")
    print("  → Run 'python src/data_standardization.py' to standardize the data")
    print("="*70)

if __name__ == "__main__":
    main()
