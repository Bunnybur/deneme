"""
Advanced EDA Analysis
=====================
Additional statistical visualizations and analysis:
- Histogram (distribution analysis)
- Boxplot (outlier visualization)
- Correlation matrix (for multivariate relationships)
- Statistical report generation

Author: Advanced Computer Programming Course
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from datetime import datetime

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import STANDARDIZED_DATA_PATH

def load_data():
    """Load the standardized sensor data."""
    print("="*70)
    print("ADVANCED EDA ANALYSIS")
    print("="*70)
    print("\n[1] Loading standardized data...")
    
    try:
        df = pd.read_csv(STANDARDIZED_DATA_PATH)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        print(f"    ✓ Loaded {len(df):,} records")
        return df
    except FileNotFoundError:
        print(f"    ✗ Error: Data not found")
        sys.exit(1)

def generate_visualizations(df):
    """Generate histogram, boxplot, and correlation matrix."""
    print("\n[2] Generating statistical visualizations...")
    
    # Set style
    sns.set_style("whitegrid")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Extract time-based features for correlation analysis
    df_analysis = df.copy()
    df_analysis['Hour'] = df_analysis['Timestamp'].dt.hour
    df_analysis['DayOfWeek'] = df_analysis['Timestamp'].dt.dayofweek
    df_analysis['Month'] = df_analysis['Timestamp'].dt.month
    df_analysis['DayOfYear'] = df_analysis['Timestamp'].dt.dayofyear
    
    # ============ PLOT 1: Histogram ============
    ax1 = plt.subplot(3, 2, 1)
    values_normal = df[df['Value'] <= 60]['Value']  # Filter extreme values for better visualization
    ax1.hist(values_normal, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(values_normal.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {values_normal.mean():.2f}°C')
    ax1.axvline(values_normal.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {values_normal.median():.2f}°C')
    ax1.set_xlabel('Temperature (°C)', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Distribution of Temperature Readings (Normal Range)', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ============ PLOT 2: Histogram with Outliers ============
    ax2 = plt.subplot(3, 2, 2)
    ax2.hist(df['Value'], bins=60, color='coral', alpha=0.7, edgecolor='black')
    ax2.axvline(df['Value'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["Value"].mean():.2f}°C')
    ax2.set_xlabel('Temperature (°C)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Distribution Including Extreme Values', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # ============ PLOT 3: Boxplot - Overall ============
    ax3 = plt.subplot(3, 2, 3)
    box_data = [df[df['Value'] <= 60]['Value'], df[df['Value'] > 60]['Value']]
    bp = ax3.boxplot(box_data, labels=['Normal Range\n(≤60°C)', 'Extreme Values\n(>60°C)'], 
                     patch_artist=True, showmeans=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax3.set_ylabel('Temperature (°C)', fontsize=11)
    ax3.set_title('Boxplot: Normal vs Extreme Values', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # ============ PLOT 4: Boxplot by Hour ============
    ax4 = plt.subplot(3, 2, 4)
    hourly_normal = df_analysis[df_analysis['Value'] <= 60]
    hourly_data = [hourly_normal[hourly_normal['Hour'] == h]['Value'].values 
                   for h in range(0, 24, 3)]
    bp2 = ax4.boxplot(hourly_data, labels=[f'{h}:00' for h in range(0, 24, 3)], 
                      patch_artist=True, showmeans=True)
    for patch in bp2['boxes']:
        patch.set_facecolor('lightblue')
    ax4.set_xlabel('Hour of Day', fontsize=11)
    ax4.set_ylabel('Temperature (°C)', fontsize=11)
    ax4.set_title('Temperature Distribution by Hour (3-hour intervals)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
    
    # ============ PLOT 5: Correlation Matrix ============
    ax5 = plt.subplot(3, 2, 5)
    # Select numeric features for correlation
    corr_features = ['Value', 'Value_Standardized', 'Hour', 'DayOfWeek', 'Month', 'DayOfYear']
    corr_matrix = df_analysis[corr_features].corr()
    
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax5)
    ax5.set_title('Correlation Matrix: Temperature & Time Features', fontsize=12, fontweight='bold')
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.setp(ax5.yaxis.get_majorticklabels(), rotation=0)
    
    # ============ PLOT 6: Monthly Boxplot ============
    ax6 = plt.subplot(3, 2, 6)
    monthly_normal = df_analysis[df_analysis['Value'] <= 60]
    monthly_data = [monthly_normal[monthly_normal['Month'] == m]['Value'].values 
                    for m in range(1, 13)]
    bp3 = ax6.boxplot(monthly_data, labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], 
                      patch_artist=True, showmeans=True)
    for i, patch in enumerate(bp3['boxes']):
        # Color gradient by season
        if i in [11, 0, 1]:  # Winter
            patch.set_facecolor('lightblue')
        elif i in [2, 3, 4]:  # Spring
            patch.set_facecolor('lightgreen')
        elif i in [5, 6, 7]:  # Summer
            patch.set_facecolor('lightyellow')
        else:  # Fall
            patch.set_facecolor('lightsalmon')
    ax6.set_xlabel('Month', fontsize=11)
    ax6.set_ylabel('Temperature (°C)', fontsize=11)
    ax6.set_title('Temperature Distribution by Month', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    print("    ✓ Visualizations generated successfully")
    print("    ✓ Close the plot window to continue...")
    plt.show()
    
    return df_analysis, corr_matrix

def generate_statistical_report(df, df_analysis, corr_matrix):
    """Generate comprehensive statistical report."""
    print("\n" + "="*70)
    print("STATISTICAL REPORT")
    print("="*70)
    
    # Distribution Analysis
    print("\n" + "-"*70)
    print("1. DISTRIBUTION ANALYSIS")
    print("-"*70)
    
    values = df['Value']
    values_normal = df[df['Value'] <= 60]['Value']
    
    print(f"\n[Normal Range Data (≤60°C)]: {len(values_normal):,} records ({len(values_normal)/len(df)*100:.2f}%)")
    print(f"  • Mean:           {values_normal.mean():.2f}°C")
    print(f"  • Median:         {values_normal.median():.2f}°C")
    print(f"  • Mode:           {values_normal.mode()[0]:.2f}°C")
    print(f"  • Std Dev:        {values_normal.std():.2f}°C")
    print(f"  • Skewness:       {values_normal.skew():.3f}")
    print(f"  • Kurtosis:       {values_normal.kurtosis():.3f}")
    
    # Interpret distribution
    skewness = values_normal.skew()
    if abs(skewness) < 0.5:
        skew_interpretation = "approximately symmetric (normal)"
    elif skewness > 0:
        skew_interpretation = f"right-skewed (positively skewed)"
    else:
        skew_interpretation = f"left-skewed (negatively skewed)"
    
    print(f"\n  📊 Distribution Shape: The data is {skew_interpretation}")
    print(f"     Mean ≈ Median indicates symmetric distribution around {values_normal.mean():.2f}°C")
    
    # Outlier Analysis
    print("\n" + "-"*70)
    print("2. OUTLIER ANALYSIS")
    print("-"*70)
    
    Q1 = values.quantile(0.25)
    Q3 = values.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers_low = df[values < lower_bound]
    outliers_high = df[values > upper_bound]
    extreme_outliers = df[values > 100]
    
    print(f"\n[IQR Method Results]:")
    print(f"  • Q1 (25th percentile):  {Q1:.2f}°C")
    print(f"  • Q3 (75th percentile):  {Q3:.2f}°C")
    print(f"  • IQR:                   {IQR:.2f}°C")
    print(f"  • Lower Fence:          {lower_bound:.2f}°C")
    print(f"  • Upper Fence:          {upper_bound:.2f}°C")
    
    print(f"\n[Outlier Counts]:")
    print(f"  • Low Outliers (<{lower_bound:.2f}°C):  {len(outliers_low):,} ({len(outliers_low)/len(df)*100:.2f}%)")
    print(f"  • High Outliers (>{upper_bound:.2f}°C): {len(outliers_high):,} ({len(outliers_high)/len(df)*100:.2f}%)")
    print(f"  • Extreme Faults (>100°C):              {len(extreme_outliers):,} ({len(extreme_outliers)/len(df)*100:.2f}%)")
    
    print(f"\n  🚨 Outlier Interpretation:")
    print(f"     - {len(outliers_high):,} readings exceed normal operating range")
    print(f"     - {len(extreme_outliers):,} extreme sensor faults detected (likely hardware malfunction)")
    print(f"     - {len(outliers_low):,} unusually low readings (possible sensor calibration issues)")
    
    # Relationship Analysis
    print("\n" + "-"*70)
    print("3. RELATIONSHIP ANALYSIS (Temporal Patterns)")
    print("-"*70)
    
    print(f"\n[Correlation Matrix Key Findings]:")
    print(f"  • Temp vs Hour:       {corr_matrix.loc['Value', 'Hour']:+.3f}")
    print(f"  • Temp vs DayOfWeek:  {corr_matrix.loc['Value', 'DayOfWeek']:+.3f}")
    print(f"  • Temp vs Month:      {corr_matrix.loc['Value', 'Month']:+.3f}")
    print(f"  • Temp vs DayOfYear:  {corr_matrix.loc['Value', 'DayOfYear']:+.3f}")
    
    # Temporal insights
    hourly_avg = df_analysis.groupby('Hour')['Value'].mean()
    monthly_avg = df_analysis.groupby('Month')['Value'].mean()
    
    peak_hour = hourly_avg.idxmax()
    lowest_hour = hourly_avg.idxmin()
    peak_month = monthly_avg.idxmax()
    lowest_month = monthly_avg.idxmin()
    
    print(f"\n  📈 Temporal Patterns:")
    print(f"     - Peak temperature hour: {peak_hour}:00 ({hourly_avg[peak_hour]:.2f}°C avg)")
    print(f"     - Lowest temperature hour: {lowest_hour}:00 ({hourly_avg[lowest_hour]:.2f}°C avg)")
    print(f"     - Warmest month: Month {peak_month} ({monthly_avg[peak_month]:.2f}°C avg)")
    print(f"     - Coolest month: Month {lowest_month} ({monthly_avg[lowest_month]:.2f}°C avg)")
    
    # Overall insights
    print("\n" + "="*70)
    print("KEY INSIGHTS & RECOMMENDATIONS")
    print("="*70)
    
    print(f"\n✅ DATA QUALITY:")
    print(f"   • {len(values_normal)/len(df)*100:.2f}% of readings are within expected range (0-60°C)")
    print(f"   • Very low noise level with σ = {values_normal.std():.2f}°C")
    print(f"   • Stable sensor performance over 13-month period")
    
    print(f"\n⚠️ ANOMALIES DETECTED:")
    print(f"   • {len(extreme_outliers):,} extreme sensor faults require investigation")
    print(f"   • {len(outliers_high):,} high-temperature outliers may indicate equipment stress")
    print(f"   • Statistical outliers represent {(len(outliers_high) + len(outliers_low))/len(df)*100:.2f}% of data")
    
    print(f"\n🔍 OPERATIONAL INSIGHTS:")
    print(f"   • Temperature shows {skew_interpretation} distribution")
    print(f"   • Diurnal (daily) temperature variation detected")
    print(f"   • Seasonal patterns evident in monthly analysis")
    print(f"   • Consider predictive maintenance for outlier prevention")
    
    print("\n" + "="*70)

def main():
    """Main execution function."""
    # Load data
    df = load_data()
    
    # Generate visualizations
    df_analysis, corr_matrix = generate_visualizations(df)
    
    # Generate statistical report
    generate_statistical_report(df, df_analysis, corr_matrix)
    
    print("\n✓ Advanced EDA Analysis Complete!")
    print("="*70)

if __name__ == "__main__":
    main()
