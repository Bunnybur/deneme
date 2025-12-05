# Quick Start Guide - Enhanced Dashboard

## Opening the Dashboard

Simply open this file in your browser:
```
dashboard_enhanced.html
```

Double-click the file or right-click → Open with → Your preferred browser

## What You'll See

### 1. Overview Section (Default View)
- **Total Records**: Count from your CSV
- **Date Range**: Time span of your data
- **Normal Range %**: Readings within 0-60°C
- **IQR Outliers %**: Statistical anomalies
- **Extreme Faults %**: Critical readings >100°C

### 2. Data Analysis Section (Click sidebar)
- Full descriptive statistics table
- IQR outlier detection breakdown
- Domain knowledge reference

### 3. Visualizations Section (Main feature!)

**Six comprehensive charts:**

1. **Complete Time Series** - All readings over time with highlighted anomalies
2. **Normal Operating Range** - Focus on 0-60°C readings
3. **Standardized Values** - Z-score view with ±3σ bounds
4. **Temperature Distribution** - Histogram showing frequency by temperature
5. **Box Plot** - Quartile visualization with outliers
6. **Monthly Trends** - Aggregated monthly avg/min/max

### 4. Model Training Section
- Run pipeline simulation
- Train model (simulation)
- View performance metrics

### 5. Logs Section
- System activity feed
- Data loading status
- Pipeline execution logs

## Interacting with Charts

- **Hover** over any data point for detailed information
- **Scroll** to see all six visualizations
- Charts are **responsive** and adjust to window size

## Color Legend

- 🔵 **Blue** = Normal readings
- 🟢 **Green** = Normal range (0-60°C), averages
- 🟠 **Orange** = IQR outliers (warnings)
- 🔴 **Red** = Extreme faults (>100°C)
- 🟣 **Purple** = Statistical/analytical views

## Troubleshooting

### Charts not loading?
- Check browser console (F12) for errors
- Ensure `sensor-fault-detection.csv` is in the same folder
- Try refreshing the page (Ctrl+R or Cmd+R)

### Slow performance?
- Normal for 62K+ records on first load
- Charts use sampling for better performance
- Close other browser tabs

### Statistics showing "..."?
- Data is still loading, wait 2-3 seconds
- Check logs section for loading status

## Files Overview

- `dashboard_enhanced.html` - Main dashboard (OPEN THIS)
- `dashboard_new.js` - JavaScript with all logic
- `dashboard.css` - Styling
- `sensor-fault-detection.csv` - Your data source

## Need the Original?

The original dashboard files were preserved:
- `dashboard.html` - Original version
- `dashboard.js` - Original JavaScript

## Next Actions

1. ✅ Open `dashboard_enhanced.html`
2. ✅ Wait for data to load (2-3 seconds)
3. ✅ Explore all 6 visualizations
4. ✅ Check statistics match your expectations
5. ✅ Try the pipeline controls in Model Training section

Enjoy your enhanced visualizations! 📊
