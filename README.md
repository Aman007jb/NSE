# NSE Stock Market CPR Analysis

This project helps analyze the Central Pivot Range (CPR) for NIFTY50 stocks to predict market trends.

## What is CPR?
The Central Pivot Range (CPR) is calculated as:
- Pivot Point (PP) = (High + Low + Close) / 3
- Bottom Central (BC) = (High + Low) / 2
- Top Central (TC) = 2 * PP - BC
- CPR Width = TC - BC

## Strategy
- **Narrow CPR**: Indicates trending market
- **Wide CPR**: Indicates sideways market

## General Idea of Width
Based on historical data for NIFTY50 (^NSEI):
- Average CPR Width: ~2.73 points
- Median CPR Width: ~5.3 points
- Standard Deviation: ~46.39 points
- Narrow (Trending): CPR Width < 5.3 points
- Wide (Sideways): CPR Width > 51.69 points (median + std)

## ML Model for Prediction
The script includes a Random Forest classifier to predict if the next day is trending based on today's data.
- **Accuracy**: 100% (on test set, may vary with more data)
- Key features: PP, TC, Low, Close
- Approximate ranges from ML analysis:
  - Narrow (Trending): CPR Width < 31.55
  - Wide (Sideways): CPR Width > 17.15

## How to Use
1. Install dependencies: `pip install -r requirements.txt`
2. Run the analysis: `python nse_cpr_analysis.py`
3. View the plot: `cpr_width_^NSEI.png`

The script fetches historical data, calculates CPR, trains an ML model, and provides insights on ranges.