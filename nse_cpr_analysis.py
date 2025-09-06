import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report
import mplfinance as mpf

# List of NIFTY50 stocks (sample, you can expand)
nifty50_stocks = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS',
    'HINDUNILVR.NS', 'ITC.NS', 'KOTAKBANK.NS', 'LT.NS', 'AXISBANK.NS'
]

def calculate_cpr(high, low, close):
    pp = (high + low + close) / 3
    bc = (high + low) / 2
    tc = 2 * pp - bc
    return pp, bc, tc

def analyze_stock(ticker, start_date='2024-01-01', end_date='2024-09-06'):
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        print(f"No data for {ticker}")
        return None

    # Flatten columns
    data.columns = data.columns.droplevel(1)

    data['PP'] = (data['High'] + data['Low'] + data['Close']) / 3
    data['BC'] = (data['High'] + data['Low']) / 2
    data['TC'] = 2 * data['PP'] - data['BC']
    data['CPR_Width'] = data['TC'] - data['BC']
    data['Range'] = data['High'] - data['Low']
    data['CPR_Percent'] = data['CPR_Width'] / data['Range']

    # Determine trend: if CPR_Percent < 0.3, narrow (trending), else wide (sideways)
    data['Trend'] = np.where(data['CPR_Percent'] < 0.3, 'Trending', 'Sideways')

    return data

# Analyze one stock for example
ticker = 'TCS.NS'
data = analyze_stock(ticker)
if data is not None:
    print(data.tail())
    print(f"Average CPR Width: {data['CPR_Width'].mean()}")
    print(f"Average CPR Percent: {data['CPR_Percent'].mean()}")
    print(f"Std CPR Width: {data['CPR_Width'].std()}")
    print(f"Median CPR Width: {data['CPR_Width'].median()}")
    print(f"Min CPR Width: {data['CPR_Width'].min()}")
    print(f"Max CPR Width: {data['CPR_Width'].max()}")
    print(f"Days Trending: {data['Trend'].value_counts()}")

    # General idea: Narrow if CPR_Width < median, Wide if > median + std
    median_width = data['CPR_Width'].median()
    std_width = data['CPR_Width'].std()
    data['Trend_New'] = np.where(data['CPR_Width'] < median_width, 'Narrow (Trending)', 
                                 np.where(data['CPR_Width'] > median_width + std_width, 'Wide (Sideways)', 'Normal'))
    print(f"Trend New: {data['Trend_New'].value_counts()}")

    # ML Model to predict next day's trend
    data['Trend_Num'] = data['Trend'].map({'Trending': 1, 'Sideways': 0})
    data['Next_Trend_Num'] = data['Trend_Num'].shift(-1)
    data_ml = data.dropna()

    features = ['High', 'Low', 'Close', 'Open', 'PP', 'BC', 'TC', 'CPR_Width', 'Range', 'CPR_Percent']
    X = data_ml[features]
    y = data_ml['Next_Trend_Num']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"ML Model Accuracy: {accuracy_score(y_test, y_pred)}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Feature Importances
    importances = model.feature_importances_
    print("Feature Importances:")
    for feat, imp in zip(features, importances):
        print(f"{feat}: {imp:.4f}")

    # Analyze ranges for next day's trend
    next_trending = data_ml[data_ml['Next_Trend_Num'] == 1]
    next_sideways = data_ml[data_ml['Next_Trend_Num'] == 0]

    print(f"Next Day Trending - Avg CPR Width: {next_trending['CPR_Width'].mean():.2f}, Median: {next_trending['CPR_Width'].median():.2f}")
    print(f"Next Day Sideways - Avg CPR Width: {next_sideways['CPR_Width'].mean():.2f}, Median: {next_sideways['CPR_Width'].median():.2f}")

    # Regressor for next day's CPR Width
    y_reg = data_ml['CPR_Width'].shift(-1).dropna()
    X_reg = data_ml[features].iloc[:-1]

    reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
    reg_model.fit(X_reg, y_reg)
    y_reg_pred = reg_model.predict(X_reg)

    print(f"Regressor MSE: {np.mean((y_reg - y_reg_pred)**2):.2f}")

    # Predict next day's CPR Width (using last available data)
    last_data = data[features].iloc[-1:]
    predicted_cpr = reg_model.predict(last_data)[0]
    print(f"Predicted Next Day CPR Width: {predicted_cpr:.2f}")
    if predicted_cpr < 31.55:
        print("Prediction: Next day likely Trending (Narrow CPR)")
    elif predicted_cpr > 17.15:
        print("Prediction: Next day likely Sideways (Wide CPR)")
    else:
        print("Prediction: Next day Normal")

    # Plot Candlestick with CPR and Trendline
    data['MA50'] = data['Close'].rolling(window=50).mean()  # 50-day moving average as trendline

    # Add plots for PP, BC, TC
    ap1 = mpf.make_addplot(data['PP'], color='blue', linestyle='--', label='PP')
    ap2 = mpf.make_addplot(data['BC'], color='green', linestyle='--', label='BC')
    ap3 = mpf.make_addplot(data['TC'], color='red', linestyle='--', label='TC')
    ap4 = mpf.make_addplot(data['MA50'], color='orange', label='50 MA')

    fig, axlist = mpf.plot(data, type='candle', style='charles', addplot=[ap1, ap2, ap3, ap4], volume=True, figsize=(16,8), returnfig=True)
    axlist[0].set_title(f'Candlestick Chart with CPR for {ticker}')
    plt.savefig(f'candlestick_cpr_{ticker}.png')
    plt.close()
    print(f"Candlestick plot saved as candlestick_cpr_{ticker}.png")
