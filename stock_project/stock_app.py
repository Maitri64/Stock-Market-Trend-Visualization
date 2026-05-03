"""
STOCK MARKET TREND VISUALIZATION & PREDICTION
Indian Stock Market Version (NSE/BSE)
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Indian Stock Market Analyzer",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Indian Stock Market Trend Visualization & Prediction")
st.markdown("""
This app analyzes **NSE/BSE stock data** and predicts future prices using Linear Regression.
Enter a stock symbol below to get started. Use **.NS** for NSE and **.BO** for BSE.
""")

st.sidebar.header("📊 User Input")

stock_symbol = st.sidebar.text_input(
    "Enter Stock Symbol",
    value="RELIANCE.NS",
    help="Examples: RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS, TATAMOTORS.NS"
)

col1, col2 = st.sidebar.columns(2)
with col1:
    years_back = st.number_input(
        "Years of historical data",
        min_value=1,
        max_value=5,
        value=2,
        step=1
    )
with col2:
    prediction_days = st.number_input(
        "Days to predict",
        min_value=1,
        max_value=10,
        value=5,
        step=1
    )

end_date = datetime.now()
start_date = end_date - timedelta(days=years_back * 365)

analyze_button = st.sidebar.button("🔍 Analyze Stock", type="primary")

@st.cache_data(ttl=3600)
def fetch_stock_data(symbol, start, end):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(start=start, end=end)
        if data.empty:
            return None, "No data found"
        return data, None
    except Exception as e:
        return None, str(e)

def calculate_moving_averages(data):
    data_copy = data.copy()
    data_copy['MA20'] = data_copy['Close'].rolling(window=20).mean()
    data_copy['MA50'] = data_copy['Close'].rolling(window=50).mean()
    return data_copy

def predict_future_prices(data, days_to_predict):
    training_days = min(30, len(data))
    recent_data = data.tail(training_days)

    X = np.arange(training_days).reshape(-1, 1)
    y = recent_data['Close'].values.astype(float)

    model = LinearRegression()
    model.fit(X, y)

    future_X = np.arange(training_days, training_days + days_to_predict).reshape(-1, 1)
    future_prices = model.predict(future_X)
    future_prices = np.array(future_prices).astype(float)

    slope = model.coef_[0]
    r2_score = model.score(X, y)

    return future_prices, slope, model, r2_score, X, y

def make_decision(current_price, predicted_price, slope, current_ma20, current_ma50):
    percent_change = ((predicted_price - current_price) / current_price) * 100
    reasons = []

    if predicted_price > current_price and slope > 0:
        decision = "BUY"
        color = "green"
        emoji = "📈"
        reasons.append(f"Price expected to increase by {percent_change:.2f}%")
        reasons.append(f"Upward trend (slope: {slope:.4f})")
    elif predicted_price < current_price and slope < 0:
        decision = "SELL"
        color = "red"
        emoji = "📉"
        reasons.append(f"Price expected to decrease by {abs(percent_change):.2f}%")
        reasons.append(f"Downward trend (slope: {slope:.4f})")
    else:
        decision = "HOLD"
        color = "orange"
        emoji = "⚖️"
        reasons.append("No clear trend direction")
        reasons.append("Wait for clearer signals")

    if not pd.isna(current_ma20):
        if current_price > current_ma20:
            reasons.append("Price above 20-day MA (bullish signal)")
        else:
            reasons.append("Price below 20-day MA (bearish signal)")

    return decision, color, emoji, reasons, percent_change

if analyze_button:
    with st.spinner(f"Fetching data for {stock_symbol}..."):
        data, error = fetch_stock_data(stock_symbol, start_date, end_date)

        if error:
            st.error(f"❌ Error fetching data: {error}")
            st.info("Try symbols like: RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS")
        elif data.empty:
            st.error("❌ No data found. Please check the symbol and try again.")
        else:
            data = calculate_moving_averages(data)
            future_prices, slope, model, r2_score, X, y = predict_future_prices(data, prediction_days)

            current_price = float(data['Close'].iloc[-1])
            current_ma20 = float(data['MA20'].iloc[-1]) if not pd.isna(data['MA20'].iloc[-1]) else current_price
            current_ma50 = float(data['MA50'].iloc[-1]) if not pd.isna(data['MA50'].iloc[-1]) else current_price
            tomorrow_price = float(future_prices[0])

            decision, color, emoji, reasons, percent_change = make_decision(
                current_price, tomorrow_price, slope, current_ma20, current_ma50
            )

            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader(f"📊 {stock_symbol.upper()} - Stock Analysis")

                fig, ax = plt.subplots(figsize=(12, 6))

                ax.plot(data.index, data['Close'], label='Closing Price', linewidth=2, color='blue')
                ax.plot(data.index, data['MA20'], label='20-Day MA', linewidth=1.5, color='orange', alpha=0.7)
                ax.plot(data.index, data['MA50'], label='50-Day MA', linewidth=1.5, color='green', alpha=0.7)

                last_date = data.index[-1]
                future_dates = [last_date + timedelta(days=i+1) for i in range(prediction_days)]
                ax.plot(future_dates, future_prices, 'r--', label='Predicted Prices', linewidth=2, marker='o', markersize=6)

                ax.axvline(x=last_date, color='gray', linestyle='--', alpha=0.5)
                ax.set_title(f'{stock_symbol.upper()} - Price Trend with {prediction_days}-Day Prediction', fontsize=14)
                ax.set_xlabel('Date')
                ax.set_ylabel('Price (INR ₹)')
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()

                st.pyplot(fig)

            with col2:
                st.subheader("📈 Key Metrics")

                price_change = current_price - float(data['Close'].iloc[-2]) if len(data) > 1 else 0

                st.metric(
                    label="Current Price",
                    value=f"₹{current_price:.2f}",
                    delta=f"₹{price_change:.2f}" if len(data) > 1 else None
                )

                st.metric(
                    label="Predicted Tomorrow's Price",
                    value=f"₹{tomorrow_price:.2f}",
                    delta=f"{percent_change:.2f}%"
                )

                st.metric(
                    label="Trend Strength (Slope)",
                    value=f"{slope:.4f}",
                    delta="Upward" if slope > 0 else "Downward"
                )

                st.write("---")
                st.write("**Moving Averages:**")
                st.write(f"📊 20-Day MA: ₹{current_ma20:.2f}")
                st.write(f"📊 50-Day MA: ₹{current_ma50:.2f}")

            st.markdown("---")
            st.subheader("🎯 Trading Recommendation")

            st.markdown(
                f"""
                <div style="
                    background-color: {color}20;
                    padding: 20px;
                    border-radius: 10px;
                    border-left: 5px solid {color};
                ">
                    <h2 style="color: {color}; margin: 0;">{emoji} {decision}</h2>
                    <ul>
                        {"".join([f"<li>{r}</li>" for r in reasons])}
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )

            with st.expander("📋 View Raw Data"):
                st.write("**Recent Data (Last 10 days):**")
                recent_data = data.tail(10)[['Close', 'MA20', 'MA50']]
                st.dataframe(recent_data.style.format("{:.2f}", na_rep="N/A"))

                st.write("**Future Predictions:**")
                future_prices_clean = [float(price) for price in future_prices]
                pred_df = pd.DataFrame({
                    'Day': [f"Day {i+1}" for i in range(prediction_days)],
                    'Predicted Price (₹)': future_prices_clean
                })
                st.dataframe(pred_df.style.format({"Predicted Price (₹)": "{:.2f}"}))

                st.write("**Model Information:**")
                st.write(f"- Model Type: Linear Regression")
                st.write(f"- Training Data: Last 30 days")
                st.write(f"- R² Score: {float(r2_score):.4f} (1 = perfect fit)")

else:
    st.info("👈 Enter a stock symbol in the sidebar and click 'Analyze Stock' to get started!")

    st.subheader("📚 Indian Stock Symbols (NSE)")
    example_stocks = pd.DataFrame({
        "Company": ["Reliance Industries", "TCS", "Infosys", "HDFC Bank", "Tata Motors",
                    "Wipro", "SBI", "Bajaj Finance", "Adani Ports", "ITC"],
        "Symbol": ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "TATAMOTORS.NS",
                   "WIPRO.NS", "SBIN.NS", "BAJFINANCE.NS", "ADANIPORTS.NS", "ITC.NS"],
        "Exchange": ["NSE", "NSE", "NSE", "NSE", "NSE", "NSE", "NSE", "NSE", "NSE", "NSE"]
    })
    st.dataframe(example_stocks, hide_index=True)

    st.subheader("📚 Indian Stock Symbols (BSE)")
    bse_stocks = pd.DataFrame({
        "Company": ["Reliance Industries", "TCS", "Infosys", "HDFC Bank", "Tata Motors"],
        "Symbol": ["RELIANCE.BO", "TCS.BO", "INFY.BO", "HDFCBANK.BO", "TATAMOTORS.BO"],
        "Exchange": ["BSE", "BSE", "BSE", "BSE", "BSE"]
    })
    st.dataframe(bse_stocks, hide_index=True)

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: gray;">
        📊 Built with Streamlit, yfinance, and Linear Regression | Indian Stock Market | DAV Project
    </div>
    """,
    unsafe_allow_html=True
)