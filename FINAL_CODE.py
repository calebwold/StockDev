import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import ollama
import tempfile
import base64
import os
from prophet import Prophet
import requests

password = "123Awol!"  # Set your password here

# Streamlit app setup
st.set_page_config(layout="wide")
st.title("StockForecastX: An AI Stock Forecasting Application")

# Displaying maintenance message
st.markdown("### 🚧 This app is currently undergoing updates. Please check back soon! 🚧")

# Password input
input_password = st.text_input("Enter password to access the app:", type="password")

# Check if the entered password matches
if input_password == password:
    st.success("Password correct! Welcome to the app.")
    # Add the rest of your app code here for users who entered the correct password
else:
    st.warning("Incorrect password. Please try again.")
    st.stop()  # Stops the app from continuing if the password is incorrect

# Function to get weather data
def get_weather(city, api_key):
    try:
        response = requests.get(
            f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=imperial"
        )
        if response.status_code == 200:
            data = response.json()
            weather_desc = data["weather"][0]["description"].capitalize()
            temp = data["main"]["temp"]
            return f"{city}: {weather_desc}, {temp}°F"
        else:
            return "Unable to fetch weather data. Please check your entered city name."
    except Exception as e:
        return f"Error fetching weather data: {e}"

# API key for OpenWeatherMap
api_key = '161a02f55af96a556113c9c2379c3f69'

# Weather input
st.sidebar.header("Weather Settings")
weather_city = st.sidebar.text_input("Enter City for Weather Info:", "Denver")

# Fetch and display weather
weather_info = get_weather(weather_city, api_key)
st.markdown(f"### 🌤 Weather Update: {weather_info}", unsafe_allow_html=True)

# Warning notice
if "warning_shown" not in st.session_state:
    st.session_state["warning_shown"] = False

if not st.session_state["warning_shown"]:
    st.warning("⚠️ Please note: This application utilizes AI-powered stock analysis and forecasting. While the AI provides valuable insights, the predictions may not always be accurate and should not be relied upon for real-world trading decisions.")
    st.session_state["warning_shown"] = True

# Sidebar inputs for stock forecasting
ticker = st.sidebar.text_input("Enter Stock Ticker Symbol (e.g., AAPL):", "AAPL")
forecast_period = st.sidebar.slider("Forecast Days", min_value=1, max_value=30, value=7)
start_date = st.sidebar.date_input("Select Start Date", pd.to_datetime("2024-01-01"))
end_date = pd.to_datetime("today")
st.sidebar.write(f"Forecasting starting from {start_date}")

# Fetch stock data
if st.sidebar.button("Fetch Data"):
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        if stock_data.empty:
            st.error("No data found for the specified ticker and date range.")
        else:
            st.session_state["stock_data"] = stock_data
            st.success("Stock data loaded successfully!")
    except Exception as e:
        st.error(f"Error fetching data: {e}")

# Display raw stock data if available
if "stock_data" in st.session_state:
    data = st.session_state["stock_data"]
    st.subheader("Raw Stock Data")
    st.write(data)

    # Prepare data for forecasting with Prophet
    df = data.reset_index()[["Date", "Close"]]
    df.columns = ["ds", "y"]  # Rename columns for Prophet compatibility

    # Fit the Prophet model and make predictions
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=forecast_period)
    forecast = model.predict(future)

    # Display forecast table
    forecast_table = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].iloc[-forecast_period:]
    forecast_table.columns = ["Date", "Predicted Price", "Lower Bound", "Upper Bound"]
    st.subheader("Stock Price Forecast")
    st.dataframe(forecast_table)

    # Provide CSV download link
    csv = forecast_table.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  
    href = f'<a href="data:file/csv;base64,{b64}" download="forecast.csv">Download Forecast as CSV</a>'
    st.markdown(href, unsafe_allow_html=True)

    # Create and plot candlestick chart
    fig = go.Figure(data=[
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="Candlestick"
        )
    ])

    # Add forecasted price as dotted line
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        name='Forecasted Price',
        line=dict(dash='dot', color='orange')
    ))

    # Display the plot
    fig.update_layout(xaxis_rangeslider_visible=False)
    st.plotly_chart(fig)

    # AI-powered analysis button
    st.subheader("AI-Powered Analysis")
    if st.button("Run AI Analysis"):
        st.error("AI analysis functionality is currently under construction.")
