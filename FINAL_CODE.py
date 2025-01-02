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


api_key = '161a02f55af96a556113c9c2379c3f69'

# Streamlit app setup
st.set_page_config(layout="wide")
st.title("AI Trader: An AI Stock Forecasting Application")

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

    # Show the next day prediction
    next_day_prediction = forecast.iloc[-forecast_period]
    st.sidebar.write(f"Next Day Prediction: {next_day_prediction['yhat']:.2f}")

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

    # Adding selected technical indicators
    def add_indicator(indicator):
        if indicator == "20-Day SMA":
            sma = data['Close'].rolling(window=20).mean()
            fig.add_trace(go.Scatter(x=data.index, y=sma, mode='lines', name='SMA (20)'))
        elif indicator == "20-Day EMA":
            ema = data['Close'].ewm(span=20).mean()
            fig.add_trace(go.Scatter(x=data.index, y=ema, mode='lines', name='EMA (20)'))
        elif indicator == "20-Day Bollinger Bands":
            sma = data['Close'].rolling(window=20).mean()
            std = data['Close'].rolling(window=20).std()
            bb_upper = sma + 2 * std
            bb_lower = sma - 2 * std
            fig.add_trace(go.Scatter(x=data.index, y=bb_upper, mode='lines', name='BB Upper'))
            fig.add_trace(go.Scatter(x=data.index, y=bb_lower, mode='lines', name='BB Lower'))
        elif indicator == "VWAP":
            data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
            fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], mode='lines', name='VWAP'))

    # Sidebar options for technical indicators
    st.sidebar.subheader("Technical Indicators")
    indicators = st.sidebar.multiselect(
        "Select Indicators:",
        ["20-Day SMA", "20-Day EMA", "20-Day Bollinger Bands", "VWAP"],
        default=["20-Day SMA"]
    )

    # Add selected indicators to the chart
    for indicator in indicators:
        add_indicator(indicator)

    # Display the plot
    fig.update_layout(xaxis_rangeslider_visible=False)
    st.plotly_chart(fig)


    # AI-powered analysis button
    st.subheader("AI-Powered Analysis")
    if st.button("Run AI Analysis"):
        with st.spinner("Analyzing the chart, please wait..."):
            try:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                    fig.write_image(tmpfile.name)
                    tmpfile_path = tmpfile.name

                with open(tmpfile_path, "rb") as image_file:
                    image_data = base64.b64encode(image_file.read()).decode('utf-8')

                messages = [{
                    'role': 'user',
                    'content': """You are a Stock Trader specializing in Technical Analysis at a top financial institution.
                                Analyze the stock chart's technical indicators and provide a buy/hold/sell recommendation.
                                Base your recommendation only on the candlestick chart and the displayed technical indicators.
                                First, provide the recommendation, then, provide your detailed reasoning.
                    """,
                    'images': [image_data]
                }]
                response = ollama.chat(model='llama3.2-vision', messages=messages)

                st.write("**AI Analysis Results:**")
                st.write(response["message"]["content"])

                os.remove(tmpfile_path)
            except Exception as e:
                st.error(f"Error with AI analysis please try again later")