

# Password protection
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
