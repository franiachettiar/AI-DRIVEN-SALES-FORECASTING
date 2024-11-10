import streamlit as st
import importlib.util
import sys
import os
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# Use the full path to the model.py file
model_path = 'D:/ATS/backend/model.py'

# Check if the file exists before loading
if not os.path.exists(model_path):
    st.error(f"File {model_path} does not exist!")
else:
    # Load the module dynamically using importlib
    spec = importlib.util.spec_from_file_location("model", model_path)
    model = importlib.util.module_from_spec(spec)
    sys.modules["model"] = model
    spec.loader.exec_module(model)

    # Now you can use TimeSeriesModel from the dynamically loaded module
    if hasattr(model, 'TimeSeriesModel'):
        from model import TimeSeriesModel
    else:
        st.error("The class 'TimeSeriesModel' is not found in model.py")

# Load your sales data using the new caching method
@st.cache_data
def load_data():
    data = pd.read_csv(r'D:/ATS/frontend/train.csv')  # Adjust path accordingly
    data['date'] = pd.to_datetime(data['date'])
    return data

# Improved title with style and logo support
def main():
    # Custom Header with Logo and Title
    st.markdown("""
        <style>
        .header-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 20px;
            background-color: #1f3b73;
            color: white;
            border-radius: 5px;
        }
        .header-container img {
            height: 60px;
        }
        .header-title {
            font-size: 28px;
            font-weight: bold;
        }
        </style>
        <div class="header-container">
            <div class="header-title">AI-Driven Sales Forecasting App</div>
        </div>
        """, unsafe_allow_html=True)

    # Load the data
    data = load_data()

    # Introduction section with better formatting
    st.markdown("""
    <style>
    .intro-text {
        text-align: center;
        font-size: 18px;
        color: #FAFAFA;
        padding-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown("<div class='intro-text'>Forecast sales data with AI models. Select your product and choose a forecasting range to see predictions and insights.</div>", unsafe_allow_html=True)

    # Product and year input sections in columns
    col1, col2 = st.columns(2)

    with col1:
        # User input: Select product
        product_list = data['family'].unique()
        selected_product = st.selectbox("Select the product family:", product_list)

    with col2:
        # User input: Prediction range in years
        forecast_years = st.slider("Forecast for how many years?", 1, 10)  # Allow 1-10 years

    # Load the model
    if 'TimeSeriesModel' in sys.modules['model'].__dict__:
        ts_model = TimeSeriesModel(data, selected_product)

        # Display Descriptive Statistics
        st.markdown("<h3 style='color: #1f3b73;'>Descriptive Statistics</h3>", unsafe_allow_html=True)
        ts_model.descriptive_statistics()

        # Display visualizations in columns for a cleaner look
        st.markdown("<h3 style='color: #1f3b73;'>Visualizations</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            st.write(f"Line Graph for {selected_product}:")
            ts_model.plot_line_graph()
            st.write(f"Scatter Plot for {selected_product}:")
            ts_model.plot_scatter()

        with col2:
            st.write(f"Histogram for {selected_product}:")
            ts_model.plot_histogram()
            st.write(f"Q-Q Plot for {selected_product}:")
            ts_model.plot_qq_plot()

        # Check stationarity
        st.markdown("<h3 style='color: #1f3b73;'>Stationarity Check</h3>", unsafe_allow_html=True)
        st.write(f"Checking stationarity for {selected_product} sales data:")
        df_product = ts_model.preprocess_data()
        ts_model.check_stationarity(df_product['sales'])

        # ACF and PACF plots
        st.markdown("<h3 style='color: #1f3b73;'>ACF and PACF Plots</h3>", unsafe_allow_html=True)
        ts_model.plot_acf_pacf(df_product['sales'])

        # Seasonal decomposition plot
        st.markdown("<h3 style='color: #1f3b73;'>Seasonal Decomposition</h3>", unsafe_allow_html=True)
        st.write(f"Seasonal decomposition for {selected_product}:")
        ts_model.seasonality_decomposition()

        # Train ARIMA model and forecast
        st.markdown("<h3 style='color: #1f3b73;'>ARIMA Model Training</h3>", unsafe_allow_html=True)
        st.write(f"Training ARIMA model for {selected_product} to forecast {forecast_years} year(s):")

        # Perform Auto ARIMA model selection, training, and forecasting
        forecast, future_forecast, aic, bic = ts_model.auto_arima_selection(forecast_years)

        # Display AIC and BIC
        st.write(f"AIC: {aic}")
        st.write(f"BIC: {bic}")

        # Forecast sales for the chosen number of years
        st.write(f"Sales Forecast for {forecast_years} year(s):")
        ts_model.plot_forecast(forecast, future_forecast)

    else:
        st.error("TimeSeriesModel class not found in model.py")

if __name__ == "__main__":
    main()
