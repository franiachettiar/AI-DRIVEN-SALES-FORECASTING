import pandas as pd
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import seaborn as sns
from pmdarima import auto_arima
import streamlit as st

class TimeSeriesModel:
    def __init__(self, data: pd.DataFrame, product: str):
        self.data = data
        self.product = product
        self.model_fit = None

    def preprocess_data(self):
        # Filter the data for the selected product family
        df_product = self.data[self.data['family'] == self.product]

        # Check for duplicate dates and aggregate sales (sum) for the same date
        df_product = df_product.groupby('date').agg({'sales': 'sum', 'onpromotion': 'sum'}).reset_index()

        # Set 'date' as the index
        df_product.set_index('date', inplace=True)

        # Adjust frequency to daily and fill missing dates
        df_product = df_product.asfreq('D', method='pad')  # Pad missing dates with previous value
        df_product['sales'] = df_product['sales'].fillna(0)  # Ensure no missing values in sales
        return df_product

    def descriptive_statistics(self):
        # Generate and display descriptive statistics
        df_product = self.preprocess_data()
        stats = df_product['sales'].describe()
        st.write("Descriptive Statistics for Sales:")
        st.write(stats)

    def check_stationarity(self, series):
        # Perform Augmented Dickey-Fuller test
        result = adfuller(series)
        st.write(f'ADF Statistic: {result[0]}')
        st.write(f'p-value: {result[1]}')
        if result[1] > 0.05:
            st.write("Data is non-stationary, applying differencing")
            series = series.diff().dropna()  # Apply differencing to make the data stationary
        return series

    def plot_acf_pacf(self, series):
        # Plot ACF and PACF with the passed series
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        plot_acf(series, lags=40, ax=axes[0])
        plot_pacf(series, lags=40, ax=axes[1])
        st.pyplot(fig)

    def plot_line_graph(self):
        df_product = self.preprocess_data()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df_product.index, df_product['sales'], label="Sales over Time")
        ax.set_title("Line Graph of Sales Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sales")
        ax.legend()
        st.pyplot(fig)

    def plot_scatter(self):
        df_product = self.preprocess_data()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(df_product.index, df_product['sales'], alpha=0.5)
        ax.set_title("Scatter Plot of Sales")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sales")
        st.pyplot(fig)

    def plot_histogram(self):
        df_product = self.preprocess_data()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df_product['sales'], bins=30, ax=ax, kde=True)
        ax.set_title("Histogram of Sales")
        ax.set_xlabel("Sales")
        st.pyplot(fig)

    def plot_qq_plot(self):
        df_product = self.preprocess_data()
        fig = qqplot(df_product['sales'], line='s')
        plt.title("Q-Q Plot of Sales")
        st.pyplot(fig)

    def seasonality_decomposition(self):
        df_product = self.preprocess_data()

        # Perform seasonal decomposition (additive model)
        decomposition = seasonal_decompose(df_product['sales'], model='additive', period=365)

        # Plot decomposition
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10))
        decomposition.observed.plot(ax=ax1)
        ax1.set_ylabel('Observed')
        decomposition.trend.plot(ax=ax2)
        ax2.set_ylabel('Trend')
        decomposition.seasonal.plot(ax=ax3)
        ax3.set_ylabel('Seasonal')
        decomposition.resid.plot(ax=ax4)
        ax4.set_ylabel('Residual')
        plt.tight_layout()

        # Display the plot
        st.pyplot(fig)

    def auto_arima_selection(self, forecast_years):
        df_product = self.preprocess_data()

        # Check if the data is stationary, apply differencing if necessary
        sales_series = self.check_stationarity(df_product['sales'])

        # Limit the search space to make Auto ARIMA faster
        st.write("Running Auto ARIMA with optimized parameters...")

        model = auto_arima(
            sales_series,
            start_p=1, max_p=3,             # Restrict AR terms (autocorrelation)
            start_q=1, max_q=3,             # Restrict MA terms (moving average)
            d=None,                         # Let auto_arima determine the order of differencing
            seasonal=True, 
            m=12,                           # Set m to 12 for monthly seasonality
            start_P=0, max_P=2,             # Seasonal AR
            start_Q=0, max_Q=2,             # Seasonal MA
            D=None,                         # Let auto_arima determine seasonal differencing
            trace=True,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True,                  # Stepwise search for faster convergence
            n_jobs=-1                       # Use all available CPU cores for parallel processing
        )

        st.write(f"Auto ARIMA selected order: {model.order}, seasonal order: {model.seasonal_order}")

        # Fit the model
        self.model_fit = model  # Model is already fitted in auto_arima

        # Forecast for the test period (20% of data)
        test_size = int(len(sales_series) * 0.2)
        forecast = self.model_fit.predict(n_periods=test_size)

        # Forecast for the number of days corresponding to the chosen forecast_years
        days_to_forecast = 365 * forecast_years
        future_forecast = self.model_fit.predict(n_periods=days_to_forecast)

        return forecast, future_forecast, self.model_fit.aic(), self.model_fit.bic()

    def plot_forecast(self, forecast, future_forecast=None):
        df_product = self.preprocess_data()
        sales_series = self.check_stationarity(df_product['sales'])
        train_size = int(len(sales_series) * 0.8)
        train, test = sales_series[:train_size], sales_series[train_size:]

        # Create a new figure for the forecast
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the actual sales for the train and test period
        ax.plot(train.index, train, label="Train Sales", color='blue')
        ax.plot(test.index, test, label="Test Sales", color='orange')

        # Plot the forecasted sales for the test period
        forecast_index = test.index[:len(forecast)]
        ax.plot(forecast_index, forecast, label="Forecasted Sales", color='green')

        # If future forecast is provided, plot it as well
        if future_forecast is not None:
            future_index = pd.date_range(start=test.index[-1], periods=len(future_forecast)+1, freq='D')[1:]
            ax.plot(future_index, future_forecast, label="Future Forecast", color='red')

        ax.set_title("Sales Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sales")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
