# AI-Driven Sales Forecasting App

This application leverages AI-powered models to forecast sales data for different product families. Users can select a product family, specify a forecasting range, and view predictions along with detailed insights and visualizations.

---

## Key Features

### 1. Sales Forecasting
- Allows users to select a product family (e.g., `AUTOMOTIVE`, `ELECTRONICS`, etc.) and forecast sales for a specified range (e.g., 1 to 10 years).
- Utilizes ARIMA and Auto ARIMA models to predict future sales trends based on historical data.

### 2. Descriptive Statistics
- Provides a summary of key statistical metrics for the selected product family.
- Helps users understand the data distribution and key characteristics.

### 3. Visualizations
Offers multiple plots to provide insights into sales trends:
- **Line Graph**: Displays overall sales trends.
- **Scatter Plot**: Highlights individual data points.
- **Histogram**: Visualizes the distribution of sales data.
- **Q-Q Plot**: Evaluates data normality.

### 4. Stationarity Check
- Performs stationarity tests (e.g., Augmented Dickey-Fuller test) to determine if the sales data is suitable for time series modeling.
- Outputs detailed results, including ADF statistics and p-values, for interpretation.

### 5. Autoregressive Integrated Moving Average (ARIMA) Model
- Trains ARIMA models for the selected product family and forecast range:
  - Auto ARIMA selects the optimal parameters for both non-seasonal and seasonal components.
- Outputs key performance metrics such as AIC and BIC for model evaluation.

### 6. Seasonal Decomposition
- Decomposes the sales data into:
  - **Trend**: Long-term sales movements.
  - **Seasonality**: Recurring patterns.
  - **Residuals**: Random noise.

### 7. Forecasting Output
- Provides forecasted sales data for the specified range.
- Visualizes predictions to aid business decision-making.

---

## Example Workflow

1. **Product Selection**: Choose a product family (e.g., `AUTOMOTIVE`, `ELECTRONICS`).
2. **Forecasting Range**: Specify the number of years for prediction (e.g., 1 to 10 years).
3. **Data Insights**:
   - View descriptive statistics, visualizations, and stationarity checks.
4. **Model Training**:
   - Train ARIMA or Auto ARIMA models.
   - Obtain optimized parameters and performance metrics.
5. **Sales Forecast**:
   - View predictions for the specified range.
   - Analyze trends with dynamic visualizations.

---

## Visualizations

1. **Line Graph**: Shows the overall sales trend over time.
2. **Scatter Plot**: Highlights individual data points in the sales dataset.
3. **Histogram**: Illustrates the distribution of sales.
4. **Q-Q Plot**: Assesses data normality for statistical modeling.
5. **ACF and PACF Plots**: Displays lag correlations for ARIMA modeling.
6. **Seasonal Decomposition**: Visualizes the trend, seasonality, and residuals in the sales data.

---

## Key Features Across Product Families

### Stationarity Check
- Identifies whether the sales data is stationary for accurate forecasting.

### ARIMA Model Optimization
- Selects optimal parameters for forecasting with Auto ARIMA.
- Provides model performance metrics to evaluate accuracy.

### Sales Forecast
- Generates predictions for the selected product family and forecast range.
- Supports various product families (e.g., `AUTOMOTIVE`, `ELECTRONICS`, etc.).
