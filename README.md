# AI DRIVEN SALES-FORECASTING


Sales Forecasting Application
This repository contains a sales forecasting application built using machine learning models and a web interface for seamless user interaction. The project is designed to predict sales trends based on historical data and provide insights through an intuitive interface.

Features
Backend
Model Training (model.py):

Implements machine learning algorithms for sales prediction.
Trains the model using historical sales data (train.csv).
Includes preprocessing, feature engineering, and model evaluation functionalities.
Forecasting:

Predicts future sales trends based on trained models.
Handles missing data and noisy datasets efficiently.
Customizable Parameters:

Allows users to adjust key model parameters to fine-tune predictions.
Frontend
Web Application (app.py):

Built with a lightweight Python web framework.
Provides a user-friendly interface for uploading datasets and visualizing predictions.
Displays key metrics like predicted sales, accuracy, and trends.
Visualization:

Includes charts and graphs to illustrate predictions.
Highlights anomalies or trends in sales data.
Prerequisites
To run this project, ensure the following dependencies are installed:

Python 3.7 or later
Libraries:
pandas
numpy
scikit-learn
matplotlib
flask (or streamlit for the web interface)
Installation
Clone this repository:

bash
Copy code
git clone https://github.com/franiachettiar/SALES-FORECASTING.git
cd SALES-FORECASTING
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Set up the environment:

Place the train.csv dataset in the frontend folder.
Usage
Running the Application
Start the backend server:

bash
Copy code
python backend/model.py
Launch the frontend:

bash
Copy code
python frontend/app.py
Open the application in your browser at http://localhost:5000.

Uploading and Training
Navigate to the "Upload" section to upload your dataset.
The model will preprocess and train on the uploaded data.
Visualizing Predictions
View the "Forecast" tab for:
Sales predictions.
Trend graphs.
Insights into data patterns.
