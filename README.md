# TV-Price-Predictor

## Overview
This project was developed as part of the CAP 4612 - Introduction to Machine Learning course at Florida International University. It represents my first practical experience building a complete machine learning application.

The program predicts the price of a television based on its hardware features and brand using the K-Nearest Neighbors (KNN) regression algorithm. It includes data preprocessing, model training, and evaluation using metrics such as Mean Squared Error (MSE) and Mean Absolute Error (MAE).

## Features
- Data loading and preprocessing using pandas and NumPy.
- Feature scaling using MinMaxScaler.
- Model training with K-Nearest Neighbors (KNN) Regression.
- Evaluation using accuracy, precision, recall, F1-score, MAE, MSE, RMSE, R².
- Cross-validation to validate model performance.
- Data visualization using Matplotlib and Seaborn.


## Required Libraries and Imported Functions
- pandas, numpy
- scikit-learn (MinMaxScaler, KNeighborsRegressor, metrics, cross-validation)
- matplotlib, seaborn

## Required Files
- TV.csv: The dataset containing TV specifications and prices.
- CAP 4612 TV Price Prediction Program.ipynb: Jupyter Notebook with the complete code for preprocessing, training, and evaluating the KNN model.

## Installation
1. Clone the repository:
   git clone <repository_link>
   cd TV_Price_Prediction_Cap4612

2. Install required libraries:
  pip install -r requirements.txt

## Usage
Run the Jupyter Notebook:
   jupyter notebook CAP_4612_TV_Price_Prediction_Program.ipynb
Follow the cells step-by-step to preprocess the data, train the model, and evaluate its performance.

## Repository Structure
TV_Price_Prediction_Cap4612/

├── data/

│   └── tv_data.csv

├── notebooks/

│   └── CAP_4612_TV_Price_Prediction_Program.ipynb

├── README.md

└── requirements.txt
