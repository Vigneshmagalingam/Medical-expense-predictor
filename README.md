
# 🏥 Medical Expenses Prediction Project

## Overview

This project aims to predict the medical expenses for patients based on their health conditions using regression techniques. The dataset used for this project is `hospital_data.xlsx`, and the target variable for prediction is the total cost to the hospital.

## Problem Statement

The goal is to develop a regression model that accurately predicts medical expenses, allowing healthcare providers to estimate costs and plan resources efficiently. The model will be evaluated using performance measures such as Sum Squared Regression (SSR), Mean Squared Error (MSE), and R2 values on test data.

## Dataset

The dataset `hospital_data.xlsx` contains relevant features such as patient health conditions, treatments, and medical history, along with the corresponding total cost to the hospital.

## Regression Technique

Any regression technique can be used for prediction, such as Linear Regression, Decision Tree Regression, or Random Forest Regression. The choice of technique will be based on model performance and suitability for the data.

## Performance Evaluation

After training the regression model, the following performance measures will be plotted and evaluated:
- Sum Squared Regression (SSR)


These metrics will provide insights into the accuracy and effectiveness of the regression model in predicting medical expenses.

## Deployment

The trained regression model will be deployed as a mobile or web application to provide a user-friendly interface for predicting medical expenses. A form will be created to collect user inputs such as patient health conditions, treatments required, and other relevant factors.

## Files Included

- `hospital_data.csv`: Dataset containing patient and medical expense information.
- `app.py` (for web application deployment): Python script for deploying the regression model as a web application.
- `app.py` (for mobile application deployment): Python script for deploying the regression model as a mobile application.
- `requirements.txt`: List of dependencies required to run the application.

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/Vigneshmagalingam/medical-expenses-prediction.git
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the regression model development notebook `Regression_Model.ipynb` to train and evaluate the model.

4. Deploy the model using either `app.py` (for web application) or `mobile_app.py` (for mobile application).

## Contributors

- Vignesh.M (vigneshmagalingamwork@gmail.com) 💼

---


