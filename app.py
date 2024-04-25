import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("C:\\Users\\vigne\\Downloads\\hospital_data.csv")

# Handle missing values if needed
df.fillna(0, inplace=True)  # Replace NaN values with 0

# Encode categorical variables if needed
binary_mapping = {
    'GENDER': {'M': 1, 'F': 0},
    'MARITAL STATUS': {'MARRIED': 1, 'UNMARRIED': 0},
    'KEY COMPLAINTS -CODE': {
        'other': 0,
        'CAD': 1,
        'RHD': 2,
        'None': 3,
        'ACHD': 4,
        'OS-ASD': 5
    }
}

# Convert categorical columns to binary (0 and 1)
df.replace(binary_mapping, inplace=True)

# Split the data into features (X) and target variable (y)
X = df.drop('TOTAL COST TO HOSPITAL ', axis=1)
y = df['TOTAL COST TO HOSPITAL ']

model = LinearRegression()
model.fit(X, y)

# Define a function for user inputs and prediction
def user_input_features():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title('Hospital Cost Prediction')

    # Get user inputs
    age = st.slider('Age', min_value=0, max_value=100, value=25)
    gender = st.radio('Gender', ('M', 'F'))
    marital_status = st.radio('Marital Status', ('MARRIED', 'UNMARRIED'))
    key_complaints_code = st.selectbox('Key Complaints Code', ('other', 'CAD', 'RHD', 'None', 'ACHD', 'OS-ASD'))
    body_weight = st.number_input('Body Weight', min_value=0.0, max_value=200.0, value=70.0)
    body_height = st.number_input('Body Height', min_value=0.0, max_value=250.0, value=170.0)
    hr_pulse = st.number_input('HR Pulse', min_value=0, max_value=200, value=80)
    rr = st.number_input('RR', min_value=0, max_value=50, value=20)

    # Create a DataFrame for the user input
    input_data = pd.DataFrame({
        'AGE': [age],
        'GENDER': [gender],
        'MARITAL STATUS': [marital_status],
        'KEY COMPLAINTS -CODE': [key_complaints_code],
        'BODY WEIGHT': [body_weight],
        'BODY HEIGHT': [body_height],
        'HR PULSE': [hr_pulse],
        'RR': [rr]
    })

    # Ensure all categorical values are present in input data for encoding
    all_categories = list(X.columns)
    for category in all_categories:
        if category not in input_data.columns:
            input_data[category] = 0

    # Encode categorical variables in input data
    input_data.replace(binary_mapping, inplace=True)
    
    return input_data

# Get user inputs and make prediction
input_df = user_input_features()
y_pred = model.predict(input_df)

# Display the prediction
st.subheader('Predicted Total Cost to Hospital:')
st.write(f"${y_pred[0]:,.2f}")

# Plotting code if needed
st.subheader('Sum Squared Regression (SSR) Plot:')
plt.figure(figsize=(8, 6))
plt.plot(input_df, y_pred, 'o')
plt.xlabel('Actual Total Cost to Hospital')
plt.ylabel('Predicted Total Cost to Hospital')
plt.grid(True)
st.pyplot()

# Optional: Display the DataFrame for the user input
st.subheader('User Input DataFrame:')
st.write(input_df)
