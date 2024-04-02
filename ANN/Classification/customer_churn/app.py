import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import os

# Load the pre-trained model
model_path = os.path.abspath('model.h5')
model = tf.keras.models.load_model(model_path)

# model = tf.keras.models.load_model('churn_prediction_model.h5')

# Function to preprocess input data
def preprocess_input(data):
    
    # Convert 'Yes' and 'No' to 1 and 0 in the appropriate columns
    data[:, 8] = np.where(data[:, 8] == 'Yes', 1, 0)  # Has Credit Card column
    data[:, 9] = np.where(data[:, 9] == 'Yes', 1, 0)  # Is Active Member column

    le = LabelEncoder()
    data[:, 2] = le.fit_transform(data[:, 2])  # Gender column

    transformer = [('enco_country', OneHotEncoder(), [1])]
    ct = ColumnTransformer(transformers=transformer, remainder='passthrough')
    data_encoded = ct.fit_transform(data)

    # Exclude columns that were one-hot encoded and columns with string values from scaling
    columns_to_scale = list(range(data_encoded.shape[1]))
    columns_to_exclude = [1, 3, 4, 5, 6, 7]  # Exclude non-numeric columns
    columns_to_scale = [col for col in columns_to_scale if col not in columns_to_exclude]

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_encoded[:, columns_to_scale])

    # Add dummy columns for missing categorical features
    num_missing_cols = 12 - data_scaled.shape[1]
    if num_missing_cols > 0:
        dummy_cols = np.zeros((data_scaled.shape[0], num_missing_cols))
        data_scaled = np.concatenate((data_scaled, dummy_cols), axis=1)

    return data_scaled

# Streamlit app
def main():
    st.title('Customer Churn Prediction')

    # Input fields for user data
    credit_score = st.number_input('Credit Score')
    geography = st.selectbox('Geography', ['France', 'Germany', 'Spain'])
    gender = st.selectbox('Gender', ['Male', 'Female'])
    age = st.number_input('Age')
    tenure = st.number_input('Tenure')
    balance = st.number_input('Balance')
    num_of_products = st.number_input('Number of Products')
    has_credit_card = st.selectbox('Has Credit Card', ['No', 'Yes'])
    is_active_member = st.selectbox('Is Active Member', ['No', 'Yes'])
    estimated_salary = st.number_input('Estimated Salary')

    # Preprocess user input
    input_data = np.array([[credit_score, geography, gender, age, tenure, balance, num_of_products,
                            has_credit_card, is_active_member, estimated_salary]])
    input_data_processed = preprocess_input(input_data)

    if st.button('Predict Churn'):
        prediction = model.predict(input_data_processed)
        churn_status = 'Churned' if prediction > 0.5 else 'Not Churned'
        st.write(f'Results: Customer will {churn_status}')

if __name__ == '__main__':
    main()

# streamlit run app.py
# streamlit run ./folder/main.py