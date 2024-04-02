# Banking Customer Churn Prediction using Artificial Neural Networks (ANN)

## Overview
This project aims to predict customer churn in the banking sector using machine learning techniques, specifically leveraging an Artificial Neural Network (ANN). Customer churn refers to the phenomenon where customers cease their relationship with a company, which is a critical concern for businesses aiming to retain their customer base.

## Implementation Details
- **Data Collection**: The dataset 'Churn_Modelling.csv' containing customer information was used for this analysis and model development.
- **Data Preprocessing**:
  - Converted numeric columns to float32 for computational efficiency.
  - Encoded categorical variables using LabelEncoder and OneHotEncoder.
  - Balanced the dataset using Synthetic Minority Oversampling Technique (SMOTE) to handle class imbalance.
  - Standardized the features using StandardScaler to ensure uniformity in data distribution.
- **Model Building**:
  - Constructed an ANN with two hidden layers (each with 6 units and ReLU activation) and an output layer with sigmoid activation for binary classification (churn or not churn).
  - Utilized the Adam optimizer and binary cross-entropy loss function for model training.
- **Model Evaluation**:
  - Evaluated the model using test data and computed the test loss and accuracy.
- **Deployment**:
  - Deployed the trained model using Streamlit, enabling interactive predictions and analysis.

## Key Files
- **churn_prediction_model.h5**: Saved ANN model for churn prediction.
- **app.py**: Streamlit web application for interactive churn prediction.

## Usage
1. Ensure Python and required libraries (TensorFlow, pandas, numpy, matplotlib, scikit-learn, imbalanced-learn) are installed.
2. Clone the repository and navigate to the project directory.
3. Run `streamlit run app.py` to launch the Streamlit app.
4. Input customer data to predict churn probability and explore model insights.

## Results
The trained model achieved significant accuracy in predicting customer churn, providing valuable insights for proactive customer retention strategies.
