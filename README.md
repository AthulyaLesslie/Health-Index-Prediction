# Health-Index-Prediction
A machine learning project for health index prediction using Colab notebooks and a Streamlit-based web app.  Analyzes health-related features to provide insights and predictions.

This project predicts the Health Index Score of a country for a specific disease in a given year using machine learning. The dataset consists of health-related factors for 20 countries, covering 20 diseases from 2020 to 2024. The model is trained to analyze various healthcare, socio-economic, and disease-related features to provide insights.

The dataset includes the following features:
Year, Country, Disease Name, Disease Category,
Most Affected Age, Population Affected, Healthcare Access (%),
Treatment Type, Average Treatment Cost (USD), Availability of Vaccines/Treatment,
Education Index, Prevalence Rate (%), Incidence Rate (%), Mortality Rate (%), Recovery Rate (%),
and Target Variable as Health Index Score

Data Preprocessing:
Handled missing values, duplicates, and irrelevant features,
Applied One-Hot Encoding and Label Encoding,
Split data into training and test sets

Machine Learning Models Used:
The model was trained and evaluated using multiple algorithms:
-K-Nearest Neighbors (KNN) & Hyperparameter-Tuned KNN
-Decision Tree & Hyperparameter-Tuned Decision Tree
-Random Forest Regressor & Hyperparameter-Tuned Random Forest
-Linear Regression
-AdaBoost (for both Linear Regression & Decision Tree)
-Support Vector Machine (SVM)

Deployment:
-The trained model was saved and integrated into a Streamlit web application for easy predictions.
-The web app is built using Streamlit in VS Code.
