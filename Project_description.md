# **Health Index Prediction**

## **Overview**
A machine learning project for **health index prediction** using **Colab notebooks** and a **Streamlit-based web app**. The model analyzes health-related features to provide insights and predictions.

This project predicts the **Health Index Score** of a country for a specific **disease** in a given **year**. The dataset includes **20 countries**, **20 diseases**, and data from **2020 to 2024**, covering key healthcare, socio-economic, and disease-related factors.

## **Dataset Details**
The dataset includes the following features:
- **General Info:**  
  - Year  
  - Country  
  - Disease Name  
  - Disease Category  
- **Health & Treatment Factors:**  
  - Most Affected Age  
  - Population Affected  
  - Healthcare Access (%)  
  - Treatment Type  
  - Average Treatment Cost (USD)  
  - Availability of Vaccines/Treatment  
- **Socio-Economic & Disease Metrics:**  
  - Education Index  
  - Prevalence Rate (%)  
  - Incidence Rate (%)  
  - Mortality Rate (%)  
  - Recovery Rate (%)  
- **Target Variable:**  
  - Health Index Score  

## **Data Preprocessing**
- Handled:
  - Missing values  
  - Duplicates  
  - Irrelevant features  
- Applied:
  - One-Hot Encoding  
  - Label Encoding  
- Split into:
  - Training set  
  - Test set  

## **Exploratory Data Analysis & Visualizations**
- Used **Matplotlib** and **Seaborn** to analyze and visualize trends in the dataset.  
- Visualized relationships between **disease prevalence, healthcare access, treatment costs, and health index scores**.  
- Plotted:
  - Heatmaps to show feature correlations  
  - Bar charts for disease-wise and country-wise health index scores  
  - Line graphs to track changes in health index over time  

## **Machine Learning Models Used**
The model was trained and evaluated using multiple algorithms:  
- ✅ **K-Nearest Neighbors (KNN) & Hyperparameter-Tuned KNN**  
- ✅ **Decision Tree & Hyperparameter-Tuned Decision Tree**  
- ✅ **Random Forest Regressor & Hyperparameter-Tuned Random Forest**  
- ✅ **Linear Regression**  
- ✅ **AdaBoost (for both Linear Regression & Decision Tree)**  
- ✅ **Support Vector Machine (SVM)**  

## **Deployment**
- The trained model was saved and integrated into a **Streamlit web application** for easy predictions.  
- The web app is built using **Streamlit** in **VS Code**.  

---

