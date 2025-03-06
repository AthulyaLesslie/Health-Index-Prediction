import streamlit as st
import pickle
import numpy as np
import pandas as pd
import base64

# Create a dictionary that maps Disease Name to Disease Category
disease_category_map = {
    "Diabetes": "Metabolic",
    "Influenza": "Viral",
    "Malaria": "Parasitic",
    "Parkinson's Disease": "Neurological",
    "Polio": "Viral",
    "Measles": "Viral",
    "Leprosy": "Bacterial",
    "Asthma": "Respiratory",
    "Ebola": "Viral",
    "Hepatitis": "Viral",
    "Hypertension": "Cardiovascular",
    "HIV/AIDS": "Viral",
    "Zika": "Viral",
    "COVID-19": "Viral",
    "Dengue": "Viral",
    "Alzheimer's Disease": "Neurological",
    "Rabies": "Viral",
    "Cholera": "Bacterial",
    "Tuberculosis": "Bacterial",
    "Cancer": "Genetic"
}

# Function to set background image
def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded_string = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        h1 {{
            text-align: center;
            color: #ffffff;
            font-size: 47px;
            text-shadow: 3px 3px 6px black;
            margin-top: 10px;
        }}
        .result-box {{
            background-color: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
            text-align: center;
            font-size: 20px;
            font-weight: bold;
        }}
        /* Remove the top black bar completely */
        header, .css-18ni7ap, .css-1r6slb0, .css-eczf16 {{
            display: none !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Apply background
set_background("bg.jpg")

# Title
st.markdown("<h1>🩺 Health Index Prediction </h1>", unsafe_allow_html=True)

# Load models and encoders
model = pickle.load(open('MODEL_DTT2.sav', 'rb'))
scaler = pickle.load(open('SD1_DT.sav', 'rb'))
ohe_c = pickle.load(open('OHEC1_DT.sav', 'rb'))
ohe_dn = pickle.load(open('OHEDN1_DT.sav', 'rb'))
ohe_dc = pickle.load(open('OHEDC1_DT.sav', 'rb'))
le_age = pickle.load(open('LEAG1_DT.sav', 'rb'))
ohe_tt = pickle.load(open('OHETT1_DT.sav', 'rb'))
le_vacc = pickle.load(open('LEVACC1_DT.sav', 'rb'))
feature_order = pickle.load(open('FEATURE_ORDER_DT.sav', 'rb'))

# Sidebar inputs
st.sidebar.header("🔍 Input Details")
year = st.sidebar.text_input("📅 Year [2000-24]")
country = st.sidebar.selectbox("🌍 Country", [""] + list(ohe_c.categories_[0]))
disease_name = st.sidebar.selectbox("🦠 Disease Name", [""] + list(ohe_dn.categories_[0]))
disease_category = disease_category_map.get(disease_name, "")
st.sidebar.selectbox("🏥 Disease Category", [disease_category], disabled=True)
healthcare_access = st.sidebar.text_input("🚑 Healthcare Access (%)")
treatment_type = st.sidebar.selectbox("💊 Treatment Type", [""] + list(ohe_tt.categories_[0]))
avg_treatment_cost = st.sidebar.text_input("💰 Average Treatment Cost (USD)")
prevalence_rate = st.sidebar.text_input("📊 Prevalence Rate (%)")
incidence_rate = st.sidebar.text_input("📈 Incidence Rate (%)")
mortality_rate = st.sidebar.text_input("📉 Mortality Rate (%)")
recovery_rate = st.sidebar.text_input("❤️‍🩹 Recovery Rate (%)")

if st.sidebar.button("🔍 Predict Health Index Score"):
    try:
        year = int(year)
        healthcare_access = float(healthcare_access)
        avg_treatment_cost = float(avg_treatment_cost)
        prevalence_rate = float(prevalence_rate)
        incidence_rate = float(incidence_rate)
        mortality_rate = float(mortality_rate)
        recovery_rate = float(recovery_rate)

        country_ohe = ohe_c.transform([[country]])[0]
        disease_name_ohe = ohe_dn.transform([[disease_name]])[0]
        disease_category_ohe = ohe_dc.transform([[disease_category]])[0]
        treatment_type_ohe = ohe_tt.transform([[treatment_type]])[0]

        features = np.array([
            year, healthcare_access,
            avg_treatment_cost, prevalence_rate,
            incidence_rate, mortality_rate, recovery_rate,
            *country_ohe, *disease_name_ohe, *disease_category_ohe, *treatment_type_ohe
        ]).reshape(1, -1)

        df_input = pd.DataFrame(features, columns=feature_order)
        features_scaled = scaler.transform(df_input)
        prediction = model.predict(features_scaled)
        health_index = prediction[0]

        if health_index <= 0.55:
            health_status = f"The health status of {country} related to {disease_name} in {year} was Poor, reflecting challenges in healthcare and treatment."
            color = "red"
            icon = "🚨"
        elif health_index < 0.64:
            health_status = f"In {year}, {country} faced a Moderate health situation regarding {disease_name}, with healthcare access and treatment options still needing attention."
            color = "orange"
            icon = "⚠️"
        elif health_index < 0.72:
            health_status = f"The health status of {country} related to {disease_name} in {year} was Good, reflecting positive healthcare conditions and treatment availability."
            color = "green"
            icon = "✅"
        else:
            health_status = f"The health status of {country} related to {disease_name} in {year} was Excellent, indicating strong healthcare systems and successful disease management."
            color = "gold"
            icon = "🌟"

        st.markdown(
            f"""
            <style>
            .result-box {{
                background-color: rgba(0, 0, 0, 0.8); /* Dark background */
                color: white;
                padding: 20px;
                border-radius: 10px;
                margin: 20px auto; /* Reduced gap */
                text-align: center;
                font-size: 20px;
                font-weight: bold;
                width: 250px; /* Slightly wider box */
                box-shadow: 4px 4px 12px rgba(0, 0, 0, 0.4);
            }}
            .score-number {{
                color: {color};
                font-weight: bold;
            }}

            .animated-box {{
                font-size: 18px;
                background-color: #1f1f1f; /* Dark background */
                border: 2px solid {color};
                padding: 15px;
                margin: 10px auto; /* Reduced gap */
                text-align: center;
                width: 80%;
                max-width: 500px; /* Keeps box width manageable */
                box-shadow: 4px 4px 12px rgba(0, 0, 0, 0.4);
                border-radius: 8px;
            }}
            </style>
            <div class='result-box'>Health Index: <span class='score-number'>{health_index:.4f}</span> {icon}</div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class='animated-box'>
                {health_status}
            </div>
            """,
            unsafe_allow_html=True
        )


    except Exception as e:
        st.error(f"Error: {str(e)}")
