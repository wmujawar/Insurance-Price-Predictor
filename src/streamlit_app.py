import os
import json
import pandas as pd
import streamlit as st

from model import Model
from utils import generate_html_performance_report

medical_conditions = ['Diabetic', 'Blood Pressure', 'Transplant', 'Chronic Disease', 'Allergies', 'History of Cancer in Family']

# Set title
st.title(':blue[Premium Price Prediction]')

# Create two columns
col1, col2= st.columns([5, 3], vertical_alignment="top", border=True, gap='medium')

# Create column 1
with col1:
    # Age
    age = st.number_input('Age (Years):', min_value=18, max_value=100)

    # Height
    height = st.number_input('Height (cm):', min_value=135, max_value=215)

    # Weight
    weight = st.number_input('Weight (kg):', min_value=30, max_value=150)

    # Number of Major Surgeries
    surgeries = st.number_input('Number of Major Surgeries:', min_value=0, max_value=3)
    
    conditions = st.pills("Medical Condition", medical_conditions, selection_mode="multi")
    
    button = st.button('Predict', use_container_width=True, type='primary')
    
    if button:
        # Infer medical conditions
        diabetes = 1 if 'Diabetic' in conditions else 0
        blood_pressure = 1 if 'Blood Pressure' in conditions else 0
        transplant = 1 if 'Transplant' in conditions else 0
        chronic_disease = 1 if 'Chronic Disease' in conditions else 0
        allergy = 1 if 'Allergies' in conditions else 0
        cancer_history = 1 if 'History of Cancer in Family' in conditions else 0
        
        # Get prediction
        requestData = dict(
            Age = int(age),
            Height = int(height),
            Weight = int(weight),
            Diabetes = diabetes,
            BloodPressureProblems = blood_pressure,
            AnyTransplants = transplant,
            AnyChronicDiseases = chronic_disease,
            KnownAllergies = allergy,
            HistoryOfCancerInFamily = cancer_history,
            NumberOfMajorSurgeries = int(surgeries)
        )
        
        df = pd.DataFrame.from_dict(requestData, orient='index').T
        
        model = Model()
        prediction = model.predict(df)
        
        # Show prediction
        st.write(f'Estimated Premium: &#8377; {prediction[0]:.2f}')


# Create column 2
with col2:
    
    parent_dir = os.path.abspath(os.path.join(os.getcwd()))
    metrics_dir = os.path.join(parent_dir, 'output')
    metrics_path = os.path.join(metrics_dir, 'metrics.json')

    with open(metrics_path, 'r') as f:
        matrix = json.load(f) 
    
    train = matrix['Train Metrics']
    test = matrix['Test Metrics']
    
    st.html('<h3>Model Performance</h3>')
    
    train_expander = st.expander("Train Performance", expanded=False)
    train_expander.html(generate_html_performance_report(train['RMSE'], train['R2'], train['Adjusted R2']))
    
    test_expander = st.expander("Test Performance", expanded=False)
    test_expander.html(generate_html_performance_report(test['RMSE'], test['R2'], test['Adjusted R2']))
