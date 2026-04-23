import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Bank Marketing Prediction", layout="wide")

@st.cache_resource # This keeps the model in memory so it's fast
def load_model():
    return joblib.load('final_project_bank_model.pkl')

model = load_model()

pipeline = model['pipeline']

# Target Threshold
threshold = model['optimal_threshold']

# 2. Mapping Dictionaries
edu_map = {0: 'Illiterate', 1: 'Basic 4y', 2: 'Basic 6y', 3: 'Basic 9y',
           4: 'High School', 5: 'Professional Course', 6: 'University Degree'
           }

age_map = {
    'Student_Age': 0, 'Young_Adult': 1, 'Adult': 2, 'Senior': 3
}

# --- UI SETUP ---
st.title("🏦 Bank Marketing Campaign Prediction")
st.write(f"Model Optimized Threshold: **{threshold}**")

with st.form("main_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("👤 Profile")
        edu_level = st.selectbox("Education Level", list(edu_map.values()))
        age_input = st.number_input("Age", min_value=0, max_value=120, value=30)
        job_category = st.selectbox("Job", ['white-collar', 'blue-collar', 'retired', 'self-employed', 'student', 'unemployed'])
        marital = st.selectbox("Marital Status", ['divorced', 'married', 'single'])

    with col2:
        st.subheader("📞 Campaign")
        contact = st.selectbox("Contact", ['cellular', 'telephone'])
        month = st.selectbox("Month", ['mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
        day = st.selectbox("Day", ['mon', 'tue', 'wed', 'thu', 'fri'])
        poutcome = st.selectbox("P-Outcome", ['nonexistent', 'failure', 'success'])
        intensity = st.selectbox("Intensity", ['low', 'high'])
        campaign = st.number_input("Campaign Count", min_value=1, value=1)
        previous = st.number_input("Previous Count", min_value=0, value=0)

    with col3:
        st.subheader("📉 Economic")
        cp_idx = st.number_input("Cons Price Idx", min_value=0.0, max_value=100.0, value = 93.0, format="%.3f")
        cc_idx = st.number_input("Cons Conf Idx", min_value=-100.0, max_value=0.0, value=-40.0, format="%.1f")
        e3m = st.number_input("Euribor 3M", min_value=0.0, max_value=5.0, value=3.5, format="%.3f")
        
        default = st.selectbox("Default", ['no', 'yes', 'unknown'])
        housing = st.selectbox("Housing", ['no', 'yes', 'unknown'])
        loan = st.selectbox("Loan", ['no', 'yes', 'unknown'])
        never_con = st.selectbox("Never Contacted?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        econ_grow = st.selectbox("Is the Economy Growing?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

    submitted = st.form_submit_button("Predict Subscription")

# Hidden
is_married = 1 if marital == 'married' else 0   # Convert marital status to binary for the model

education_level = [k for k, v in edu_map.items() if v == edu_level][0]  # Convert the label back to the number for the model

age_bin = pd.cut([age_input], bins=[0, 25, 40, 60, 1000], labels=['Student_Age', 'Young_Adult', 'Adult', 'Senior'])[0] # Convert age to age_bin category

if submitted:

    input_data = {
        'campaign': float(campaign),
        'previous': float(previous),
        'cons.price.idx': float(cp_idx),
        'cons.conf.idx': float(cc_idx),
        'euribor3m': float(e3m),
        'job_category': str(job_category),
        'marital': str(marital),
        'default': str(default),
        'housing': str(housing),
        'loan': str(loan),
        'contact': str(contact),
        'month': str(month),
        'day_of_week': str(day),
        'poutcome': str(poutcome),
        'age_bin': str(age_bin),
        'campaign_intensity': str(intensity),
        'education_level': float(education_level),
        'is_married': float(is_married),
        'never_contacted': float(never_con),
        'economy_growing': float(econ_grow),
        
        # The "Passenger" columns the model expectss:
        'age': float(age_input),     # Map to age_input
        'job': str(job_category),    # Map to job_category
        'education': str(edu_level)  # Map to edu_level
    }

    df_input = pd.DataFrame([input_data])

    feature_order = [
    'age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact',
    'month', 'day_of_week', 'campaign', 'previous', 'poutcome', 'cons.price.idx',
    'cons.conf.idx', 'euribor3m', 'job_category', 'education_level', 'is_married',
    'age_bin', 'never_contacted', 'campaign_intensity', 'economy_growing'
    ]

    df_input = df_input[feature_order]

    nominal_features = ['job', 'education', 'job_category', 'marital', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome', 'age_bin', 'campaign_intensity']
    for col in nominal_features:
        df_input[col] = df_input[col].astype(str)

    numeric_and_others = [c for c in feature_order if c not in nominal_features]
    for col in numeric_and_others:
        df_input[col] = pd.to_numeric(df_input[col], errors='coerce').fillna(0).astype(float)


    try:
        # prediction probability for the positive class (subscribing to term deposit)
        prediction_prob = pipeline.predict_proba(df_input)[0, 1]

        # Use the business threshold
        is_subscriber = prediction_prob >= threshold

        # Show Result
        if is_subscriber:
            st.success(f"🎯 High Potential Lead! (Probability: {prediction_prob:.2%})")
            st.write("Recommendation: Proceed with Call.")
        else:
            st.error(f"⚠️ Low Potential Lead. (Probability: {prediction_prob:.2%})")
            st.write("Recommendation: Do not call (Cost Optimization).")

    except Exception as e:
        st.error(f"Something went wrong: {e}")
        st.write("Debug - Data sent to the model:")
        st.dataframe(df_input)