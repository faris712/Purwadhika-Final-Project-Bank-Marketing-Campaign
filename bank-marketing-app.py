import streamlit as st
import pandas as pd
import pickle

# Load the Model Bundle
@st.cache_resource # This keeps the model in memory so it's fast
def load_model():
    with open('final_bank_project_model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

model_bundle = load_model()
pipeline = model_bundle['pipeline']

# Target Threshold
threshold = model_bundle['optimal_threshold']

st.title("Bank Term Deposit Predictor")
st.write(f"Model Optimized Threshold: **{threshold}**")

# Collect inputs (examples of how to set up the sidebar)
st.sidebar.header("User Input Features")

# Numeric
campaign = st.sidebar.number_input("Campaign Contacts", value=1)
previous = st.sidebar.number_input("Previous Contacts", value=0)
cons_price = st.sidebar.number_input("Cons. Price Index", value=93.0)
cons_conf = st.sidebar.number_input("Cons. Conf. Index", value=-42.0)
euribor = st.sidebar.number_input("Euribor 3M Rate", value=4.8)

# Nominal (Dropdowns)
job = st.sidebar.selectbox("Job", options=['white-collar', 'blue-collar', 'retired', 'self-employed', 'student', 'unemployed'])
marital = st.sidebar.selectbox("Marital Status", options=['divorced', 'married', 'single'])
default = st.sidebar.selectbox("Has Credit in Default?", options=['no', 'yes', 'unknown'])
housing = st.sidebar.selectbox("Has Housing Loan?", options=['no', 'yes', 'unknown'])
loan = st.sidebar.selectbox("Has Personal Loan?", options=['no', 'yes', 'unknown'])
contact = st.sidebar.selectbox("Contact Channel", options=['cellular', 'telephone'])
month = st.sidebar.selectbox("Month of Last Contact", options=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
day_of_week = st.sidebar.selectbox("Day of Week of Last Contact", options=['mon', 'tue', 'wed', 'thu', 'fri'])
poutcome = st.sidebar.selectbox("Previous Outcome", options=['failure', 'nonexistent', 'success'])
# ... repeat for marital, default, housing, loan, contact, month, day_of_week, poutcome ...

# Ordinal & Binary
age_bin = st.sidebar.selectbox("Age Group", options=['Student_Age', 'Young_Adult', 'Adult', 'Senior'])
intensity = st.sidebar.selectbox("Campaign Intensity", options=['low', 'high'])

# Create a dictionary for mapping education levels
edu_map = {0: 'Illiterate', 1: 'Basic 4y', 2: 'Basic 6y', 3: 'Basic 9y', 4: 'High School', 5: 'Professional Course', 6: 'University Degree'}
# Use the map in the slider (or a selectbox)
edu_label = st.sidebar.selectbox("Education Level", options=list(edu_map.values()))
# Convert the label back to the number for the model
edu = [k for k, v in edu_map.items() if v == edu_label][0]

married = st.sidebar.selectbox("Is Married?", [0, 1])
never_contacted = st.sidebar.selectbox("Never Contacted?", [0, 1])
econ_grow = st.sidebar.selectbox("Economy Growing?", [0, 1])

# --- CRITICAL STEP: The DataFrame Construction ---
# The keys MUST match your 'eng_num', 'eng_nom', etc. lists exactly.
input_dict = {
    'campaign': campaign,
    'previous': previous,
    'cons.price.idx': cons_price,
    'cons.conf.idx': cons_conf,
    'euribor3m': euribor,
    'job_category': job,
    'marital': marital,
    'default': default,
    'housing': housing,
    'loan': loan,
    'contact': contact,
    'month': month,
    'day_of_week': day_of_week,
    'poutcome': poutcome,
    'age_bin': age_bin,
    'campaign_intensity': intensity,
    'education_level': edu,
    'is_married': married,
    'never_contacted': never_contacted,
    'economy_growing': econ_grow
}

input_df = pd.DataFrame([input_dict])

# prediction probability for the positive class (subscribing to term deposit)
prediction_prob = pipeline.predict_proba(input_df)[0, 1]

# Use the business threshold
is_subscriber = prediction_prob >= threshold

# Show Result
if is_subscriber:
    st.success(f"🎯 High Potential Lead! (Probability: {prediction_prob:.2%})")
    st.write("Recommendation: Proceed with Call.")
else:
    st.error(f"⚠️ Low Potential Lead. (Probability: {prediction_prob:.2%})")
    st.write("Recommendation: Do not call (Cost Optimization).")