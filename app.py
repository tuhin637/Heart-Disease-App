import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
    }
    .main-header {
        color: white;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #b3b3b3;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background-color: #262730;
        color: white;
        border: 1px solid #4a4a4a;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-size: 1rem;
        font-weight: 500;
        width: 100%;
        margin-top: 1rem;
    }
    .result-box {
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin-top: 2rem;
    }
    .result-title {
        color: #000;
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .result-status {
        color: #000;
        font-size: 1.3rem;
        font-weight: 600;
    }
    .debug-text {
        color: #b3b3b3;
        font-size: 0.9rem;
        margin-top: 1rem;
    }
    label {
        color: white !important;
        font-size: 0.95rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Simple pre-trained model (for demo without large files)
@st.cache_resource
def load_demo_model():
    """Load a lightweight demo model"""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    # Create dummy training data (17 features)
    np.random.seed(42)
    X_dummy = np.random.randn(1000, 17)
    y_dummy = np.random.randint(0, 2, 1000)
    model.fit(X_dummy, y_dummy)
    return model

model = load_demo_model()

# Header
st.markdown('<h1 class="main-header">Heart Disease Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Enter patient data to predict the risk of heart disease.</p>', unsafe_allow_html=True)

st.info("ℹ️ Demo Version: Using a lightweight model for demonstration purposes.")

# Input fields
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.5)

smoking = st.selectbox("Do you smoke?", 
                       options=[0, 1], 
                       format_func=lambda x: "No" if x == 0 else "Yes",
                       index=0)

alcohol_drinking = st.selectbox("Do you consume alcohol?", 
                                options=[0, 1], 
                                format_func=lambda x: "No" if x == 0 else "Yes",
                                index=0)

stroke = st.selectbox("Have you ever had a stroke?", 
                      options=[0, 1], 
                      format_func=lambda x: "No" if x == 0 else "Yes",
                      index=0)

physical_health = st.slider("Physical Health (number of bad days in past month)", 0, 30, 0)

mental_health = st.slider("Mental Health (number of bad days in past month)", 0, 30, 0)

diff_walking = st.selectbox("Difficulty Walking?", 
                            options=[0, 1], 
                            format_func=lambda x: "No" if x == 0 else "Yes",
                            index=0)

sex = st.selectbox("Sex", 
                   options=[0, 1], 
                   format_func=lambda x: "Female" if x == 0 else "Male",
                   index=1)

age_categories = ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", 
                  "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80 or older"]
age_category = st.selectbox("Age Category", 
                            options=list(range(len(age_categories))), 
                            format_func=lambda x: age_categories[x],
                            index=6)

races = ["American Indian/Alaskan Native", "Asian", "Black", "Hispanic", "Other", "White"]
race = st.selectbox("Race", 
                    options=list(range(len(races))), 
                    format_func=lambda x: races[x],
                    index=5)

diabetic_options = ["No", "No, borderline diabetes", "Yes", "Yes (during pregnancy)"]
diabetic = st.selectbox("Diabetic", 
                        options=list(range(len(diabetic_options))), 
                        format_func=lambda x: diabetic_options[x],
                        index=0)

physical_activity = st.selectbox("Physically Active?", 
                                 options=[0, 1], 
                                 format_func=lambda x: "No" if x == 0 else "Yes",
                                 index=1)

gen_health_options = ["Excellent", "Fair", "Good", "Poor", "Very good"]
gen_health = st.selectbox("General Health", 
                          options=list(range(len(gen_health_options))), 
                          format_func=lambda x: gen_health_options[x],
                          index=2)

sleep_time = st.slider("Average Sleep Time (hours)", 0, 24, 7)

asthma = st.selectbox("Asthma", 
                      options=[0, 1], 
                      format_func=lambda x: "No" if x == 0 else "Yes",
                      index=0)

kidney_disease = st.selectbox("Kidney Disease", 
                              options=[0, 1], 
                              format_func=lambda x: "No" if x == 0 else "Yes",
                              index=0)

skin_cancer = st.selectbox("Skin Cancer", 
                           options=[0, 1], 
                           format_func=lambda x: "No" if x == 0 else "Yes",
                           index=0)

# Predict button
if st.button("Predict"):
    try:
        # Create input array
        input_data = np.array([[
            bmi, smoking, alcohol_drinking, stroke, physical_health, mental_health,
            diff_walking, sex, age_category, race, diabetic, physical_activity,
            gen_health, sleep_time, asthma, kidney_disease, skin_cancer
        ]])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        # Debug info
        st.markdown(f'<p class="debug-text">Debug: Probability = {probability[1]:.3f}</p>', unsafe_allow_html=True)
        
        # Result box
        if prediction == 1:
            result_text = "Heart Disease Positive."
            result_color = "#ff6b6b"
            emoji = "⚠️"
        else:
            result_text = "Heart Disease Negative."
            result_color = "#90ee90"
            emoji = "✅"
        
        st.markdown(f"""
            <div class="result-box" style="background-color: {result_color};">
                <p class="result-title">{emoji} Result of Disease!!</p>
                <p class="result-status">{result_text}</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Additional metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Risk Probability", f"{probability[1]*100:.1f}%")
        with col2:
            st.metric("Confidence", f"{max(probability)*100:.1f}%")
            
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>⚠️ This is a demo version for educational purposes only.</p>
        <p>Always consult healthcare professionals for medical advice.</p>
    </div>
""", unsafe_allow_html=True)