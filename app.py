import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Healthcare AI: Disease Predictor", page_icon="🩺", layout="wide")

# --- DATA & MODEL LOADING (Cached for speed) ---
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigree', 'Age', 'Outcome']
    df = pd.read_csv(url, names=column_names)
    return df

@st.cache_resource
def train_model(df):
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, X_test_scaled, y_test, X.columns

df = load_data()
model, scaler, X_test_scaled, y_test, feature_names = train_model(df)

# --- SIDEBAR ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=100)
st.sidebar.title("Navigation")
st.sidebar.info("This dashboard uses a Random Forest Machine Learning model to predict disease risk based on patient vitals.")

# --- MAIN DASHBOARD ---
st.title("🩺 Healthcare Data Analysis & Disease Prediction")
st.markdown("Explore the patient dataset, evaluate the AI model, and simulate live patient diagnoses.")

# Create 3 Tabs for a clean presentation
tab1, tab2, tab3 = st.tabs(["📊 Exploratory Data Analysis", "🧠 Model Performance", "👨‍⚕️ Patient Simulator"])

# ==========================================
# TAB 1: EDA
# ==========================================
with tab1:
    st.header("Patient Population Overview")
    
    # Top Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Patients in Dataset", len(df))
    col2.metric("Healthy Patients", len(df[df['Outcome']==0]))
    col3.metric("Diagnosed Patients", len(df[df['Outcome']==1]))
    
    st.markdown("---")
    
    # Interactive Plotly Charts
    colA, colB = st.columns(2)
    
    with colA:
        st.subheader("Glucose Levels by Diagnosis")
        fig_hist = px.histogram(df, x="Glucose", color="Outcome", barmode="overlay", 
                                color_discrete_sequence=["#2ecc71", "#e74c3c"],
                                labels={"Outcome": "Diagnosis (1=Sick)"})
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with colB:
        st.subheader("Age vs BMI Distribution")
        fig_scatter = px.scatter(df, x="Age", y="BMI", color="Outcome", opacity=0.7,
                                 color_discrete_sequence=["#2ecc71", "#e74c3c"])
        st.plotly_chart(fig_scatter, use_container_width=True)

# ==========================================
# TAB 2: MODEL PERFORMANCE
# ==========================================
with tab2:
    st.header("AI Model Evaluation")
    
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    st.success(f"**Overall Model Accuracy:** {acc*100:.2f}%")
    
    colA, colB = st.columns(2)
    
    with colA:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                           labels=dict(x="Predicted", y="Actual"),
                           x=['Healthy', 'At-Risk'], y=['Healthy', 'At-Risk'])
        st.plotly_chart(fig_cm, use_container_width=True)
        
    with colB:
        st.subheader("Feature Importance")
        importances = model.feature_importances_
        # Sort features by importance
        indices = np.argsort(importances)
        fig_feat = px.bar(x=importances[indices], y=feature_names[indices], orientation='h',
                          labels={'x':'Importance', 'y':'Feature'}, color_discrete_sequence=["#3498db"])
        st.plotly_chart(fig_feat, use_container_width=True)

# ==========================================
# TAB 3: PATIENT SIMULATOR (The "Wow" Factor)
# ==========================================
with tab3:
    st.header("Live Patient Risk Simulator")
    st.markdown("Enter a patient's vitals below to get a real-time AI risk prediction.")
    
    # Create a nice form layout
    with st.form("patient_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
            glucose = st.slider("Glucose Level", 50, 250, 120)
            bp = st.slider("Blood Pressure", 50, 150, 80)
            
        with col2:
            skin = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
            insulin = st.slider("Insulin Level", 0, 900, 80)
            bmi = st.slider("BMI", 10.0, 70.0, 25.0)
            
        with col3:
            pedigree = st.slider("Diabetes Pedigree (Genetics)", 0.0, 3.0, 0.5)
            age = st.slider("Age", 1, 120, 30)
            
        submit = st.form_submit_button("Predict Risk Score")

    # Run Prediction when button is clicked
    if submit:
        # 1. Gather inputs into an array
        patient_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, pedigree, age]])
        
        # 2. Scale the input using the same scaler from training!
        patient_scaled = scaler.transform(patient_data)
        
        # 3. Get probability
        risk_probability = model.predict_proba(patient_scaled)[0][1] * 100
        
        st.markdown("---")
        
        # Display a beautiful Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_probability,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Disease Risk Probability (%)"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "black"},
                'steps': [
                    {'range': [0, 33], 'color': "lightgreen"},
                    {'range': [33, 66], 'color': "yellow"},
                    {'range': [66, 100], 'color': "red"}],
                'threshold' : {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': risk_probability}
            }
        ))
        
        col_res1, col_res2 = st.columns([1, 2])
        with col_res1:
            if risk_probability > 50:
                st.error("🚨 **High Risk Detected!**")
                st.markdown("The AI recommends immediate medical consultation based on the patient's elevated metrics (particularly Glucose and BMI).")
            else:
                st.success("✅ **Low Risk**")
                st.markdown("The patient's metrics fall within normal, healthy ranges according to the model.")
                
        with col_res2:
            st.plotly_chart(fig_gauge, use_container_width=True)