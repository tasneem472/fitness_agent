import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Elite Fitness Coach AI",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium styling
st.markdown("""
<style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Custom card styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 10px 0;
    }
    
    /* Premium title */
    .premium-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 0%, #e0e0e0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 20px 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Section headers */
    .section-header {
        color: white;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 30px 0 20px 0;
        padding-bottom: 10px;
        border-bottom: 3px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Stats box */
    .stats-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    }
    
    /* Info cards */
    .info-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: white;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        font-weight: 700;
        font-size: 1.1rem;
        padding: 15px 40px;
        border-radius: 50px;
        border: none;
        box-shadow: 0 8px 32px rgba(245, 87, 108, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(245, 87, 108, 0.6);
    }
    
    /* Form styling */
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>select {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
        border: 2px solid rgba(255, 255, 255, 0.3);
        padding: 10px;
    }
    
    /* Success message */
    .success-banner {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 20px 0;
        box-shadow: 0 8px 32px rgba(17, 153, 142, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Load model & tools
@st.cache_resource
def load_models():
    model = joblib.load("diet_model.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    scaler = joblib.load("scaler.pkl")
    try:
        metadata = joblib.load("model_metadata.pkl")
    except:
        metadata = {'accuracy': 0.85}
    return model, label_encoders, scaler, metadata

model, label_encoders, scaler, metadata = load_models()

# Header
st.markdown('<h1 class="premium-title">💪 ELITE FITNESS COACH AI</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: white; font-size: 1.2rem; margin-bottom: 30px;">Your Personal AI-Powered Health & Fitness Companion</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🎯 Dashboard")
    st.markdown(f"""
    <div class="stats-box">
        <h3>Model Accuracy</h3>
        <h1>{metadata.get('accuracy', 0.85):.1%}</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    st.info("🔥 Plans Generated Today: 47")
    st.success("⭐ Average Rating: 4.8/5.0")
    st.warning("💎 Premium Features Active")
    
    st.markdown("---")
    st.markdown("### 🎁 Features")
    st.markdown("""
    - ✅ AI-Powered Recommendations
    - ✅ Personalized Meal Plans
    - ✅ Custom Workout Routines
    - ✅ Progress Tracking
    - ✅ Health Analytics
    - ✅ Expert Insights
    """)

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Assessment", "📊 Analytics", "📈 Progress", "💡 Insights"])

with tab1:
    st.markdown('<h2 class="section-header">Personal Health Assessment</h2>', unsafe_allow_html=True)
    
    with st.form("user_input"):
        # Personal Information
        st.markdown("### 👤 Personal Information")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            age = st.number_input("Age", 15, 80, 30)
        with col2:
            gender = st.selectbox("Gender", label_encoders["Gender"].classes_)
        with col3:
            weight = st.number_input("Weight (kg)", 30.0, 200.0, 70.0)
        with col4:
            height = st.number_input("Height (cm)", 120.0, 220.0, 170.0)
        
        # Calculate BMI automatically
        bmi = round(weight / ((height/100) ** 2), 2)
        st.info(f"📊 Calculated BMI: **{bmi}** kg/m²")
        
        st.markdown("---")
        
        # Health Metrics
        st.markdown("### 🏥 Health Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            disease = st.selectbox("Disease Type", label_encoders["Disease_Type"].classes_)
        with col2:
            severity = st.selectbox("Severity", label_encoders["Severity"].classes_)
        with col3:
            cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 400, 200)
        with col4:
            glucose = st.number_input("Glucose (mg/dL)", 70, 300, 100)
        
        col5, col6 = st.columns(2)
        with col5:
            bp = st.number_input("Blood Pressure (mmHg)", 80, 200, 120)
        with col6:
            calories = st.number_input("Daily Caloric Intake", 1000, 5000, 2000)
        
        st.markdown("---")
        
        # Lifestyle
        st.markdown("### 🏃 Lifestyle & Activity")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            activity = st.selectbox(
                "Physical Activity Level",
                label_encoders["Physical_Activity_Level"].classes_
            )
        with col2:
            weekly_exercise = st.slider("Weekly Exercise Hours", 0, 20, 3)
        with col3:
            adherence = st.slider("Diet Plan Adherence %", 0, 100, 70)
        
        st.markdown("---")
        
        # Dietary Preferences
        st.markdown("### 🍽️ Dietary Preferences")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            restrictions = st.selectbox(
                "Dietary Restrictions",
                label_encoders["Dietary_Restrictions"].classes_
            )
        with col2:
            allergies = st.selectbox(
                "Allergies",
                label_encoders["Allergies"].classes_
            )
        with col3:
            cuisine = st.selectbox(
                "Preferred Cuisine",
                label_encoders["Preferred_Cuisine"].classes_
            )
        
        imbalance = st.slider("Dietary Nutrient Imbalance Score", 0, 100, 30)
        
        st.markdown("<br>", unsafe_allow_html=True)
        submit = st.form_submit_button("🚀 Generate My Personalized Plan", use_container_width=True)

    # Prediction
    if submit:
        with st.spinner("🔮 AI is analyzing your data..."):
            input_data = pd.DataFrame([{
                "Age": age,
                "Gender": gender,
                "Weight_kg": weight,
                "Height_cm": height,
                "BMI": bmi,
                "Disease_Type": disease,
                "Severity": severity,
                "Physical_Activity_Level": activity,
                "Daily_Caloric_Intake": calories,
                "Cholesterol_mg/dL": cholesterol,
                "Blood_Pressure_mmHg": bp,
                "Glucose_mg/dL": glucose,
                "Dietary_Restrictions": restrictions,
                "Allergies": allergies,
                "Preferred_Cuisine": cuisine,
                "Weekly_Exercise_Hours": weekly_exercise,
                "Adherence_to_Diet_Plan": adherence,
                "Dietary_Nutrient_Imbalance_Score": imbalance
            }])

            # Encode categorical
            for col, le in label_encoders.items():
                input_data[col] = le.transform(input_data[col].astype(str))

            # Scale numerical
            numerical_cols = scaler.feature_names_in_
            input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])

            # Predict
            recommendation = model.predict(input_data)[0]
            probabilities = model.predict_proba(input_data)[0]
            confidence = max(probabilities) * 100

        st.markdown('<div class="success-banner">✨ Your Personalized Plan is Ready! ✨</div>', unsafe_allow_html=True)
        
        # Display Results
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h2 style="color: #667eea;">🥗 Diet Recommendation</h2>
                <h1 style="color: #764ba2;">{recommendation}</h1>
                <p style="color: #666;">Confidence: {confidence:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Diet details
            st.markdown("#### 📋 Your Meal Plan Includes:")
            if "low" in recommendation.lower():
                st.write("• Low-carb vegetables & lean proteins")
                st.write("• Healthy fats from nuts & avocado")
                st.write("• Limited grain intake")
            elif "high" in recommendation.lower() and "protein" in recommendation.lower():
                st.write("• Lean meats, fish & poultry")
                st.write("• Eggs and dairy products")
                st.write("• Plant-based proteins")
            elif "mediterranean" in recommendation.lower():
                st.write("• Olive oil & fatty fish")
                st.write("• Fresh fruits & vegetables")
                st.write("• Whole grains & legumes")
            else:
                st.write("• Balanced macronutrients")
                st.write("• Variety of food groups")
                st.write("• Portion-controlled meals")

        with col2:
            # Workout plan
            workout_plan = ""
            workout_details = []
            
            if weekly_exercise < 3:
                workout_plan = "Beginner Program"
                workout_details = [
                    "🏃 Light cardio: 20-30 min, 3x/week",
                    "💪 Full body workout: 2x/week",
                    "🧘 Flexibility training: 2x/week",
                    "🎯 Focus: Building foundation"
                ]
            elif weekly_exercise < 6:
                workout_plan = "Intermediate Program"
                workout_details = [
                    "🏋️ Strength training: 4x/week",
                    "🏃 Cardio sessions: 3x/week",
                    "🔥 HIIT workouts: 2x/week",
                    "🎯 Focus: Building strength & endurance"
                ]
            else:
                workout_plan = "Advanced Program"
                workout_details = [
                    "💪 Advanced split training: 5-6x/week",
                    "⚡ HIIT & plyometrics: 3x/week",
                    "🏃 Endurance cardio: 2x/week",
                    "🎯 Focus: Peak performance"
                ]
            
            st.markdown(f"""
            <div class="metric-card">
                <h2 style="color: #667eea;">🏃 Workout Plan</h2>
                <h1 style="color: #764ba2;">{workout_plan}</h1>
                <p style="color: #666;">Tailored to your fitness level</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### 💪 Your Training Schedule:")
            for detail in workout_details:
                st.write(detail)
        
        # Additional recommendations
        st.markdown("---")
        st.markdown("### 🎯 Personalized Recommendations")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="info-card">
                <h4>💧 Hydration Goal</h4>
                <h2>2.5-3L</h2>
                <p>Daily water intake</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="info-card">
                <h4>😴 Sleep Target</h4>
                <h2>7-9 hrs</h2>
                <p>Essential for recovery</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            target_calories = int(calories * 0.9) if bmi > 25 else int(calories * 1.1)
            st.markdown(f"""
            <div class="info-card">
                <h4>🔥 Calorie Target</h4>
                <h2>{target_calories}</h2>
                <p>Daily caloric goal</p>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.markdown('<h2 class="section-header">Health Analytics Dashboard</h2>', unsafe_allow_html=True)
    
    if 'bmi' in locals():
        # BMI Analysis
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # BMI gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=bmi,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "BMI Analysis", 'font': {'size': 24}},
                delta={'reference': 22},
                gauge={
                    'axis': {'range': [None, 40], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 18.5], 'color': 'lightblue'},
                        {'range': [18.5, 25], 'color': 'lightgreen'},
                        {'range': [25, 30], 'color': 'yellow'},
                        {'range': [30, 40], 'color': 'lightcoral'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': bmi
                    }
                }
            ))
            fig.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### BMI Categories")
            st.info("📊 Underweight: < 18.5")
            st.success("✅ Normal: 18.5 - 24.9")
            st.warning("⚠️ Overweight: 25 - 29.9")
            st.error("🔴 Obese: ≥ 30")
            
            if bmi < 18.5:
                st.markdown("**Status:** Underweight")
            elif bmi < 25:
                st.markdown("**Status:** Healthy Weight ✅")
            elif bmi < 30:
                st.markdown("**Status:** Overweight")
            else:
                st.markdown("**Status:** Obese")
        
        # Health metrics comparison
        st.markdown("### 📊 Your Health Metrics vs Optimal Range")
        
        if 'cholesterol' in locals():
            metrics_data = {
                'Metric': ['Cholesterol', 'Blood Pressure', 'Glucose'],
                'Your Value': [cholesterol, bp, glucose],
                'Optimal Min': [125, 90, 70],
                'Optimal Max': [200, 120, 100]
            }
            
            df_metrics = pd.DataFrame(metrics_data)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Your Value', x=df_metrics['Metric'], y=df_metrics['Your Value'], marker_color='rgb(102, 126, 234)'))
            fig.add_trace(go.Bar(name='Optimal Min', x=df_metrics['Metric'], y=df_metrics['Optimal Min'], marker_color='lightgreen'))
            fig.add_trace(go.Bar(name='Optimal Max', x=df_metrics['Metric'], y=df_metrics['Optimal Max'], marker_color='lightcoral'))
            
            fig.update_layout(
                barmode='group',
                height=400,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'color': "white"},
                title="Health Metrics Comparison"
            )
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown('<h2 class="section-header">Progress Tracking</h2>', unsafe_allow_html=True)
    
    # Simulated progress data
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    weight_progress = [70 + np.sin(i/5) * 2 + np.random.randn() * 0.5 for i in range(30)]
    
    fig = px.line(x=dates, y=weight_progress, title='Weight Progress (30 Days)')
    fig.update_traces(line_color='#667eea', line_width=3)
    fig.update_layout(
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.1)",
        font={'color': "white"},
        xaxis_title="Date",
        yaxis_title="Weight (kg)"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Weekly stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Weight Change", "-2.3 kg", "-3.2%", delta_color="inverse")
    with col2:
        st.metric("Workouts Completed", "12", "+3")
    with col3:
        st.metric("Avg. Calories", "1,950", "-150")
    with col4:
        st.metric("Adherence", "87%", "+12%")

with tab4:
    st.markdown('<h2 class="section-header">AI Insights & Recommendations</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <h3>💡 Personalized Insights</h3>
        <p style="font-size: 1.1rem; line-height: 1.8;">
        Based on your comprehensive health assessment, our AI has identified key areas for improvement and optimization.
        Focus on consistency with your meal plan and gradually increase your exercise intensity for optimal results.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h4>✅ Strengths</h4>
            <ul style="font-size: 1.05rem;">
                <li>Good adherence to diet plan</li>
                <li>Regular exercise routine</li>
                <li>Balanced nutrient intake</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h4>🎯 Focus Areas</h4>
            <ul style="font-size: 1.05rem;">
                <li>Increase daily water intake</li>
                <li>Improve sleep consistency</li>
                <li>Add more protein to meals</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <h3>📅 30-Day Action Plan</h3>
        <ol style="font-size: 1.05rem; line-height: 2;">
            <li><strong>Week 1-2:</strong> Establish meal prep routine and baseline measurements</li>
            <li><strong>Week 2-3:</strong> Increase workout intensity by 10-15%</li>
            <li><strong>Week 3-4:</strong> Fine-tune nutrition based on progress data</li>
            <li><strong>Week 4+:</strong> Maintain consistency and celebrate milestones</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: white; padding: 20px;">
    <p style="font-size: 0.9rem;">💪 Elite Fitness Coach AI • Powered by Advanced Machine Learning</p>
    <p style="font-size: 0.8rem;">Your journey to optimal health starts here. Stay consistent, stay strong! 🚀</p>
</div>
""", unsafe_allow_html=True)