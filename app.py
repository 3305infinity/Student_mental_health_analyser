# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# from model_training import AdvancedStudentHealthModel
# import google.generativeai as genai
# import os

# # Page configuration
# st.set_page_config(
#     page_title="Student Mental Health Analyzer Pro",
#     page_icon="🎓",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for better styling
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 3rem;
#         color: #1f77b4;
#         text-align: center;
#         margin-bottom: 2rem;
#     }
#     .metric-card {
#         background-color: #f0f2f6;
#         padding: 1.5rem;
#         border-radius: 10px;
#         border-left: 5px solid #1f77b4;
#     }
#     .risk-high {
#         background-color: #ffcccc;
#         padding: 1rem;
#         border-radius: 10px;
#         border-left: 5px solid #ff0000;
#     }
#     .risk-medium {
#         background-color: #fff4cc;
#         padding: 1rem;
#         border-radius: 10px;
#         border-left: 5px solid #ffaa00;
#     }
#     .risk-low {
#         background-color: #ccffcc;
#         padding: 1rem;
#         border-radius: 10px;
#         border-left: 5px solid #00aa00;
#     }
#     .model-performance {
#         background-color: #e6f3ff;
#         padding: 1rem;
#         border-radius: 10px;
#         margin: 0.5rem 0;
#     }
# </style>
# """, unsafe_allow_html=True)

# @st.cache_resource
# def initialize_model():
#     return AdvancedStudentHealthModel()

# def setup_gemini(api_key):
#     try:
#         genai.configure(api_key=api_key)
#         return genai.GenerativeModel('gemini-pro')
#     except Exception as e:
#         st.error(f"Gemini setup error: {str(e)}")
#         return None

# def get_ai_advice(user_data, prediction, gemini_model):
#     prompt = f"""
#     Student Mental Health Analysis - Provide comprehensive advice in Hindi/English mix:
    
#     STUDENT PROFILE:
#     - Age: {user_data['Age']}
#     - Gender: {user_data['Gender']}
#     - Academic Pressure: {user_data['Academic Pressure']}/5
#     - Study Satisfaction: {user_data['Study Satisfaction']}/5  
#     - Sleep: {user_data['Sleep Duration']}
#     - Diet: {user_data['Dietary Habits']}
#     - Study Hours: {user_data['Study Hours']} hours/day
#     - Financial Stress: {user_data['Financial Stress']}/5
#     - Family Mental Health History: {user_data['Family History of Mental Illness']}
#     - Past Suicidal Thoughts: {user_data['Have you ever had suicidal thoughts ?']}
    
#     HEALTH ASSESSMENT:
#     - Depression Risk: {prediction['prediction']}
#     - Risk Level: {prediction['risk_level']}
#     - Confidence: {prediction['confidence']:.1%}
    
#     Please provide:
#     1. IMMEDIATE ACTIONS (2-3 turant steps)
#     2. LONG-TERM STRATEGIES (lambe samay ke upay)
#     3. STUDY-LIFE BALANCE TIPS (padhai aur life balance)
#     4. MENTAL HEALTH RESOURCES (sahyog ke sadhan)
#     5. POSITIVE ENCOURAGEMENT (sakaratmak salah)
    
#     Keep it practical, compassionate and student-friendly. Use Hindi/English mix.
#     """
    
#     try:
#         response = gemini_model.generate_content(prompt)
#         return response.text
#     except Exception as e:
#         return f"AI advice temporarily unavailable. Error: {str(e)}"

# def main():
#     st.markdown('<h1 class="main-header">🎓 Student Mental Health Analyzer Pro</h1>', unsafe_allow_html=True)
    
#     # Initialize
#     health_model = initialize_model()
    
#     # Sidebar
#     with st.sidebar:
#         st.header("⚙️ Configuration")
        
#         # Gemini API
#         gemini_key = st.text_input("Enter Gemini API Key:", type="password",
#                                 help="Get from Google AI Studio")
#         if gemini_key:
#             st.session_state.gemini_model = setup_gemini(gemini_key)
#             if st.session_state.gemini_model:
#                 st.success("✅ Gemini Connected!")
        
#         st.markdown("---")
#         st.header("🤖 ML Model Training")
        
#         csv_path = "Depression Student Dataset.csv"
        
#         if st.button("🚀 Train All ML Models", type="primary", use_container_width=True):
#             with st.spinner("Training 8 different ML models... This may take a minute"):
#                 if os.path.exists(csv_path):
#                     results = health_model.train_all_models(csv_path)
#                     st.session_state.model_trained = True
#                     st.session_state.health_model = health_model
#                     st.session_state.model_results = results
                    
#                     # Show best model
#                     best_model = health_model.best_model
#                     best_score = health_model.best_score
#                     st.success(f"✅ Training Complete! Best Model: {best_model} ({best_score:.3f})")
#                 else:
#                     st.error(f"❌ CSV file not found: {csv_path}")
        
#         if st.button("📥 Load Trained Models", use_container_width=True):
#             if health_model.load_models():
#                 st.session_state.model_trained = True
#                 st.session_state.health_model = health_model
#                 st.success("✅ Models Loaded!")
#             else:
#                 st.error("❌ No trained models found")
        
#         if st.session_state.get('model_trained', False):
#             st.markdown("---")
#             st.header("📈 Model Performance")
#             if st.session_state.get('model_results'):
#                 for name, result in st.session_state.model_results.items():
#                     with st.container():
#                         st.markdown(f"""
#                         <div class="model-performance">
#                             <b>{name}</b><br>
#                             Accuracy: {result['accuracy']:.3f} | CV Score: {result['cv_mean']:.3f}
#                         </div>
#                         """, unsafe_allow_html=True)
    
#     # Main content
#     if not st.session_state.get('model_trained', False):
#         col1, col2 = st.columns([2, 1])
#         with col1:
#             st.info("👈 Please train ML models using the sidebar first")
#             st.markdown("""
#             ### 🎯 Features:
#             - **8 ML Models**: Random Forest, Gradient Boosting, SVM, Logistic Regression, Decision Tree, KNN, Naive Bayes, AdaBoost
#             - **Performance Comparison**: Auto-selects best performing model
#             - **AI-Powered Advice**: Gemini integration for personalized recommendations
#             - **Beautiful Visualizations**: Interactive charts and metrics
#             - **Risk Analysis**: Comprehensive health assessment
#             """)
#         with col2:
#             st.image("https://cdn-icons-png.flaticon.com/512/3079/3079165.png", width=150)
#         return
    
#     # Input forms in tabs
#     tab1, tab2, tab3 = st.tabs(["👤 Student Info", "📚 Academic Details", "🏥 Health History"])
    
#     with tab1:
#         col1, col2 = st.columns(2)
#         with col1:
#             gender = st.selectbox("Gender", ["Male", "Female", "Other"])
#             age = st.slider("Age", 16, 35, 20)
#             sleep_duration = st.selectbox("Sleep Duration", 
#                                        ["Less than 5 hours", "5-6 hours", "6-7 hours", "7-8 hours", "More than 8 hours"])
#         with col2:
#             dietary_habits = st.selectbox("Diet Quality", 
#                                         ["Unhealthy", "Moderate", "Healthy", "Very Healthy", "Very Unhealthy"])
#             study_hours = st.slider("Daily Study Hours", 0, 16, 6)
    
#     with tab2:
#         col1, col2 = st.columns(2)
#         with col1:
#             academic_pressure = st.slider("Academic Pressure", 1.0, 5.0, 3.0, 0.1,
#                                        help="1 = Low pressure, 5 = High pressure")
#             study_satisfaction = st.slider("Study Satisfaction", 1.0, 5.0, 3.0, 0.1,
#                                         help="1 = Very dissatisfied, 5 = Very satisfied")
#         with col2:
#             financial_stress = st.slider("Financial Stress", 1.0, 5.0, 2.0, 0.1,
#                                        help="1 = No stress, 5 = High stress")
    
#     with tab3:
#         col1, col2 = st.columns(2)
#         with col1:
#             family_history = st.radio("Family History of Mental Illness", 
#                                    ["No", "Yes"], horizontal=True)
#         with col2:
#             suicidal_thoughts = st.radio("Past Suicidal Thoughts", 
#                                       ["No", "Yes"], horizontal=True)
    
#     # Analyze button
#     col1, col2, col3 = st.columns([1, 2, 1])
#     with col2:
#         analyze_clicked = st.button("🔍 Analyze Mental Health Risk", 
#                                  type="primary", use_container_width=True)
    
#     if analyze_clicked:
#         user_data = {
#             'Gender': gender,
#             'Age': age,
#             'Academic Pressure': academic_pressure,
#             'Study Satisfaction': study_satisfaction,
#             'Sleep Duration': sleep_duration,
#             'Dietary Habits': dietary_habits,
#             'Study Hours': study_hours,
#             'Financial Stress': financial_stress,
#             'Family History of Mental Illness': family_history,
#             'Have you ever had suicidal thoughts ?': suicidal_thoughts
#         }
        
#         with st.spinner("🤖 Analyzing with best ML model..."):
#             prediction = st.session_state.health_model.predict(user_data)
        
#         # Display results
#         st.markdown("---")
        
#         # Risk Summary
#         col1, col2, col3, col4 = st.columns(4)
        
#         with col1:
#             st.metric("Depression Risk", prediction['prediction'])
#         with col2:
#             risk_color = "🔴" if prediction['risk_level'] == 'High' else "🟡" if prediction['risk_level'] == 'Medium' else "🟢"
#             st.metric("Risk Level", f"{risk_color} {prediction['risk_level']}")
#         with col3:
#             st.metric("Confidence", f"{prediction['confidence']:.1%}")
#         with col4:
#             st.metric("Best Model", prediction['best_model_used'])
        
#         # Risk visualization
#         st.subheader("📊 Risk Analysis")
#         fig = make_subplots(rows=1, cols=2, 
#                           subplot_titles=('Risk Probabilities', 'Factor Analysis'),
#                           specs=[[{"type": "pie"}, {"type": "bar"}]])
        
#         # Pie chart
#         fig.add_trace(go.Pie(
#             labels=list(prediction['probabilities'].keys()),
#             values=list(prediction['probabilities'].values()),
#             hole=0.4,
#             marker_colors=['#00cc96', '#ef553b']
#         ), 1, 1)
        
#         # Bar chart for factors
#         factors = ['Academic Pressure', 'Financial Stress', 'Sleep Quality', 'Study Satisfaction']
#         scores = [
#             user_data['Academic Pressure'],
#             user_data['Financial Stress'],
#             8.0 if '8+' in user_data['Sleep Duration'] else 
#             7.5 if '7-8' in user_data['Sleep Duration'] else
#             6.0 if '6-7' in user_data['Sleep Duration'] else
#             5.5 if '5-6' in user_data['Sleep Duration'] else 4.5,
#             user_data['Study Satisfaction']
#         ]
        
#         fig.add_trace(go.Bar(
#             x=factors,
#             y=scores,
#             marker_color=['#ff9999', '#ffcc99', '#99ff99', '#99ccff']
#         ), 1, 2)
        
#         fig.update_layout(height=400, showlegend=False)
#         st.plotly_chart(fig, use_container_width=True)
        
#         # AI Advice
#         if 'gemini_model' in st.session_state:
#             st.subheader("🤖 AI Health Advisor Suggestions")
#             with st.spinner("Generating personalized AI recommendations..."):
#                 advice = get_ai_advice(user_data, prediction, st.session_state.gemini_model)
            
#             st.success(advice)
#         else:
#             st.warning("🔑 Add Gemini API key for AI-powered health advice")
        
#         # Action Plan
#         st.subheader("🎯 Recommended Action Plan")
        
#         if prediction['risk_level'] == 'High':
#             st.markdown("""
#             <div class="risk-high">
#             <h4>🚨 HIGH RISK - IMMEDIATE ACTION NEEDED</h4>
#             <ul>
#             <li><b>Contact campus counseling immediately</b></li>
#             <li>Speak with trusted faculty/advisor today</li>
#             <li>Reach out to family/friends for support</li>
#             <li>Call mental health helpline if needed</li>
#             <li>Ensure 7-8 hours sleep and proper nutrition</li>
#             </ul>
#             </div>
#             """, unsafe_allow_html=True)
#         elif prediction['risk_level'] == 'Medium':
#             st.markdown("""
#             <div class="risk-medium">
#             <h4>⚠️ MEDIUM RISK - PREVENTIVE ACTIONS</h4>
#             <ul>
#             <li>Schedule appointment with campus health services</li>
#             <li>Practice daily stress management techniques</li>
#             <li>Maintain consistent sleep schedule</li>
#             <li>Stay connected with supportive friends</li>
#             <li>Consider talking to a counselor</li>
#             </ul>
#             </div>
#             """, unsafe_allow_html=True)
#         else:
#             st.markdown("""
#             <div class="risk-low">
#             <h4>✅ LOW RISK - WELLNESS MAINTENANCE</h4>
#             <ul>
#             <li>Continue healthy lifestyle habits</li>
#             <li>Regular physical activity (30 mins daily)</li>
#             <li>Balanced study routine with breaks</li>
#             <li>Maintain social connections</li>
#             <li>Practice mindfulness and relaxation</li>
#             </ul>
#             </div>
#             """, unsafe_allow_html=True)

# if __name__ == "__main__":
#     main()



import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from model_training import AdvancedStudentHealthModel
import google.generativeai as genai
import os
# -------------------------------
# Streamlit page configuration
st.set_page_config(
    page_title="Student Mental Health Analyzer Pro",
    page_icon="🎓",
    layout="wide"
)


def setup_gemini():
    # ✅ First try to read from environment variable
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    # ✅ If not set in environment, fallback to hardcoded key
    if not GEMINI_API_KEY:
        GEMINI_API_KEY = "AIzaSyDXYDzDMAyWwwOwq5VEj1YQ3ZIY0jvirnE"  # <-- Put your key here safely

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # ✅ Use a supported model
        return genai.GenerativeModel("gemini-2.5-flash")
    except Exception as e:
        st.error(f"Gemini setup error: {str(e)}")
        return None

def get_ai_advice(user_data, prediction, gemini_model):
    # prompt = f"""
    # STUDENT PROFILE:
    # Age: {user_data['Age']}, Gender: {user_data['Gender']}
    # Sleep: {user_data['Sleep Duration']}, Diet: {user_data['Dietary Habits']}
    # Study Hours: {user_data['Study Hours']}, Academic Pressure: {user_data['Academic Pressure']}
    # Study Satisfaction: {user_data['Study Satisfaction']}, Financial Stress: {user_data['Financial Stress']}
    # Family History: {user_data['Family History of Mental Illness']}, Past Suicidal Thoughts: {user_data['Have you ever had suicidal thoughts ?']}

    # TASK:
    # 1. Assess mental health risk (Low, Medium, High)
    # 2. Suggest 3 immediate actions (1-2 sentences each)
    # 3. Suggest 3 long-term strategies (bullet points)
    # 4. Suggest 3 study-life balance tips
    # 5. Provide 1-2 motivational sentences

    # OUTPUT FORMAT:
    # - Immediate Actions:
    #   - ...
    # - Long-Term Strategies:
    #   - ...
    # - Study-Life Balance Tips:
    #   - ...
    # - Motivation:
    #   - ...
    # """
    prompt = f"""
Student Mental Health Analysis:
Profile:
- Age: {user_data['Age']}
- Gender: {user_data['Gender']}
- Academic Pressure: {user_data['Academic Pressure']}/5
- Study Satisfaction: {user_data['Study Satisfaction']}/5
- Sleep: {user_data['Sleep Duration']}
- Diet: {user_data['Dietary Habits']}
- Study Hours: {user_data['Study Hours']} hours/day
- Financial Stress: {user_data['Financial Stress']}/5
- Family Mental Health History: {user_data['Family History of Mental Illness']}
- Past Suicidal Thoughts: {user_data['Have you ever had suicidal thoughts ?']}
ML Assessment:
- Predicted Depression: {prediction['prediction']}
- Risk Level: {prediction['risk_level']}
- Confidence: {prediction['confidence']:.1%}
Instructions for AI:
1. Analyze the risk level carefully based on ML prediction.
2. If risk is HIGH, give **practical, urgent actions**.
3. If risk is MEDIUM, give **preventive measures**.
4. If risk is LOW, give **wellness maintenance tips**.
5. Provide 3-5 actionable points for each section.
6. Give advice in English, concise and student-friendly.
7. Avoid generic statements like "sleep well", make actionable suggestions.
OUTPUT FORMAT:
    - Immediate Actions:
      - ...
    - Long-Term Strategies:
      - ...
    - Study-Life Balance Tips:
      - ...
    - Motivation:
      - ...
"""
    
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI advice temporarily unavailable. Error: {str(e)}"



@st.cache_resource
def initialize_model():
    model = AdvancedStudentHealthModel()
    if model.load_models():
        return model
    return None

# -------------------------------
def main():
    st.title("🎓 Student Mental Health Analyzer Pro")
    
    health_model = initialize_model()
    
    if not health_model:
        st.warning("Please train or load models first using your model_training.py")
        return
    
    gemini_model = setup_gemini()

    # -------------------------------
    # Input form
    st.subheader("👤 Student Info")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        age = st.slider("Age", 16, 35, 20)
        sleep_duration = st.selectbox("Sleep Duration", 
                                      ["Less than 5 hours", "5-6 hours", "6-7 hours", "7-8 hours", "More than 8 hours"])
    with col2:
        dietary_habits = st.selectbox("Diet Quality", ["Unhealthy", "Moderate", "Healthy", "Very Healthy"])
        study_hours = st.slider("Daily Study Hours", 0, 16, 6)
        academic_pressure = st.slider("Academic Pressure", 1.0, 5.0, 3.0)
        study_satisfaction = st.slider("Study Satisfaction", 1.0, 5.0, 3.0)
        financial_stress = st.slider("Financial Stress", 1.0, 5.0, 2.0)
        family_history = st.radio("Family History of Mental Illness", ["No", "Yes"], horizontal=True)
        suicidal_thoughts = st.radio("Past Suicidal Thoughts", ["No", "Yes"], horizontal=True)

    # -------------------------------
    if st.button("🔍 Analyze Mental Health Risk"):
        user_data = {
            'Gender': gender,
            'Age': age,
            'Sleep Duration': sleep_duration,
            'Dietary Habits': dietary_habits,
            'Study Hours': study_hours,
            'Academic Pressure': academic_pressure,
            'Study Satisfaction': study_satisfaction,
            'Financial Stress': financial_stress,
            'Family History of Mental Illness': family_history,
            'Have you ever had suicidal thoughts ?': suicidal_thoughts
        }

        prediction = health_model.predict(user_data)

        st.subheader("📊 Risk Summary")
        st.metric("Depression Risk", prediction['prediction'])
        st.metric("Risk Level", prediction['risk_level'])
        st.metric("Confidence", f"{prediction['confidence']:.1%}")
        st.metric("Best Model", prediction['best_model_used'])

        # -------------------------------
        # Risk Visualization
        fig = go.Figure()
        fig.add_trace(go.Pie(
            labels=list(prediction['probabilities'].keys()),
            values=list(prediction['probabilities'].values()),
            hole=0.4,
            marker_colors=['#00cc96', '#ef553b']
        ))
        st.plotly_chart(fig, use_container_width=True)

        # -------------------------------
        # AI Advice
        if gemini_model:
            with st.expander("🤖 AI Health Advisor Suggestions", expanded=True):
                advice = get_ai_advice(user_data, prediction, gemini_model)
                st.markdown(advice)
        else:
            st.warning("Add Gemini API key to get AI-powered advice")

# -------------------------------
if __name__ == "__main__":
    main()
