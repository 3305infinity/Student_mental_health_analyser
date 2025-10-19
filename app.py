# import streamlit as st
# import pandas as pd
# from model_training import AdvancedStudentHealthModel
# import plotly.graph_objects as go
# import os
# import google.generativeai as genai
# from murf import Murf

# # -------------------------------
# # Streamlit page configuration
# st.set_page_config(
#     page_title="Student Mental Health Analyzer Pro",
#     page_icon="🎓",
#     layout="wide"
# )

# # -------------------------------
# # Gemini AI Setup
# def setup_gemini():
#     GEMINI_API_KEY = "AIzaSyDXYDzDMAyWwwOwq5VEj1YQ3ZIY0jvirnE"  # Keep your key here
#     try:
#         genai.configure(api_key=GEMINI_API_KEY)
#         return genai.GenerativeModel("gemini-2.5-flash")
#     except Exception as e:
#         st.error(f"Gemini setup error: {str(e)}")
#         return None

# # -------------------------------
# # Murf TTS function with chunking for long text
# def murf_tts_chunks(text, voice="en-US-natalie", format="MP3"):
#     MURF_API_KEY = "ap2_4b300076-ea74-4f7a-ad5a-57ad1699473a"  # Keep your key here
#     client = Murf(api_key=MURF_API_KEY)
    
#     max_len = 3000
#     chunks = [text[i:i+max_len] for i in range(0, len(text), max_len)]
#     audio_urls = []
    
#     for chunk in chunks:
#         try:
#             resp = client.text_to_speech.generate(
#                 text=chunk,
#                 voice_id=voice,
#                 format=format,
#                 sample_rate=44100.0
#             )
#             audio_url = resp.get("audio_url")
#             if audio_url:
#                 audio_urls.append(audio_url)
#         except Exception as e:
#             st.warning(f"Murf TTS chunk error: {str(e)}")
    
#     return audio_urls

# # -------------------------------
# # Gemini AI advice
# def get_ai_advice(user_data, prediction, gemini_model):
#     prompt = f"""
# Student Mental Health Analysis:
# Profile:
# - Age: {user_data['Age']}
# - Gender: {user_data['Gender']}
# - Academic Pressure: {user_data['Academic Pressure']}/5
# - Study Satisfaction: {user_data['Study Satisfaction']}/5
# - Sleep: {user_data['Sleep Duration']}
# - Diet: {user_data['Dietary Habits']}
# - Study Hours: {user_data['Study Hours']} hours/day
# - Financial Stress: {user_data['Financial Stress']}/5
# - Family Mental Health History: {user_data['Family History of Mental Illness']}
# - Past Suicidal Thoughts: {user_data['Have you ever had suicidal thoughts ?']}

# ML Assessment:
# - Predicted Depression: {prediction['prediction']}
# - Risk Level: {prediction['risk_level']}
# - Confidence: {prediction['confidence']:.1%}

# Instructions for AI:
# 1. Analyze the risk level carefully based on ML prediction.
# 2. If risk is HIGH, give practical, urgent actions.
# 3. If risk is MEDIUM, give preventive measures.
# 4. If risk is LOW, give wellness maintenance tips.
# 5. Provide 3-5 actionable points for each section.
# 6. Give advice in English, concise and student-friendly.
# 7. Avoid generic statements like "sleep well", make actionable suggestions.

# Output Format:
# - Immediate Actions:
#   - ...
# - Long-Term Strategies:
#   - ...
# - Study-Life Balance Tips:
#   - ...
# - Motivation:
#   - ...
# """
#     try:
#         response = gemini_model.generate_content(prompt)
#         return response.text
#     except Exception as e:
#         return f"AI advice temporarily unavailable. Error: {str(e)}"

# # -------------------------------
# @st.cache_resource
# def initialize_model():
#     model = AdvancedStudentHealthModel()
#     if model.load_models():
#         return model
#     st.warning("No trained models found. Train first using train_models.py")
#     return None

# # -------------------------------
# def main():
#     st.title("🎓 Student Mental Health Analyzer Pro")
#     health_model = initialize_model()
#     if not health_model:
#         return

#     gemini_model = setup_gemini()

#     # -------------------------------
#     # Input form
#     st.subheader("👤 Student Info")
#     col1, col2 = st.columns(2)
#     with col1:
#         gender = st.selectbox("Gender", ["Male", "Female", "Other"])
#         age = st.slider("Age", 16, 35, 20)
#         sleep_duration = st.selectbox("Sleep Duration", 
#                                       ["Less than 5 hours", "5-6 hours", "6-7 hours", "7-8 hours", "More than 8 hours"])
#     with col2:
#         dietary_habits = st.selectbox("Diet Quality", ["Unhealthy", "Moderate", "Healthy", "Very Healthy"])
#         study_hours = st.slider("Daily Study Hours", 0, 16, 6)
#         academic_pressure = st.slider("Academic Pressure", 1.0, 5.0, 3.0)
#         study_satisfaction = st.slider("Study Satisfaction", 1.0, 5.0, 3.0)
#         financial_stress = st.slider("Financial Stress", 1.0, 5.0, 2.0)
#         family_history = st.radio("Family History of Mental Illness", ["No", "Yes"], horizontal=True)
#         suicidal_thoughts = st.radio("Past Suicidal Thoughts", ["No", "Yes"], horizontal=True)

#     # -------------------------------
#     if st.button("🔍 Analyze Mental Health Risk"):
#         user_data = {
#             'Gender': gender,
#             'Age': age,
#             'Sleep Duration': sleep_duration,
#             'Dietary Habits': dietary_habits,
#             'Study Hours': study_hours,
#             'Academic Pressure': academic_pressure,
#             'Study Satisfaction': study_satisfaction,
#             'Financial Stress': financial_stress,
#             'Family History of Mental Illness': family_history,
#             'Have you ever had suicidal thoughts ?': suicidal_thoughts
#         }

#         prediction = health_model.predict(user_data)

#         st.subheader("📊 Risk Summary")
#         st.metric("Depression Risk", prediction['prediction'])
#         st.metric("Risk Level", prediction['risk_level'])
#         st.metric("Confidence", f"{prediction['confidence']:.1%}")
#         st.metric("Best Model", prediction['best_model_used'])

#         # -------------------------------
#         # Risk Visualization
#         fig = go.Figure()
#         fig.add_trace(go.Pie(
#             labels=list(prediction['probabilities'].keys()),
#             values=list(prediction['probabilities'].values()),
#             hole=0.4,
#             marker_colors=['#00cc96', '#ef553b']
#         ))
#         st.plotly_chart(fig, use_container_width=True)

#         # -------------------------------
#         # AI Advice
#         if gemini_model:
#             with st.expander("🤖 AI Health Advisor Suggestions", expanded=True):
#                 advice = get_ai_advice(user_data, prediction, gemini_model)
#                 st.markdown(advice)

#                 # Murf TTS with chunking
#                 audio_urls = murf_tts_chunks(advice)
#                 for url in audio_urls:
#                     st.audio(url, format="audio/mp3")
#         else:
#             st.warning("Add Gemini API key to get AI-powered advice")

# # -------------------------------
# if __name__ == "__main__":
#     main()
import streamlit as st
import pandas as pd
from model_training import AdvancedStudentHealthModel
import plotly.graph_objects as go
import os
import requests
import google.generativeai as genai
from murf import Murf

# -------------------------------
# Streamlit page configuration
st.set_page_config(
    page_title="Student Mental Health Analyzer Pro",
    page_icon="🎓",
    layout="wide"
)

# -------------------------------
# Gemini AI Setup
def setup_gemini():
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDXYDzDMAyWwwOwq5VEj1YQ3ZIY0jvirnE")
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        return genai.GenerativeModel("gemini-2.5-flash")
    except Exception as e:
        st.error(f"Gemini setup error: {str(e)}")
        return None

# -------------------------------
# Murf TTS function
def murf_tts(text, voice="en-US-natalie", format="MP3"):
    MURF_API_KEY = os.getenv("MURF_API_KEY", "ap2_4b300076-ea74-4f7a-ad5a-57ad1699473a")
    client = Murf(api_key=MURF_API_KEY)

    max_chars = 3000  # Murf API limit
    audio_urls = []

    chunks = [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

    for chunk in chunks:
        try:
            resp = client.text_to_speech.generate(
                format=format,
                text=chunk,
                voice_id=voice
            )
            audio_urls.append(resp.url)
        except Exception as e:
            st.warning(f"Murf TTS chunk error: {e}")
            return None

    # If multiple chunks, return list, else single URL
    return audio_urls if len(audio_urls) > 1 else audio_urls[0]

# -------------------------------
# Gemini AI advice
def get_ai_advice(user_data, prediction, gemini_model):
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
2. If risk is HIGH, give practical, urgent actions.
3. If risk is MEDIUM, give preventive measures.
4. If risk is LOW, give wellness maintenance tips.
5. Provide 3-5 actionable points for each section.
6. Give advice in English, concise and student-friendly.
7. Avoid generic statements like "sleep well", make actionable suggestions.

Output Format:
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

# -------------------------------
@st.cache_resource
def initialize_model():
    model = AdvancedStudentHealthModel()
    if model.load_models():
        return model
    st.warning("No trained models found. Train first using train_models.py")
    return None

# -------------------------------
def main():
    st.title("🎓 Student Mental Health Analyzer Pro")
    health_model = initialize_model()
    if not health_model:
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

                # Murf TTS
                audio_urls = murf_tts(advice)
                if audio_urls:
                    if isinstance(audio_urls, list):
                        for url in audio_urls:
                            st.audio(url, format="audio/mp3")
                    else:
                        st.audio(audio_urls, format="audio/mp3")
        else:
            st.warning("Add Gemini API key to get AI-powered advice")

# -------------------------------
if __name__ == "__main__":
    main()
