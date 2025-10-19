# 🎓 Student Mental Health Analyzer Pro

A cutting-edge **Python & Streamlit web app** that predicts student mental health risks using **machine learning** and provides **AI-powered advice**. With lifelike speech output, this project combines **ML models**, **Google Gemini AI**, and **Murf TTS** for a fully interactive experience.

---

## 🔥 Why This Project Stands Out

- **Advanced ML Techniques**:  
  The app uses a custom-trained ML model (`AdvancedStudentHealthModel`) to predict **depression risk** and assess confidence in predictions. This allows for personalized and data-driven mental health insights.

- **AI-Powered Advice with Gemini**:  
  Gemini AI analyzes user profiles and ML predictions to provide **actionable, student-friendly recommendations**. This ensures advice is not generic but tailored to the risk level.

- **Lifelike Speech with Murf TTS**:  
  Murf converts Gemini’s textual advice into **natural-sounding speech**, enhancing accessibility and engagement. Over 130 voices and multiple languages allow customization for real-world applications.

- **Interactive Visualizations**:  
  Risk probabilities are visualized using **Plotly pie charts**, making results easy to understand at a glance.

---

## 🚀 Key Features

- Predicts **Depression Risk**, **Risk Level**, and **Confidence**.
- Provides **Best Model Used** for transparency.
- Displays **actionable AI advice** based on ML prediction.
- Optional **audio advice** using Murf TTS.
- Fully responsive and student-friendly **Streamlit interface**.

---

## 🛠 Tech Stack

- **Python 3.10+**
- **Streamlit** – UI framework
- **Pandas & Plotly** – Data processing & visualization
- **Google Gemini AI** – AI-generated actionable advice
- **Murf AI Python SDK** – Text-to-speech output
- Custom **ML models** in `model_training.py`

---

## ⚡ Setup & Installation

1. **Clone the repo**
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
Install dependencies

bash
Copy code
pip install -r requirements.txt
Set API Keys (Optional)

bash
Copy code
export GEMINI_API_KEY="your-gemini-key"
export MURF_API_KEY="your-murf-key"
Recommended to use environment variables for security.

Run the app

bash
Copy code
streamlit run app.py
🧠 How It Works
User fills in their profile: age, gender, sleep, study habits, diet, etc.

ML model predicts depression risk, confidence, and risk level.

Gemini AI generates actionable advice tailored to risk level.

Murf TTS converts text advice to speech (optional).

Results and recommendations are displayed with interactive charts.
