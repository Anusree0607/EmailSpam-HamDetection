import streamlit as st
import pickle
import numpy as np
import base64
import time

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Email Spam Detection",
    page_icon="📧",
    layout="centered"
)

# ---------- THEME DETECTION ----------
theme = st.get_option("theme.base")

if theme == "light":
    overlay_color = "rgba(255, 255, 255, 0.85)"
else:
    overlay_color = "rgba(20, 30, 50, 0.85)"

# ---------- FUNCTION TO LOAD BACKGROUND IMAGE ----------
def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

        * {{
            font-family: 'Poppins', sans-serif;
        }}

        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        .block-container {{
            background: {overlay_color};
            padding: 40px;
            border-radius: 20px;
            backdrop-filter: blur(12px);
        }}

        h1, h2, h3 {{
            color: var(--text-color);
            font-weight: 700;
        }}

        h1 {{
            text-align: center;
            font-size: 2.4em;
        }}

        .stButton>button {{
            background: linear-gradient(135deg, #1e88e5 0%, #1565c0 100%);
            color: white;
            font-weight: 600;
            border-radius: 12px;
            height: 3em;
            width: 100%;
            border: none;
            transition: 0.3s ease;
        }}

        .stButton>button:hover {{
            transform: translateY(-2px);
        }}

        .stTextArea textarea {{
            background-color: rgba(255,255,255,0.1) !important;
            color: var(--text-color) !important;
            border-radius: 10px !important;
            padding: 12px !important;
        }}

        label {{
            color: var(--text-color) !important;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        .result-card {{
            padding: 20px;
            border-radius: 15px;
            animation: fadeIn 0.8s ease-in-out;
            margin-top: 15px;
        }}

        .footer {{
            text-align:center;
            margin-top:30px;
            font-size:14px;
            color: var(--text-color);
        }}

        </style>
        """,
        unsafe_allow_html=True
    )

# ---------- SET BACKGROUND ----------
set_background("bg.jfif")

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    model = pickle.load(open("spam_model.pkl", "rb"))
    vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model()

# ---------- TITLE ----------
st.markdown("<h1>📧 Email Spam Detection System</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;'>Enterprise-Grade Email Classification</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# ---------- INPUT ----------
option = st.radio(
    "Select Input Method:",
    ("Paste Email Text", "Upload Email File (.txt)")
)

email_content = ""

if option == "Paste Email Text":
    email_content = st.text_area(
        "Enter Email Content Below:",
        height=200,
        placeholder="Type or paste the email message here..."
    )

elif option == "Upload Email File (.txt)":
    uploaded_file = st.file_uploader("Upload Email File", type=["txt"])
    if uploaded_file is not None:
        email_content = uploaded_file.read().decode("utf-8")
        with st.expander("Preview Uploaded Email"):
            st.write(email_content)

st.markdown("---")

# ---------- PREDICTION ----------
if st.button("🔍 Analyze Email"):

    if email_content.strip() == "":
        st.warning("Please provide email content.")
    else:
        with st.spinner("Analyzing email..."):
            input_data = vectorizer.transform([email_content])
            prediction = model.predict(input_data)

            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(input_data)[0]
                confidence = np.max(prob)
            else:
                confidence = 0.99

        st.markdown("## 📊 Analysis Result")

        # Animated progress
        progress_bar = st.progress(0)
        for i in range(int(confidence * 100) + 1):
            progress_bar.progress(i)
            time.sleep(0.01)

        # Result card
        if prediction[0] == 1:
            st.markdown(
                f"""
                <div class="result-card" style="
                    background: rgba(255,0,0,0.1);
                    border-left: 6px solid #ff4b4b;">
                    <h2 style="color:#ff4b4b;">🚨 SPAM EMAIL DETECTED</h2>
                    <p>This email appears to be spam.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="result-card" style="
                    background: rgba(0,200,100,0.1);
                    border-left: 6px solid #00c853;">
                    <h2 style="color:#00c853;">✅ LEGITIMATE EMAIL (HAM)</h2>
                    <p>This email appears safe and legitimate.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown(f"### 🎯 Confidence Score: {confidence*100:.2f}%")

# ---------- FOOTER ----------
st.markdown(
    """
    <div class="footer">
        🔧 Built with Scikit-learn & Streamlit | 🚀 Powered by Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)
