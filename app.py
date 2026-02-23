import streamlit as st
import pickle
import numpy as np
import base64
import time

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Email Spam Detection",
    layout="centered"
)

# ---------- THEME DETECTION ----------
theme = st.get_option("theme.base")

if theme == "light":
    overlay_color = "rgba(255, 255, 255, 0.95)"
    text_color = "#111111"
    sub_text_color = "#333333"
else:
    overlay_color = "rgba(20, 30, 50, 0.90)"
    text_color = "#ffffff"
    sub_text_color = "#dddddd"

# ---------- BACKGROUND FUNCTION ----------
def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>

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
            backdrop-filter: blur(10px);
        }}

        h1 {{
            color: {text_color} !important;
            text-align: center;
            font-size: 2.2em;
        }}

        p, label, div {{
            color: {text_color} !important;
        }}

        .subtext {{
            color: {sub_text_color} !important;
            text-align: center;
            font-size: 1em;
            margin-bottom: 15px;
        }}

        .stTextArea textarea {{
            background-color: rgba(0,0,0,0.05) !important;
            color: {text_color} !important;
            border-radius: 10px !important;
        }}

        .stButton>button {{
            background-color: #1565c0;
            color: white;
            font-weight: 600;
            border-radius: 10px;
            height: 3em;
            width: 100%;
            border: none;
        }}

        .stButton>button:hover {{
            background-color: #0d47a1;
        }}

        .result-card {{
            padding: 20px;
            border-radius: 15px;
            margin-top: 15px;
            animation: fadeIn 0.6s ease-in-out;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        .footer {{
            text-align:center;
            margin-top:30px;
            font-size:14px;
            color:{sub_text_color};
        }}

        </style>
        """,
        unsafe_allow_html=True
    )

# ---------- APPLY BACKGROUND ----------
set_background("bg.jfif")

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    model = pickle.load(open("spam_model.pkl", "rb"))
    vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model()

# ---------- TITLE ----------
st.markdown("<h1>Email Spam Detection System</h1>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtext'>Machine Learning Based Email Classification</div>",
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
        height=200
    )

elif option == "Upload Email File (.txt)":
    uploaded_file = st.file_uploader("Upload Email File", type=["txt"])
    if uploaded_file is not None:
        email_content = uploaded_file.read().decode("utf-8")
        with st.expander("Preview Uploaded Email"):
            st.write(email_content)

st.markdown("---")

# ---------- PREDICTION ----------
if st.button("Analyze Email"):

    if email_content.strip() == "":
        st.warning("Please provide email content.")
    else:
        with st.spinner("Processing..."):
            input_data = vectorizer.transform([email_content])
            prediction = model.predict(input_data)

            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(input_data)[0]
                confidence = np.max(prob)
            else:
                confidence = 0.99

        st.markdown("## Analysis Result")

        progress_bar = st.progress(0)
        for i in range(int(confidence * 100) + 1):
            progress_bar.progress(i)
            time.sleep(0.01)

        if prediction[0] == 1:
            st.markdown(
                f"""
                <div class="result-card" style="
                    background: rgba(255,0,0,0.08);
                    border-left: 6px solid #d32f2f;">
                    <h2 style="color:#d32f2f;">SPAM EMAIL DETECTED</h2>
                    <p>This email appears to be spam.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="result-card" style="
                    background: rgba(0,150,80,0.08);
                    border-left: 6px solid #2e7d32;">
                    <h2 style="color:#2e7d32;">LEGITIMATE EMAIL (HAM)</h2>
                    <p>This email appears safe and legitimate.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown(f"### Confidence Score: {confidence*100:.2f}%")

# ---------- FOOTER ----------
st.markdown(
    """
    <div class="footer">
        Built with Scikit-learn and Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
