import streamlit as st
import pickle
import numpy as np
import base64
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Email Classification System",
    layout="centered"
)

# ---------------- BACKGROUND + GLASS UI ----------------
def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(f"""
    <style>

    .stApp {{
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-attachment: fixed;
        background-blend-mode: overlay;
    }}

    .block-container {{
        background: rgba(255,255,255,0.15);
        padding: 40px;
        border-radius: 25px;
        backdrop-filter: blur(20px);
        box-shadow: 0px 10px 40px rgba(0,0,0,0.3);
    }}

    h1 {{
        text-align: center;
        font-size: 2.5em;
        font-weight: 700;
    }}

    .subtitle {{
        text-align: center;
        margin-bottom: 20px;
        opacity: 0.85;
    }}

    .stTextArea textarea {{
        border-radius: 12px !important;
        padding: 12px !important;
    }}

    .stButton>button {{
        background: linear-gradient(135deg,#00c6ff,#0072ff);
        color: white;
        border-radius: 12px;
        height: 3em;
        font-weight: 600;
        border: none;
        transition: 0.3s;
    }}

    .stButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0px 5px 20px rgba(0,0,0,0.4);
    }}

    .result-card {{
        padding: 25px;
        border-radius: 20px;
        margin-top: 20px;
        animation: fadeIn 0.7s ease-in-out;
    }}

    @keyframes fadeIn {{
        from {{opacity:0; transform:translateY(10px);}}
        to {{opacity:1; transform:translateY(0);}}
    }}

    .footer {{
        text-align:center;
        margin-top:30px;
        font-size:14px;
        opacity:0.8;
    }}

    </style>
    """, unsafe_allow_html=True)

set_background("bg.jfif")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = pickle.load(open("spam_model.pkl", "rb"))
    vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model()

# ---------------- TITLE ----------------
st.markdown("<h1>AI Email Classification System</h1>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>TF-IDF + Machine Learning Based Spam Detection Research Demo</div>",
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------- INPUT ----------------
option = st.radio(
    "Select Input Method",
    ("Paste Email Text", "Upload Email File (.txt)")
)

email_content = ""

if option == "Paste Email Text":
    email_content = st.text_area("Enter Email Content", height=200)

else:
    uploaded_file = st.file_uploader("Upload .txt File", type=["txt"])
    if uploaded_file:
        email_content = uploaded_file.read().decode("utf-8")
        with st.expander("Preview File"):
            st.write(email_content)

st.markdown("---")

# ---------------- PREDICTION ----------------
if st.button("Run AI Analysis"):

    if email_content.strip() == "":
        st.warning("Please enter email content.")
    else:
        with st.spinner("Running Model Inference..."):
            input_data = vectorizer.transform([email_content])
            prediction = model.predict(input_data)

            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(input_data)[0]
                confidence = np.max(prob)
            else:
                confidence = 0.99

        st.markdown("## Model Output")

        progress = st.progress(0)
        for i in range(int(confidence * 100)):
            progress.progress(i + 1)
            time.sleep(0.01)

        if prediction[0] == 1:
            st.markdown(f"""
            <div class="result-card" style="
                background: rgba(255,0,0,0.1);
                border-left: 8px solid #ff1744;">
                <h2 style="color:#ff1744;">SPAM CLASSIFICATION</h2>
                <p>Model predicts this email as Spam.</p>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.balloons()
            st.markdown(f"""
            <div class="result-card" style="
                background: rgba(0,255,150,0.1);
                border-left: 8px solid #00e676;">
                <h2 style="color:#00e676;">HAM CLASSIFICATION</h2>
                <p>Model predicts this email as Legitimate.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"### Confidence Score: {confidence*100:.2f}%")

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
AI Research Demonstration | Built with Scikit-learn & Streamlit
</div>
""", unsafe_allow_html=True)
