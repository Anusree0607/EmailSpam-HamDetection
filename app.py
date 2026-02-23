import streamlit as st
import pickle
import numpy as np
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Email Classification System",
    layout="centered"
)

# ---------------- FAST BACKGROUND IMAGE ----------------
st.markdown("""
<style>
.stApp {
    background-image: url("bg.jfif");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Light glass effect without heavy blur */
.block-container {
    background: rgba(0, 0, 0, 0.45);
    padding: 40px;
    border-radius: 20px;
}

/* Title */
h1 {
    text-align: center;
    font-size: 2.3em;
    font-weight: 700;
}

/* Subtitle */
.subtitle {
    text-align: center;
    margin-bottom: 20px;
    opacity: 0.85;
}

/* Textarea */
.stTextArea textarea {
    border-radius: 12px !important;
    padding: 12px !important;
}

/* Button */
.stButton>button {
    background: linear-gradient(135deg,#00c6ff,#0072ff);
    color: white;
    border-radius: 12px;
    height: 3em;
    font-weight: 600;
    border: none;
}

/* Result Card */
.result-card {
    padding: 20px;
    border-radius: 15px;
    margin-top: 20px;
    animation: fadeIn 0.4s ease-in-out;
}

@keyframes fadeIn {
    from {opacity:0; transform:translateY(10px);}
    to {opacity:1; transform:translateY(0);}
}

.footer {
    text-align:center;
    margin-top:30px;
    font-size:14px;
    opacity:0.8;
}
</style>
""", unsafe_allow_html=True)

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

        # Faster animation
        progress = st.progress(0)
        for i in range(int(confidence * 100)):
            progress.progress(i + 1)
            time.sleep(0.002)

        if prediction[0] == 1:
            st.markdown("""
            <div class="result-card" style="
                background: rgba(255,0,0,0.15);
                border-left: 6px solid #ff1744;">
                <h2 style="color:#ff1744;">SPAM CLASSIFICATION</h2>
                <p>Model predicts this email as Spam.</p>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.balloons()
            st.markdown("""
            <div class="result-card" style="
                background: rgba(0,255,150,0.15);
                border-left: 6px solid #00e676;">
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
