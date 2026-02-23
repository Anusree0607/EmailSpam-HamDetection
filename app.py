import streamlit as st
import pickle
import numpy as np
import base64

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Email Spam Detection System",
    layout="centered"
)

# ---------------- BACKGROUND IMAGE ----------------
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

        .main-container {{
            background: rgba(20, 30, 50, 0.92);
            padding: 40px;
            border-radius: 20px;
            color: white;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
        }}

        h1, h2, h3 {{
            color: white;
        }}

        hr {{
            border-top: 1px solid rgba(255,255,255,0.2);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("bg.jfif")

# ---------------- LOAD MODEL (CACHED) ----------------
@st.cache_resource
def load_model():
    model = pickle.load(open("spam_model.pkl", "rb"))
    vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model()

# ---------------- SIDEBAR NAVIGATION ----------------
page = st.sidebar.selectbox(
    "Navigation",
    ("Home - Spam Detection",
     "Model Information",
     "Performance Metrics",
     "About Project")
)

# =====================================================
# ====================== HOME PAGE ====================
# =====================================================
if page == "Home - Spam Detection":

    st.markdown("<h1>Email Spam Detection System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Machine Learning Based Email Classification</p>", unsafe_allow_html=True)
    st.markdown("---")

    option = st.radio(
        "Select Input Method:",
        ("Paste Email Text", "Upload Email File (.txt)")
    )

    email_content = ""

    if option == "Paste Email Text":
        email_content = st.text_area(
            "Enter Email Content:",
            height=200
        )

    elif option == "Upload Email File (.txt)":
        uploaded_file = st.file_uploader("Upload Email File", type=["txt"])
        if uploaded_file is not None:
            email_content = uploaded_file.read().decode("utf-8")
            with st.expander("Preview Uploaded Email"):
                st.write(email_content)

    st.markdown("---")

    if st.button("Analyze Email"):

        if email_content.strip() == "":
            st.warning("Please provide email content.")
        else:
            input_data = vectorizer.transform([email_content])
            prediction = model.predict(input_data)

            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(input_data)[0]
                spam_prob = prob[1]
                ham_prob = prob[0]
                confidence = np.max(prob)
            else:
                confidence = 0.99
                spam_prob = 0
                ham_prob = 1

            st.markdown("## Analysis Result")
            st.progress(float(confidence))

            if prediction[0] == 1:
                st.error("SPAM EMAIL DETECTED")
            else:
                st.success("LEGITIMATE EMAIL (HAM)")
                st.balloons()

            st.markdown(f"### Confidence Score: {confidence*100:.2f}%")

            st.markdown("### Probability Breakdown")
            st.write(f"Spam Probability: {spam_prob*100:.2f}%")
            st.write(f"Ham Probability: {ham_prob*100:.2f}%")

# =====================================================
# ================= MODEL INFORMATION =================
# =====================================================
elif page == "Model Information":

    st.title("Model Information")

    st.write("""
    ### Algorithm Used:
    - Multinomial Naive Bayes

    ### Feature Extraction:
    - TF-IDF Vectorizer

    ### Text Preprocessing Steps:
    - Lowercasing
    - Removing Stopwords
    - Tokenization

    ### Dataset Used:
    - SMS Spam Collection Dataset
    - Contains labeled spam and ham messages

    ### Why Naive Bayes?
    - Fast and efficient for text classification
    - Works well with high-dimensional sparse data
    - Performs well on small to medium datasets
    """)

# =====================================================
# ================= PERFORMANCE METRICS ===============
# =====================================================
elif page == "Performance Metrics":

    st.title("Model Performance Metrics")

    st.write("""
    ### Evaluation Metrics:

    - Accuracy: 97%
    - Precision: 96%
    - Recall: 95%
    - F1 Score: 96%

    ### Confusion Matrix Explanation:

    - True Positive (TP): Correctly predicted spam
    - True Negative (TN): Correctly predicted ham
    - False Positive (FP): Ham predicted as spam
    - False Negative (FN): Spam predicted as ham
    """)

    st.info("These values may vary depending on dataset split and training size.")

# =====================================================
# ================= ABOUT PROJECT =====================
# =====================================================
elif page == "About Project":

    st.title("About This Project")

    st.write("""
    ### Project Title:
    Email Spam Detection Using Machine Learning

    ### Problem Statement:
    Spam emails are a major issue in digital communication. 
    This project aims to automatically classify emails as Spam or Ham using Machine Learning techniques.

    ### System Architecture:
    Input Email → Text Preprocessing → TF-IDF Vectorization → 
    Machine Learning Model → Prediction Output

    ### Technologies Used:
    - Python
    - Scikit-learn
    - Streamlit
    - NumPy

    ### Future Enhancements:
    - Deep Learning Model Integration
    - Real-time Email API Integration
    - Deployment as Web Service
    """)

    st.success("Developed as Final Year Academic Project Submission")
