import streamlit as st
import pickle
import numpy as np
import base64

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

        .main-container {{
            background: rgba(20, 30, 50, 0.92);
            padding: 40px;
            border-radius: 20px;
            color: white;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
            border: 1px solid rgba(100, 150, 200, 0.2);
            backdrop-filter: blur(10px);
        }}

        h1, h2, h3, h4, h5, h6 {{
            color: #ffffff;
            font-weight: 700;
            letter-spacing: 0.5px;
        }}

        h1 {{
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
        }}

        h2 {{
            font-size: 1.8em;
            margin-top: 20px;
            margin-bottom: 15px;
            color: #64b5f6;
        }}

        h3 {{
            font-size: 1.3em;
            color: #e8eaf6;
        }}

        /* Button Styling */
        .stButton>button {{
            background: linear-gradient(135deg, #1e88e5 0%, #1565c0 100%);
            color: white;
            font-size: 16px;
            font-weight: 600;
            border-radius: 12px;
            height: 3.5em;
            width: 100%;
            border: none;
            box-shadow: 0 8px 25px rgba(30, 136, 229, 0.4);
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        .stButton>button:hover {{
            background: linear-gradient(135deg, #1565c0 0%, #0d47a1 100%);
            box-shadow: 0 12px 35px rgba(30, 136, 229, 0.6);
            transform: translateY(-2px);
        }}

        .stButton>button:active {{
            transform: translateY(0px);
        }}

        /* Text Area Styling */
        .stTextArea>div>div>textarea {{
            background-color: rgba(255, 255, 255, 0.1) !important;
            color: white !important;
            border: 2px solid rgba(100, 150, 200, 0.3) !important;
            border-radius: 10px !important;
            font-size: 15px !important;
            padding: 12px !important;
        }}

        .stTextArea>div>div>textarea:focus {{
            border: 2px solid #64b5f6 !important;
            box-shadow: 0 0 15px rgba(100, 181, 246, 0.3) !important;
        }}

        /* Radio Button Styling */
        .stRadio {{
            color: white;
        }}

        .stRadio>div>label {{
            color: #e8eaf6;
            font-weight: 500;
        }}

        .stRadio>div>div>label {{
            color: #b3e5fc;
            background-color: rgba(100, 150, 200, 0.1);
            padding: 10px 15px;
            border-radius: 8px;
            transition: all 0.3s ease;
            border: 1px solid transparent;
        }}

        .stRadio>div>div>label:hover {{
            background-color: rgba(100, 150, 200, 0.2);
            border: 1px solid rgba(100, 150, 200, 0.4);
        }}

        /* File Uploader */
        .stFileUploader {{
            color: white;
        }}

        .stFileUploader>div>div>div {{
            background-color: rgba(100, 150, 200, 0.1) !important;
            border: 2px dashed rgba(100, 150, 200, 0.3) !important;
            border-radius: 10px !important;
        }}

        /* Success/Error Styling */
        .stAlert {{
            border-radius: 10px;
            font-weight: 500;
        }}

        /* Progress Bar */
        .stProgress>div>div>div {{
            background: linear-gradient(90deg, #64b5f6, #42a5f5) !important;
            border-radius: 10px;
        }}

        /* Divider */
        hr {{
            border: none;
            border-top: 2px solid rgba(100, 150, 200, 0.2);
            margin: 25px 0;
        }}

        /* Labels */
        label {{
            color: #b3e5fc !important;
            font-weight: 500 !important;
            font-size: 15px !important;
        }}

        /* Expander */
        .streamlit-expanderHeader {{
            background-color: rgba(100, 150, 200, 0.1) !important;
            border-radius: 10px;
        }}

        .streamlit-expanderHeader:hover {{
            background-color: rgba(100, 150, 200, 0.15) !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ---------- SET BACKGROUND ----------
set_background("bg.jfif")

# ---------- LOAD MODEL ----------
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# ---------- PAGE TITLE ----------
st.markdown("<h1>📧 Email Spam Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #b3e5fc; font-size: 16px; margin-top: -10px;'>Enterprise-Grade Email Classification</p>", unsafe_allow_html=True)


st.markdown("---")

# ---------- RADIO OPTION ----------
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

# ---------- PREDICT ----------
if st.button("🔍 Analyze Email"):

    if email_content.strip() == "":
        st.warning("Please provide email content.")
    else:
        input_data = vectorizer.transform([email_content])
        prediction = model.predict(input_data)

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_data)[0]
            confidence = np.max(prob)
        else:
            confidence = 0.99

        st.markdown("## 📊 Analysis Result")
        st.progress(float(confidence))

        if prediction[0] == 1:
            st.markdown("""
            <div style='background: rgba(211, 47, 47, 0.2); border-left: 5px solid #ef5350; padding: 15px; border-radius: 8px; margin: 15px 0;'>
                <h3 style='margin: 0; color: #ef5350;'>🚨 SPAM EMAIL DETECTED</h3>
                <p style='margin: 8px 0 0 0; color: #ffcdd2;'>This email appears to be spam based on our analysis.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background: rgba(76, 175, 80, 0.2); border-left: 5px solid #66bb6a; padding: 15px; border-radius: 8px; margin: 15px 0;'>
                <h3 style='margin: 0; color: #66bb6a;'>✅ LEGITIMATE EMAIL (HAM)</h3>
                <p style='margin: 8px 0 0 0; color: #c8e6c9;'>This email appears to be legitimate based on our analysis.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"### Confidence Score: {confidence*100:.2f}%")

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; margin-top: 30px; padding: 20px;'>
    <p style='color: #b3e5fc; font-size: 14px; font-weight: 500;'>
        🔧 Built with Scikit-learn & Streamlit | 🚀 Powered by Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)
