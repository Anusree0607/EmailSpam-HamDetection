#Email Spam-Ham Detection

A Machine Learning based web application that classifies emails as **Spam** or **Ham (Not Spam)** using Natural Language Processing and a trained ML model.

---

## Project Overview

Email spam filtering is an important task in modern communication systems.  
This project uses **TF-IDF vectorization** and a **Machine Learning classifier** to detect whether a given email message is spam or legitimate.

The user enters email text in a web interface, and the system predicts the result instantly.

---

## Project Structure
EmailSpam-HamDetection/
│── app.py
│── bg.jfif
│── spam_ham_dataset.csv
│── spam_model.pkl
│── tfidf_vectorizer.pkl
│── README.md


---

## 🛠️ Technologies Used

- Python
- Flask
- Scikit-learn
- Pandas
- TF-IDF Vectorizer
- Pickle (Model Saving)

---

## How to Run the Project

1️⃣ Clone the Repository

```bash
git clone https://github.com/Anusree0607/EmailSpam-HamDetection.git
cd EmailSpam-HamDetection

2️⃣ Install Required Libraries
pip install flask scikit-learn pandas

3️⃣ Run the Application
python app.py

4️⃣ Open in Browser
http://127.0.0.1:5000
Enter your email text and check whether it is Spam or Ham.

### How It Works

Text input is cleaned and processed.

TF-IDF converts text into numerical features.

The trained ML model predicts Spam or Ham.

Result is displayed on the web interface.

### Dataset

The dataset used is:

spam_ham_dataset.csv

It contains labeled email messages for training the classifier.

### Future Improvements

Improve UI design

Add model training script

Deploy project online

Improve accuracy using advanced algorithms

Add email subject & sender features

### Author

Anusree Erva

GitHub: https://github.com/Anusree0607
### License

This project is for educational purposes.

---

###  After Pasting:

1. Scroll down
2. Click **Commit changes**
3. Refresh your repository page

Your GitHub will now look professional 

If you want, I can now help you:
- Add project screenshot section
- Add badges
- Make it resume-ready
- Remove unnecessary files for better practice 
