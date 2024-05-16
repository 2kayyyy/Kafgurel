import streamlit as st
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import os
import subprocess

# Load the tokenizer and model
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

# Define the label map
label_map = {0: 'English', 1: 'RomanNep', 2: 'None'}
emoji_map = {'English': '🇺🇸', 'RomanNep': '🇳🇵', 'None': '❓'}

def predict_language(text):
    encoded_input = tokenizer(text, truncation=True, padding=True, return_tensors='pt')
    output = model(**encoded_input)
    predicted_label = output.logits.argmax(-1).item()
    predicted_label_name = label_map[predicted_label]
    return predicted_label_name

def git_commit_and_push(file_path, commit_message):
    # Retrieve GitHub credentials from environment variables
    username = os.getenv('GITHUB_USERNAME')
    token = os.getenv('GITHUB_TOKEN')
    repo_name = "Kafgurel"  # Replace with your repository name

    if username and token:
        repo_url = f"https://{username}:{token}@github.com/{username}/{repo_name}.git"
        subprocess.run(['git', 'config', '--global', 'user.email', '"you@example.com"'])
        subprocess.run(['git', 'config', '--global', 'user.name', '"Your Name"'])
        subprocess.run(['git', 'add', file_path])
        subprocess.run(['git', 'commit', '-m', commit_message])
        subprocess.run(['git', 'push', repo_url])
    else:
        print("GitHub credentials are not set in the environment variables")

# Initialize Streamlit app
st.title('Language Prediction App')

user_input = st.text_input("Enter some text")

# Initialize session state for storing the initial prediction
if 'initial_prediction' not in st.session_state:
    st.session_state.initial_prediction = None

if user_input:
    # Only predict if the user input has changed
    if user_input != st.session_state.get('last_input', None):
        language = predict_language(user_input)
        st.session_state.initial_prediction = language
        st.session_state.last_input = user_input
    else:
        language = st.session_state.initial_prediction

    st.write(f"The predicted language is: {language} {emoji_map[language]}")

    # Capture feedback selection outside of the form
    feedback = st.radio("Is the detected language correct?", ('Yes', 'No'), key='feedback_radio')

    # Conditional display of correct language selection if feedback is No
    correct_language = None
    if feedback == 'No':
        correct_language = st.selectbox("Which language is correct?", ('English', 'RomanNep', 'None'), key='correct_language_select')

    # Submit button for feedback
    if st.button('Submit Feedback'):
        csv_file = 'roman_nep-en.csv'
        if feedback == 'Yes':
            st.markdown('<div class="feedback-success">Thank you for your input!</div>', unsafe_allow_html=True)
            with open(csv_file, mode='a') as f:
                f.write(f"\n{user_input},{language}")
            git_commit_and_push(csv_file, 'Update user feedback')
        elif feedback == 'No' and correct_language:
            st.markdown('<div class="feedback-success">Thank you for your input!</div>', unsafe_allow_html=True)
            with open(csv_file, mode='a') as f:
                f.write(f"\n{user_input},{correct_language}")
            git_commit_and_push(csv_file, 'Update user feedback')
else:
    st.session_state.initial_prediction = None

# Add a footer
st.markdown(
    """
    <style>
    body {
        font-family: Arial, sans-serif;
        background-color: #f0f2f6;
    }
    .reportview-container .main footer {visibility: hidden;}
    .footer:after {
        content: 'Made with love by Anmol, Kaushal, Aryan'; 
        visibility: visible;
        display: block;
        position: relative;
        padding: 10px;
        top: 2px;
        text-align: center;
        color: #555;
        font-size: 14px;
    }
    .stRadio > div {
        display: flex;
        justify-content: center;
        padding: 10px;
    }
    .stSelectbox, .stTextInput > div > div > input, .stButton > button {
        width: 80%;
        max-width: 400px;
        margin: 10px auto;
        border-radius: 5px;
    }
    .stTextInput > div > div > input {
        padding: 10px;
        border: 1px solid #ccc;
    }
    .stButton > button {
        padding: 10px;
        background-color: #4CAF50;
        color: white;
        border: none;
        transition: background-color 0.3s;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .stMarkdown {
        text-align: center;
        max-width: 400px;
        margin: 20px auto;
    }
    .feedback-success {
        border: 1px solid #4CAF50;
        border-radius: 5px;
        padding: 10px;
        background-color: #dff0d8;
        color: #3c763d;
        text-align: center;
        max-width: 400px;
        margin: 20px auto;
    }
    </style>
    <div class="footer"></div>
    """,
    unsafe_allow_html=True
)
