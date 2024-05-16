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
    username = os.getenv('GITHUB_USERNAME')
    token = os.getenv('GITHUB_TOKEN')
    if username and token:
        repo_url = f"https://{username}:{token}@github.com/{username}/Kafgurel.git"
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

    with st.form(key='feedback_form'):
        feedback = st.radio("Is the detected language correct?", ('Yes', 'No'))

        if feedback == 'No':
            correct_language = st.selectbox("Which language is correct?", ('English', 'RomanNep', 'None'))
        else:
            correct_language = None

        submit_button = st.form_submit_button(label='Submit Feedback')

        if submit_button:
            csv_file = 'roman_nep-en.csv'
            if feedback == 'Yes':
                st.success('Thank you for your input!')
                with open(csv_file, mode='a') as f:
                    f.write(f"\n{user_input},{language}")
                git_commit_and_push(csv_file, 'Update user feedback')
            else:
                st.success('Thank you for your input!')
                with open(csv_file, mode='a') as f:
                    f.write(f"\n{user_input},{correct_language}")
                git_commit_and_push(csv_file, 'Update user feedback')

# Add a footer
st.markdown(
    """
    <style>
    .reportview-container .main footer {visibility: hidden;}
    .footer:after {
        content: 'Made with love by Anmol, Kaushal, Aryan'; 
        visibility: visible;
        display: block;
        position: relative;
        padding: 5px;
        top: 2px;
    }
    </style>
    <div class="footer"></div>
    """,
    unsafe_allow_html=True
)
