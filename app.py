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
    # Encode the new input text
    encoded_input = tokenizer(text, truncation=True, padding=True, return_tensors='pt')

    # Make the prediction
    output = model(**encoded_input)
    predicted_label = output.logits.argmax(-1).item()

    # Get the label name from the label_map
    predicted_label_name = label_map[predicted_label]

    return predicted_label_name

def git_commit_and_push(file_path, commit_message):
    subprocess.run(['git', 'add', file_path])
    subprocess.run(['git', 'commit', '-m', commit_message])
    subprocess.run(['git', 'push'])

# Initialize Streamlit app
st.title('Language Prediction App')

user_input = st.text_input("Enter some text")
if user_input:
    language = predict_language(user_input)
    st.write(f"The predicted language is: {language} {emoji_map[language]}")

    feedback = st.radio("Is the detected language correct?", ('Yes', 'No'))

    if st.button('Submit Feedback'):
        # Append the data to a CSV file
        csv_file = 'roman_nep-en.csv'
        if feedback == 'Yes':
            st.success('Thank you for your input!')
            with open(csv_file, mode='a') as f:
                f.write(f"\n{user_input},{language}")
            git_commit_and_push(csv_file, 'Update user feedback')
        else:
            correct_language = st.selectbox(
                "Which language is correct?",
                ('English', 'RomanNep', 'None')
            )
            if st.button('Submit Correct Language'):
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
