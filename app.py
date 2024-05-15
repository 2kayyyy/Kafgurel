import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
import os
import logging
import subprocess

# Configure logging
logging.basicConfig(filename='new_entries.log', level=logging.INFO, format='%(asctime)s - %(message)s')


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])
    return text


def git_push():
    try:
        subprocess.run(["git", "config", "--global", "user.email", "aryankafle004@gmail.com"], check=True)
        subprocess.run(["git", "config", "--global", "user.name", "2kayyyy"], check=True)
        subprocess.run(["git", "add", "roman_nep-en.csv"], check=True)
        subprocess.run(["git", "commit", "-m", "Update dataset with new entries"], check=True)
        subprocess.run(["git", "push"], check=True)
        logging.info("Changes pushed to GitHub successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"An error occurred while pushing changes to GitHub: {e}")


def main():
    st.title("Language Detection: English, Roman Nepali, or None")

    # Load dataset
    data = pd.read_csv('roman_nep-en.csv')

    # Handle missing values
    data.dropna(subset=['Value', 'Label'], inplace=True)

    # Preprocess data
    data['Value'] = data['Value'].apply(preprocess_text)
    X = data['Value']
    y = data['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = make_pipeline(CountVectorizer(), MultinomialNB())
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: {accuracy:.2f}")

    # User input for prediction
    user_input = st.text_input("Enter text to detect language:")
    if st.button("Detect"):
        prediction = model.predict([user_input])[0]
        st.write(f"Predicted Language: {prediction}")

        # Ask user to confirm the detected language
        correct_label = st.selectbox("Is the detected language correct?", ["Yes", "No"])
        if correct_label == "No":
            # Ask user for the correct label
            correct_label = st.selectbox("Please select the correct language", ["English", "RomanNep", "None"])
        else:
            correct_label = prediction

        # Button to add data to dataset
        if st.button("Add to Dataset"):
            if user_input and correct_label:
                # Append new data to the CSV file
                new_data = pd.DataFrame([[user_input, correct_label]], columns=['Value', 'Label'])
                if not os.path.isfile('roman_nep-en.csv'):
                    new_data.to_csv('roman_nep-en.csv', index=False)
                else:
                    new_data.to_csv('roman_nep-en.csv', mode='a', header=False, index=False)
                st.success("New data added to the dataset!")
                # Log the new entry
                logging.info(f"Added new entry: {user_input} - {correct_label}")
                print(f"Logging: Added new entry: {user_input} - {correct_label}")
                # Push changes to GitHub
                git_push()


if __name__ == "__main__":
    main()