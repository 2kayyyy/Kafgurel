import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score


def main():
    st.title("Language Detection: English or Roman Nepali")

    # Load dataset
    data = pd.read_csv('roman_nep-en.csv')

    # Handle missing values
    data.dropna(subset=['Value', 'Label'], inplace=True)

    # Display dataset
    st.write("Sample Data:", data.head())

    # Preprocess data
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

    # User input
    user_input = st.text_input("Enter text to detect language:")
    if st.button("Detect"):
        prediction = model.predict([user_input])
        st.write(f"Predicted Language: {prediction[0]}")


if __name__ == "__main__":
    main()