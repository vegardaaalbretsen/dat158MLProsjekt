import streamlit as st
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import numpy as np
import pandas as pd
import nltk
import os

# Set NLTK to use the local nltk_data directory
nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
nltk.data.path.append(nltk_data_path)

# Load the pre-trained model and vectorizer
with open('logistic_regression_model_tokens.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer_tokens.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Define emotion labels
output_labels = ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"]

# Initialize the stemmer
stemmer = PorterStemmer()

# Set up the Streamlit app layout
st.title("Text Analysis App")
st.write("Enter some text below to analyze sentiment or classify it into either Sadness, Joy, Love, Anger, Fear, or Surprise.")

# Text input for the user
user_input = st.text_area("Enter your text here:")

import re

def preprocess_text(text):
    # Use regex to tokenize by finding word characters
    tokens = re.findall(r'\b\w+\b', text.lower())  # Tokenize using regex
    stemmed_tokens = [stemmer.stem(word) for word in tokens]  # Apply stemming
    return ' '.join(stemmed_tokens)  # Join back to a single string

if st.button("Analyze Text"):
    if user_input:
        # Preprocess the input text
        processed_input = preprocess_text(user_input)
        
        # Transform the processed text using the TF-IDF vectorizer
        input_vector = vectorizer.transform([processed_input])
        
        # Get probabilities for each emotion
        probabilities = model.predict_proba(input_vector)[0]
        
        # Sort emotions by probability in descending order
        sorted_indices = np.argsort(probabilities)[::-1]
        
        # Display top predictions
        st.write("Top Predictions:")
        for idx in sorted_indices:
            st.write(f"{output_labels[idx]}: {probabilities[idx]:.2%}")
    else:
        st.write("Please enter some text.")

st.header("Gi oss tilbakemelding")
st.write("Var denne tjenesten nyttig?")

# Opprett tommel opp og tommel ned-knapper
if st.button("👍 Ja"):
    feedback_type = "Tommel Opp"
    st.success("Takk for din positive tilbakemelding!")
elif st.button("👎 Nei"):
    feedback_type = "Tommel Ned"
    st.warning("Vi setter pris på tilbakemeldingen og jobber med å forbedre oss.")

# Lagre tilbakemelding i en CSV-fil hvis en knapp ble trykket
if 'feedback_type' in locals():
    feedback_file = 'feedback_summary.csv'

    # Sjekk om filen allerede finnes
    if os.path.exists(feedback_file):
        feedback_data = pd.read_csv(feedback_file)
    else:
        feedback_data = pd.DataFrame(columns=["FeedbackType"])

    # Legg til ny tilbakemelding
    new_feedback = pd.DataFrame([[feedback_type]], columns=["FeedbackType"])
    feedback_data = pd.concat([feedback_data, new_feedback], ignore_index=True)

    # Lagre tilbake til CSV-fil
    feedback_data.to_csv(feedback_file, index=False)