"""
Filename: app.py
Description: Emotion Classification App
Authors: Vegard Aa Albretsen, Erlend Vitsø
Date: November 15, 2024

Generative AI has been used.
"""
import streamlit as st
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import numpy as np
import pandas as pd
import nltk
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json
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

# Define Google Sheets credentials and scope
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/spreadsheets",
         "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]

# Load credentials from Streamlit secrets
credentials_dict = st.secrets["gcp_service_account"]
credentials = ServiceAccountCredentials.from_json_keyfile_dict(credentials_dict, scope)

# Authorize and open the Google Sheet
client = gspread.authorize(credentials)
sheet = client.open("Feedback Dat158").sheet1  # Replace with your Google Sheet name


# Define Streamlit UI
st.title("Text Analysis App")
st.write("Welcome! Please let us know if you found this service helpful.")

# Feedback buttons
if st.button("👍 Yes"):
    feedback_type = "Thumbs Up"
    st.success("Thank you for your positive feedback!")
elif st.button("👎 No"):
    feedback_type = "Thumbs Down"
    st.warning("We appreciate your feedback and are working to improve.")

# Append feedback to Google Sheets if a button was clicked
if 'feedback_type' in locals():
    sheet.append_row([feedback_type])  # Add feedback type as a new row