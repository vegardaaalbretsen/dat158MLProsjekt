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
import os
from PIL import Image

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
        
        # Get prediction for the emotion
        prediction = model.predict(input_vector)[0]

        # Check if the prediction is "1" (e.g., corresponding to a specific emotion like "Joy")
        if prediction == 1:
            # Display the image if the prediction is "1"
            image = Image.open("images/joy.jpg")  # Replace with your actual image file path
            st.image(image, caption="Joy detected!", width=300)

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

# Feedback section
st.header("Give Us Feedback")
st.write("Was this service mega cool?")

# Feedback buttons
if st.button("👍 Yes"):
    feedback_type = "Thumbs Up"
    st.success("Thank you for your positive feedback!")
elif st.button("👎 No"):
    feedback_type = "Thumbs Down"
    st.warning("We appreciate your feedback and are working to improve.")

# Save feedback to a CSV file if a button was clicked
if 'feedback_type' in locals():
    feedback_file = 'feedback_summary.csv'

    # Check if the feedback file exists
    if os.path.exists(feedback_file):
        feedback_data = pd.read_csv(feedback_file)
    else:
        feedback_data = pd.DataFrame(columns=["FeedbackType"])

    # Append new feedback
    new_feedback = pd.DataFrame([[feedback_type]], columns=["FeedbackType"])
    feedback_data = pd.concat([feedback_data, new_feedback], ignore_index=True)

    # Save updated feedback data to CSV
    feedback_data.to_csv(feedback_file, index=False)
