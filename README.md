
# Emotion Classification with Machine Learning

This project applies machine learning to classify text into six emotions: **anger**, **fear**, **joy**, **love**, **sadness**, and **surprise**. The model is designed to analyze customer feedback and other text data, providing insights to improve user engagement, customer service, and content personalization.

### Try the App
Access the live app on Streamlit: [dat158mlprosjekt.streamlit.app](https://dat158mlprosjekt.streamlit.app/)

### Authors
Vegard Aa Albretsen, Erlend Vitsø

## Overview

The primary goal is to develop a text-based emotion detection model to help businesses understand user sentiment. Applications include customer support, marketing insights, and improved content recommendations.

### Key Features
- **Emotion Detection**: Classifies text into one of six emotions.
- **Model Deployment**: The model is deployed on Streamlit for interactive use.
- **Insights**: Offers business insights by identifying user sentiment and adjusting responses accordingly.

## Usage Instructions
**Try it via the website, OR:**
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/h599002/dat158MLProsjekt.git
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the App**:
   ```bash
   streamlit run app.py
   ```

## Model Details

The model pipeline includes:
- **Data Preprocessing**: Tokenization, stemming, and feature extraction using TF-IDF.
- **Classification Model**: Trained using Logistic Regression for optimal performance in emotion detection.

Data Preprocessing: Tokenization, stemming, and feature extraction using TF-IDF.
Classification Model: Trained using Logistic Regression for optimal performance in emotion detection.
Dataset
Source: Kaggle dataset with 416,809 records from X (formerly Twitter).
Class Distribution:
Sadness: 121,187
Joy: 141,067
Love: 34,554
Anger: 57,317
Fear: 47,712
Surprise: 14,972
Key Metrics
Performance metrics for the model include accuracy, precision, and recall, with accuracy reaching approximately 90%

- model_train/: Contains scripts for training models.
- saved_models/: Stores trained models and vectorizers.
- plots/: Contains generated plots, such as confusion matrices.
