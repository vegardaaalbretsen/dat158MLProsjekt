import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

nltk.download('punkt')

# Sjekk om en lagret fil allerede eksisterer
processed_file_path = 'MLProsjekt/balanced_dataset.csv'

if os.path.exists(processed_file_path):
    # Les inn den ferdigbehandlede filen
    data = pd.read_csv(processed_file_path)
    print("Lastet inn ferdigbehandlede data.")
else:
    # Last inn data og gjør tokenisering og stemming
    data = pd.read_csv('MLProsjekt/text.csv')
    stemmer = PorterStemmer()
    data['tokens'] = data['text'].apply(lambda x: [stemmer.stem(word) for word in word_tokenize(x)])

    # Lagre den bearbeidede filen for fremtidig bruk
    data.to_csv(processed_file_path, index=False)
    print("Data behandlet og lagret.")

# Del opp datasettet i trenings- og testdata
X_train_text, X_test_text, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Sett opp TF-IDF-omformer og tilpass på treningsdata
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(X_train_text)  # Fit-transform på treningsdata
X_test = vectorizer.transform(X_test_text)        # Transform på testdata uten fit

# Tren en enkel logistisk regresjonsmodell
model = LogisticRegression(max_iter=100, C = 0.5)
model.fit(X_train, y_train)

# Lag prediksjoner på testsettet
y_pred = model.predict(X_test)

# Beregn og skriv ut basis-score
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Basis accuracy: {accuracy}")
print("Classification Report:\n", report)

import pickle

# Save the model and vectorizer with pickle
with open('MLProsjekt/logistic_regression_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('MLProsjekt/tfidf_vectorizer_balanced.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)