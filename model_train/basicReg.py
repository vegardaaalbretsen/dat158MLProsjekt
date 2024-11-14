from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
from nltk.corpus import stopwords
import nltk
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

nltk.download('stopwords')

# Load the dataset
data = pd.read_csv('./data/balanced_dataset.csv')

# Prepare features and labels
X = data['tokens']
y = data['label']

# Vectorizer configuration
vectorizer = TfidfVectorizer(
    max_features=5000,  # Max number of features
    stop_words=stopwords.words('english'),  # Use stopwords
    ngram_range=(1, 2),  # Unigrams and bigrams
    min_df=2,  # Min document frequency
    max_df=0.7,  # Max document frequency
    lowercase=True,
    strip_accents='unicode'
)

# Transform text data
X = vectorizer.fit_transform(data['text'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
model = LogisticRegression(C=1, penalty='l2', solver='lbfgs')
model.fit(X_train, y_train)
# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Display results
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"], 
            yticklabels=["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"])
plt.title("Confusion Matrix using tokenized and stemmed words")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.show()

# Save the model and vectorizer
model_path = './saved_model/logistic_regression_model_tokens.pkl'
vectorizer_path = './saved_model/vectorizer_tokens.pkl'

with open(model_path, 'wb') as model_file:
    pickle.dump(model, model_file)
    print(f"Model saved to {model_path}")

with open(vectorizer_path, 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)
    print(f"Vectorizer saved to {vectorizer_path}")
