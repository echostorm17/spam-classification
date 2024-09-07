import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pickle

import subprocess

kaggle_url = 'mfaisalqureshi/spam-email'
file_name = 'spam.csv'

subprocess.run(['kaggle', 'datasets', 'download', '-d', kaggle_url, '-f', file_name])

df = pd.read_csv(file_name)

def clean_text(text):
    # Remove special characters, numbers, and punctuation
    text = re.sub(r'\W', ' ', text)
    # Convert to lowercase
    text = text.lower()
    return text

# Apply text cleaning
df['message'] = df['Message'].apply(clean_text)

# Display the first few cleaned messages
df.head()

# Encode labels: 'ham' -> 0, 'spam' -> 1
df['label'] = df['Category'].map({'ham': 0, 'spam': 1})

# Display label distribution
df['label'].value_counts()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Display the size of the train and test sets
print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

# Initialize the CountVectorizer
vectorizer = CountVectorizer()

# Fit and transform the training data
X_train_vect = vectorizer.fit_transform(X_train)

# Transform the test data
X_test_vect = vectorizer.transform(X_test)

# Display the shape of the vectorized data
print(f"Training data shape: {X_train_vect.shape}")
print(f"Testing data shape: {X_test_vect.shape}")

# Initialize the Multinomial Naive Bayes model
model = MultinomialNB()

# Train the model on the training data
model.fit(X_train_vect, y_train)

# Display the training completion message
print("Model training completed.")

# Make predictions on the test data
y_pred = model.predict(X_test_vect)

# Display the first few predictions
y_pred[:10]