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