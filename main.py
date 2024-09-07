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