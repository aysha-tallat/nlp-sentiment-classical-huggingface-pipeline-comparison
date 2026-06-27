import nltk
import numpy as np
from datasets import load_dataset
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import re


train_data = load_dataset("stanfordnlp/imdb", split="train[:5000]")
test_data  = load_dataset("stanfordnlp/imdb", split="test[:1000]")

# debug
print(set(train_data["label"]))
print(train_data[0])