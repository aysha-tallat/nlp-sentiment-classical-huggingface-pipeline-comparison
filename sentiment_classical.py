import nltk
import numpy as np
from datasets import load_dataset
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import re

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# --- data ---
train_data = load_dataset("stanfordnlp/imdb", split="train").shuffle(seed=42).select(range(5000))
test_data  = load_dataset("stanfordnlp/imdb", split="test").shuffle(seed=42).select(range(1000))

# --- preprocessing ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)       # strip HTML
    text = re.sub(r"[^a-z\s]", " ", text)       # keep letters only
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)


X_train = [clean(t) for t in train_data["text"]]
X_test  = [clean(t) for t in test_data["text"]]
y_train = train_data["label"]
y_test  = test_data["label"]


# --- features ---
vectorizer = TfidfVectorizer(max_features=20_000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

# --- model ---
model = LogisticRegression(max_iter=1000, C=1.0)
model.fit(X_train_vec, y_train)

# --- evaluate ---
preds = model.predict(X_test_vec)
print(classification_report(y_test, preds, target_names=["negative", "positive"]))

# --- inspect top features ---
feature_names = vectorizer.get_feature_names_out()
coefs = model.coef_[0]
top_pos = np.argsort(coefs)[-10:][::-1]
top_neg = np.argsort(coefs)[:10]

print("\nTop positive words:", [feature_names[i] for i in top_pos])
print("Top negative words:", [feature_names[i] for i in top_neg])