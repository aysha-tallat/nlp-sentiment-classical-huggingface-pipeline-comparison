import re
import nltk
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from datasets import load_dataset
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import pipeline as hf_pipeline
from flask import send_from_directory

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

app = Flask(__name__)
CORS(app)



@app.route("/")
def index():
    return send_from_directory(".", "index.html")

# ── preprocessing ──────────────────────────────────────────────────────────────
lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words("english"))

def clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)

# ── train classical pipeline on startup ────────────────────────────────────────
print("Loading dataset and training classical model...")
train_data = load_dataset("stanfordnlp/imdb", split="train").shuffle(seed=42).select(range(5000))

X_train = [clean(t) for t in train_data["text"]]
y_train  = train_data["label"]

vectorizer = TfidfVectorizer(max_features=20_000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)

clf = LogisticRegression(max_iter=1000, C=1.0)
clf.fit(X_train_vec, y_train)

feature_names = vectorizer.get_feature_names_out()
coefs         = clf.coef_[0]
print("Classical model ready.")

# ── load HuggingFace pipeline ───────────────────────────────────────────────────
print("Loading HuggingFace model...")
hf_classifier = hf_pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    truncation=True,
    max_length=128,
)
print("HuggingFace model ready.")


# ── helpers ────────────────────────────────────────────────────────────────────
def classical_analysis(text: str) -> dict:
    cleaned   = clean(text)
    vec       = vectorizer.transform([cleaned])
    label_idx = clf.predict(vec)[0]
    proba     = clf.predict_proba(vec)[0]
    label     = "POSITIVE" if label_idx == 1 else "NEGATIVE"
    confidence = float(proba[label_idx])

    # words in this input and their contribution (coef * tfidf weight)
    nonzero_cols = vec.nonzero()[1]
    word_scores  = []
    for col in nonzero_cols:
        word   = feature_names[col]
        score  = float(coefs[col])          # positive coef → positive word
        tfidf  = float(vec[0, col])
        word_scores.append({"word": word, "score": score, "tfidf": tfidf})

    # top 5 words that drove the prediction
    word_scores.sort(key=lambda x: abs(x["score"]), reverse=True)
    top_words = word_scores[:5]

    return {
        "label":      label,
        "confidence": round(confidence * 100, 1),
        "top_words":  top_words,
    }


def hf_analysis(text: str) -> dict:
    result     = hf_classifier(text)[0]
    label      = result["label"]        # "POSITIVE" or "NEGATIVE"
    confidence = round(result["score"] * 100, 1)

    # simple token highlighting: words not in stopwords, sorted by position
    tokens = [w for w in re.findall(r"[a-z]+", text.lower()) if w not in stop_words]

    return {
        "label":      label,
        "confidence": confidence,
        "tokens":     tokens[:10],      # first 10 meaningful tokens shown
    }


# ── route ──────────────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    return jsonify({
        "classical": classical_analysis(text),
        "hf":        hf_analysis(text),
    })


if __name__ == "__main__":
    app.run(debug=False, port=5000)
