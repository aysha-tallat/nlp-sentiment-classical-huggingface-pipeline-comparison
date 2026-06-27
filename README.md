# NLP Sentiment Analysis — IMDb Movie Reviews

Comparing a classical ML pipeline against a HuggingFace transformer on the IMDb dataset.

---

## Project Structure

```
nlp-sentiment/
├── sentiment_classical.py   # TF-IDF + Logistic Regression
├── sentiment_hf.py          # DistilBERT via HuggingFace pipeline
├── nlp_sentiment.yml        # conda environment
└── README.md
```

---

## Setup

```bash
conda env create -f nlp_sentiment.yml
conda activate nlp-sentiment

# install transformers and torch separately
pip install transformers
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## Dataset

**IMDb Movie Reviews** — 25k train / 25k test, binary sentiment (positive / negative).

```python
# load with shuffle to ensure both classes are present
train_data = load_dataset("stanfordnlp/imdb", split="train").shuffle(seed=42).select(range(5000))
test_data  = load_dataset("stanfordnlp/imdb", split="test").shuffle(seed=42).select(range(1000))
```

> `shuffle(seed=42)` is required — the raw dataset is sorted by label, so slicing without shuffle returns only one class.

---

## Pipeline 1 — Classical ML

**Stack:** text preprocessing → TF-IDF → Logistic Regression

### Preprocessing

```python
def clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)      # strip HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)     # keep letters only
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)
```

| Step | What it does |
|---|---|
| `lower()` | unifies `"Great"` and `"great"` as the same feature |
| strip HTML | IMDb reviews contain raw `<br />` tags |
| keep letters | removes punctuation and numbers |
| lemmatize | `"running"` → `"run"`, `"movies"` → `"movie"` |
| remove stopwords | drops `"the"`, `"is"`, `"and"` — no sentiment signal |

### TF-IDF Vectorization

```python
vectorizer = TfidfVectorizer(max_features=20_000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)       # never fit on test data
```

- **TF (Term Frequency)** — how often a word appears in this review
- **IDF (Inverse Document Frequency)** — penalizes words that appear in every review
- **TF-IDF = TF × IDF** — high score = frequent here, rare elsewhere = meaningful
- `ngram_range=(1, 2)` — captures `"good"` AND `"not good"` as features
- Output: sparse matrix of shape `(5000, 20000)`

### Model

```python
model = LogisticRegression(max_iter=1000, C=1.0)
model.fit(X_train_vec, y_train)
```

- `max_iter=1000` — default 100 often fails to converge on large sparse text data
- `C=1.0` — regularization strength, controls overfitting

### Results

```
              precision    recall  f1-score   support
    negative       0.88      0.83      0.85       512
    positive       0.83      0.88      0.86       488
    accuracy                           0.85      1000
```

### Top features learned by the model

```
positive: great, best, excellent, love, well, wonderful, amazing, favorite, perfect, fun
negative: bad, worst, waste, nothing, poor, awful, boring, even, terrible, oh
```

These are the words with the highest and lowest learned coefficients — exactly what the model uses to decide sentiment. This is what makes logistic regression explainable.

---

## Pipeline 2 — HuggingFace Transformer

**Stack:** HuggingFace `pipeline()` with pre-trained DistilBERT — no training needed.

```python
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    truncation=True,
    max_length=128,   # reduces inference time ~4x vs 512 with minimal accuracy loss
    batch_size=8,
)

texts  = list(dataset["text"])    # must be plain Python list
labels = list(dataset["label"])

results = classifier(texts)
label_map = {"POSITIVE": 1, "NEGATIVE": 0}
preds = [label_map[r["label"]] for r in results]
```

Three things happen inside every `pipeline()` call:
1. **Tokenize** — subword tokenization (WordPiece), converts text to vocab IDs
2. **Forward pass** — attention layers produce contextual embeddings
3. **Decode** — classification head outputs label + confidence score

### Results

```
              precision    recall  f1-score   support
    negative       0.87      0.90      0.89       512
    positive       0.89      0.86      0.88       488
    accuracy                           0.88      1000
```

---

## Comparison

| | Classical (TF-IDF + LR) | DistilBERT |
|---|---|---|
| Accuracy | 85% | **88%** |
| Negative F1 | 0.85 | **0.89** |
| Positive F1 | 0.86 | **0.88** |
| Training needed | yes (5000 samples) | no (zero-shot) |
| Inference speed | instant | ~8 min on CPU |
| Explainable | yes (inspect coefficients) | no |
| Model size | few KB | 268 MB |

### Why the 3% gap exists

TF-IDF is bag-of-words — word order is lost. `"not good"` becomes two independent features: `not` (stopword, removed) and `good` (positive weight) → predicts positive.

DistilBERT reads the full sequence. Its attention mechanism lets every token attend to every other token — `"not"` directly modifies `"good"`, the model understands the negation → predicts negative correctly.

### When to use which

**Use classical when:**
- Explainability is required (regulated industries, debugging)
- Very high inference volume where speed matters
- Text is short and domain-specific with clear keywords

**Use transformers when:**
- Accuracy is the priority
- No labeled training data available (zero-shot)
- Text has complex language: negation, sarcasm, long context

---

## Known Issues & Fixes

| Error | Fix |
|---|---|
| `HfUriError: Repository id must be 'namespace/name'` | use `stanfordnlp/imdb` instead of `imdb` |
| `ValueError: only one class in data` | add `.shuffle(seed=42)` before `.select()` |
| `ModuleNotFoundError: transformers` | run `pip install transformers` separately from torch |
| `--index-url` breaks transformers install | install torch and transformers in separate pip commands |
| Pipeline receives dataset object | wrap with `list()`: `texts = list(dataset["text"])` |

---

## Run

```bash
# classical pipeline
python3 sentiment_classical.py

# transformer pipeline
python3 sentiment_hf.py

#to see them working run flask app
python3 app.py
```
