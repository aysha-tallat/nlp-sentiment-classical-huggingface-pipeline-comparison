from datasets import load_dataset
from transformers import pipeline
from sklearn.metrics import classification_report

# --- data (same 1k test split) ---
dataset = load_dataset("stanfordnlp/imdb", split="test").shuffle(seed=42).select(range(1000))
texts  = list(dataset["text"])
labels = list(dataset["label"])

# --- model (zero-shot: no training needed) ---
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    truncation=True,
    max_length=512,
    batch_size=32,
)

# --- predict ---
results = classifier(texts)

label_map = {"POSITIVE": 1, "NEGATIVE": 0}
preds = [label_map[r["label"]] for r in results]

# --- evaluate ---
print(classification_report(labels, preds, target_names=["negative", "positive"]))

# --- inspect a few predictions ---
for text, result in zip(texts[:3], results[:3]):
    print(f"\n[{result['label']} | {result['score']:.2f}]")
    print(text[:120], "...")