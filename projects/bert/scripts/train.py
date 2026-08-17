"""Small CPU experiment: TF-IDF baseline vs. DistilBERT on AG News."""

import argparse
import csv
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer


LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data(max_train, max_test, seed):
    data = load_dataset("ag_news")
    train = data["train"].shuffle(seed=seed).select(range(min(max_train, len(data["train"]))))
    test = data["test"].shuffle(seed=seed).select(range(min(max_test, len(data["test"]))))
    return train, test


def score(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }


def run_tfidf(train, test, artifact_dir):
    vectorizer = TfidfVectorizer(max_features=20_000, ngram_range=(1, 2), min_df=2)
    x_train = vectorizer.fit_transform(train["text"])
    x_test = vectorizer.transform(test["text"])
    model = LogisticRegression(max_iter=200)
    model.fit(x_train, train["label"])
    result = score(test["label"], model.predict(x_test))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with (artifact_dir / "tfidf_logistic.pkl").open("wb") as file:
        pickle.dump({"vectorizer": vectorizer, "model": model}, file)
    return {"model": "tfidf-logistic", "fraction": 1.0, **result}


def run_distilbert(
    train, test, fraction, model_name, epochs, batch_size, max_length, seed, artifact_dir
):
    count = max(1, int(len(train) * fraction))
    train = train.shuffle(seed=seed).select(range(count))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)
    model.config.id2label = dict(enumerate(LABEL_NAMES))
    model.config.label2id = {label: index for index, label in enumerate(LABEL_NAMES)}
    model.to("cpu")

    def encode(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)

    train = train.map(encode, batched=True, remove_columns=["text"])
    test = test.map(encode, batched=True, remove_columns=["text"])
    train.set_format("torch")
    test.set_format("torch")
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    model.train()
    for _ in range(epochs):
        for batch in train_loader:
            labels = batch.pop("label")
            output = model(**batch, labels=labels)
            output.loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    model.eval()
    predictions, labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch_labels = batch.pop("label")
            predictions.extend(model(**batch).logits.argmax(dim=1).tolist())
            labels.extend(batch_labels.tolist())
    result = score(labels, predictions)
    checkpoint_dir = artifact_dir / f"distilbert_{int(fraction * 100)}pct"
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    return {"model": "distilbert", "fraction": fraction, **result}


def save_results(results, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["model", "fraction", "accuracy", "macro_f1"])
        writer.writeheader()
        writer.writerows(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train", type=int, default=3000)
    parser.add_argument("--max-test", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs",
        help="Directory for results.csv and reusable prediction artifacts.",
    )
    args = parser.parse_args()

    seed_everything(args.seed)
    train, test = load_data(args.max_train, args.max_test, args.seed)
    artifact_dir = args.output_dir / "models"
    results = [run_tfidf(train, test, artifact_dir)]
    for fraction in (0.10, 0.25, 1.0):
        print(f"Training DistilBERT with {fraction:.0%} of the data")
        results.append(
            run_distilbert(
                train, test, fraction, args.model, args.epochs,
                args.batch_size, args.max_length, args.seed, artifact_dir
            )
        )
    save_results(results, args.output_dir / "results.csv")
    for result in results:
        print(result)


if __name__ == "__main__":
    main()
