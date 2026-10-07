"""Reproduce a text-only baseline without outcome-derived location ratios."""
import argparse
import json
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, average_precision_score, roc_auc_score

TEXT_COLUMNS = ["title", "company_profile", "description", "requirements", "benefits"]

def make_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=30000, min_df=2)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])

def train(data_path, output):
    df = pd.read_csv(data_path)
    df["text"] = df[TEXT_COLUMNS].fillna("").agg(" ".join, axis=1).str.strip()
    df = df[df.text.ne("")].drop_duplicates(subset=["text"])
    if not df.fraudulent.isin([0, 1]).all():
        raise ValueError("fraudulent must contain binary 0/1 labels")
    X_train, X_test, y_train, y_test = train_test_split(df.text, df.fraudulent, test_size=0.25, stratify=df.fraudulent, random_state=42)
    model = make_pipeline()
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {
        "model": "TF-IDF + balanced logistic regression",
        "split": "stratified random 75/25, seed 42; exact duplicate texts removed",
        "train_rows": len(X_train), "test_rows": len(X_test),
        "average_precision": average_precision_score(y_test, probabilities),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "classification_report": classification_report(y_test, model.predict(X_test), output_dict=True, zero_division=0),
        "limitations": "Random split on a historical dataset; not time-based or company-disjoint. Similar postings can remain across splits. Not a fraud verdict or a deployment benchmark.",
    }
    Path(output).write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(Path(__file__).parent / "data/fake_job_postings.csv"))
    parser.add_argument("--output", default="baseline_metrics.json")
    args = parser.parse_args()
    train(args.data, args.output)
