"""
Training entry point for the ML support-ticket classifier.

Trains TF-IDF + LogisticRegression on data/tickets.csv and saves the
pipeline to models/ticket_classifier.joblib (auto-loaded by the app).
"""

import argparse

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from src.ticket_classification import TicketClassifier, load_ticket_data

DEFAULT_DATA = "data/tickets.csv"


def main(filepath=None):
    df = load_ticket_data(filepath) if filepath else load_ticket_data()
    print(f"Loaded {len(df)} labeled tickets, {df['category'].nunique()} categories")

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["category"]
    )

    classifier = TicketClassifier()
    classifier.train_ml(train_df, text_col="ticket_text", label_col="category", cv=4)
    classifier.save_ml()

    preds = [classifier.classify(t)["category"] for t in test_df["ticket_text"]]
    print("\nHold-out classification report:")
    print(classification_report(test_df["category"], preds, zero_division=0))
    print("Model saved to models/ticket_classifier.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the ML ticket classifier")
    parser.add_argument("--data", default=None, help="Path to a CSV with ticket_text,category columns")
    args = parser.parse_args()
    main(args.data)
