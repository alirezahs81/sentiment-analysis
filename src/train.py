"""
Training pipeline entry point for sentiment analysis.

Runs the complete flow: load data -> split -> train BERT classifier
-> evaluate -> persist model weights and reports.
"""

import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from src.data_loading import load_sample_data
from src.model_training import BERTweetClassifier, Trainer

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

MODEL_NAME = "bert-base-uncased"
RANDOM_STATE = 42


def train_full_pipeline(
    df: pd.DataFrame = None,
    epochs: int = 3,
    learning_rate: float = 2e-5,
    batch_size: int = 16,
    max_len: int = 128,
) -> dict:
    """Run the complete training pipeline and return artifacts."""
    print("=" * 60)
    print("SENTIMENT ANALYSIS - TRAINING PIPELINE")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] Loading data...")
    if df is None:
        df = load_sample_data()
    print(f"   Loaded {len(df)} samples")

    # 2. Split data
    print("\n[2/5] Splitting data...")
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE, stratify=df["label"]
    )
    print(f"   Train: {len(train_df)} samples")
    print(f"   Val: {len(val_df)} samples")

    # 3. Initialize model and tokenizer
    print("\n[3/5] Initializing model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = BERTweetClassifier(model_name=MODEL_NAME, num_classes=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"   Using device: {device}")

    # 4. Train
    print("\n[4/5] Training...")
    print("-" * 40)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_df=train_df,
        val_df=val_df,
        epochs=epochs,
        learning_rate=learning_rate,
        max_len=max_len,
        batch_size=batch_size,
        device=device,
    )
    history = trainer.train(show_progress=True)

    # 5. Evaluate and persist
    print("\n" + "-" * 40)
    print("Evaluation on validation set:")
    metrics = trainer.evaluate(val_df)

    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print("\nClassification Report:")
    report_df = pd.DataFrame(metrics["report"]).transpose()
    print(report_df.round(4))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    trainer.save_model(MODEL_DIR / "sentiment_model.pt")

    history_path = REPORTS_DIR / "training_history.json"
    history_path.write_text(json.dumps(history, indent=4), encoding="utf-8")
    print(f"  Training history saved to {history_path}")

    report_path = REPORTS_DIR / "classification_report.json"
    report_path.write_text(json.dumps(metrics["report"], indent=4), encoding="utf-8")
    print(f"  Classification report saved to {report_path}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print(f"Best Val Accuracy: {max(history['val_accuracy']):.4f}")
    print("=" * 60)

    return {"history": history, "metrics": metrics, "model": trainer.model, "tokenizer": tokenizer}


if __name__ == "__main__":
    train_full_pipeline()
