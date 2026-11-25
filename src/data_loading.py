import pandas as pd
import os
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

def load_sample_data():
    """Load sample sentiment data if no custom dataset provided."""
    # Sample Persian/English sentiment data
    sample_data = {
        "text": [
            "I love this product, it's amazing!",
            "This is the worst experience ever.",
            "It's okay, nothing special.",
            "Great quality and fast delivery.",
            "I'm very disappointed with this purchase.",
            "Good value for money.",
            "Terrible customer service, would not recommend.",
            "Excellent product, exactly what I expected.",
        ],
        "label": [1, 0, 0, 1, 0, 1, 0, 1],  # 1=positive, 0=negative
    }
    df = pd.DataFrame(sample_data)
    return df

def load_from_csv(filepath):
    """Load data from CSV file."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    return pd.read_csv(filepath)


def load_nltk_movie_reviews():
    """
    Load the NLTK movie_reviews corpus as a real sentiment dataset
    (~2,000 labeled movie reviews, 1,000 positive / 1,000 negative).

    Downloads the corpus on first use (~4 MB). Falls back to the
    bundled sample data when the corpus is unavailable (e.g. offline).
    """
    import nltk
    from nltk.corpus import movie_reviews

    try:
        nltk.download("movie_reviews", quiet=True)
        docs = [
            (" ".join(movie_reviews.words(fileid)), 1 if category == "pos" else 0)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)
        ]
        df = pd.DataFrame(docs, columns=["text", "label"]).sample(
            frac=1.0, random_state=42
        ).reset_index(drop=True)
        return df
    except LookupError:
        print("movie_reviews corpus unavailable - falling back to sample data")
        return load_sample_data()

def save_processed_data(df, filepath):
    """Save processed data to CSV."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")