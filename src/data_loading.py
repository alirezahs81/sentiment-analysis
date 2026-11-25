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

def save_processed_data(df, filepath):
    """Save processed data to CSV."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")