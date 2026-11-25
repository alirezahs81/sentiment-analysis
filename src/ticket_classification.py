"""
Ticket Classification Module

Handles classification of support tickets into categories such as:
- Technical Issues
- Billing/Payments
- Account Access
- Feature Requests
- General Inquiries
- Urgency/Priority labeling
"""

import re
import string
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

try:
    import joblib
except ImportError:  # joblib ships with scikit-learn
    joblib = None

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
TICKET_MODEL_PATH = MODELS_DIR / "ticket_classifier.joblib"


# Ticket category mappings
TICKET_CATEGORIES = {
    0: "Technical Issues",
    1: "Billing & Payments",
    2: "Account Access",
    3: "Feature Requests",
    4: "General Inquiries",
    5: "Urgent/Support Priority",
}

TICKET_PRIORITY = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "urgent": 4,
}

DEFAULT_CATEGORY = "General Inquiries"


class TicketPreprocessor:
    """Preprocessor specialized for support ticket text."""
    
    def __init__(self, language='english'):
        self.language = language
        self._init_stopwords()
    
    def _init_stopwords(self):
        """Initialize stopwords for the selected language."""
        import nltk
        try:
            nltk.download('stopwords', quiet=True)
            if self.language == 'english':
                self.stop_words = set(nltk.corpus.stopwords.words('english'))
            else:
                self.stop_words = set()
        except:
            # Basic fallback stopwords
            self.stop_words = set(['the', 'and', 'or', 'but', 'is', 'are',
                                  'was', 'were', 'been', 'being', 'to', 'of',
                                  'in', 'on', 'for', 'with', 'as', 'at'])
    
    def clean_ticket_text(self, text):
        """Clean and normalize ticket text."""
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\-@.#%]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_keywords(self, text, top_n=10):
        """Extract potential keywords from ticket text."""
        words = text.split()
        # Filter out stopwords and short words
        filtered = [w for w in words if w not in self.stop_words and len(w) > 3]
        
        # Simple frequency count
        freq = defaultdict(int)
        for word in filtered:
            freq[word] += 1
        
        # Return top N
        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return sorted_words[:top_n]


class TicketClassifier:
    """
    Ticket classification using rule-based approaches 
    or can be extended with ML/NLP models.
    """
    
    def __init__(self, model_name='rule-based'):
        self.model_name = model_name
        self.preprocessor = TicketPreprocessor()
        self.category_keywords = self._load_category_keywords()
        self.ml_pipeline = None  # fitted TF-IDF + LogisticRegression, if available

    # ---------------- ML path ----------------

    def train_ml(self, df, text_col='ticket_text', label_col='category', cv=5):
        """
        Train a TF-IDF + LogisticRegression classifier on labeled tickets
        and switch the classifier to the ML method. Falls back to the
        rule-based method when too few examples are available.
        """
        df = df.dropna(subset=[text_col, label_col])
        if len(df) < 10 or df[label_col].nunique() < 2:
            print("Not enough labeled tickets - staying with the rule-based method")
            return None

        self.ml_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ('clf', LogisticRegression(max_iter=1000, C=10)),
        ])

        scores = None
        if cv and len(df) >= cv * 2:
            try:
                scores = cross_val_score(
                    self.ml_pipeline, df[text_col], df[label_col], cv=cv, scoring='f1_macro'
                )
                print(f"ML ticket classifier CV F1-macro: {scores.mean():.3f} ± {scores.std():.3f}")
            except ValueError:
                scores = None

        self.ml_pipeline.fit(df[text_col], df[label_col])
        self.model_name = 'tfidf-logreg'
        return scores

    def save_ml(self, filepath=TICKET_MODEL_PATH):
        """Persist the fitted ML pipeline with joblib."""
        if self.ml_pipeline is None or joblib is None:
            return False
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.ml_pipeline, filepath)
        return True

    def load_ml(self, filepath=TICKET_MODEL_PATH):
        """Load a previously saved ML pipeline; returns True on success."""
        if joblib is None:
            return False
        filepath = Path(filepath)
        if not filepath.exists():
            return False
        self.ml_pipeline = joblib.load(filepath)
        self.model_name = 'tfidf-logreg'
        return True

    # ---------------- classification ----------------
    
    def _load_category_keywords(self):
        """Load keyword mappings for each ticket category."""
        return {
            "Technical Issues": [
                "error", "bug", "crash", "not working", "broken", 
                "fix", "problem", "issue", "fail", "doesn't work"
            ],
            "Billing & Payments": [
                "payment", "charge", "bill", "invoice", "pay", "cost",
                "price", "refund", "subscription", "payment method"
            ],
            "Account Access": [
                "login", "password", "access", "account", "locked",
                "sign in", "credentials", "username"
            ],
            "Feature Requests": [
                "feature", "want", "suggestion", "improve", "enhance",
                "add", "like to see", "new feature"
            ],
            "General Inquiries": [
                "question", "info", "hello", "hi", "thanks",
                "contact", "reach out"
            ],
            "Urgent/Support Priority": [
                "urgent", "asap", "immediately", "critical", "emergency",
                "stop working", "cannot use"
            ]
        }
    
    def rule_based_classify(self, text):
        """
        Classify ticket using keyword matching.
        Returns (category, confidence_score).
        """
        cleaned = self.preprocessor.clean_ticket_text(text)

        if not cleaned:
            return DEFAULT_CATEGORY, 0.0

        category_scores = defaultdict(int)

        # Substring match so word forms like "crashing" match "crash"
        # and multi-word phrases like "not working" are handled.
        for category, kws in self.category_keywords.items():
            for kw in kws:
                if kw in cleaned:
                    category_scores[category] += 1
        
        if not category_scores:
            return DEFAULT_CATEGORY, 0.0
        
        # Get best category
        best_category = max(category_scores, key=category_scores.get)
        confidence = category_scores[best_category] / max(sum(category_scores.values()), 1)
        
        return best_category, min(confidence, 1.0)
    
    def classify(self, text):
        """
        Public classification method.
        Uses the trained ML model when available, otherwise rule-based.
        Returns dict with category and confidence.
        """
        if self.ml_pipeline is not None:
            proba = self.ml_pipeline.predict_proba([text])[0]
            best = proba.argmax()
            category = self.ml_pipeline.classes_[best]
            confidence = float(proba[best])
            return {
                'text': text,
                'category': category,
                'priority': self._estimate_priority(text),
                'confidence': confidence,
                'method': 'tfidf-logreg',
            }

        category, confidence = self.rule_based_classify(text)

        return {
            'text': text,
            'category': category,
            'priority': self._estimate_priority(text),
            'confidence': confidence,
            'method': 'rule-based',
        }
    
    def _estimate_priority(self, text):
        """Estimate urgency/priority level."""
        if not isinstance(text, str):
            return "medium"

        urgent_keywords = ['urgent', 'asap', 'immediately', 'critical',
                           'emergency', 'cannot', 'broken', 'stop']
        text_lower = text.lower()
        
        for kw in urgent_keywords:
            if kw in text_lower:
                return "high"
        
        # Check for polite/normal requests
        normal_keywords = ['hello', 'hi', 'thanks', 'question']
        for kw in normal_keywords:
            if kw in text_lower:
                return "low"
        
        return "medium"


class TicketDataset:
    """Dataset class for ticket classification training."""
    
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
        )
        
        return {
            'input_ids': torch.tensor(encoding['input_ids']),
            'attention_mask': torch.tensor(encoding['attention_mask']),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# Sample ticket data for demonstration
SAMPLE_TICKETS = pd.DataFrame({
    'ticket_text': [
        "My payment was charged twice last month, need refund ASAP!",
        "Can't login to my account, password reset not working",
        "Love the new feature, want to suggest another integration",
        "App crashes every time I try to upload a photo",
        "Hello, I have a question about my subscription plan",
        "System error when trying to checkout, help needed!"
    ],
    'category': [1, 2, 3, 0, 4, 1],  # Billing, Account, Feature, Tech, General, Billing
    'priority': [3, 3, 1, 3, 1, 3],  # high, high, low, high, low, high
    'label': [1, 2, 3, 0, 4, 1]  # integer labels
})


def create_sample_ticket_data(filepath=None):
    """Create sample ticket dataset if no file provided."""
    if filepath:
        df = pd.read_csv(filepath)
        return df
    return SAMPLE_TICKETS.copy()


def load_ticket_data(filepath=Path(__file__).resolve().parent.parent / "data" / "tickets.csv"):
    """Load the bundled labeled ticket dataset (or a custom CSV)."""
    return pd.read_csv(filepath)


def train_ticket_classifier(filepath=None, save=True):
    """Train the ML ticket classifier on the bundled dataset."""
    df = load_ticket_data(filepath) if filepath else load_ticket_data()
    classifier = TicketClassifier()
    classifier.train_ml(df, text_col='ticket_text', label_col='category')
    if save:
        classifier.save_ml()
    return classifier


if __name__ == "__main__":
    # Quick demo
    classifier = TicketClassifier()
    
    test_tickets = [
        "Payment charged twice, need refund immediately!",
        "Forgot password, can't access account",
        "Great app, want dark mode feature",
        "App keeps crashing on startup"
    ]
    
    print("=" * 60)
    print("TICKET CLASSIFICATION DEMO")
    print("=" * 60)
    
    for ticket in test_tickets:
        result = classifier.classify(ticket)
        print(f"\nTicket: {ticket[:60]}...")
        print(f"  → Category: {result['category']}")
        print(f"  → Priority: {result['priority']}")
        print(f"  → Confidence: {result['confidence']:.2%}")
        print(f"  → Method: {result['method']}")
    
    print("\n" + "=" * 60)