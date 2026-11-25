import torch
from pathlib import Path
from transformers import AutoTokenizer

from src.preprocessing import TextPreprocessor
from src.model_training import BERTweetClassifier

MODEL_PATH = Path(__file__).parent.parent / "models" / "sentiment_model.pt"
TOKENIZER_NAME = "bert-base-uncased"

class SentimentAnalyzer:
    """Main class for sentiment analysis inference."""
    
    def __init__(self, model_path=MODEL_PATH, device='auto'):
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'auto' else device)
        self.model_path = Path(model_path)
        
        # Initialize preprocessor
        self.preprocessor = TextPreprocessor(language='english')
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        
        # Load model
        self.model = BERTweetClassifier(num_classes=2)
        
        if self.model_path.exists():
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            print(f"Model loaded from {self.model_path}")
        else:
            print(f"Warning: Model not found at {self.model_path}. Using random weights.")
        
        self.model.to(self.device)
        self.model.eval()
    
    def preprocess_text(self, text):
        """Preprocess input text."""
        return self.preprocessor.preprocess(text, clean=True, remove_stopwords_flag=True, stem=False)
    
    def analyze(self, text):
        """Analyze sentiment of given text."""
        # Preprocess
        cleaned_text = self.preprocess_text(text)
        
        # Tokenize
        encoding = self.tokenizer(
            cleaned_text,
            add_special_tokens=True,
            max_length=128,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
        )
        
        input_ids = torch.tensor(encoding['input_ids']).unsqueeze(0).to(self.device)
        attention_mask = torch.tensor(encoding['attention_mask']).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            prediction = int(torch.argmax(outputs, dim=1).cpu().numpy()[0])
        
        # Map to label
        label_map = {0: "Negative", 1: "Positive"}
        confidence = float(max(probabilities))
        
        return {
            'text': text,
            'cleaned_text': cleaned_text,
            'sentiment': label_map[prediction],
            'confidence': confidence,
            'probabilities': {
                'negative': float(probabilities[0]),
                'positive': float(probabilities[1])
            }
        }
    
    def batch_analyze(self, texts, batch_size=16):
        """Analyze sentiment for multiple texts."""
        results = []
        for text in texts:
            result = self.analyze(text)
            results.append(result)
        return results