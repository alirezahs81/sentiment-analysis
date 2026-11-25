import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download NLTK resources
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

class TextPreprocessor:
    def __init__(self, language='english'):
        self.language = language
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        if language == 'english':
            try:
                self.stop_words = set(stopwords.words('english'))
            except LookupError:
                # Fallback if NLTK corpora are not available
                self.stop_words = set()
        elif language == 'persian':
            # Persian stopwords - basic list
            self.stop_words = set([
                'و', 'در', 'است', 'که', 'را', 'می', 'بود', 'بودید', 'باشد', 'بوده',
                'می‌باشد', 'می‌بуд', 'باشم', 'باشه', 'باشد', ' بودند', 'been'
            ])
        else:
            self.stop_words = set()
    
    def clean_text(self, text):
        """Basic text cleaning: lowercasing, removing URLs, special chars."""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags (Twitter style)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Keep only letters, digits, and spaces (for English);
        # for Persian, keep Persian/Arabic characters too
        if self.language == 'persian':
            # Keep Persian/Arabic characters, digits, and spaces
            text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\d\s]', ' ', text)
        else:
            # Keep only ASCII letters, digits, and spaces; replace everything else
            text = re.sub(r'[^a-z0-9\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_stopwords(self, text):
        """Remove stopwords from text."""
        words = text.split()
        filtered_words = [w for w in words if w not in self.stop_words]
        return ' '.join(filtered_words)
    
    def stem(self, text):
        """Apply stemming."""
        words = text.split()
        stemmed_words = [self.stemmer.stem(w) for w in words]
        return ' '.join(stemmed_words)
    
    def lemmatize(self, text):
        """Apply lemmatization."""
        words = text.split()
        lemmatized_words = [self.lemmatizer.lemmatize(w) for w in words]
        return ' '.join(lemmatized_words)
    
    def preprocess(self, text, 
                   clean=True, 
                   remove_stopwords_flag=True, 
                   stem=True, 
                   lemmatize=False):
        """Full preprocessing pipeline."""
        if clean:
            text = self.clean_text(text)
        
        if remove_stopwords_flag:
            text = self.remove_stopwords(text)
        
        if stem:
            text = self.stem(text)
        
        if lemmatize:
            text = self.lemmatize(text)
        
        return text