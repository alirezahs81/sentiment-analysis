import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
from transformers.optimization import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
from pathlib import Path
import time

MODEL_DIR = Path(__file__).parent.parent / "models"

class SentimentDataset(Dataset):
    """Dataset for sentiment classification."""
    
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
            'labels': torch.tensor(label, dtype=torch.float)
        }

class BERTweetClassifier(nn.Module):
    """BERT-based classifier for sentiment analysis."""
    
    def __init__(self, model_name='bert-base-uncased', num_classes=2):
        super(BERTweetClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        pooled_output = output.pooler_output
        output = self.drop(pooled_output)
        return self.out(output)

class Trainer:
    """Training loop for sentiment model."""
    
    def __init__(self, model, tokenizer, train_df, val_df, 
                 epochs=3, learning_rate=2e-5, max_len=128, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.max_len = max_len
        self.device = device
        
        self.train_dataset = SentimentDataset(
            train_df['text'].values,
            train_df['label'].values,
            tokenizer,
            max_len
        )
        self.val_dataset = SentimentDataset(
            val_df['text'].values,
            val_df['label'].values,
            tokenizer,
            max_len
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=16,
            shuffle=True
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=16,
            shuffle=False
        )
        
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.train_loader) * epochs
        )
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch in self.train_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            loss = self.criterion(outputs, labels.long())
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def eval_epoch(self):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.criterion(outputs, labels.long())
                total_loss += loss.item()
                
                preds = torch.softmax(outputs, dim=1).argmax(dim=1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
        
        accuracy = accuracy_score(all_labels, all_preds)
        return total_loss / len(self.val_loader), accuracy, all_preds, all_labels
    
    def train(self, show_progress=True):
        print(f"Starting training on {self.device}...")
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}

        for epoch in range(self.epochs):
            train_loss = self.train_epoch()
            val_loss, val_accuracy, preds, labels = self.eval_epoch()

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)

            if show_progress:
                print(f"Epoch {epoch+1}/{self.epochs}:")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_loss:.4f}")
                print(f"  Val Accuracy: {val_accuracy:.4f}")

        return history
    
    def predict(self, text):
        """Predict sentiment for a single text."""
        self.model.eval()
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
        )
        
        input_ids = torch.tensor(encoding['input_ids']).unsqueeze(0).to(self.device)
        attention_mask = torch.tensor(encoding['attention_mask']).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            prediction = int(torch.argmax(outputs, dim=1).cpu().numpy()[0])
        
        return {
            'prediction': prediction,  # 0=negative, 1=positive
            'probabilities': {
                'negative': float(probabilities[0]),
                'positive': float(probabilities[1])
            }
        }