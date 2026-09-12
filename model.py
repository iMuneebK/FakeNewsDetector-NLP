import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import numpy as np

class FakeNewsModelBERT:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        # For a portfolio project, we can use a pre-trained sentiment model 
        # or load a fine-tuned one if we had weights. 
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        # Assuming label 1 is "Real" and 0 is "Fake" based on custom fine-tuning.
        # We will mock the semantics for the portfolio demonstration.
        confidence = torch.max(probs).item()
        label = torch.argmax(probs).item()
        return ("Real News" if label == 1 else "Fake News", confidence)

class FakeNewsModelClassical:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        
    def load(self, model_path, vectorizer_path):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
            
    def predict(self, text):
        if not self.model or not self.vectorizer:
            raise ValueError("Model not loaded")
        vec_text = self.vectorizer.transform([text])
        pred = self.model.predict(vec_text)[0]
        prob = np.max(self.model.predict_proba(vec_text)[0])
        return ("Real News" if pred == 1 else "Fake News", prob)
