import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
from preprocessor import TextPreprocessor

def train_classical_model():
    print("Training Logistic Regression baseline model...")
    # Generate dummy data for demonstration
    data = {
        'text': [
            "The earth is flat and scientists are lying.", 
            "The stock market saw a 5% increase today due to tech earnings.",
            "Aliens landed in New York yesterday and gave everyone free pizza.",
            "The president signed the new healthcare bill into law this morning."
        ],
        'label': [0, 1, 0, 1]  # 0 = Fake, 1 = Real
    }
    df = pd.DataFrame(data)
    
    preprocessor = TextPreprocessor()
    df['clean_text'] = df['text'].apply(preprocessor.clean_text)
    
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['clean_text'])
    y = df['label']
    
    model = LogisticRegression()
    model.fit(X, y)
    
    os.makedirs('models', exist_ok=True)
    with open('models/logreg_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('models/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
        
    print("Classical model training complete.")

if __name__ == "__main__":
    train_classical_model()
