import streamlit as st
import os

from model import FakeNewsModelBERT, FakeNewsModelClassical
from preprocessor import TextPreprocessor
from utils import get_explanation
from train import train_classical_model

st.set_page_config(page_title="Fake News Detector", page_icon="🕵️‍♂️")

# Train classical model on startup if it doesn't exist
if not os.path.exists("models/logreg_model.pkl"):
    train_classical_model()

@st.cache_resource
def load_models():
    bert_model = FakeNewsModelBERT()
    
    classical_model = FakeNewsModelClassical()
    classical_model.load("models/logreg_model.pkl", "models/tfidf_vectorizer.pkl")
    
    preprocessor = TextPreprocessor()
    return bert_model, classical_model, preprocessor

def main():
    st.title("🕵️‍♂️ AI Fake News Detector")
    st.markdown("Analyze news articles to determine credibility using Transformer models and Classical ML.")
    
    bert_model, classical_model, preprocessor = load_models()
    
    model_choice = st.sidebar.selectbox("Choose Model", ["BERT (Transformer)", "Logistic Regression"])
    
    article_text = st.text_area("Paste news article text here:", height=300)
    
    if st.button("Analyze Article", type="primary"):
        if not article_text.strip():
            st.warning("Please enter some text to analyze.")
            return
            
        with st.spinner("Analyzing..."):
            clean_text = preprocessor.clean_text(article_text)
            
            if model_choice == "BERT (Transformer)":
                label, confidence = bert_model.predict(article_text)
            else:
                label, confidence = classical_model.predict(clean_text)
                
            # Display results
            st.subheader("Results")
            
            if label == "Real News":
                st.success(f"**{label}**")
            else:
                st.error(f"**{label}**")
                
            st.progress(confidence)
            st.write(f"**Confidence:** {confidence:.2%}")
            
            st.info("**Explanation:**\n" + get_explanation(article_text, label, confidence))

if __name__ == "__main__":
    main()
