# 🕵️‍♂️ AI Fake News Detector

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B)

An intelligent application to combat misinformation by classifying news articles as Real or Fake, offering both state-of-the-art Transformer models and classical Machine Learning approaches.

## ✨ Features
- **Dual-Model Architecture:** Compare predictions between a DistilBERT Transformer and a classic TF-IDF + Logistic Regression model.
- **Explainability:** Provides confidence scores and heuristic explanations for predictions based on text length and sensationalist keyword tracking.
- **Robust NLP Pipeline:** Includes automated text cleaning (stopword removal, punctuation stripping, HTML tag removal) using NLTK.
- **Interactive UI:** Clean, responsive web interface built with Streamlit for real-time article analysis.

## 🏗️ Architecture
```mermaid
graph LR
    A[News Article] --> B(Streamlit App)
    B --> C{Text Preprocessor}
    C --> D[BERT Model]
    C --> E[Logistic Regression]
    D --> F[Prediction & Confidence]
    E --> F
    F --> G(Explainability Engine)
    G --> B
```

## 🚀 Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/fake-news-detector.git
   cd fake-news-detector
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```
   *(Note: The classical model will automatically train on dummy data upon the first startup).*

## 📊 Results / Demo
*(Include screenshots of your running application here)*
- Example of a real news classification.
- Example of a fake news classification with the explainability module highlighting sensationalist words.

## 🛠️ Tech Stack
- **Framework:** PyTorch, Scikit-Learn
- **NLP:** Hugging Face Transformers, NLTK
- **Frontend:** Streamlit
- **Data Manipulation:** Pandas, NumPy
