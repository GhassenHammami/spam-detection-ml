import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter
import warnings

# Configuration
warnings.filterwarnings('ignore')
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
sns.set_style('whitegrid')

def load_and_clean_data(filepath):
    """Loads dataset and performs basic cleaning."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Ensure your dataset is at {filepath}")
        
    df = pd.read_csv(filepath)
    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', axis=1, inplace=True)
    df.columns = ['label', 'text', 'class']
    return df

def preprocess_text(df):
    """Tokenizes and removes stopwords."""
    stop_words = set(stopwords.words('english'))
    df['text'] = df['text'].apply(
        lambda x: ' '.join([word for word in word_tokenize(x.lower()) if word.isalnum() and word not in stop_words])
    )
    return df

def train_models(X_train, y_train):
    """Handles vectorization and model training."""
    # Vectorization
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_vec = tfidf.fit_transform(X_train)
    
    # Logistic Regression with GridSearch
    lr = LogisticRegression(max_iter=1000)
    grid = {"C": [0.1, 1.0, 10.0], "solver": ["liblinear"]}
    logreg_cv = GridSearchCV(lr, grid, cv=3)
    logreg_cv.fit(X_train_vec, y_train)
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(X_train_vec, y_train)
    
    return tfidf, logreg_cv, rf

def evaluate(model, tfidf, X, y, title):
    """Evaluates and prints results."""
    X_vec = tfidf.transform(X)
    preds = model.predict(X_vec)
    print(f"\n--- {title} ---")
    print(f"Accuracy: {accuracy_score(y, preds):.4f}")
    print(classification_report(y, preds))

if __name__ == "__main__":
    # 1. Setup
    DATA_PATH = 'data/spam_ham_dataset.csv'
    df = load_and_clean_data(DATA_PATH)
    
    # 2. Preprocess
    print("Pre-processing text...")
    df = preprocess_text(df)
    
    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['class'], test_size=0.2, random_state=42, stratify=df['class']
    )
    
    # 4. Train
    print("Training models (this may take a moment)...")
    tfidf, lr_model, rf_model = train_models(X_train, y_train)
    
    # 5. Final Test Evaluation
    evaluate(lr_model, tfidf, X_test, y_test, "Logistic Regression (Test Set)")
    evaluate(rf_model, tfidf, X_test, y_test, "Random Forest (Test Set)")