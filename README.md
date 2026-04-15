# Email Spam Classifier (NLP)

A high-performance machine learning pipeline designed to classify emails as Ham (legitimate) or Spam. This project demonstrates a full NLP workflow, from text preprocessing to hyperparameter tuning and model evaluation.

---

## Project Structure

The repository is organized as follows:

```text
spam-detection-ml/
├── data/
│   └── spam_ham_dataset.csv       # Dataset source (see below)
├── notebooks/
│   └── spam_detection.ipynb       # Google Colab notebook version
├── src/
│   └── classifier.py              # Main training and evaluation script
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation
```

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/spam-detection-ml.git
   cd spam-detection-ml
   ```

2. **Install dependencies:**
   Make sure you have Python 3.8+ installed.
   ```bash
   pip install -r requirements.txt
   ```

3. **Data Preparation:**
   The dataset used in this project is the **Spam Mails Dataset** (Enron-1) by Venkatesh Garnepudi. You can download it here:
   [Kaggle: Spam Mails Dataset](https://www.kaggle.com/datasets/venky73/spam-mails-dataset/data)
   
   Ensure the `spam_ham_dataset.csv` is placed inside the `data/` folder.

---

## The NLP Pipeline

### 1. Preprocessing
Text data is cleaned by converting to lowercase, removing punctuation, and filtering out common English Stopwords using `nltk`.

### 2. Feature Extraction
We utilize **TF-IDF (Term Frequency-Inverse Document Frequency)** with an N-gram range of (1, 2) to capture context and common phrases.

### 3. Machine Learning Models
* **Logistic Regression:** Tuned via `GridSearchCV`.
* **Random Forest:** Ensemble method with 200 trees and balanced class weights.

---

## Performance & Visualizations

### Model Accuracy
Final evaluation on the Test Set:

| Model | Accuracy | Precision (Spam) | Recall (Spam) | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | ~97.5% | 0.97 | 0.96 | 0.96 |
| **Random Forest** | ~97.1% | 0.99 | 0.92 | 0.95 |

### Confusion Matrix
The confusion matrix below illustrates the performance of our classifiers in distinguishing between Ham and Spam emails:

<img width="1348" height="495" alt="image" src="https://github.com/user-attachments/assets/55316337-e5a1-41e0-9321-db68d07429b1" />

### Feature Importance
This chart shows which specific keywords (like "enron", "http", or "deal") most strongly influence the model's decision-making process:

<img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/2aff09aa-1164-4e72-9ff1-b5743ce5c144" />

---

## Usage

### Local Execution
To run the complete training and evaluation pipeline locally:
```bash
python src/classifier.py
```

### Google Colab
For an interactive version of this project including all visualizations and step-by-step analysis, use the notebook in `notebooks/` or open it in Google Colab.

---

## License
This project is licensed under the [MIT License](LICENSE).
