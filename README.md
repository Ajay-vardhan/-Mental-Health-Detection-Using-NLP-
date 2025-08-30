# 🧠 Mental Health Detection Using NLP

This repository contains multiple machine learning and deep learning approaches for **multi-class mental health status detection** based on user text data.  
Our models can classify text into **7 categories**:
- Anxiety
- Bipolar
- Depression
- Normal
- Personality Disorder
- Stress
- Suicidal

---

## 📂 Contents
- **`DistilBERT_model_91%.ipynb`** → Transformer-based fine-tuned DistilBERT achieving 91% accuracy.
- **`biLSTM+pretrained word2vec model.ipynb`** → BiLSTM with pretrained Word2Vec embeddings.
- **`An Ensemble Machine Learning Pipeline for Multi-Class Mental Health Status Detection.ipynb`** → Traditional ML + Deep Learning ensemble approach.

---

## 📊 Dataset
The dataset consists of user-generated statements related to mental health, labeled into 7 categories.  
Before training, we:
- Removed stopwords, URLs, and special characters.
- Balanced the dataset to reduce class imbalance.
- Tokenized and padded sequences for deep learning models.

---

## 🔍 Exploratory Data Analysis (EDA)
We performed:
- **Class distribution analysis** (bar & pie charts)
- **Word clouds** for each class
- **Text length distribution**
- **Most frequent words per category**

---

## 🧪 Models Implemented

### 1️⃣ DistilBERT (Transformers)
- Tokenizer: `DistilBertTokenizerFast`
- Architecture: `TFDistilBertForSequenceClassification`
- Optimizer: AdamW with learning rate scheduler
- **Performance**:  
  - Accuracy: **91.06%**
  - Strong performance on most classes except slight drop for Depression & Suicidal

### 2️⃣ BiLSTM + Pretrained Word2Vec
- Embedding: Pretrained Google News Word2Vec
- Bi-directional LSTM with dropout regularization
- Dense softmax classification layer

### 3️⃣ Ensemble Machine Learning
- Models: Logistic Regression, Random Forest, XGBoost
- Voting ensemble with soft probabilities
- Improved macro-average F1 score

---

## 📈 Results

| Class               | Precision | Recall | F1-score |
|---------------------|-----------|--------|----------|
| Anxiety             | 0.96      | 0.97   | 0.97     |
| Bipolar             | 0.97      | 0.98   | 0.98     |
| Depression          | 0.76      | 0.72   | 0.74     |
| Normal              | 0.95      | 0.95   | 0.95     |
| Personality Disorder| 0.99      | 0.98   | 0.98     |
| Stress              | 0.95      | 0.97   | 0.96     |
| Suicidal            | 0.78      | 0.80   | 0.79     |

**Overall Accuracy:** 91.06%  
**Macro F1-score:** 0.91

---



