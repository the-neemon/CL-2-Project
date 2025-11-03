# CL-2-Project  
### Sentiment Analysis on Social Media Texts with Semantic Interpretation  

A comparative study combining **lexicon-based semantic features** with **machine learning approaches** for sentiment classification on Twitter data.  
This project explores how integrating **semantic interpretation techniques**—such as negation handling, intensifiers, and pre-trained embeddings—can improve sentiment analysis performance compared to traditional lexical methods.

---

## Project Overview
The project aims to evaluate how combining **semantic features** with standard text-based representations affects the accuracy and robustness of sentiment classification models on social media text.  

**Key Objectives:**
- Develop preprocessing pipeline for noisy Twitter data  
- Engineer semantic, contextual, and traditional lexical features  
- Compare model performance with and without semantic enrichment  
- Conduct feature ablation, error analysis, and qualitative interpretation  

---

## Methods

### **Data Preprocessing**
- Removal of URLs, mentions, hashtags, and special characters using NLTK  
- Tokenization, normalization, stopword removal, and lemmatization  
- Handling emojis and emoticons to retain sentiment cues  

### **Feature Engineering**
- **Traditional features:** N-grams, POS tags  
- **Contextual features:** Negation and intensifier detection  
- **Semantic features:** Word2Vec/GloVe embeddings and lexicon polarity scores (VADER, NRC)  
- Each model is first trained with all features; the best-performing one is then re-evaluated under feature ablation (isolating each feature type).  

### **Models**
- Naive Bayes  
- Logistic Regression  
- Random Forest  

### **Evaluation Metrics**
- Accuracy, Precision, Recall, F1-score (with k-fold cross-validation)  
- Error and success analysis to understand model behavior  
- Qualitative analysis of semantic influence on classification  

---

## Datasets

### **1 Primary Dataset: Sentiment140**
- **Source:** [Kaggle – Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)  
- **Description:** 1.6M labeled tweets (subset of 50K–100K used)  
- **Labels:** Positive (4), Neutral (2), Negative (0) → remapped to {1, 0, -1}  
- **Purpose:** Model training and internal evaluation  
- **Split:** 80% training, 20% testing (with balanced sampling)  

### **2 Secondary Dataset: Twitter US Airline Sentiment**
- **Source:** [Kaggle – US Airline Sentiment Dataset](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)  
- **Description:** 14K labeled tweets about airline services  
- **Labels:** Positive, Neutral, Negative  
- **Purpose:** Cross-domain validation and generalization testing  
