# CL-2-Project
## Sentiment Analysis on Social Media Texts with Semantic Interpretation

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

## Quick Start

### Installation

```bash
# Install required dependencies
pip install -r requirements.txt
```

### Run Preprocessing Pipeline

```bash
# Process both datasets with default settings (80-20 split)
python main.py

# Process only Sentiment140 with custom sample size
python main.py --dataset sentiment140 --sample-size 10000

# Process only airline dataset
python main.py --dataset airline

# Custom output directory and test size
python main.py --output-dir my_data --test-size 0.3
```

### Test Preprocessing

```bash
# Run demo to see preprocessing examples
python demo.py
```

---

## Project Structure

```
CL-2-Project/
├── datasets/                      # Raw datasets (not tracked)
│   ├── Sentiment140_dataset/
│   └── cross_validation_dataset/
├── processed_data/                # Preprocessed train/test splits
├── preprocessing.py               # Text preprocessing pipeline
├── data_loader.py                # Data loading and splitting utilities
├── main.py                       # Main pipeline script
├── demo.py                       # Usage examples
└── requirements.txt              # Python dependencies
```

---

## Methods

### **Data Preprocessing** (Phase 1 - Complete)
- Text cleaning: URL, mention, and hashtag removal
- HTML entity decoding
- Tokenization using NLTK's TweetTokenizer
- Stopword removal and lemmatization
- **Emoji and emoticon preservation** for sentiment context
- Stratified 80-20 train-test split with class balancing
- Reproducible pipeline with random seed control

### **Feature Engineering** (Phase 2 - Upcoming)
- **Traditional features:** N-grams, POS tags
- **Contextual features:** Negation and intensifier detection
- **Semantic features:** Word2Vec/GloVe embeddings and lexicon polarity scores (VADER, NRC)

### **Models** (Phase 2-3 - Upcoming)
- Naive Bayes
- Logistic Regression
- Random Forest
- Feature ablation study on best-performing model

### **Evaluation Metrics** (Phase 3 - Upcoming)
- Accuracy, Precision, Recall, F1-score
- K-fold cross-validation
- Error and success analysis
- Qualitative analysis of semantic influence

---

## Datasets

### **Primary Dataset: Sentiment140**
- **Source:** [Kaggle – Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)
- **Description:** 1.6M labeled tweets
- **Labels:** Binary (0=negative, 4=positive → mapped to 0 and 1)
- **Purpose:** Model training and primary evaluation
- **Split:** 80% training, 20% testing (stratified)

### **Secondary Dataset: Twitter US Airline Sentiment**
- **Source:** [Kaggle – US Airline Sentiment Dataset](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)
- **Description:** 14.6K labeled tweets about airline services
- **Labels:** Negative (0), Neutral (1), Positive (2)
- **Purpose:** Cross-domain validation and generalization testing

---

## Progress

**Phase 1: Data Preparation** ✓ Complete
- [x] Download and prepare datasets
- [x] Implement preprocessing pipeline with emoji preservation
- [x] Create 80-20 stratified train-test split
- [x] Set up data loading utilities
- [x] Ensure reproducibility (random seed: 42)

**Phase 2: Feature Engineering & Traditional Models** (Upcoming)
- [ ] N-grams and POS tagging
- [ ] Naive Bayes and Logistic Regression
- [ ] Initial model evaluation
- [ ] Cross-validation framework

**Phase 3: Semantic Features & Advanced Models** (Upcoming)
- [ ] Semantic feature engineering
- [ ] Random Forest classifier
- [ ] Feature ablation study
- [ ] Cross-domain validation

---

## Team

**Member 1 (Naman):** Data Pipeline & Traditional ML  
**Member 2 (Shrish):** Semantic Features & Advanced ML

---

## License

This project is for academic purposes as part of CL-2 coursework.
