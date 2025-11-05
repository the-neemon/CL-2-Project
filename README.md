# CL-2-Project
## Sentiment Analysis on Social Media Texts with Semantic Interpretation

A comparative study combining **lexicon-based semantic features** with **machine learning approaches** for sentiment classification on Twitter data.

This project explores how integrating **semantic interpretation techniques**—such as negation handling, intensifiers, and pre-trained embeddings—can improve sentiment analysis performance compared to traditional lexical methods.

**Team Members:**
- **Naman:** Data Pipeline & Traditional ML (Phase 1 Complete)
- **Shrish:** Semantic Features & Advanced ML (Phase 1 Complete)

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

### Test Integration

```bash
# Test that all modules work together
python test_integration.py
```

### Run Preprocessing Pipeline

```bash
# Process both datasets with default settings (80-20 split)
python scripts/main.py

# Process only Sentiment140 with custom sample size
python scripts/main.py --dataset sentiment140 --sample-size 10000

# Process full datasets
python scripts/process_full_data.py
```

---

## Project Structure

```
CL-2-Project/
├── preprocessing/                 # Data preprocessing (Naman - Phase 1)
│   ├── __init__.py
│   ├── preprocessing.py          # TweetPreprocessor class
│   └── data_loader.py            # Data loading and splitting
│
├── features/                      # Feature extraction (Shrish - Phase 1)
│   ├── __init__.py
│   ├── contextual_features.py    # Negation, intensifiers, emphasis
│   ├── semantic_embeddings.py    # Word2Vec, GloVe embeddings
│   ├── lexicon_scoring.py        # VADER, NRC emotion lexicons
│   └── feature_pipeline.py       # Unified feature extraction
│
├── scripts/                       # Execution scripts
│   ├── main.py                   # Main preprocessing pipeline
│   └── process_full_data.py      # Full dataset processor
│
├── datasets/                      # Raw datasets (not tracked)
│   ├── Sentiment140_dataset/
│   └── cross_validation_dataset/
│
├── test_integration.py           # Integration tests
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## Phase 1 Implementation (Complete ✓)

### Preprocessing Module (Naman)
- ✅ **Text Cleaning:** URL, mention, hashtag removal, HTML entity decoding
- ✅ **Tokenization:** NLTK's TweetTokenizer for Twitter-specific text
- ✅ **Normalization:** Stopword removal and lemmatization
- ✅ **Emoji Preservation:** Retains emojis and emoticons for sentiment context
- ✅ **Train-Test Split:** Stratified 80-20 split with class balancing
- ✅ **Reproducibility:** Fixed random seed (42) for consistent results

### Feature Extraction Module (Shrish)
- ✅ **Contextual Features:** 
  - Negation detection and scope analysis
  - Intensifier and diminisher identification
  - Emphasis pattern detection (caps, exclamations, etc.)
  - Twitter-specific features (mentions, hashtags, retweets)
  
- ✅ **Semantic Embeddings:**
  - Word2Vec (Google News 300d)
  - GloVe (Wiki Gigaword 300d)
  - Sentiment similarity scoring
  - Dimensionality reduction (PCA)
  
- ✅ **Lexicon-Based Scoring:**
  - VADER sentiment analyzer
  - NRC Emotion Lexicon (10 emotions)
  - Custom polarity scoring
  - Sentiment modifiers

- ✅ **Feature Pipeline:**
  - Unified extraction interface
  - Configurable feature types
  - Feature scaling and normalization
  - Save/load capabilities

---

## Usage Examples

### Python API - Preprocessing

```python
from preprocessing import TweetPreprocessor, SentimentDataLoader

# Preprocess individual tweets
preprocessor = TweetPreprocessor()
processed = preprocessor.preprocess("@user This is amazing! 😍 #happy")
# Output: "amazing 😍 happy"

# Load and preprocess datasets
loader = SentimentDataLoader('datasets')
df = loader.load_sentiment140(sample_size=10000)
train_df, test_df = loader.create_train_test_split(df, test_size=0.2)
```

### Python API - Feature Extraction

```python
from features import (
    ContextualFeatures,
    LexiconBasedScoring,
    SemanticEmbeddings,
    FeatureExtractionPipeline
)

# Extract contextual features
contextual = ContextualFeatures()
features = contextual.extract_contextual_features("Not bad at all!")

# Extract lexicon features
lexicon = LexiconBasedScoring()
lexicon.initialize_lexicons()
scores = lexicon.extract_lexicon_features("This is amazing!")

# Unified feature pipeline
pipeline = FeatureExtractionPipeline()
pipeline.initialize_extractors()
all_features = pipeline.extract_all_features(["Sample tweet"])
```

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

## Key Features

### Preprocessing
✓ **Emoji preservation**: 😍 😡 🎉 are retained  
✓ **Emoticon preservation**: :) :( <3 are retained  
✓ **URL removal**: http links stripped  
✓ **Mention removal**: @username removed  
✓ **Hashtag processing**: #happy → happy  
✓ **HTML decoding**: &amp; → &  
✓ **Stopword removal**: common words removed  
✓ **Lemmatization**: running → run  
✓ **Stratified splits**: class balance maintained  
✓ **Reproducible**: fixed random seed

### Feature Extraction
✓ **12 contextual features**: negation, intensifiers, emphasis  
✓ **22 lexicon features**: VADER, NRC emotions, custom scoring  
✓ **34 semantic features**: Word2Vec, GloVe, sentiment similarity  
✓ **Traditional features**: TF-IDF, BoW, POS tags (configurable)  
✓ **Unified pipeline**: easy integration and scaling  

---

## Testing

```bash
# Run integration tests
python test_integration.py

# Expected output:
# ✓ Preprocessing: PASS
# ✓ Features: PASS  
# ✓ Integration: PASS
# ✓ ALL TESTS PASSED!
```

---

## Next Steps (Phase 2)

**Naman:**
- [ ] Implement N-gram feature extraction
- [ ] Train Naive Bayes classifier
- [ ] Train Logistic Regression
- [ ] Set up cross-validation framework
- [ ] Initial model evaluation

**Shrish:**
- [ ] Train Random Forest classifier
- [ ] Feature ablation study
- [ ] Cross-domain validation
- [ ] Comparative analysis

---

## Dependencies

```
numpy>=1.21.0          # Numerical operations
pandas>=1.3.0          # Data manipulation
scikit-learn>=1.0.0    # Machine learning
nltk>=3.6.0            # NLP toolkit
gensim>=4.0.0          # Word embeddings
vaderSentiment>=3.3.2  # Sentiment analysis
```

---

## License

This project is for academic purposes as part of CL-2 coursework.

---

## Acknowledgments

- Sentiment140 dataset creators
- VADER sentiment analysis tool
- NRC Emotion Lexicon
- Word2Vec and GloVe embedding projects
