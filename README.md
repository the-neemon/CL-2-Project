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

### Train Phase 2 Models

```bash
# Train Naive Bayes and Logistic Regression with default settings
# Uses 50K samples, 5000 features, 5-fold CV
python scripts/train_phase2.py

# Train with custom settings
python scripts/train_phase2.py --dataset sentiment140 --sample-size 100000 --max-features 10000 --cv-folds 10

# Train on full dataset (1.6M tweets)
python scripts/train_phase2.py --sample-size 0

# Train on airline dataset
python scripts/train_phase2.py --dataset airline

# Run evaluation and cross-validation example
python example_evaluation.py

# Demo: Best model identification & success analysis
python demo_best_model_and_success.py
```

### Comparative Analysis (Phase 3 - Naman)

```bash
# Compare traditional features vs. semantic-enhanced features
python scripts/comparative_analysis.py --sample-size 10000

# Run comparative analysis on full dataset
python scripts/comparative_analysis.py --sample-size 0

# Compare on airline dataset
python scripts/comparative_analysis.py --dataset airline --sample-size 0
```

### Run Phase 3 (Cross-Domain Validation & Qualitative Analysis)

```bash
# Run complete Phase 3 pipeline (recommended)
python scripts/run_phase3.py

# Run with custom settings
python scripts/run_phase3.py --source-sample-size 100000 --max-features 10000 --n-qualitative-samples 20

# Run only cross-domain validation
python scripts/cross_domain_validation.py

# Run only qualitative analysis
python scripts/qualitative_analysis.py --n-samples 15
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
├── features/                      # Feature extraction (Phase 1 & 2)
│   ├── __init__.py
│   ├── contextual_features.py    # Negation, intensifiers, emphasis (Shrish)
│   ├── semantic_embeddings.py    # Word2Vec, GloVe embeddings (Shrish)
│   ├── lexicon_scoring.py        # VADER, NRC emotion lexicons (Shrish)
│   ├── feature_pipeline.py       # Unified feature extraction (Shrish)
│   └── traditional_features.py   # N-grams, POS tagging (Naman - Phase 2)
│
├── models/                        # ML models (Naman - Phase 2 & 3)
│   ├── __init__.py
│   ├── traditional_models.py     # Naive Bayes, Logistic Regression
│   ├── success_analysis.py       # Success pattern analysis (Phase 2)
│   └── error_analysis.py         # Error pattern analysis (Phase 3)
│
├── scripts/                       # Execution scripts
│   ├── main.py                   # Main preprocessing pipeline
│   ├── process_full_data.py      # Full dataset processor
│   ├── train_phase2.py           # Train Phase 2 models (Naman)
│   ├── comparative_analysis.py   # Comparative analysis (Naman - Phase 3)
│   ├── train_with_ablation.py    # Feature ablation study (Shrish)
│   ├── cross_domain_validation.py # Phase 3: Cross-domain validation (Shrish)
│   ├── qualitative_analysis.py   # Phase 3: Qualitative analysis (Shrish)
│   └── run_phase3.py             # Phase 3: Complete pipeline (Shrish)
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

## Phase 2 Implementation (Complete ✓)

### Traditional Feature Engineering (Naman)
- ✅ **N-gram Features:**
  - Unigram (single word) features
  - Bigram (two-word phrase) features
  - TF-IDF vectorization
  - Configurable vocabulary size (default: 5000 features)
  - Document frequency filtering (min_df=2, max_df=0.95)
  - **Memory-optimized sparse matrix support** for large datasets (1.6M tweets)

- ✅ **POS Tagging:**
  - Sentiment-bearing POS categories:
    - Adjectives (JJ, JJR, JJS) - descriptors
    - Adverbs (RB, RBR, RBS) - intensifiers/modifiers
    - Verbs (VB, VBD, VBG, VBN, VBP, VBZ) - actions/states
    - Nouns (NN, NNS, NNP, NNPS) - entities/topics
  - Normalized POS tag counts per text

### Traditional ML Models (Naman)
- ✅ **Naive Bayes:**
  - Multinomial Naive Bayes for text classification
  - Laplace smoothing (alpha=1.0)
  - Probabilistic predictions with class priors
  - Sparse matrix support for memory efficiency
  
- ✅ **Logistic Regression:**
  - L2 regularization (C=1.0)
  - LBFGS solver for optimization
  - Multi-core parallel training
  - Sparse matrix support for large-scale training
  
- ✅ **Model Evaluation:**
  - K-fold cross-validation (default: 5 folds)
  - Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
  - Confusion matrices and classification reports
  - Model comparison framework
  - **Best model identification** based on maximum F1-score
  
- ✅ **Success Analysis:**
  - Analyze correctly classified instances
  - Per-class success rates
  - Confidence distribution analysis
  - High-confidence prediction identification
  - Text characteristic patterns
  - Model agreement/disagreement analysis
  
- ✅ **Model Persistence:**
  - Save/load trained models (pickle)
  - Feature extractor serialization
  - Results export (CSV, JSON)

---

## Phase 3 Implementation (Complete ✓)

### Integration & Analysis (Naman)
- ✅ **Semantic Feature Integration:**
  - Integrated contextual features (negation, intensifiers)
  - Integrated semantic embeddings (Word2Vec, GloVe)
  - Integrated lexicon-based scoring (VADER, NRC)
  - Unified feature pipeline support

- ✅ **Comparative Analysis:**
  - Performance comparison: Traditional vs. Semantic-enhanced features
  - Feature ablation study integration
  - Impact analysis of semantic features
  - Automated comparison tables and metrics
  - Side-by-side performance evaluation
  
- ✅ **Error Analysis:**
  - Analyze misclassified instances
  - Per-class error rates and patterns
  - Confidence analysis for errors
  - High-confidence error identification (≥90% threshold)
  - Text characteristic analysis for errors
  - Confusion pattern detection
  - Model error comparison
  - Sample error inspection with detailed reporting
  
- ✅ **Documentation:**
  - Comprehensive preprocessing pipeline documentation
  - Traditional feature extraction documentation
  - Model training and evaluation documentation
  - Usage examples and API reference
  - Integration guides

### Cross-Domain Validation (Shrish)
- ✅ **Train on Sentiment140, Test on Airline Dataset:**
  - Source domain: Sentiment140 (general Twitter sentiment)
  - Target domain: Twitter US Airline Sentiment (domain-specific)
  - Binary classification: Negative (0) vs Positive (1)
  - Multiple model comparison (Logistic Regression, Naive Bayes, Random Forest)
  - Domain gap analysis (in-domain vs cross-domain performance)
  
- ✅ **Performance Metrics:**
  - Accuracy, Precision, Recall, F1-Score
  - ROC-AUC for probability calibration
  - Domain transfer effectiveness evaluation

### Qualitative Analysis (Shrish)
- ✅ **Representative Tweet Inspection:**
  - Sample analysis across prediction categories:
    - True Negatives (Correct)
    - True Positives (Correct)
    - False Positives (Errors)
    - False Negatives (Errors)
  - Confidence-stratified sampling (high/medium/low confidence)
  
- ✅ **Semantic Feature Interpretation:**
  - **Negation Analysis:** Detection and context extraction
  - **Intensifier Patterns:** "very", "really", "extremely" usage
  - **VADER Lexicon Scores:** Positive/negative/neutral signals
  - **Emphasis Detection:** ALL CAPS, exclamation marks, questions
  - **Text Complexity:** Length, mentions, hashtags, URLs
  
- ✅ **Pattern Discovery:**
  - Compare semantic patterns in correct vs incorrect predictions
  - Identify features that contribute to success/failure
  - Generate actionable insights for model improvement
  - Export comprehensive analysis reports (JSON + Markdown)

### Outputs Generated
- `comparison_table.csv` - Feature comparison results (Naman)
- `error_analysis.json` - Detailed error analysis (Naman)
- `success_analysis.json` - Success pattern analysis (Naman)
- `cross_domain_validation.json` - Full cross-domain results (Shrish)
- `qualitative_analysis.json` - Detailed sample analyses (Shrish)
- `phase3_comprehensive_report.json` - Combined insights (Shrish)
- `PHASE3_REPORT.md` - Human-readable summary report (Shrish)

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
    FeatureExtractionPipeline,
    TraditionalFeatureExtractor
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

# Extract traditional N-gram and POS features
extractor = TraditionalFeatureExtractor(
    ngram_range=(1, 2),
    max_features=5000
)
X_train = extractor.fit_transform(train_texts)
X_test = extractor.transform(test_texts)
```

### Python API - Model Training

```python
from models import SentimentClassifier, compare_models, SuccessAnalyzer, ErrorAnalyzer

# Train Naive Bayes
nb_model = SentimentClassifier(model_type='naive_bayes', alpha=1.0)
nb_model.fit(X_train, y_train)
metrics = nb_model.evaluate(X_test, y_test)

# Train Logistic Regression
lr_model = SentimentClassifier(model_type='logistic_regression', C=1.0)
lr_model.fit(X_train, y_train)
predictions = lr_model.predict(X_test)

# Compare multiple models with cross-validation
models = {
    'Naive Bayes': nb_model,
    'Logistic Regression': lr_model
}
results_df = compare_models(models, X_train, y_train, X_test, y_test, cv=5)

# Success Analysis (Phase 2)
success_analyzer = SuccessAnalyzer()
success_results = success_analyzer.analyze_correct_predictions(
    model=lr_model,
    X=X_test,
    y=y_test,
    texts=test_texts
)

# Error Analysis (Phase 3 - Naman)
error_analyzer = ErrorAnalyzer()
error_results = error_analyzer.analyze_errors(
    model=lr_model,
    X=X_test,
    y_true=y_test,
    texts=test_texts,
    model_name="Logistic Regression"
)

# Compare error patterns between models
error_comparison = error_analyzer.compare_model_errors(
    model1=nb_model,
    model2=lr_model,
    X=X_test,
    y_true=y_test,
    texts=test_texts,
    model1_name="Naive Bayes",
    model2_name="Logistic Regression"
)

# Export analyses
success_analyzer.export_analysis('success_analysis.json')
error_analyzer.export_analysis('error_analysis.json')
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

## Project Status

**Phase 1 - Data Preparation:** ✅ Complete  
**Phase 2 - Traditional ML:** ✅ Complete  
**Phase 3 - Integration & Analysis:** ✅ Complete  

**Key Achievements:**
- ✅ Processed 1.6M tweets from Sentiment140 dataset
- ✅ Memory-optimized sparse matrix implementation (46.9 GiB → ~50 MB)
- ✅ Trained models on full dataset
- ✅ Comprehensive success and error analysis
- ✅ Semantic feature integration ready
- ✅ Comparative analysis framework complete
- ✅ Cross-domain validation implemented
- ✅ Qualitative analysis complete

**Final Deliverables:**
- ✅ Comparative performance analysis (Naman)
- ✅ Error and success analysis framework (Naman)
- ✅ Feature ablation study (Shrish)
- ✅ Cross-domain validation (Shrish)
- ✅ Qualitative analysis (Shrish)
- ✅ Complete documentation
- [ ] Project presentation/report (In Progress)

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