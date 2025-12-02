# Sentiment Analysis on Social Media with Semantic Interpretation

A comparative study evaluating traditional ML approaches versus semantic-enhanced features for Twitter sentiment classification.

**Authors:** Naman & Shrish  
**Course:** CL-2 (Computational Linguistics)  

---

## Overview

This project implements a complete sentiment analysis pipeline that:
- Processes 1.6M tweets with emoji/emoticon preservation
- Extracts traditional (N-grams, POS) and semantic features (Word2Vec, GloVe, VADER, negation, intensifiers)
- Trains and evaluates multiple ML models (Naive Bayes, Logistic Regression, Random Forest)
- Performs comparative analysis, error/success analysis, and cross-domain validation

**Key Results:**
- Best Model: Naive Bayes (F1: 0.71, Accuracy: 0.71)
- Cross-Domain Validation: 62.3% accuracy on airline sentiment
- Memory-optimized for 1.6M tweets using sparse matrices

---

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run Preprocessing
```bash
# Process sample dataset
python scripts/main.py --dataset sentiment140 --sample-size 10000

# Process full dataset
python scripts/process_full_data.py
```

### Train Models (Phase 2)
```bash
# Train with default settings (50K samples, 5-fold CV)
python scripts/train_phase2.py

# Train on full dataset
python scripts/train_phase2.py --sample-size 0

# Custom configuration
python scripts/train_phase2.py --sample-size 100000 --max-features 10000
```

### Comparative Analysis (Phase 3)
```bash
# Compare traditional vs semantic features
python scripts/comparative_analysis.py --sample-size 10000

# Cross-domain validation
python scripts/run_phase3.py
```

### Testing
```bash
# Run Phase 3 test suite
python test_phase3.py
```

---

## Project Structure

```
CL-2-Project/
├── preprocessing/          # Data loading and text preprocessing
├── features/               # Feature extraction (traditional + semantic)
├── models/                 # ML models, success/error analysis
├── scripts/                # Execution scripts
├── datasets/               # Raw data (Sentiment140, Airline)
├── trained_models/         # Saved models and results
├── analysis_results/       # Phase 3 validation results
└── comparative_results/    # Comparative analysis outputs
```

---

## Features Implemented

### Phase 1: Data Preparation
- ✅ Text preprocessing (URL/mention removal, lemmatization)
- ✅ Emoji/emoticon preservation
- ✅ Stratified 80-20 split with reproducibility
- ✅ Contextual features (negation, intensifiers)
- ✅ Semantic embeddings (Word2Vec, GloVe)
- ✅ Lexicon scoring (VADER, NRC emotions)

### Phase 2: Model Training & Evaluation
- ✅ Traditional features (N-grams, POS tags)
- ✅ Naive Bayes & Logistic Regression models
- ✅ Random Forest with feature ablation
- ✅ Random Forest regularization for better generalization
- ✅ 5-fold cross-validation
- ✅ Best model identification (F1-score based)
- ✅ Success pattern analysis
- ✅ Memory optimization (sparse matrices)

### Phase 3: Analysis & Validation
- ✅ Comparative analysis (traditional vs semantic)
- ✅ Error analysis on misclassifications
- ✅ Cross-domain validation (Sentiment140 → Airline)
- ✅ Qualitative analysis with semantic interpretation
- ✅ Comprehensive reporting

---

## Usage Examples

### Preprocessing
```python
from preprocessing import TweetPreprocessor, SentimentDataLoader

# Preprocess text
preprocessor = TweetPreprocessor()
clean_text = preprocessor.preprocess("@user This is great! 😊 #happy")
# Output: "great 😊 happy"

# Load dataset
loader = SentimentDataLoader('datasets')
df = loader.load_sentiment140(sample_size=10000)
train_df, test_df = loader.create_train_test_split(df, test_size=0.2)
```

### Feature Extraction
```python
from features import TraditionalFeatureExtractor, FeatureExtractionPipeline

# Traditional features (N-grams + POS)
extractor = TraditionalFeatureExtractor(ngram_range=(1, 2), max_features=5000)
X_train = extractor.fit_transform(train_texts)
X_test = extractor.transform(test_texts)

# All features (traditional + semantic)
pipeline = FeatureExtractionPipeline()
pipeline.initialize_extractors()
X_all = pipeline.extract_all_features(texts)
```

### Model Training
```python
from models import SentimentClassifier, SuccessAnalyzer, ErrorAnalyzer

# Train model
model = SentimentClassifier(model_type='naive_bayes')
model.fit(X_train, y_train)

# Evaluate
metrics = model.evaluate(X_test, y_test)
print(f"F1-Score: {metrics['f1_score']:.4f}")

# Analyze predictions
success_analyzer = SuccessAnalyzer()
success_results = success_analyzer.analyze_correct_predictions(
    model=model, X=X_test, y=y_test, texts=test_texts
)

error_analyzer = ErrorAnalyzer()
error_results = error_analyzer.analyze_errors(
    model=model, X=X_test, y_true=y_test, texts=test_texts
)
```

---

## Results Summary

### Model Performance (10K sample)
| Model | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|---------|
| Naive Bayes | 0.712 | 0.712 | 0.787 |
| Logistic Regression | 0.709 | 0.709 | 0.791 |

### Success Analysis
- **Correctly Classified:** 71.2%
- **High-Confidence Predictions (≥90%):** 3.6%
- **Average Text Length (Correct):** 7.1 words

### Error Analysis
- **Error Rate:** 28.8%
- **Common Patterns:** Mixed sentiment expressions, sarcasm, context-dependent phrases
- **Model Agreement:** 67% (both models correct on same samples)

### Cross-Domain Validation
- **Best Generalization:** Naive Bayes (Target F1: 0.71)
- **Best Source Performance:** Random Forest Original (Source F1: 0.75)
- **Fixed Overfitting:** Random Forest Regularized reduces domain gap from +0.11 to ~+0.03

**Key Insight:** Original Random Forest overfits to source domain. Regularized version (max_depth=10, min_samples_split=20) achieves better cross-domain generalization while maintaining competitive performance.

---

## Datasets

### Sentiment140 (Primary)
- **Size:** 1.6M tweets
- **Labels:** Binary (0=negative, 1=positive)
- **Purpose:** Training and evaluation

### Twitter US Airline Sentiment (Validation)
- **Size:** 14.6K tweets
- **Labels:** Negative, Neutral, Positive
- **Purpose:** Cross-domain validation

---

## Dependencies

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
nltk>=3.6.0
gensim>=4.0.0
vaderSentiment>=3.3.2
```

---

## Output Files

**Model Outputs:**
- `trained_models/phase2/` - Trained models (.pkl), results (CSV/JSON)

**Analysis Results:**
- `comparative_results/` - Traditional vs semantic comparison
- `analysis_results/` - Cross-domain validation, qualitative analysis

**Documentation:**
- `PROJECT_COMPLETION_SUMMARY.md` - Full project completion report
- `PHASE3_REPORT.md` - Phase 3 analysis report