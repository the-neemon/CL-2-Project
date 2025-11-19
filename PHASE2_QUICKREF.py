"""
Quick Reference: Phase 2 Evaluation & Cross-Validation

This file serves as a quick reference for the two implemented Phase 2 tasks.
"""

# ============================================================================
# TASK 1: Model Evaluation with Accuracy, Precision, Recall, and F1-Score
# ============================================================================

"""
LOCATION: models/traditional_models.py
METHOD: SentimentClassifier.evaluate()

EXAMPLE:
"""
from models.traditional_models import SentimentClassifier

# Train model
model = SentimentClassifier(model_type='naive_bayes')
model.fit(X_train, y_train)

# Evaluate with all 4 metrics
metrics = model.evaluate(X_test, y_test, verbose=True)

# Access individual metrics
accuracy = metrics['accuracy']      # Overall correctness
precision = metrics['precision']    # Weighted precision
recall = metrics['recall']          # Weighted recall
f1 = metrics['f1_score']           # Weighted F1-score

# Additional methods
cm = model.get_confusion_matrix(X_test, y_test)  # Confusion matrix
report = model.get_classification_report(X_test, y_test)  # Detailed report


# ============================================================================
# TASK 2: Cross-Validation Framework
# ============================================================================

"""
LOCATION: models/traditional_models.py
FUNCTION: cross_validate_model()

EXAMPLE:
"""
from models.traditional_models import cross_validate_model

# Perform 5-fold stratified cross-validation
cv_results = cross_validate_model(
    model,
    X_train,
    y_train,
    cv=5,           # Number of folds
    verbose=True    # Print results
)

# Access results (mean ± std)
print(f"Accuracy:  {cv_results['accuracy_mean']:.4f} ± {cv_results['accuracy_std']:.4f}")
print(f"Precision: {cv_results['precision_mean']:.4f} ± {cv_results['precision_std']:.4f}")
print(f"Recall:    {cv_results['recall_mean']:.4f} ± {cv_results['recall_std']:.4f}")
print(f"F1-Score:  {cv_results['f1_mean']:.4f} ± {cv_results['f1_std']:.4f}")


# ============================================================================
# COMPLETE WORKFLOW EXAMPLE
# ============================================================================

from preprocessing.data_loader import SentimentDataLoader
from preprocessing.preprocessing import TweetPreprocessor
from features.traditional_features import TraditionalFeatureExtractor
from models.traditional_models import SentimentClassifier, cross_validate_model

# 1. Load and preprocess data
loader = SentimentDataLoader(dataset_dir='datasets')
preprocessor = TweetPreprocessor()
df = loader.load_sentiment140(sample_size=10000)
train_df, test_df = loader.create_train_test_split(df, test_size=0.2)
train_df = loader.preprocess_dataframe(train_df, preprocessor)
test_df = loader.preprocess_dataframe(test_df, preprocessor)

# 2. Extract features
extractor = TraditionalFeatureExtractor(max_features=2000, ngram_range=(1, 2))
X_train = extractor.fit_transform(train_df['processed_text'].values)
X_test = extractor.transform(test_df['processed_text'].values)
y_train = train_df['sentiment'].values
y_test = test_df['sentiment'].values

# 3. Train model
model = SentimentClassifier(model_type='logistic_regression')
model.fit(X_train, y_train)

# 4. TASK 1: Evaluate on test set
print("Test Set Evaluation:")
test_metrics = model.evaluate(X_test, y_test, verbose=True)

# 5. TASK 2: Cross-validation
print("\nCross-Validation:")
cv_results = cross_validate_model(model, X_train, y_train, cv=5, verbose=True)


# ============================================================================
# COMMAND LINE USAGE
# ============================================================================

"""
# Run the example script
$ python example_evaluation.py

# Use the training script (includes both tasks automatically)
$ python scripts/train_phase2.py --sample-size 10000 --cv-folds 5

# Train on full dataset
$ python scripts/train_phase2.py --sample-size 0
"""
