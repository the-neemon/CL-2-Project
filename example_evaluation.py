"""
Example: Model Evaluation and Cross-Validation

Demonstrates the two key Phase 2 tasks:
1. Initial model evaluation with accuracy, precision, recall, and F1-score
2. Cross-validation framework

Author: Naman
Phase: 2
"""

from preprocessing.data_loader import SentimentDataLoader
from preprocessing.preprocessing import TweetPreprocessor
from features.traditional_features import TraditionalFeatureExtractor
from models.traditional_models import SentimentClassifier, cross_validate_model


def main():
    """Demonstrate evaluation and cross-validation."""
    
    print("="*70)
    print("PHASE 2: MODEL EVALUATION & CROSS-VALIDATION EXAMPLE")
    print("="*70)
    
    # Load and preprocess data
    print("\n[Step 1] Loading and preprocessing data...")
    loader = SentimentDataLoader(dataset_dir='datasets')
    preprocessor = TweetPreprocessor()
    
    df = loader.load_sentiment140(sample_size=5000)
    train_df, test_df = loader.create_train_test_split(df, test_size=0.2)
    train_df = loader.preprocess_dataframe(train_df, preprocessor)
    test_df = loader.preprocess_dataframe(test_df, preprocessor)
    
    print(f"✓ Train: {len(train_df)} samples")
    print(f"✓ Test:  {len(test_df)} samples")
    
    # Extract features
    print("\n[Step 2] Extracting traditional features...")
    extractor = TraditionalFeatureExtractor(
        max_features=2000,
        ngram_range=(1, 2)
    )
    X_train = extractor.fit_transform(train_df['processed_text'].values)
    X_test = extractor.transform(test_df['processed_text'].values)
    y_train = train_df['sentiment'].values
    y_test = test_df['sentiment'].values
    
    print(f"✓ Feature matrix shape: {X_train.shape}")
    
    # Task 1: Model Evaluation
    print("\n" + "="*70)
    print("TASK 1: INITIAL MODEL EVALUATION")
    print("="*70)
    
    # Train model
    model = SentimentClassifier(model_type='naive_bayes')
    model.fit(X_train, y_train)
    
    # Evaluate with all metrics
    print("\nTest Set Evaluation:")
    metrics = model.evaluate(X_test, y_test, verbose=True)
    
    # Show confusion matrix
    print("\nConfusion Matrix:")
    cm = model.get_confusion_matrix(X_test, y_test)
    print(cm)
    
    # Task 2: Cross-Validation
    print("\n" + "="*70)
    print("TASK 2: CROSS-VALIDATION FRAMEWORK")
    print("="*70)
    
    cv_model = SentimentClassifier(model_type='logistic_regression')
    cv_results = cross_validate_model(
        cv_model,
        X_train,
        y_train,
        cv=5,  # 5-fold cross-validation
        verbose=True
    )
    
    print("\nCross-Validation Summary:")
    print(f"  Mean Accuracy:  {cv_results['accuracy_mean']:.4f} "
          f"± {cv_results['accuracy_std']:.4f}")
    print(f"  Mean Precision: {cv_results['precision_mean']:.4f} "
          f"± {cv_results['precision_std']:.4f}")
    print(f"  Mean Recall:    {cv_results['recall_mean']:.4f} "
          f"± {cv_results['recall_std']:.4f}")
    print(f"  Mean F1-Score:  {cv_results['f1_mean']:.4f} "
          f"± {cv_results['f1_std']:.4f}")
    
    print("\n" + "="*70)
    print("✅ Both tasks completed successfully!")
    print("="*70)


if __name__ == '__main__':
    main()
