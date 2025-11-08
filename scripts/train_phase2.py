"""
Training script for Phase 2: Traditional Feature Engineering & Models.

Trains and evaluates Naive Bayes and Logistic Regression models
using traditional N-gram and POS features.

Author: Naman
Phase: 2
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from preprocessing.data_loader import SentimentDataLoader
from features.traditional_features import TraditionalFeatureExtractor
from models.traditional_models import SentimentClassifier, compare_models


def load_preprocessed_data(dataset: str = 'sentiment140',
                          sample_size: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load preprocessed training and test data.
    
    Args:
        dataset: Dataset name ('sentiment140' or 'airline')
        sample_size: Optional sample size for testing
        
    Returns:
        Tuple of (train_df, test_df)
    """
    print(f"\nLoading {dataset} dataset...")
    
    from preprocessing.preprocessing import TweetPreprocessor
    
    loader = SentimentDataLoader(dataset_dir='datasets')
    preprocessor = TweetPreprocessor(preserve_case=False, reduce_len=True)
    
    if dataset == 'sentiment140':
        # Load raw data
        df = loader.load_sentiment140(sample_size=sample_size)
        
        # Split
        train_df, test_df = loader.create_train_test_split(
            df, test_size=0.2, stratify=True
        )
        
        # Preprocess
        train_df = loader.preprocess_dataframe(train_df, preprocessor)
        test_df = loader.preprocess_dataframe(test_df, preprocessor)
        
    elif dataset == 'airline':
        # Load raw data
        df = loader.load_airline_sentiment()
        
        if sample_size:
            df = df.sample(n=min(sample_size, len(df)), random_state=42)
        
        # Split
        train_df, test_df = loader.create_train_test_split(
            df, test_size=0.2, stratify=True
        )
        
        # Preprocess
        train_df = loader.preprocess_dataframe(train_df, preprocessor)
        test_df = loader.preprocess_dataframe(test_df, preprocessor)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    print(f"✓ Loaded {len(train_df)} training samples")
    print(f"✓ Loaded {len(test_df)} test samples")
    
    return train_df, test_df


def train_phase2_models(dataset: str = 'sentiment140',
                       sample_size: Optional[int] = 50000,
                       max_features: int = 5000,
                       cv_folds: int = 5,
                       save_models: bool = True) -> Dict[str, Any]:
    """
    Train Phase 2 traditional models (Naive Bayes & Logistic Regression).
    
    Args:
        dataset: Dataset to use
        sample_size: Sample size (None for full dataset)
        max_features: Maximum number of N-gram features
        cv_folds: Number of cross-validation folds
        save_models: Whether to save trained models
        
    Returns:
        Dictionary with results and trained models
    """
    print("="*70)
    print("PHASE 2: TRADITIONAL FEATURE ENGINEERING & MODELS")
    print("="*70)
    
    # Load data
    train_df, test_df = load_preprocessed_data(dataset, sample_size)
    
    # Extract texts and labels
    X_train_text = train_df['processed_text'].tolist()
    y_train = train_df['sentiment'].values
    
    X_test_text = test_df['processed_text'].tolist()
    y_test = test_df['sentiment'].values
    
    print(f"\nClass distribution (train): {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"Class distribution (test):  {dict(zip(*np.unique(y_test, return_counts=True)))}")
    
    # Extract traditional features
    print("\n" + "="*70)
    print("FEATURE EXTRACTION")
    print("="*70)
    
    feature_extractor = TraditionalFeatureExtractor(
        ngram_range=(1, 2),  # Unigrams and bigrams
        max_features=max_features,
        min_df=2,
        max_df=0.95,
        use_idf=True
    )
    
    # Fit and transform features
    X_train = feature_extractor.fit_transform(X_train_text)
    X_test = feature_extractor.transform(X_test_text)
    
    print(f"\n✓ Feature extraction complete")
    print(f"  Training shape: {X_train.shape}")
    print(f"  Test shape: {X_test.shape}")
    
    # Initialize models
    print("\n" + "="*70)
    print("MODEL TRAINING & EVALUATION")
    print("="*70)
    
    models = {
        'Naive Bayes': SentimentClassifier(
            model_type='naive_bayes',
            alpha=1.0
        ),
        'Logistic Regression': SentimentClassifier(
            model_type='logistic_regression',
            C=1.0,
            penalty='l2',
            max_iter=1000
        )
    }
    
    # Train and compare models
    results_df = compare_models(
        models=models,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        cv=cv_folds
    )
    
    # Print detailed reports
    print("\n" + "="*70)
    print("DETAILED CLASSIFICATION REPORTS")
    print("="*70)
    
    for name, model in models.items():
        print(f"\n{name}:")
        print("-" * 60)
        print(model.get_classification_report(X_test, y_test))
        
        print(f"\nConfusion Matrix:")
        print(model.get_confusion_matrix(X_test, y_test))
    
    # Save models and feature extractor
    if save_models:
        output_dir = Path('trained_models/phase2')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*70)
        print("SAVING MODELS")
        print("="*70)
        
        # Save feature extractor
        feature_extractor.save(output_dir / 'feature_extractor.pkl')
        
        # Save models
        for name, model in models.items():
            filename = name.lower().replace(' ', '_') + '.pkl'
            model.save(output_dir / filename)
        
        # Save results
        results_df.to_csv(output_dir / 'results.csv', index=False)
        print(f"✓ Results saved to {output_dir / 'results.csv'}")
    
    return {
        'results': results_df,
        'models': models,
        'feature_extractor': feature_extractor,
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test
    }


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Train Phase 2 traditional models for sentiment analysis'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='sentiment140',
        choices=['sentiment140', 'airline'],
        help='Dataset to use (default: sentiment140)'
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=50000,
        help='Sample size (default: 50000, use 0 for full dataset)'
    )
    
    parser.add_argument(
        '--max-features',
        type=int,
        default=5000,
        help='Maximum number of N-gram features (default: 5000)'
    )
    
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Number of cross-validation folds (default: 5)'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save trained models'
    )
    
    args = parser.parse_args()
    
    # Handle sample size
    sample_size = None if args.sample_size == 0 else args.sample_size
    
    # Train models
    results = train_phase2_models(
        dataset=args.dataset,
        sample_size=sample_size,
        max_features=args.max_features,
        cv_folds=args.cv_folds,
        save_models=not args.no_save
    )
    
    print("\n" + "="*70)
    print("PHASE 2 TRAINING COMPLETE!")
    print("="*70)
    print("\nBest Model (by F1-Score):")
    best_idx = results['results']['test_f1'].idxmax()
    best_model = results['results'].iloc[best_idx]
    print(f"  Model: {best_model['model']}")
    print(f"  F1-Score: {best_model['test_f1']:.4f}")
    print(f"  Accuracy: {best_model['test_accuracy']:.4f}")


if __name__ == '__main__':
    main()
