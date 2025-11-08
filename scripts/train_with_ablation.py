"""
Training script with Random Forest and Feature Ablation Study.

This script:
1. Loads preprocessed data
2. Extracts all feature types
3. Trains Random Forest classifier
4. Performs comprehensive feature ablation study

Usage:
    python scripts/train_with_ablation.py

Author: Shrish
Phase: 2 - Model Training & Feature Analysis
"""

import argparse
from pathlib import Path
import sys
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from features.feature_pipeline import FeatureExtractionPipeline
from models.traditional_models import SentimentClassifier, FeatureAblationStudy, compare_models


def load_preprocessed_data(data_dir: str = 'processed_data', 
                          dataset: str = 'sentiment140'):
    """
    Load preprocessed data.
    
    Args:
        data_dir: Directory containing preprocessed data
        dataset: Dataset name ('sentiment140' or 'airline')
        
    Returns:
        Tuple of (X_train, y_train, X_test, y_test, texts_train, texts_test)
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    print(f"Loading preprocessed data from: {data_dir}")
    print(f"Dataset: {dataset}")
    
    # Load data files
    train_file = data_path / f'{dataset}_train.csv'
    test_file = data_path / f'{dataset}_test.csv'
    
    if not train_file.exists() or not test_file.exists():
        raise FileNotFoundError(f"Train or test file not found for {dataset}")
    
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    
    print(f"✓ Loaded {len(train_df)} training samples")
    print(f"✓ Loaded {len(test_df)} test samples")
    
    # Extract labels and texts
    y_train = train_df['sentiment'].values
    y_test = test_df['sentiment'].values
    
    # Get processed text
    text_col = 'processed_text' if 'processed_text' in train_df.columns else 'text'
    texts_train = train_df[text_col].values
    texts_test = test_df[text_col].values
    
    return texts_train, y_train, texts_test, y_test


def extract_features_by_type(pipeline: FeatureExtractionPipeline,
                            texts_train: np.ndarray,
                            texts_test: np.ndarray) -> dict:
    """
    Extract features by type for ablation study.
    
    Args:
        pipeline: Initialized feature extraction pipeline
        texts_train: Training texts
        texts_test: Test texts
        
    Returns:
        Dictionary with 'train' and 'test' feature dictionaries
    """
    print("\nExtracting features by type...")
    print("="*70)
    
    # Extract all features separately
    train_features = pipeline.extract_all_features(
        texts_train, 
        fit_vectorizers=True, 
        return_dict=True
    )
    
    test_features = pipeline.extract_all_features(
        texts_test,
        fit_vectorizers=False,
        return_dict=True
    )
    
    # Print feature dimensions
    print("\nFeature dimensions:")
    for feature_type in train_features.keys():
        train_shape = train_features[feature_type].shape
        test_shape = test_features[feature_type].shape
        print(f"  {feature_type:15s}: Train {train_shape}, Test {test_shape}")
    
    return {'train': train_features, 'test': test_features}


def train_random_forest(X_train: np.ndarray,
                       y_train: np.ndarray,
                       X_test: np.ndarray,
                       y_test: np.ndarray,
                       **rf_params) -> SentimentClassifier:
    """
    Train and evaluate Random Forest classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        **rf_params: Random Forest parameters
        
    Returns:
        Trained Random Forest classifier
    """
    print("\n" + "="*70)
    print("TRAINING RANDOM FOREST CLASSIFIER")
    print("="*70)
    
    # Initialize Random Forest
    rf_model = SentimentClassifier(
        model_type='random_forest',
        **rf_params
    )
    
    # Train
    rf_model.fit(X_train, y_train)
    
    # Evaluate
    print("\nEvaluating on test set...")
    test_metrics = rf_model.evaluate(X_test, y_test, verbose=True)
    
    # Print classification report
    print("\nDetailed Classification Report:")
    print(rf_model.get_classification_report(X_test, y_test))
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    cm = rf_model.get_confusion_matrix(X_test, y_test)
    print(cm)
    
    return rf_model


def compare_all_models(X_train: np.ndarray,
                      y_train: np.ndarray,
                      X_test: np.ndarray,
                      y_test: np.ndarray) -> pd.DataFrame:
    """
    Compare Naive Bayes, Logistic Regression, and Random Forest.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Comparison DataFrame
    """
    print("\n" + "="*70)
    print("COMPARING ALL MODELS")
    print("="*70)
    
    models = {
        'Naive Bayes': SentimentClassifier(
            model_type='naive_bayes',
            alpha=1.0
        ),
        'Logistic Regression': SentimentClassifier(
            model_type='logistic_regression',
            C=1.0,
            max_iter=1000
        ),
        'Random Forest': SentimentClassifier(
            model_type='random_forest',
            n_estimators=100,
            max_depth=None,
            min_samples_split=2
        )
    }
    
    comparison_df = compare_models(
        models=models,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        cv=5
    )
    
    return comparison_df


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Train Random Forest and perform feature ablation study'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='processed_data',
        help='Directory containing preprocessed data'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['sentiment140', 'airline'],
        default='sentiment140',
        help='Which dataset to use'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Directory to save results'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=None,
        help='Sample size for testing (use subset of data)'
    )
    parser.add_argument(
        '--n-estimators',
        type=int,
        default=100,
        help='Number of trees in Random Forest (default: 100)'
    )
    parser.add_argument(
        '--max-depth',
        type=int,
        default=None,
        help='Maximum depth of trees (default: None - unlimited)'
    )
    parser.add_argument(
        '--skip-ablation',
        action='store_true',
        help='Skip feature ablation study'
    )
    parser.add_argument(
        '--compare-models',
        action='store_true',
        help='Compare all models (NB, LR, RF)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print("="*70)
    print("RANDOM FOREST TRAINING & FEATURE ABLATION STUDY")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  N estimators: {args.n_estimators}")
    print(f"  Max depth: {args.max_depth}")
    if args.sample_size:
        print(f"  Sample size: {args.sample_size:,}")
    print()
    
    # Load data
    print("\n" + "="*70)
    print("STEP 1: LOADING DATA")
    print("="*70)
    texts_train, y_train, texts_test, y_test = load_preprocessed_data(
        args.data_dir, 
        args.dataset
    )
    
    # Sample data if requested
    if args.sample_size and args.sample_size < len(texts_train):
        print(f"\nSampling {args.sample_size} training samples...")
        indices = np.random.choice(len(texts_train), args.sample_size, replace=False)
        texts_train = texts_train[indices]
        y_train = y_train[indices]
        print(f"✓ Sampled to {len(texts_train)} samples")
    
    # Initialize feature extraction pipeline
    print("\n" + "="*70)
    print("STEP 2: INITIALIZING FEATURE EXTRACTION PIPELINE")
    print("="*70)
    
    pipeline = FeatureExtractionPipeline()
    pipeline.initialize_extractors(
        init_contextual=True,
        init_semantic=True,
        init_lexicon=True
    )
    
    # Extract features by type
    print("\n" + "="*70)
    print("STEP 3: EXTRACTING FEATURES")
    print("="*70)
    
    features_by_type = extract_features_by_type(pipeline, texts_train, texts_test)
    
    # Combine all features for main training
    print("\nCombining all features...")
    X_train = pipeline.extract_all_features(texts_train, fit_vectorizers=True, return_dict=False)
    X_test = pipeline.extract_all_features(texts_test, fit_vectorizers=False, return_dict=False)
    
    print(f"✓ Combined feature shape - Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Save feature pipeline
    pipeline_path = output_path / f'feature_pipeline_{timestamp}.pkl'
    pipeline.save_pipeline(str(pipeline_path))
    
    # Train Random Forest
    print("\n" + "="*70)
    print("STEP 4: TRAINING RANDOM FOREST")
    print("="*70)
    
    rf_model = train_random_forest(
        X_train, y_train, X_test, y_test,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth
    )
    
    # Save model
    model_path = output_path / f'random_forest_{timestamp}.pkl'
    rf_model.save(str(model_path))
    
    # Compare all models if requested
    if args.compare_models:
        print("\n" + "="*70)
        print("STEP 5: COMPARING ALL MODELS")
        print("="*70)
        
        comparison_df = compare_all_models(X_train, y_train, X_test, y_test)
        comparison_path = output_path / f'model_comparison_{timestamp}.csv'
        comparison_df.to_csv(comparison_path, index=False)
        print(f"\n✓ Model comparison saved to {comparison_path}")
    
    # Feature ablation study
    if not args.skip_ablation:
        print("\n" + "="*70)
        print("STEP 6: FEATURE ABLATION STUDY")
        print("="*70)
        
        ablation_study = FeatureAblationStudy(
            model_type='random_forest',
            n_estimators=args.n_estimators,
            max_depth=args.max_depth
        )
        
        ablation_results = ablation_study.run_ablation_study(
            feature_dict=features_by_type,
            y_train=y_train,
            y_test=y_test,
            cv=5,
            test_individual=True,
            test_combined=True,
            test_leave_one_out=True
        )
        
        # Save ablation results
        ablation_path = output_path / f'ablation_study_{timestamp}.csv'
        ablation_study.save_results(str(ablation_path))
        
        # Plot results
        try:
            plot_path = output_path / f'ablation_plot_{timestamp}.png'
            ablation_study.plot_results(str(plot_path))
        except Exception as e:
            print(f"Could not create plot: {e}")
    
    print("\n" + "="*70)
    print("✓ TRAINING AND ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  - Model: {model_path.name}")
    print(f"  - Pipeline: {pipeline_path.name}")
    if not args.skip_ablation:
        print(f"  - Ablation results: {ablation_path.name}")
    if args.compare_models:
        print(f"  - Model comparison: {comparison_path.name}")
    print()


if __name__ == '__main__':
    main()
