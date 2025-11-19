"""
Comparative Analysis: Traditional vs. Semantic-Enhanced Features

Compares model performance with and without semantic features to evaluate
the impact of semantic enrichment on sentiment classification.

Usage:
    python scripts/comparative_analysis.py --sample-size 10000

Author: Naman
Phase: 3 - Integration & Analysis
"""

import argparse
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from preprocessing.text_preprocessing import load_sentiment140
from features.traditional_features import TraditionalFeatureExtractor
from features.feature_pipeline import FeatureExtractionPipeline
from models.traditional_models import SentimentClassifier, compare_models
from models.success_analysis import SuccessAnalyzer
from models.error_analysis import ErrorAnalyzer


def main():
    parser = argparse.ArgumentParser(description='Comparative Analysis: Traditional vs Semantic Features')
    parser.add_argument('--sample-size', type=int, default=10000,
                       help='Number of samples to use (0 for full dataset)')
    parser.add_argument('--dataset', type=str, default='sentiment140',
                       choices=['sentiment140', 'airline'],
                       help='Dataset to use')
    parser.add_argument('--output-dir', type=str, default='comparative_results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    print("="*70)
    print("COMPARATIVE ANALYSIS: TRADITIONAL VS SEMANTIC-ENHANCED FEATURES")
    print("="*70)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load and preprocess data
    print(f"\nLoading {args.dataset} dataset...")
    X_train_text, y_train, X_test_text, y_test = load_sentiment140(
        sample_size=args.sample_size,
        preprocess=True
    )
    
    print(f"✓ Loaded {len(X_train_text)} training samples")
    print(f"✓ Loaded {len(X_test_text)} test samples")
    
    # Initialize models
    models = {
        'Naive Bayes': SentimentClassifier(model_type='naive_bayes', alpha=1.0),
        'Logistic Regression': SentimentClassifier(model_type='logistic_regression', 
                                                   C=1.0, max_iter=1000)
    }
    
    results_comparison = {}
    
    # ===================================================================
    # EXPERIMENT 1: Traditional Features Only
    # ===================================================================
    print("\n" + "="*70)
    print("EXPERIMENT 1: TRADITIONAL FEATURES ONLY")
    print("="*70)
    
    print("\nExtracting traditional features (N-grams + POS)...")
    traditional_extractor = TraditionalFeatureExtractor(
        ngram_range=(1, 2),
        max_features=5000
    )
    
    X_train_trad = traditional_extractor.fit_transform(X_train_text)
    X_test_trad = traditional_extractor.transform(X_test_text)
    
    print(f"✓ Training shape: {X_train_trad.shape}")
    print(f"✓ Test shape: {X_test_trad.shape}")
    
    print("\nTraining models with traditional features...")
    results_trad = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_trad, y_train)
        
        # Evaluate
        metrics = model.evaluate(X_test_trad, y_test)
        results_trad[name] = metrics
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-score: {metrics['f1']:.4f}")
    
    # ===================================================================
    # EXPERIMENT 2: Traditional + Semantic Features
    # ===================================================================
    print("\n" + "="*70)
    print("EXPERIMENT 2: TRADITIONAL + SEMANTIC FEATURES")
    print("="*70)
    
    print("\nInitializing feature pipeline with semantic features...")
    try:
        pipeline = FeatureExtractionPipeline(
            ngram_range=(1, 2),
            max_features=5000,
            init_contextual=True,
            init_semantic=True,
            init_lexicon=True
        )
        
        print("\nExtracting all features...")
        X_train_full = pipeline.fit_transform(X_train_text)
        X_test_full = pipeline.transform(X_test_text)
        
        print(f"✓ Training shape: {X_train_full.shape}")
        print(f"✓ Test shape: {X_test_full.shape}")
        
        # Retrain models with full features
        models_full = {
            'Naive Bayes': SentimentClassifier(model_type='naive_bayes', alpha=1.0),
            'Logistic Regression': SentimentClassifier(model_type='logistic_regression',
                                                       C=1.0, max_iter=1000)
        }
        
        print("\nTraining models with semantic-enhanced features...")
        results_full = {}
        
        for name, model in models_full.items():
            print(f"\nTraining {name}...")
            model.fit(X_train_full, y_train)
            
            # Evaluate
            metrics = model.evaluate(X_test_full, y_test)
            results_full[name] = metrics
            
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-score: {metrics['f1']:.4f}")
        
        semantic_features_available = True
        
    except Exception as e:
        print(f"\n⚠️  Could not initialize semantic features: {e}")
        print("Continuing with traditional features only...")
        semantic_features_available = False
        results_full = {}
    
    # ===================================================================
    # COMPARATIVE RESULTS
    # ===================================================================
    print("\n" + "="*70)
    print("COMPARATIVE RESULTS")
    print("="*70)
    
    # Create comparison table
    comparison_data = []
    
    for model_name in models.keys():
        trad_metrics = results_trad[model_name]
        
        row = {
            'Model': model_name,
            'Feature Set': 'Traditional Only',
            'Accuracy': trad_metrics['accuracy'],
            'Precision': trad_metrics['precision'],
            'Recall': trad_metrics['recall'],
            'F1-Score': trad_metrics['f1']
        }
        comparison_data.append(row)
        
        if semantic_features_available and model_name in results_full:
            full_metrics = results_full[model_name]
            row = {
                'Model': model_name,
                'Feature Set': 'Traditional + Semantic',
                'Accuracy': full_metrics['accuracy'],
                'Precision': full_metrics['precision'],
                'Recall': full_metrics['recall'],
                'F1-Score': full_metrics['f1']
            }
            comparison_data.append(row)
            
            # Calculate improvement
            improvement = {
                'Model': model_name,
                'Feature Set': 'Improvement (%)',
                'Accuracy': (full_metrics['accuracy'] - trad_metrics['accuracy']) * 100,
                'Precision': (full_metrics['precision'] - trad_metrics['precision']) * 100,
                'Recall': (full_metrics['recall'] - trad_metrics['recall']) * 100,
                'F1-Score': (full_metrics['f1'] - trad_metrics['f1']) * 100
            }
            comparison_data.append(improvement)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    print("\n📊 Performance Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Save comparison table
    comparison_path = output_dir / 'comparison_table.csv'
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\n✓ Comparison table saved to {comparison_path}")
    
    # ===================================================================
    # SUCCESS & ERROR ANALYSIS
    # ===================================================================
    print("\n" + "="*70)
    print("SUCCESS & ERROR ANALYSIS")
    print("="*70)
    
    # Analyze best traditional model
    best_trad_model = max(results_trad.items(), key=lambda x: x[1]['f1'])
    best_model_name = best_trad_model[0]
    
    print(f"\nBest Traditional Model: {best_model_name} (F1={best_trad_model[1]['f1']:.4f})")
    
    # Success analysis
    print("\n--- SUCCESS ANALYSIS ---")
    success_analyzer = SuccessAnalyzer()
    success_analyzer.analyze_correct_predictions(
        model=models[best_model_name],
        X=X_train_trad,
        y_true=y_train,
        texts=X_train_text,
        model_name=best_model_name
    )
    
    # Error analysis
    print("\n--- ERROR ANALYSIS ---")
    error_analyzer = ErrorAnalyzer()
    error_analyzer.analyze_errors(
        model=models[best_model_name],
        X=X_test_trad,
        y_true=y_test,
        texts=X_test_text,
        model_name=best_model_name
    )
    
    # Save analyses
    success_path = output_dir / 'success_analysis.json'
    success_analyzer.export_analysis(str(success_path))
    
    error_path = output_dir / 'error_analysis.json'
    error_analyzer.export_analysis(str(error_path))
    
    # ===================================================================
    # MODEL COMPARISON (ERROR PATTERNS)
    # ===================================================================
    if len(models) >= 2:
        print("\n" + "="*70)
        print("MODEL ERROR COMPARISON")
        print("="*70)
        
        model_names = list(models.keys())
        error_analyzer.compare_model_errors(
            model1=models[model_names[0]],
            model2=models[model_names[1]],
            X=X_test_trad,
            y_true=y_test,
            texts=X_test_text,
            model1_name=model_names[0],
            model2_name=model_names[1]
        )
    
    print("\n" + "="*70)
    print("✅ COMPARATIVE ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - comparison_table.csv")
    print(f"  - success_analysis.json")
    print(f"  - error_analysis.json")


if __name__ == '__main__':
    main()
