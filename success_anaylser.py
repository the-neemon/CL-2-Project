#!/usr/bin/env python3
"""
TASK IMPLEMENTATION: Best Model Identification & Success Analysis

This script directly implements the two main tasks:
1. ✅ Identify best-performing model based on maximum F1-score
2. ✅ Conduct success analysis on correctly classified instances

Key Features:
- Trains multiple sentiment classification models
- Evaluates and ranks them by F1-score
- Identifies the best-performing model
- Conducts comprehensive success analysis
- Provides detailed insights into correct predictions

Author: AI Assistant
Date: November 19, 2025
"""

import sys
sys.path.insert(0, '.')

import numpy as np
from typing import Dict, List, Tuple, Any
from preprocessing.data_loader import SentimentDataLoader
from preprocessing.preprocessing import TweetPreprocessor
from features.traditional_features import TraditionalFeatureExtractor
from models.traditional_models import SentimentClassifier
from models.success_analysis import SuccessAnalyzer


def task_1_identify_best_model(X_train, y_train, X_test, y_test) -> Tuple[str, SentimentClassifier, Dict[str, Dict[str, float]]]:
    """
    TASK 1: Identify best-performing model based on maximum F1-score
    
    Returns:
        - Best model name
        - Best model object
        - All model results
    """
    print("="*80)
    print("🎯 TASK 1: IDENTIFY BEST-PERFORMING MODEL BY F1-SCORE")
    print("="*80)
    
    # Define models to evaluate
    models_to_test = {
        'Multinomial Naive Bayes': SentimentClassifier('naive_bayes'),
        'Logistic Regression': SentimentClassifier('logistic_regression'),
        'Logistic Regression (L1)': SentimentClassifier('logistic_regression', penalty='l1', solver='liblinear'),
        'Random Forest': SentimentClassifier('random_forest', n_estimators=100, max_depth=10),
        'Random Forest (Balanced)': SentimentClassifier('random_forest', n_estimators=100, class_weight='balanced')
    }
    
    print(f"📋 Training and evaluating {len(models_to_test)} models...")
    
    model_results = {}
    trained_models = {}
    
    # Train and evaluate each model
    for model_name, model in models_to_test.items():
        print(f"\n🔄 {model_name}:")
        print(f"   Training... ", end="")
        
        # Train model
        model.fit(X_train, y_train)
        trained_models[model_name] = model
        print("✅")
        
        # Evaluate model
        print(f"   Evaluating... ", end="")
        metrics = model.evaluate(X_test, y_test, verbose=False)
        model_results[model_name] = metrics
        print(f"F1={metrics['f1_score']:.4f}")
    
    # Rank models by F1-score
    print(f"\n📊 MODEL PERFORMANCE RANKING (by F1-score):")
    print(f"{'Rank':<4} {'Model':<30} {'F1-Score':<8} {'Accuracy':<8} {'ROC-AUC':<8}")
    print("-" * 70)
    
    sorted_models = sorted(model_results.items(), key=lambda x: x[1]['f1_score'], reverse=True)
    
    for rank, (name, metrics) in enumerate(sorted_models, 1):
        roc_auc = f"{metrics.get('roc_auc', 'N/A'):.4f}" if 'roc_auc' in metrics else 'N/A'
        print(f"{rank:<4} {name:<30} {metrics['f1_score']:<8.4f} "
              f"{metrics['accuracy']:<8.4f} {roc_auc:<8}")
    
    # Identify best model
    best_model_name = sorted_models[0][0]
    best_model = trained_models[best_model_name]
    best_metrics = sorted_models[0][1]
    
    print(f"\n🏆 BEST MODEL IDENTIFIED: {best_model_name}")
    print(f"   ✅ Maximum F1-Score: {best_metrics['f1_score']:.4f}")
    print(f"   📈 Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"   🎯 Precision: {best_metrics['precision']:.4f}")
    print(f"   🔄 Recall: {best_metrics['recall']:.4f}")
    if 'roc_auc' in best_metrics:
        print(f"   📊 ROC-AUC: {best_metrics['roc_auc']:.4f}")
    
    return best_model_name, best_model, model_results


def task_2_success_analysis(best_model_name: str, best_model: SentimentClassifier,
                           X_test, y_test, texts, feature_names) -> Dict[str, Any]:
    """
    TASK 2: Conduct success analysis on correctly classified instances
    
    Returns:
        Success analysis results
    """
    print("\n" + "="*80)
    print("🔍 TASK 2: SUCCESS ANALYSIS ON CORRECTLY CLASSIFIED INSTANCES")
    print("="*80)
    
    print(f"📊 Analyzing correctly classified instances for: {best_model_name}")
    
    # Initialize success analyzer
    analyzer = SuccessAnalyzer()
    
    # Conduct comprehensive success analysis
    analysis_results = analyzer.analyze_correct_predictions(
        best_model,
        X_test,
        y_test,
        texts,
        feature_names
    )
    
    # Additional success pattern analysis
    print(f"\n🔬 ADDITIONAL SUCCESS PATTERN ANALYSIS")
    print("-" * 60)
    
    # Get predictions and probabilities
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)
    
    # Identify correctly classified instances
    correct_mask = y_pred == y_test
    correct_indices = np.where(correct_mask)[0]
    
    # Analyze prediction confidence for correct instances
    correct_confidences = np.max(y_proba[correct_mask], axis=1)
    
    print(f"📈 Success by Confidence Level:")
    confidence_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    for i, threshold in enumerate(confidence_thresholds):
        high_conf_correct = np.sum(correct_confidences >= threshold)
        total_correct = len(correct_confidences)
        percentage = (high_conf_correct / total_correct) * 100
        print(f"   Confidence ≥ {threshold}: {high_conf_correct}/{total_correct} "
              f"({percentage:.1f}% of correct predictions)")
    
    # Analyze text length patterns for correct predictions
    correct_texts = texts[correct_mask]
    correct_lengths = [len(text.split()) for text in correct_texts]
    
    print(f"\n📝 Text Length Analysis for Correct Predictions:")
    print(f"   Mean length: {np.mean(correct_lengths):.1f} words")
    print(f"   Median length: {np.median(correct_lengths):.1f} words")
    print(f"   Min length: {np.min(correct_lengths)} words")
    print(f"   Max length: {np.max(correct_lengths)} words")
    
    # Feature importance analysis (if available)
    if hasattr(best_model.model, 'coef_'):
        coef = best_model.model.coef_[0]
        print(f"\n⭐ TOP FEATURES CONTRIBUTING TO SUCCESS:")
        print("   Most Positive Features (for positive sentiment):")
        top_positive_idx = np.argsort(coef)[-5:][::-1]
        for i, idx in enumerate(top_positive_idx, 1):
            feature_name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
            print(f"      {i}. {feature_name} (coef: {coef[idx]:.4f})")
        
        print("   Most Negative Features (for negative sentiment):")
        top_negative_idx = np.argsort(coef)[:5]
        for i, idx in enumerate(top_negative_idx, 1):
            feature_name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
            print(f"      {i}. {feature_name} (coef: {coef[idx]:.4f})")
    
    # Sample some highly confident correct predictions
    very_confident_correct = (correct_confidences >= 0.9)
    if np.sum(very_confident_correct) > 0:
        print(f"\n✨ SAMPLE VERY HIGH-CONFIDENCE CORRECT PREDICTIONS (≥0.9):")
        very_conf_indices = correct_indices[very_confident_correct][:3]
        for i, idx in enumerate(very_conf_indices, 1):
            confidence = np.max(y_proba[idx])
            true_label = "Positive" if y_test[idx] == 1 else "Negative"
            text = texts[idx]
            print(f"   [{i}] {true_label} (confidence: {confidence:.4f})")
            print(f"       Text: \"{text[:80]}...\"" if len(text) > 80 else f"       Text: \"{text}\"")
    
    return analysis_results


def main():
    """
    Main execution function that implements both tasks
    """
    print("🚀 SENTIMENT ANALYSIS: BEST MODEL IDENTIFICATION & SUCCESS ANALYSIS")
    print("="*80)
    print("Implementing Phase 2 Requirements:")
    print("1. ✅ Identify best-performing model based on maximum F1-score")
    print("2. ✅ Conduct success analysis on correctly classified instances")
    print("="*80)
    
    # === DATA PREPARATION ===
    print("\n📥 LOADING AND PREPARING DATA...")
    
    # Load data
    loader = SentimentDataLoader(dataset_dir='datasets')
    preprocessor = TweetPreprocessor()
    
    df = loader.load_sentiment140(sample_size=3000)  # Manageable size for demo
    train_df, test_df = loader.create_train_test_split(df, test_size=0.2)
    train_df = loader.preprocess_dataframe(train_df, preprocessor)
    test_df = loader.preprocess_dataframe(test_df, preprocessor)
    
    print(f"✅ Dataset prepared:")
    print(f"   Training samples: {len(train_df)}")
    print(f"   Test samples: {len(test_df)}")
    
    # Extract features
    print(f"\n🛠️  EXTRACTING FEATURES...")
    extractor = TraditionalFeatureExtractor(max_features=1500, ngram_range=(1, 2))
    X_train = extractor.fit_transform(train_df['processed_text'].values)
    X_test = extractor.transform(test_df['processed_text'].values)
    y_train = train_df['sentiment'].values
    y_test = test_df['sentiment'].values
    feature_names = extractor.get_feature_names()
    
    print(f"✅ Feature extraction complete:")
    print(f"   Feature matrix shape: {X_train.shape}")
    print(f"   Number of features: {len(feature_names)}")
    
    # === TASK 1: IDENTIFY BEST MODEL ===
    best_model_name, best_model, all_results = task_1_identify_best_model(
        X_train, y_train, X_test, y_test
    )
    
    # === TASK 2: SUCCESS ANALYSIS ===
    success_analysis = task_2_success_analysis(
        best_model_name, best_model, X_test, y_test, 
        test_df['processed_text'].values, feature_names
    )
    
    # === SUMMARY ===
    print("\n" + "="*80)
    print("📋 FINAL SUMMARY")
    print("="*80)
    print(f"✅ TASK 1 COMPLETE: Best model identified")
    print(f"   🏆 Winner: {best_model_name}")
    print(f"   📊 F1-Score: {all_results[best_model_name]['f1_score']:.4f}")
    print(f"   🎯 Accuracy: {all_results[best_model_name]['accuracy']:.4f}")
    
    print(f"\n✅ TASK 2 COMPLETE: Success analysis conducted")
    print(f"   📈 Correctly classified: {success_analysis['correct_count']}/{success_analysis['total_samples']} "
          f"({success_analysis['accuracy']:.2%})")
    print(f"   🔍 High-confidence predictions: {success_analysis['high_confidence_count']}")
    print(f"   📝 Average text length (correct): {success_analysis['correct_text_length_mean']:.1f} words")
    
    print(f"\n🎯 KEY INSIGHTS:")
    print(f"   • {len(all_results)} models evaluated for optimal F1-score")
    print(f"   • Success patterns identified in correctly classified instances")
    print(f"   • Feature importance analysis reveals key sentiment indicators")
    print(f"   • Confidence analysis shows model reliability patterns")
    
    print(f"\n✨ ANALYSIS COMPLETE! Both Phase 2 tasks successfully implemented.")


if __name__ == '__main__':
    main()
