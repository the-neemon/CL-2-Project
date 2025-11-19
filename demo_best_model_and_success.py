"""
Demonstration: Best Model Identification and Success Analysis

Shows the two Phase 2 tasks:
1. Identify best-performing model based on maximum F1-score
2. Conduct success analysis on correctly classified instances

Author: Naman
Phase: 2
"""

import sys
sys.path.insert(0, '.')

from preprocessing.data_loader import SentimentDataLoader
from preprocessing.preprocessing import TweetPreprocessor
from features.traditional_features import TraditionalFeatureExtractor
from models.traditional_models import SentimentClassifier
from models.success_analysis import SuccessAnalyzer
import numpy as np


def main():
    """Demonstrate best model selection and success analysis."""
    
    print("="*80)
    print("DEMO: BEST MODEL IDENTIFICATION & SUCCESS ANALYSIS")
    print("="*80)
    
    # Load small sample for quick demo
    print("\n[1] Loading data...")
    loader = SentimentDataLoader(dataset_dir='datasets')
    preprocessor = TweetPreprocessor()
    
    df = loader.load_sentiment140(sample_size=3000)
    train_df, test_df = loader.create_train_test_split(df, test_size=0.2)
    train_df = loader.preprocess_dataframe(train_df, preprocessor)
    test_df = loader.preprocess_dataframe(test_df, preprocessor)
    
    print(f"✓ Train: {len(train_df)} samples")
    print(f"✓ Test:  {len(test_df)} samples")
    
    # Extract features
    print("\n[2] Extracting features...")
    extractor = TraditionalFeatureExtractor(max_features=1500, ngram_range=(1, 2))
    X_train = extractor.fit_transform(train_df['processed_text'].values)
    X_test = extractor.transform(test_df['processed_text'].values)
    y_train = train_df['sentiment'].values
    y_test = test_df['sentiment'].values
    
    # Train multiple models
    print("\n[3] Training models...")
    models = {
        'Naive Bayes': SentimentClassifier(model_type='naive_bayes'),
        'Logistic Regression': SentimentClassifier(model_type='logistic_regression')
    }
    
    results = {}
    for name, model in models.items():
        print(f"\n  Training {name}...")
        model.fit(X_train, y_train)
        metrics = model.evaluate(X_test, y_test, verbose=False)
        results[name] = metrics
        print(f"    F1-Score: {metrics['f1_score']:.4f}")
    
    # TASK 1: Identify best model by F1-score
    print("\n" + "="*80)
    print("TASK 1: IDENTIFY BEST-PERFORMING MODEL BY F1-SCORE")
    print("="*80)
    
    best_name = max(results.keys(), key=lambda k: results[k]['f1_score'])
    best_model = models[best_name]
    best_metrics = results[best_name]
    
    print(f"\n🏆 Best Model: {best_name}")
    print(f"  ✓ F1-Score:  {best_metrics['f1_score']:.4f}")
    print(f"  ✓ Accuracy:  {best_metrics['accuracy']:.4f}")
    print(f"  ✓ Precision: {best_metrics['precision']:.4f}")
    print(f"  ✓ Recall:    {best_metrics['recall']:.4f}")
    
    print(f"\n📊 All Model Rankings by F1-Score:")
    sorted_models = sorted(results.items(), key=lambda x: x[1]['f1_score'], 
                          reverse=True)
    for rank, (name, metrics) in enumerate(sorted_models, 1):
        print(f"  {rank}. {name:20s} F1={metrics['f1_score']:.4f}")
    
    # TASK 2: Success analysis
    print("\n" + "="*80)
    print("TASK 2: SUCCESS ANALYSIS ON CORRECTLY CLASSIFIED INSTANCES")
    print("="*80)
    
    analyzer = SuccessAnalyzer()
    
    # Analyze the best model
    print(f"\n📈 Analyzing {best_name}...")
    analysis = analyzer.analyze_correct_predictions(
        best_model,
        X_test,
        y_test,
        test_df['processed_text'].values
    )
    
    # Compare the two models
    if len(models) >= 2:
        model_names = list(models.keys())
        print("\n" + "="*80)
        print(f"BONUS: COMPARING {model_names[0]} vs {model_names[1]}")
        print("="*80)
        
        comparison = analyzer.compare_success_patterns(
            model_names[0],
            models[model_names[0]],
            model_names[1],
            models[model_names[1]],
            X_test,
            y_test,
            test_df['processed_text'].values
        )
        
        print(f"\n💡 Key Insights:")
        print(f"  • Agreement on correct: {comparison['both_correct']} samples")
        print(f"  • {model_names[0]} unique successes: "
              f"{comparison['only_model1_correct']}")
        print(f"  • {model_names[1]} unique successes: "
              f"{comparison['only_model2_correct']}")
    
    print("\n" + "="*80)
    print("✅ DEMONSTRATION COMPLETE")
    print("="*80)
    print("""
Summary:
  ✓ Task 1: Best model identified based on maximum F1-score
  ✓ Task 2: Success analysis conducted on correctly classified instances
  ✓ Bonus: Model comparison showing unique strengths
    """)


if __name__ == '__main__':
    main()
