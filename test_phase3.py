"""
Quick test script for Phase 3 implementations.

Tests:
1. Error Analysis module
2. Comparative Analysis integration
3. Success Analysis integration

Author: Naman
Phase: 3 - Integration & Analysis
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from models import ErrorAnalyzer, SuccessAnalyzer
from models.traditional_models import SentimentClassifier


def test_error_analysis():
    """Test ErrorAnalyzer module."""
    print("\n" + "="*70)
    print("TEST 1: Error Analysis Module")
    print("="*70)
    
    try:
        # Create simple mock data (positive values for Naive Bayes)
        np.random.seed(42)
        X = np.abs(np.random.randn(100, 10))  # Positive values only
        y_true = np.random.randint(0, 2, 100)
        texts = [f"sample text {i}" for i in range(100)]
        
        # Train a simple model (use Logistic Regression for simplicity)
        model = SentimentClassifier(model_type='logistic_regression')
        model.fit(X, y_true)
        
        # Test error analysis
        analyzer = ErrorAnalyzer()
        results = analyzer.analyze_errors(
            model=model,
            X=X,
            y_true=y_true,
            texts=texts,
            model_name="Test Model",
            verbose=False
        )
        
        # Verify results structure
        assert 'total_samples' in results
        assert 'error_rate' in results
        assert 'class_analysis' in results
        assert 'confidence_analysis' in results
        assert 'confusion_patterns' in results
        
        print("✅ ErrorAnalyzer: PASS")
        print(f"   - Analyzed {results['total_samples']} samples")
        print(f"   - Error rate: {results['error_rate']*100:.2f}%")
        return True
        
    except Exception as e:
        print(f"❌ ErrorAnalyzer: FAIL - {e}")
        return False


def test_success_analysis():
    """Test SuccessAnalyzer module."""
    print("\n" + "="*70)
    print("TEST 2: Success Analysis Module")
    print("="*70)
    
    try:
        # Create simple mock data (positive values for compatibility)
        np.random.seed(42)
        X = np.abs(np.random.randn(100, 10))
        y_true = np.random.randint(0, 2, 100)
        texts = [f"sample text {i}" for i in range(100)]
        
        # Train a simple model
        model = SentimentClassifier(model_type='logistic_regression')
        model.fit(X, y_true)
        
        # Test success analysis
        analyzer = SuccessAnalyzer()
        results = analyzer.analyze_correct_predictions(
            model=model,
            X=X,
            y=y_true,
            texts=texts
        )
        
        # Verify results structure
        assert 'total_samples' in results
        assert 'correct_count' in results
        assert 'accuracy' in results
        assert 'class_analysis' in results
        
        print("✅ SuccessAnalyzer: PASS")
        print(f"   - Analyzed {results['total_samples']} samples")
        print(f"   - Accuracy: {results['accuracy']*100:.2f}%")
        return True
        
    except Exception as e:
        print(f"❌ SuccessAnalyzer: FAIL - {e}")
        return False


def test_model_comparison():
    """Test model comparison features."""
    print("\n" + "="*70)
    print("TEST 3: Model Comparison")
    print("="*70)
    
    try:
        # Create simple mock data (positive values for compatibility)
        np.random.seed(42)
        X = np.abs(np.random.randn(100, 10))
        y_true = np.random.randint(0, 2, 100)
        texts = [f"sample text {i}" for i in range(100)]
        
        # Train two models
        model1 = SentimentClassifier(model_type='logistic_regression', C=0.1)
        model1.fit(X, y_true)
        
        model2 = SentimentClassifier(model_type='logistic_regression', C=1.0)
        model2.fit(X, y_true)
        
        # Test error comparison
        error_analyzer = ErrorAnalyzer()
        error_comparison = error_analyzer.compare_model_errors(
            model1=model1,
            model2=model2,
            X=X,
            y_true=y_true,
            texts=texts,
            model1_name="Naive Bayes",
            model2_name="Logistic Regression",
            verbose=False
        )
        
        assert 'both_wrong' in error_comparison
        assert 'only_model1_wrong' in error_comparison
        assert 'only_model2_wrong' in error_comparison
        
        # Test success comparison
        success_analyzer = SuccessAnalyzer()
        success_comparison = success_analyzer.compare_success_patterns(
            "LR (C=0.1)",
            model1,
            "LR (C=1.0)",
            model2,
            X,
            y_true,
            texts
        )
        
        assert 'both_correct' in success_comparison
        assert 'agreement_rate' in success_comparison
        
        print("✅ Model Comparison: PASS")
        print(f"   - Error overlap: {error_comparison['both_wrong']} samples")
        print(f"   - Agreement rate: {success_comparison['agreement_rate']*100:.2f}%")
        return True
        
    except Exception as e:
        print(f"❌ Model Comparison: FAIL - {e}")
        return False


def test_export_functions():
    """Test export functions."""
    print("\n" + "="*70)
    print("TEST 4: Export Functions")
    print("="*70)
    
    try:
        # Create simple mock data (positive values for compatibility)
        np.random.seed(42)
        X = np.abs(np.random.randn(50, 10))
        y_true = np.random.randint(0, 2, 50)
        texts = [f"sample text {i}" for i in range(50)]
        
        # Train model
        model = SentimentClassifier(model_type='logistic_regression')
        model.fit(X, y_true)
        
        # Test error analysis export
        error_analyzer = ErrorAnalyzer()
        error_analyzer.analyze_errors(
            model=model,
            X=X,
            y_true=y_true,
            texts=texts,
            model_name="Test Model",
            verbose=False
        )
        
        test_output_dir = Path('test_outputs')
        test_output_dir.mkdir(exist_ok=True)
        
        error_path = test_output_dir / 'test_error_analysis.json'
        error_analyzer.export_analysis(str(error_path))
        
        assert error_path.exists()
        
        # Test success analysis export
        success_analyzer = SuccessAnalyzer()
        success_analyzer.analyze_correct_predictions(
            model=model,
            X=X,
            y=y_true,
            texts=texts
        )
        
        success_path = test_output_dir / 'test_success_analysis.json'
        success_analyzer.export_analysis(str(success_path))
        
        assert success_path.exists()
        
        # Cleanup
        error_path.unlink()
        success_path.unlink()
        test_output_dir.rmdir()
        
        print("✅ Export Functions: PASS")
        print("   - Error analysis export successful")
        print("   - Success analysis export successful")
        return True
        
    except Exception as e:
        print(f"❌ Export Functions: FAIL - {e}")
        return False


def main():
    """Run all Phase 3 tests."""
    print("="*70)
    print("PHASE 3 IMPLEMENTATION TEST SUITE")
    print("="*70)
    
    results = []
    
    # Run tests
    results.append(("Error Analysis", test_error_analysis()))
    results.append(("Success Analysis", test_success_analysis()))
    results.append(("Model Comparison", test_model_comparison()))
    results.append(("Export Functions", test_export_functions()))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:30s} {status}")
    
    print("\n" + "="*70)
    if passed == total:
        print(f"✅ ALL TESTS PASSED! ({passed}/{total})")
        print("="*70)
        print("\n🎉 Phase 3 implementation is ready!")
    else:
        print(f"⚠️  SOME TESTS FAILED ({passed}/{total} passed)")
        print("="*70)
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
