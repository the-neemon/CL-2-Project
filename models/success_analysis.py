"""
Success Analysis for Sentiment Classification Models.

Conducts detailed analysis of correctly classified instances to understand
what makes the model succeed.

Author: Naman
Phase: 2
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter
import re


class SuccessAnalyzer:
    """
    Analyzer for correctly classified instances.
    
    Provides insights into what patterns, features, and characteristics
    lead to successful predictions.
    """
    
    def __init__(self):
        """Initialize the success analyzer."""
        self.analysis_results = {}
    
    def analyze_correct_predictions(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        texts: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze correctly classified instances.
        
        Args:
            model: Trained SentimentClassifier
            X: Feature matrix
            y: True labels
            texts: Original text samples
            feature_names: Names of features (for interpretation)
            
        Returns:
            Dictionary with analysis results
        """
        print("\n" + "="*70)
        print("SUCCESS ANALYSIS: CORRECTLY CLASSIFIED INSTANCES")
        print("="*70)
        
        # Get predictions
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)
        
        # Identify correctly classified instances
        correct_mask = y_pred == y
        correct_indices = np.where(correct_mask)[0]
        
        num_correct = len(correct_indices)
        total = len(y)
        accuracy = num_correct / total
        
        print(f"\n📊 Overall Statistics:")
        print(f"  Total samples: {total}")
        print(f"  Correctly classified: {num_correct} ({accuracy:.2%})")
        print(f"  Incorrectly classified: {total - num_correct} "
              f"({1-accuracy:.2%})")
        
        # Analyze by class
        print(f"\n📋 Success by Class:")
        class_analysis = {}
        for label in np.unique(y):
            class_mask = y == label
            class_correct = correct_mask & class_mask
            num_class = np.sum(class_mask)
            num_class_correct = np.sum(class_correct)
            class_acc = num_class_correct / num_class if num_class > 0 else 0
            
            class_analysis[label] = {
                'total': num_class,
                'correct': num_class_correct,
                'accuracy': class_acc
            }
            
            label_name = "Negative" if label == 0 else "Positive"
            print(f"  {label_name} (class {label}): "
                  f"{num_class_correct}/{num_class} ({class_acc:.2%})")
        
        # Analyze confidence levels
        print(f"\n📈 Confidence Analysis:")
        correct_proba = np.max(y_proba[correct_mask], axis=1)
        incorrect_proba = np.max(y_proba[~correct_mask], axis=1)
        
        print(f"  Correct predictions:")
        print(f"    Mean confidence: {np.mean(correct_proba):.4f}")
        print(f"    Median confidence: {np.median(correct_proba):.4f}")
        print(f"    Std confidence: {np.std(correct_proba):.4f}")
        
        if len(incorrect_proba) > 0:
            print(f"  Incorrect predictions:")
            print(f"    Mean confidence: {np.mean(incorrect_proba):.4f}")
            print(f"    Median confidence: {np.median(incorrect_proba):.4f}")
            print(f"    Std confidence: {np.std(incorrect_proba):.4f}")
        
        # High confidence correct predictions
        high_conf_threshold = 0.9
        high_conf_correct = correct_proba >= high_conf_threshold
        num_high_conf = np.sum(high_conf_correct)
        
        print(f"\n🎯 High Confidence Correct Predictions (≥{high_conf_threshold}):")
        print(f"  Count: {num_high_conf} ({num_high_conf/num_correct:.2%} "
              f"of correct)")
        
        # Analyze text characteristics
        print(f"\n📝 Text Characteristics:")
        correct_texts = texts[correct_mask]
        incorrect_texts = texts[~correct_mask]
        
        correct_lengths = [len(text.split()) for text in correct_texts]
        incorrect_lengths = [len(text.split()) for text in incorrect_texts]
        
        print(f"  Correctly classified texts:")
        print(f"    Mean length: {np.mean(correct_lengths):.1f} words")
        print(f"    Median length: {np.median(correct_lengths):.1f} words")
        
        if len(incorrect_lengths) > 0:
            print(f"  Incorrectly classified texts:")
            print(f"    Mean length: {np.mean(incorrect_lengths):.1f} words")
            print(f"    Median length: {np.median(incorrect_lengths):.1f} words")
        
        # Sample high-confidence correct predictions
        print(f"\n✨ Sample High-Confidence Correct Predictions:")
        self._show_sample_predictions(
            correct_indices, y, y_pred, y_proba, texts,
            num_samples=5, min_confidence=high_conf_threshold
        )
        
        # Store results
        results = {
            'total_samples': total,
            'correct_count': num_correct,
            'accuracy': accuracy,
            'class_analysis': class_analysis,
            'correct_confidence_mean': float(np.mean(correct_proba)),
            'correct_confidence_median': float(np.median(correct_proba)),
            'correct_confidence_std': float(np.std(correct_proba)),
            'high_confidence_count': int(num_high_conf),
            'correct_text_length_mean': float(np.mean(correct_lengths)),
            'correct_text_length_median': float(np.median(correct_lengths))
        }
        
        if len(incorrect_proba) > 0:
            results['incorrect_confidence_mean'] = float(
                np.mean(incorrect_proba))
            results['incorrect_confidence_median'] = float(
                np.median(incorrect_proba))
            results['incorrect_text_length_mean'] = float(
                np.mean(incorrect_lengths))
        
        self.analysis_results = results
        return results
    
    def _show_sample_predictions(
        self,
        indices: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        texts: np.ndarray,
        num_samples: int = 5,
        min_confidence: float = 0.0
    ):
        """Show sample predictions."""
        confidences = np.max(y_proba[indices], axis=1)
        high_conf_mask = confidences >= min_confidence
        
        if np.sum(high_conf_mask) == 0:
            print("  No samples meet the confidence threshold")
            return
        
        high_conf_indices = indices[high_conf_mask]
        high_conf_confidences = confidences[high_conf_mask]
        
        # Sort by confidence
        sorted_idx = np.argsort(high_conf_confidences)[::-1]
        sample_indices = high_conf_indices[sorted_idx[:num_samples]]
        
        for i, idx in enumerate(sample_indices, 1):
            true_label = y_true[idx]
            pred_label = y_pred[idx]
            confidence = np.max(y_proba[idx])
            text = texts[idx]
            
            label_name = "Negative" if true_label == 0 else "Positive"
            
            print(f"\n  [{i}] Confidence: {confidence:.4f}")
            print(f"      True/Pred: {label_name} (both {true_label})")
            print(f"      Text: \"{text[:100]}{'...' if len(text) > 100 else ''}\"")
    
    def compare_success_patterns(
        self,
        model1_name: str,
        model1,
        model2_name: str,
        model2,
        X: np.ndarray,
        y: np.ndarray,
        texts: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compare success patterns between two models.
        
        Args:
            model1_name: Name of first model
            model1: First trained model
            model2_name: Name of second model
            model2: Second trained model
            X: Feature matrix
            y: True labels
            texts: Original texts
            
        Returns:
            Comparison analysis results
        """
        print("\n" + "="*70)
        print(f"COMPARING SUCCESS PATTERNS: {model1_name} vs {model2_name}")
        print("="*70)
        
        # Get predictions
        y_pred1 = model1.predict(X)
        y_pred2 = model2.predict(X)
        
        # Identify correct predictions
        correct1 = y_pred1 == y
        correct2 = y_pred2 == y
        
        # Analyze agreement
        both_correct = correct1 & correct2
        only_model1 = correct1 & ~correct2
        only_model2 = ~correct1 & correct2
        both_wrong = ~correct1 & ~correct2
        
        total = len(y)
        
        print(f"\n📊 Agreement Analysis:")
        print(f"  Both correct: {np.sum(both_correct)} "
              f"({np.sum(both_correct)/total:.2%})")
        print(f"  Only {model1_name} correct: {np.sum(only_model1)} "
              f"({np.sum(only_model1)/total:.2%})")
        print(f"  Only {model2_name} correct: {np.sum(only_model2)} "
              f"({np.sum(only_model2)/total:.2%})")
        print(f"  Both incorrect: {np.sum(both_wrong)} "
              f"({np.sum(both_wrong)/total:.2%})")
        
        # Show samples where models differ
        print(f"\n🔍 Samples where {model1_name} succeeds but {model2_name} fails:")
        only_model1_indices = np.where(only_model1)[0]
        if len(only_model1_indices) > 0:
            for i, idx in enumerate(only_model1_indices[:3], 1):
                print(f"  [{i}] True: {y[idx]}, {model1_name}: {y_pred1[idx]}, "
                      f"{model2_name}: {y_pred2[idx]}")
                print(f"      \"{texts[idx][:80]}...\"")
        else:
            print("  None found")
        
        return {
            'both_correct': int(np.sum(both_correct)),
            'only_model1_correct': int(np.sum(only_model1)),
            'only_model2_correct': int(np.sum(only_model2)),
            'both_incorrect': int(np.sum(both_wrong))
        }
    
    def export_analysis(self, filepath: str):
        """Export analysis results to JSON."""
        import json
        
        # Convert numpy types to Python native types for JSON serialization
        def convert_to_native(obj):
            """Recursively convert numpy types to Python native types."""
            if isinstance(obj, dict):
                return {str(k): convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_native(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        serializable_results = convert_to_native(self.analysis_results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"\n✓ Analysis exported to {filepath}")
