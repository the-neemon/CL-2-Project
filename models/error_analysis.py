"""
Error Analysis for Sentiment Classification Models.

Analyzes misclassified instances to understand model weaknesses,
identify error patterns, and provide insights for improvement.

Author: Naman
Phase: 3 - Integration & Analysis
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from collections import Counter
import json
from scipy import sparse


class ErrorAnalyzer:
    """
    Analyze incorrectly predicted samples to understand error patterns.
    
    Provides detailed analysis of:
    - Overall error statistics
    - Per-class error rates
    - Confidence distributions for errors
    - Text characteristics of misclassified samples
    - Common error patterns and examples
    """
    
    def __init__(self):
        """Initialize ErrorAnalyzer."""
        self.analysis_results = {}
    
    def analyze_errors(
        self,
        model,
        X,
        y_true: np.ndarray,
        texts: Optional[List[str]] = None,
        model_name: str = "Model",
        verbose: bool = True
    ) -> Dict:
        """
        Analyze incorrectly predicted samples.
        
        Args:
            model: Trained model with predict() and predict_proba() methods
            X: Feature matrix (can be sparse or dense)
            y_true: True labels
            texts: Optional list of original texts for analysis
            model_name: Name of the model for display
            verbose: Whether to print analysis to console
            
        Returns:
            Dictionary containing error analysis results
        """
        # Get predictions
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)
        
        # Identify incorrect predictions
        incorrect_mask = y_pred != y_true
        correct_mask = ~incorrect_mask
        
        # Get confidence scores (max probability)
        confidences = np.max(y_proba, axis=1)
        
        # Basic statistics
        total_samples = len(y_true)
        incorrect_count = np.sum(incorrect_mask)
        correct_count = total_samples - incorrect_count
        error_rate = incorrect_count / total_samples
        
        # Per-class error analysis
        unique_classes = np.unique(y_true)
        class_analysis = {}
        
        for cls in unique_classes:
            cls_mask = y_true == cls
            cls_total = np.sum(cls_mask)
            cls_incorrect = np.sum(incorrect_mask & cls_mask)
            cls_correct = cls_total - cls_incorrect
            
            class_analysis[int(cls)] = {
                'total': int(cls_total),
                'incorrect': int(cls_incorrect),
                'correct': int(cls_correct),
                'error_rate': float(cls_incorrect / cls_total) if cls_total > 0 else 0.0
            }
        
        # Confidence analysis for errors
        incorrect_confidences = confidences[incorrect_mask]
        correct_confidences = confidences[correct_mask]
        
        # Identify high-confidence errors (likely systematic issues)
        high_conf_threshold = 0.9
        high_conf_errors = np.sum((incorrect_mask) & (confidences >= high_conf_threshold))
        
        # Text length analysis (if texts provided)
        text_analysis = {}
        if texts is not None:
            text_lengths = np.array([len(str(t).split()) for t in texts])
            text_analysis = {
                'incorrect_length_mean': float(np.mean(text_lengths[incorrect_mask])) if np.any(incorrect_mask) else 0.0,
                'incorrect_length_median': float(np.median(text_lengths[incorrect_mask])) if np.any(incorrect_mask) else 0.0,
                'correct_length_mean': float(np.mean(text_lengths[correct_mask])) if np.any(correct_mask) else 0.0,
                'correct_length_median': float(np.median(text_lengths[correct_mask])) if np.any(correct_mask) else 0.0
            }
        
        # Confusion patterns
        confusion_patterns = {}
        for true_cls in unique_classes:
            for pred_cls in unique_classes:
                if true_cls != pred_cls:
                    pattern_mask = (y_true == true_cls) & (y_pred == pred_cls)
                    count = np.sum(pattern_mask)
                    if count > 0:
                        key = f"True_{int(true_cls)}_Pred_{int(pred_cls)}"
                        confusion_patterns[key] = {
                            'count': int(count),
                            'percentage': float(count / total_samples * 100)
                        }
        
        # Store results
        results = {
            'model_name': model_name,
            'total_samples': int(total_samples),
            'incorrect_count': int(incorrect_count),
            'correct_count': int(correct_count),
            'error_rate': float(error_rate),
            'accuracy': float(1 - error_rate),
            'class_analysis': class_analysis,
            'confidence_analysis': {
                'incorrect_mean': float(np.mean(incorrect_confidences)) if len(incorrect_confidences) > 0 else 0.0,
                'incorrect_median': float(np.median(incorrect_confidences)) if len(incorrect_confidences) > 0 else 0.0,
                'incorrect_std': float(np.std(incorrect_confidences)) if len(incorrect_confidences) > 0 else 0.0,
                'correct_mean': float(np.mean(correct_confidences)) if len(correct_confidences) > 0 else 0.0,
                'correct_median': float(np.median(correct_confidences)) if len(correct_confidences) > 0 else 0.0,
                'correct_std': float(np.std(correct_confidences)) if len(correct_confidences) > 0 else 0.0,
                'high_confidence_errors': int(high_conf_errors),
                'high_confidence_error_rate': float(high_conf_errors / incorrect_count) if incorrect_count > 0 else 0.0
            },
            'text_analysis': text_analysis,
            'confusion_patterns': confusion_patterns
        }
        
        self.analysis_results[model_name] = results
        
        if verbose:
            self._print_analysis(results, texts, y_true, y_pred, confidences, incorrect_mask)
        
        return results
    
    def _print_analysis(
        self,
        results: Dict,
        texts: Optional[List[str]],
        y_true: np.ndarray,
        y_pred: np.ndarray,
        confidences: np.ndarray,
        incorrect_mask: np.ndarray
    ):
        """Print formatted error analysis."""
        print("\n" + "="*70)
        print(f"ERROR ANALYSIS: {results['model_name']}")
        print("="*70)
        
        print(f"\n📊 Overall Statistics:")
        print(f"  Total samples: {results['total_samples']}")
        print(f"  Incorrect predictions: {results['incorrect_count']} ({results['error_rate']*100:.2f}%)")
        print(f"  Correct predictions: {results['correct_count']} ({results['accuracy']*100:.2f}%)")
        
        print(f"\n📈 Per-Class Error Analysis:")
        for cls, stats in results['class_analysis'].items():
            print(f"  Class {cls}:")
            print(f"    Total: {stats['total']}")
            print(f"    Incorrect: {stats['incorrect']} ({stats['error_rate']*100:.2f}%)")
            print(f"    Correct: {stats['correct']} ({(1-stats['error_rate'])*100:.2f}%)")
        
        conf_analysis = results['confidence_analysis']
        print(f"\n🎯 Confidence Analysis:")
        print(f"  Incorrect predictions:")
        print(f"    Mean confidence: {conf_analysis['incorrect_mean']:.4f}")
        print(f"    Median confidence: {conf_analysis['incorrect_median']:.4f}")
        print(f"    Std deviation: {conf_analysis['incorrect_std']:.4f}")
        print(f"  Correct predictions:")
        print(f"    Mean confidence: {conf_analysis['correct_mean']:.4f}")
        print(f"    Median confidence: {conf_analysis['correct_median']:.4f}")
        print(f"    Std deviation: {conf_analysis['correct_std']:.4f}")
        
        print(f"\n⚠️  High Confidence Errors (≥0.9):")
        print(f"  Count: {conf_analysis['high_confidence_errors']}")
        print(f"  Percentage of errors: {conf_analysis['high_confidence_error_rate']*100:.2f}%")
        
        if results['text_analysis']:
            print(f"\n📝 Text Length Analysis:")
            text_stats = results['text_analysis']
            print(f"  Incorrect predictions:")
            print(f"    Mean length: {text_stats['incorrect_length_mean']:.2f} words")
            print(f"    Median length: {text_stats['incorrect_length_median']:.2f} words")
            print(f"  Correct predictions:")
            print(f"    Mean length: {text_stats['correct_length_mean']:.2f} words")
            print(f"    Median length: {text_stats['correct_length_median']:.2f} words")
        
        if results['confusion_patterns']:
            print(f"\n🔀 Confusion Patterns:")
            sorted_patterns = sorted(
                results['confusion_patterns'].items(),
                key=lambda x: x[1]['count'],
                reverse=True
            )
            for pattern, stats in sorted_patterns[:5]:  # Top 5 patterns
                print(f"  {pattern}: {stats['count']} ({stats['percentage']:.2f}%)")
        
        # Show sample high-confidence errors
        if texts is not None:
            print(f"\n❌ Sample High-Confidence Errors:")
            high_conf_error_mask = incorrect_mask & (confidences >= 0.9)
            if np.any(high_conf_error_mask):
                error_indices = np.where(high_conf_error_mask)[0]
                sample_indices = error_indices[:min(5, len(error_indices))]
                
                for idx in sample_indices:
                    true_label = y_true[idx]
                    pred_label = y_pred[idx]
                    conf = confidences[idx]
                    text = texts[idx] if idx < len(texts) else "N/A"
                    print(f"\n  Text: '{text}'")
                    print(f"  True: {true_label}, Predicted: {pred_label}, Confidence: {conf:.4f}")
    
    def compare_model_errors(
        self,
        model1,
        model2,
        X,
        y_true: np.ndarray,
        texts: Optional[List[str]] = None,
        model1_name: str = "Model 1",
        model2_name: str = "Model 2",
        verbose: bool = True
    ) -> Dict:
        """
        Compare error patterns between two models.
        
        Args:
            model1: First trained model
            model2: Second trained model
            X: Feature matrix
            y_true: True labels
            texts: Optional list of original texts
            model1_name: Name of first model
            model2_name: Name of second model
            verbose: Whether to print comparison
            
        Returns:
            Dictionary containing comparison results
        """
        # Get predictions
        y_pred1 = model1.predict(X)
        y_pred2 = model2.predict(X)
        
        # Identify errors
        errors1 = y_pred1 != y_true
        errors2 = y_pred2 != y_true
        
        # Error overlap analysis
        both_wrong = errors1 & errors2
        only_model1_wrong = errors1 & ~errors2
        only_model2_wrong = ~errors1 & errors2
        both_correct = ~errors1 & ~errors2
        
        comparison = {
            'model1_name': model1_name,
            'model2_name': model2_name,
            'total_samples': int(len(y_true)),
            'both_wrong': int(np.sum(both_wrong)),
            'only_model1_wrong': int(np.sum(only_model1_wrong)),
            'only_model2_wrong': int(np.sum(only_model2_wrong)),
            'both_correct': int(np.sum(both_correct)),
            'model1_error_rate': float(np.mean(errors1)),
            'model2_error_rate': float(np.mean(errors2)),
            'error_overlap': float(np.sum(both_wrong) / max(np.sum(errors1), np.sum(errors2))) if max(np.sum(errors1), np.sum(errors2)) > 0 else 0.0
        }
        
        if verbose:
            print("\n" + "="*70)
            print(f"ERROR COMPARISON: {model1_name} vs {model2_name}")
            print("="*70)
            
            print(f"\n📊 Error Overlap Analysis:")
            print(f"  Both models wrong: {comparison['both_wrong']} ({comparison['both_wrong']/len(y_true)*100:.2f}%)")
            print(f"  Only {model1_name} wrong: {comparison['only_model1_wrong']} ({comparison['only_model1_wrong']/len(y_true)*100:.2f}%)")
            print(f"  Only {model2_name} wrong: {comparison['only_model2_wrong']} ({comparison['only_model2_wrong']/len(y_true)*100:.2f}%)")
            print(f"  Both models correct: {comparison['both_correct']} ({comparison['both_correct']/len(y_true)*100:.2f}%)")
            
            print(f"\n📈 Error Rates:")
            print(f"  {model1_name}: {comparison['model1_error_rate']*100:.2f}%")
            print(f"  {model2_name}: {comparison['model2_error_rate']*100:.2f}%")
            print(f"  Error overlap: {comparison['error_overlap']*100:.2f}%")
            
            # Show samples where only one model is wrong
            if texts is not None and np.any(only_model1_wrong):
                print(f"\n✨ Samples where only {model1_name} is wrong:")
                indices = np.where(only_model1_wrong)[0][:3]
                for idx in indices:
                    print(f"  Text: '{texts[idx]}'")
                    print(f"  True: {y_true[idx]}, {model1_name} predicted: {y_pred1[idx]}, {model2_name} predicted: {y_pred2[idx]}")
            
            if texts is not None and np.any(only_model2_wrong):
                print(f"\n✨ Samples where only {model2_name} is wrong:")
                indices = np.where(only_model2_wrong)[0][:3]
                for idx in indices:
                    print(f"  Text: '{texts[idx]}'")
                    print(f"  True: {y_true[idx]}, {model1_name} predicted: {y_pred1[idx]}, {model2_name} predicted: {y_pred2[idx]}")
        
        return comparison
    
    def export_analysis(self, output_path: str):
        """
        Export error analysis results to JSON.
        
        Args:
            output_path: Path to save JSON file
        """
        with open(output_path, 'w') as f:
            json.dump(self.analysis_results, f, indent=2)
        
        print(f"\n✓ Error analysis exported to {output_path}")
