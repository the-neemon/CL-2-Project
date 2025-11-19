"""
Phase 3: Qualitative Analysis - Manual Inspection with Semantic Interpretation

This script performs detailed qualitative analysis by:
1. Identifying representative tweets (correct/incorrect predictions)
2. Analyzing semantic features (negations, intensifiers, contextual patterns)
3. Interpreting model decisions through feature importance
4. Generating human-readable insights

Author: Phase 3 Implementation
Date: November 2025
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
from collections import Counter, defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from preprocessing.data_loader import SentimentDataLoader
from preprocessing.preprocessing import TweetPreprocessor
from features.feature_pipeline import FeatureExtractionPipeline
from features.contextual_features import ContextualFeatures
from features.lexicon_scoring import LexiconBasedScoring
from models.traditional_models import SentimentClassifier


class QualitativeAnalyzer:
    """
    Performs qualitative analysis on sentiment predictions with
    semantic interpretation.
    """
    
    def __init__(self, output_dir: str = "analysis_results"):
        """
        Initialize qualitative analyzer.
        
        Args:
            output_dir: Directory for saving analysis results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.preprocessor = TweetPreprocessor()
        self.contextual_extractor = ContextualFeatures()
        self.lexicon_scorer = LexiconBasedScoring()
        
        self.analysis_results = {}
    
    def extract_semantic_features(self, text: str) -> Dict[str, Any]:
        """
        Extract detailed semantic features from a tweet.
        
        Args:
            text: Raw tweet text
            
        Returns:
            Dictionary with semantic features
        """
        # Preprocess
        processed = self.preprocessor.preprocess(text)
        tokens = processed.split()
        
        # Contextual features
        contextual = self.contextual_extractor.extract_features([text])[0]
        
        # Lexicon scores
        lexicon = self.lexicon_scorer.score_tweet(text)
        
        # Additional semantic patterns
        semantic_info = {
            'original_text': text,
            'processed_text': processed,
            'token_count': len(tokens),
            
            # Contextual patterns
            'negation_count': contextual['negation_count'],
            'negation_contexts': self._find_negation_contexts(text),
            'intensifier_count': contextual['intensifier_count'],
            'intensifiers_found': self._find_intensifiers(text),
            'all_caps_count': contextual['all_caps_count'],
            'exclamation_count': contextual['exclamation_count'],
            'question_count': contextual['question_count'],
            
            # Lexicon scores
            'vader_compound': lexicon['vader_compound'],
            'vader_positive': lexicon['vader_pos'],
            'vader_negative': lexicon['vader_neg'],
            'vader_neutral': lexicon['vader_neu'],
            
            # Emotional indicators
            'has_strong_positive': lexicon['vader_compound'] > 0.5,
            'has_strong_negative': lexicon['vader_compound'] < -0.5,
            'emotionally_neutral': -0.1 < lexicon['vader_compound'] < 0.1,
            
            # Complexity indicators
            'avg_word_length': np.mean([len(w) for w in tokens]) if tokens else 0,
            'has_mentions': '@' in text,
            'has_hashtags': '#' in text,
            'has_urls': 'http' in text.lower() or 'www' in text.lower()
        }
        
        return semantic_info
    
    def _find_negation_contexts(self, text: str) -> List[str]:
        """Find phrases containing negations."""
        negations = ['not', "n't", 'no', 'never', 'neither', 'nor', 'nothing', 'nowhere']
        text_lower = text.lower()
        words = text_lower.split()
        
        contexts = []
        for i, word in enumerate(words):
            if any(neg in word for neg in negations):
                start = max(0, i - 2)
                end = min(len(words), i + 3)
                context = ' '.join(words[start:end])
                contexts.append(context)
        
        return contexts
    
    def _find_intensifiers(self, text: str) -> List[str]:
        """Find intensifiers in text."""
        intensifiers = [
            'very', 'really', 'extremely', 'absolutely', 'totally',
            'completely', 'highly', 'utterly', 'incredibly', 'especially',
            'particularly', 'remarkably', 'exceptionally'
        ]
        
        text_lower = text.lower()
        words = text_lower.split()
        
        found = []
        for word in words:
            for intensifier in intensifiers:
                if intensifier in word:
                    found.append(intensifier)
        
        return list(set(found))
    
    def analyze_prediction_samples(
        self,
        df: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray = None,
        n_samples: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze representative samples of predictions.
        
        Args:
            df: DataFrame with original tweets
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
            n_samples: Number of samples per category
            
        Returns:
            Dictionary with analysis results
        """
        print("\n" + "="*80)
        print("QUALITATIVE ANALYSIS: REPRESENTATIVE SAMPLES")
        print("="*80)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_samples': len(y_true),
            'correct_predictions': int(np.sum(y_true == y_pred)),
            'incorrect_predictions': int(np.sum(y_true != y_pred)),
            'accuracy': float(np.mean(y_true == y_pred)),
            'sample_analyses': []
        }
        
        # Define sample categories
        categories = [
            ('True Negative - Correct', (y_true == 0) & (y_pred == 0)),
            ('True Positive - Correct', (y_true == 1) & (y_pred == 1)),
            ('False Positive - Error', (y_true == 0) & (y_pred == 1)),
            ('False Negative - Error', (y_true == 1) & (y_pred == 0))
        ]
        
        for category_name, mask in categories:
            print(f"\n{'='*60}")
            print(f"Category: {category_name}")
            print('='*60)
            
            indices = np.where(mask)[0]
            if len(indices) == 0:
                print("  No samples in this category")
                continue
            
            print(f"  Total samples: {len(indices)}")
            
            # Sample randomly or by confidence
            if y_proba is not None and len(indices) > n_samples:
                # Get confidence for these samples
                confidences = np.max(y_proba[indices], axis=1)
                # Sample from different confidence ranges
                sampled_indices = self._stratified_confidence_sample(
                    indices, confidences, n_samples
                )
            else:
                sampled_indices = np.random.choice(
                    indices,
                    size=min(n_samples, len(indices)),
                    replace=False
                )
            
            category_samples = []
            
            for idx in sampled_indices:
                text = df.iloc[idx]['text']
                true_label = int(y_true[idx])
                pred_label = int(y_pred[idx])
                confidence = float(np.max(y_proba[idx])) if y_proba is not None else None
                
                # Extract semantic features
                semantic_info = self.extract_semantic_features(text)
                
                sample_analysis = {
                    'index': int(idx),
                    'true_label': true_label,
                    'predicted_label': pred_label,
                    'confidence': confidence,
                    'semantic_features': semantic_info
                }
                
                category_samples.append(sample_analysis)
                
                # Print sample
                print(f"\n  Sample #{len(category_samples)}:")
                print(f"    Text: {text[:100]}{'...' if len(text) > 100 else ''}")
                print(f"    True: {'Positive' if true_label == 1 else 'Negative'}, "
                      f"Pred: {'Positive' if pred_label == 1 else 'Negative'}")
                if confidence:
                    print(f"    Confidence: {confidence:.3f}")
                print(f"    Negations: {semantic_info['negation_count']} "
                      f"{semantic_info['negation_contexts']}")
                print(f"    Intensifiers: {semantic_info['intensifiers_found']}")
                print(f"    VADER Score: {semantic_info['vader_compound']:.3f}")
            
            results['sample_analyses'].append({
                'category': category_name,
                'sample_count': len(category_samples),
                'samples': category_samples
            })
        
        return results
    
    def _stratified_confidence_sample(
        self,
        indices: np.ndarray,
        confidences: np.ndarray,
        n_samples: int
    ) -> np.ndarray:
        """Sample from different confidence ranges."""
        # Divide into confidence ranges
        high_conf = indices[confidences >= 0.8]
        med_conf = indices[(confidences >= 0.6) & (confidences < 0.8)]
        low_conf = indices[confidences < 0.6]
        
        # Sample proportionally
        samples = []
        ranges = [high_conf, med_conf, low_conf]
        samples_per_range = n_samples // 3
        
        for range_indices in ranges:
            if len(range_indices) > 0:
                n = min(samples_per_range, len(range_indices))
                sampled = np.random.choice(range_indices, size=n, replace=False)
                samples.extend(sampled)
        
        # Fill remaining slots if needed
        while len(samples) < n_samples and len(samples) < len(indices):
            remaining = np.setdiff1d(indices, samples)
            if len(remaining) > 0:
                samples.append(np.random.choice(remaining))
            else:
                break
        
        return np.array(samples)
    
    def analyze_semantic_patterns(
        self,
        df: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze semantic patterns across correct/incorrect predictions.
        
        Args:
            df: DataFrame with tweets
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Pattern analysis results
        """
        print("\n" + "="*80)
        print("SEMANTIC PATTERN ANALYSIS")
        print("="*80)
        
        correct_mask = y_true == y_pred
        incorrect_mask = ~correct_mask
        
        patterns = {
            'correct_predictions': {},
            'incorrect_predictions': {},
            'comparison': {}
        }
        
        # Analyze both groups
        for group_name, mask in [('correct_predictions', correct_mask),
                                  ('incorrect_predictions', incorrect_mask)]:
            group_texts = df[mask]['text'].tolist()
            
            # Aggregate semantic features
            negation_counts = []
            intensifier_counts = []
            vader_scores = []
            text_lengths = []
            
            for text in group_texts[:1000]:  # Sample for efficiency
                semantic = self.extract_semantic_features(text)
                negation_counts.append(semantic['negation_count'])
                intensifier_counts.append(semantic['intensifier_count'])
                vader_scores.append(semantic['vader_compound'])
                text_lengths.append(semantic['token_count'])
            
            patterns[group_name] = {
                'sample_size': len(group_texts),
                'avg_negations': float(np.mean(negation_counts)),
                'avg_intensifiers': float(np.mean(intensifier_counts)),
                'avg_vader_score': float(np.mean(vader_scores)),
                'avg_text_length': float(np.mean(text_lengths)),
                'strong_positive_ratio': float(np.mean([s > 0.5 for s in vader_scores])),
                'strong_negative_ratio': float(np.mean([s < -0.5 for s in vader_scores])),
                'neutral_ratio': float(np.mean([abs(s) < 0.1 for s in vader_scores]))
            }
        
        # Compare patterns
        print("\nPattern Comparison:")
        print("-" * 60)
        
        for metric in ['avg_negations', 'avg_intensifiers', 'avg_vader_score', 'avg_text_length']:
            correct_val = patterns['correct_predictions'][metric]
            incorrect_val = patterns['incorrect_predictions'][metric]
            diff = correct_val - incorrect_val
            
            patterns['comparison'][metric] = {
                'correct': correct_val,
                'incorrect': incorrect_val,
                'difference': float(diff)
            }
            
            print(f"  {metric}:")
            print(f"    Correct:   {correct_val:.3f}")
            print(f"    Incorrect: {incorrect_val:.3f}")
            print(f"    Difference: {diff:+.3f}")
        
        return patterns
    
    def generate_insights(
        self,
        sample_analysis: Dict[str, Any],
        pattern_analysis: Dict[str, Any]
    ) -> List[str]:
        """
        Generate human-readable insights from analysis.
        
        Args:
            sample_analysis: Results from sample analysis
            pattern_analysis: Results from pattern analysis
            
        Returns:
            List of insight strings
        """
        insights = []
        
        # Overall performance
        insights.append(
            f"Overall Accuracy: {sample_analysis['accuracy']:.2%} "
            f"({sample_analysis['correct_predictions']}/{sample_analysis['total_samples']})"
        )
        
        # Negation patterns
        neg_diff = pattern_analysis['comparison']['avg_negations']['difference']
        if abs(neg_diff) > 0.1:
            if neg_diff > 0:
                insights.append(
                    f"Correctly classified tweets have {abs(neg_diff):.2f} more negations on average, "
                    "suggesting the model handles negation well."
                )
            else:
                insights.append(
                    f"Incorrectly classified tweets have {abs(neg_diff):.2f} more negations on average, "
                    "indicating negation handling may need improvement."
                )
        
        # Intensifier patterns
        int_diff = pattern_analysis['comparison']['avg_intensifiers']['difference']
        if abs(int_diff) > 0.05:
            if int_diff > 0:
                insights.append(
                    f"Intensifiers are more common in correctly classified tweets, "
                    "suggesting they help with accurate sentiment detection."
                )
            else:
                insights.append(
                    f"Intensifiers appear more in misclassified tweets, "
                    "which may indicate over-emphasis causes confusion."
                )
        
        # VADER score patterns
        vader_diff = pattern_analysis['comparison']['avg_vader_score']['difference']
        insights.append(
            f"VADER sentiment scores are {'higher' if vader_diff > 0 else 'lower'} "
            f"for correctly classified tweets (diff: {vader_diff:+.3f})."
        )
        
        # Text length
        len_diff = pattern_analysis['comparison']['avg_text_length']['difference']
        if abs(len_diff) > 1:
            insights.append(
                f"{'Shorter' if len_diff < 0 else 'Longer'} tweets are easier to classify correctly "
                f"(avg length diff: {len_diff:+.1f} tokens)."
            )
        
        return insights
    
    def save_analysis(
        self,
        sample_analysis: Dict[str, Any],
        pattern_analysis: Dict[str, Any],
        insights: List[str],
        filename: str = "qualitative_analysis.json"
    ):
        """Save qualitative analysis results."""
        output_path = self.output_dir / filename
        
        results = {
            'sample_analysis': sample_analysis,
            'pattern_analysis': pattern_analysis,
            'insights': insights,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nAnalysis saved to: {output_path}")
        
        return output_path
    
    def run_full_analysis(
        self,
        df: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray = None,
        n_samples: int = 10
    ) -> Dict[str, Any]:
        """
        Run complete qualitative analysis pipeline.
        
        Args:
            df: DataFrame with tweets
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities
            n_samples: Samples per category
            
        Returns:
            Complete analysis results
        """
        # Sample analysis
        sample_analysis = self.analyze_prediction_samples(
            df, y_true, y_pred, y_proba, n_samples
        )
        
        # Pattern analysis
        pattern_analysis = self.analyze_semantic_patterns(
            df, y_true, y_pred
        )
        
        # Generate insights
        insights = self.generate_insights(sample_analysis, pattern_analysis)
        
        print("\n" + "="*80)
        print("KEY INSIGHTS")
        print("="*80)
        for i, insight in enumerate(insights, 1):
            print(f"\n{i}. {insight}")
        
        # Save results
        self.save_analysis(sample_analysis, pattern_analysis, insights)
        
        return {
            'sample_analysis': sample_analysis,
            'pattern_analysis': pattern_analysis,
            'insights': insights
        }


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Phase 3: Qualitative Analysis'
    )
    parser.add_argument(
        '--predictions-file',
        type=str,
        help='Path to saved predictions file (from cross-domain validation)'
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='datasets',
        help='Directory containing datasets'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=10,
        help='Number of samples to analyze per category'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='analysis_results',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("PHASE 3: QUALITATIVE ANALYSIS")
    print("="*80)
    
    # For demo purposes, load airline dataset and make simple predictions
    from preprocessing.data_loader import SentimentDataLoader
    from features.feature_pipeline import FeaturePipeline
    
    print("\nLoading airline sentiment dataset...")
    loader = SentimentDataLoader(args.dataset_dir)
    df = loader.load_airline_sentiment()
    
    # Convert to binary
    df = df[df['sentiment'].isin([0, 2])].copy()
    df['sentiment'] = (df['sentiment'] == 2).astype(int)
    
    # Sample for analysis
    df = df.sample(n=1000, random_state=42).reset_index(drop=True)
    
    print("\nPreprocessing and extracting features...")
    preprocessor = TweetPreprocessor()
    df = loader.preprocess_dataframe(df, preprocessor)
    
    feature_pipeline = FeaturePipeline()
    X = feature_pipeline.fit_transform(df['processed_text'].tolist(), max_features=5000)
    y = df['sentiment'].values
    
    print("\nTraining model for qualitative analysis...")
    model = SentimentClassifier(model_type='logistic_regression')
    model.fit(X, y)
    
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    
    print("\nRunning qualitative analysis...")
    analyzer = QualitativeAnalyzer(output_dir=args.output_dir)
    results = analyzer.run_full_analysis(
        df, y, y_pred, y_proba, n_samples=args.n_samples
    )
    
    print("\n" + "="*80)
    print("QUALITATIVE ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
