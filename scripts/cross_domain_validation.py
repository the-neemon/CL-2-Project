"""
Phase 3: Cross-Domain Validation & Qualitative Analysis

This script performs:
1. Cross-domain validation: Train on Sentiment140, test on Twitter US Airline Sentiment
2. Qualitative analysis: Manual inspection of representative tweets with semantic interpretation

Author: Phase 3 Implementation
Date: November 2025
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from preprocessing.data_loader import SentimentDataLoader
from preprocessing.preprocessing import TweetPreprocessor
from features.feature_pipeline import FeatureExtractionPipeline
from models.traditional_models import SentimentClassifier


class CrossDomainValidator:
    """
    Performs cross-domain validation by training on one dataset
    and testing on another (domain transfer).
    """
    
    def __init__(self, dataset_dir: str, output_dir: str = "analysis_results"):
        """
        Initialize cross-domain validator.
        
        Args:
            dataset_dir: Directory containing datasets
            output_dir: Directory for saving results
        """
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.data_loader = SentimentDataLoader(dataset_dir)
        self.preprocessor = TweetPreprocessor()
        self.feature_pipeline = FeatureExtractionPipeline()
        
        self.results = {}
        
    def load_and_prepare_datasets(
        self,
        source_sample_size: int = 50000,
        target_binary: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load source (Sentiment140) and target (Airline) datasets.
        
        Args:
            source_sample_size: Number of samples from Sentiment140
            target_binary: If True, convert airline to binary classification
            
        Returns:
            Tuple of (source_df, target_df)
        """
        print("="*80)
        print("LOADING DATASETS")
        print("="*80)
        
        # Load Sentiment140 (source domain)
        print("\n[1/2] Loading Sentiment140 dataset (source domain)...")
        source_df = self.data_loader.load_sentiment140(sample_size=source_sample_size)
        
        # Load Airline sentiment (target domain)
        print("\n[2/2] Loading Twitter US Airline Sentiment dataset (target domain)...")
        target_df = self.data_loader.load_airline_sentiment()
        
        if target_binary:
            # Convert to binary: negative (0), positive (2), remove neutral (1)
            print("\nConverting to binary classification (removing neutral)...")
            original_size = len(target_df)
            target_df = target_df[target_df['sentiment'].isin([0, 2])].copy()
            # Remap: 0->0 (negative), 2->1 (positive)
            target_df['sentiment'] = (target_df['sentiment'] == 2).astype(int)
            print(f"  Removed {original_size - len(target_df)} neutral tweets")
            print(f"  Final size: {len(target_df)}")
            print(f"  Distribution:\n{target_df['sentiment'].value_counts()}")
        
        return source_df, target_df
    
    def preprocess_datasets(
        self,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocess both datasets.
        
        Args:
            source_df: Source domain dataframe
            target_df: Target domain dataframe
            
        Returns:
            Tuple of preprocessed dataframes
        """
        print("\n" + "="*80)
        print("PREPROCESSING")
        print("="*80)
        
        print("\n[1/2] Preprocessing source domain (Sentiment140)...")
        source_df = self.data_loader.preprocess_dataframe(
            source_df,
            self.preprocessor,
            text_column='text',
            output_column='processed_text'
        )
        
        print("\n[2/2] Preprocessing target domain (Airline)...")
        target_df = self.data_loader.preprocess_dataframe(
            target_df,
            self.preprocessor,
            text_column='text',
            output_column='processed_text'
        )
        
        return source_df, target_df
    
    def extract_features(
        self,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
        max_features: int = 5000
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract features from both datasets using unified pipeline.
        
        Args:
            source_df: Preprocessed source dataframe
            target_df: Preprocessed target dataframe
            max_features: Maximum number of TF-IDF features
            
        Returns:
            Tuple of (X_source, y_source, X_target, y_target)
        """
        print("\n" + "="*80)
        print("FEATURE EXTRACTION")
        print("="*80)
        
        print("\n[1/2] Extracting features from source domain...")
        X_source = self.feature_pipeline.fit_transform(
            source_df['processed_text'].tolist(),
            max_features=max_features
        )
        y_source = source_df['sentiment'].values
        
        print(f"  Source features shape: {X_source.shape}")
        
        print("\n[2/2] Extracting features from target domain (using fitted pipeline)...")
        X_target = self.feature_pipeline.transform(
            target_df['processed_text'].tolist()
        )
        y_target = target_df['sentiment'].values
        
        print(f"  Target features shape: {X_target.shape}")
        
        return X_source, y_source, X_target, y_target
    
    def train_and_evaluate_models(
        self,
        X_source: np.ndarray,
        y_source: np.ndarray,
        X_target: np.ndarray,
        y_target: np.ndarray
    ) -> Dict[str, Any]:
        """
        Train models on source domain and evaluate on target domain.
        
        Args:
            X_source: Source domain features
            y_source: Source domain labels
            X_target: Target domain features
            y_target: Target domain labels
            
        Returns:
            Dictionary with cross-domain validation results
        """
        print("\n" + "="*80)
        print("CROSS-DOMAIN VALIDATION")
        print("="*80)
        
        # Define models to evaluate
        models_config = [
            ('Logistic Regression', 'logistic_regression', {}),
            ('Logistic Regression (L1)', 'logistic_regression', {'penalty': 'l1', 'solver': 'liblinear'}),
            ('Naive Bayes', 'naive_bayes', {}),
            ('Random Forest', 'random_forest', {'n_estimators': 100, 'max_depth': 20})
        ]
        
        results = {
            'cross_domain_results': [],
            'source_domain': 'Sentiment140',
            'target_domain': 'Twitter US Airline Sentiment',
            'timestamp': datetime.now().isoformat()
        }
        
        for model_name, model_type, params in models_config:
            print(f"\n{'='*60}")
            print(f"Model: {model_name}")
            print('='*60)
            
            # Train on source domain
            print(f"\nTraining on {results['source_domain']}...")
            classifier = SentimentClassifier(model_type=model_type, **params)
            classifier.fit(X_source, y_source)
            
            # Evaluate on source domain (in-domain performance)
            print(f"\nIn-domain evaluation (source -> source):")
            y_pred_source = classifier.predict(X_source)
            source_metrics = {
                'accuracy': accuracy_score(y_source, y_pred_source),
                'precision': precision_score(y_source, y_pred_source, average='weighted'),
                'recall': recall_score(y_source, y_pred_source, average='weighted'),
                'f1_score': f1_score(y_source, y_pred_source, average='weighted')
            }
            
            print(f"  Accuracy:  {source_metrics['accuracy']:.4f}")
            print(f"  Precision: {source_metrics['precision']:.4f}")
            print(f"  Recall:    {source_metrics['recall']:.4f}")
            print(f"  F1-Score:  {source_metrics['f1_score']:.4f}")
            
            # Evaluate on target domain (cross-domain performance)
            print(f"\nCross-domain evaluation (source -> target):")
            y_pred_target = classifier.predict(X_target)
            target_metrics = {
                'accuracy': accuracy_score(y_target, y_pred_target),
                'precision': precision_score(y_target, y_pred_target, average='weighted'),
                'recall': recall_score(y_target, y_pred_target, average='weighted'),
                'f1_score': f1_score(y_target, y_pred_target, average='weighted')
            }
            
            print(f"  Accuracy:  {target_metrics['accuracy']:.4f}")
            print(f"  Precision: {target_metrics['precision']:.4f}")
            print(f"  Recall:    {target_metrics['recall']:.4f}")
            print(f"  F1-Score:  {target_metrics['f1_score']:.4f}")
            
            # Calculate domain gap
            domain_gap = source_metrics['f1_score'] - target_metrics['f1_score']
            print(f"\n  Domain Gap (F1): {domain_gap:.4f}")
            
            # Store results
            model_results = {
                'model_name': model_name,
                'model_type': model_type,
                'model_params': params,
                'source_metrics': source_metrics,
                'target_metrics': target_metrics,
                'domain_gap_f1': float(domain_gap),
                'predictions': {
                    'y_true': y_target.tolist(),
                    'y_pred': y_pred_target.tolist()
                }
            }
            
            results['cross_domain_results'].append(model_results)
        
        # Find best model
        best_model_idx = max(
            range(len(results['cross_domain_results'])),
            key=lambda i: results['cross_domain_results'][i]['target_metrics']['f1_score']
        )
        results['best_model'] = results['cross_domain_results'][best_model_idx]['model_name']
        results['best_f1_score'] = results['cross_domain_results'][best_model_idx]['target_metrics']['f1_score']
        
        print("\n" + "="*80)
        print("CROSS-DOMAIN VALIDATION SUMMARY")
        print("="*80)
        print(f"\nBest Model: {results['best_model']}")
        print(f"Best Cross-Domain F1-Score: {results['best_f1_score']:.4f}")
        
        return results
    
    def save_results(self, results: Dict[str, Any], filename: str = "cross_domain_validation.json"):
        """Save cross-domain validation results to file."""
        output_path = self.output_dir / filename
        
        # Remove predictions from saved results (too large)
        results_to_save = results.copy()
        for model_result in results_to_save['cross_domain_results']:
            model_result.pop('predictions', None)
        
        with open(output_path, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
        
        return output_path
    
    def run_cross_domain_validation(
        self,
        source_sample_size: int = 50000,
        max_features: int = 5000
    ) -> Dict[str, Any]:
        """
        Run complete cross-domain validation pipeline.
        
        Args:
            source_sample_size: Number of samples from source domain
            max_features: Maximum TF-IDF features
            
        Returns:
            Dictionary with validation results
        """
        # Load datasets
        source_df, target_df = self.load_and_prepare_datasets(
            source_sample_size=source_sample_size,
            target_binary=True
        )
        
        # Preprocess
        source_df, target_df = self.preprocess_datasets(source_df, target_df)
        
        # Extract features
        X_source, y_source, X_target, y_target = self.extract_features(
            source_df, target_df, max_features=max_features
        )
        
        # Train and evaluate
        results = self.train_and_evaluate_models(
            X_source, y_source, X_target, y_target
        )
        
        # Store dataframes for qualitative analysis
        self.source_df = source_df
        self.target_df = target_df
        self.X_target = X_target
        self.y_target = y_target
        self.results = results
        
        # Save results
        self.save_results(results)
        
        return results


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Phase 3: Cross-Domain Validation'
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='datasets',
        help='Directory containing datasets'
    )
    parser.add_argument(
        '--source-sample-size',
        type=int,
        default=50000,
        help='Number of samples from Sentiment140 (source domain)'
    )
    parser.add_argument(
        '--max-features',
        type=int,
        default=5000,
        help='Maximum number of TF-IDF features'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='analysis_results',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("PHASE 3: CROSS-DOMAIN VALIDATION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Source Domain: Sentiment140")
    print(f"  Target Domain: Twitter US Airline Sentiment")
    print(f"  Source Sample Size: {args.source_sample_size:,}")
    print(f"  Max Features: {args.max_features:,}")
    print(f"  Output Directory: {args.output_dir}")
    
    # Initialize validator
    validator = CrossDomainValidator(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir
    )
    
    # Run cross-domain validation
    results = validator.run_cross_domain_validation(
        source_sample_size=args.source_sample_size,
        max_features=args.max_features
    )
    
    print("\n" + "="*80)
    print("CROSS-DOMAIN VALIDATION COMPLETE")
    print("="*80)
    print(f"\nBest performing model: {results['best_model']}")
    print(f"Cross-domain F1-score: {results['best_f1_score']:.4f}")
    print(f"\nResults saved to: {args.output_dir}/cross_domain_validation.json")


if __name__ == "__main__":
    main()
