"""
Traditional machine learning models for sentiment classification.

Implements Naive Bayes and Logistic Regression classifiers
with evaluation metrics and k-fold cross-validation.

Author: Naman
Phase: 2 - Traditional Models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import pickle
import time

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)


class SentimentClassifier:
    """
    Base wrapper for sentiment classification models.
    
    Provides consistent interface for training, evaluation,
    and prediction across different model types.
    """
    
    def __init__(self, model_type: str = 'naive_bayes', **model_params):
        """
        Initialize classifier.
        
        Args:
            model_type: Type of model ('naive_bayes' or 'logistic_regression')
            **model_params: Additional parameters for the model
        """
        self.model_type = model_type
        self.model_params = model_params
        self.model = None
        self.is_fitted = False
        self.classes_ = None
        self.training_time_ = None
        
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the underlying sklearn model."""
        if self.model_type == 'naive_bayes':
            # Multinomial Naive Bayes (good for text features)
            self.model = MultinomialNB(
                alpha=self.model_params.get('alpha', 1.0)
            )
        elif self.model_type == 'logistic_regression':
            # Logistic Regression with regularization
            self.model = LogisticRegression(
                penalty=self.model_params.get('penalty', 'l2'),
                C=self.model_params.get('C', 1.0),
                solver=self.model_params.get('solver', 'lbfgs'),
                max_iter=self.model_params.get('max_iter', 1000),
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SentimentClassifier':
        """
        Train the classifier.
        
        Args:
            X: Feature matrix [n_samples, n_features]
            y: Labels [n_samples]
            
        Returns:
            self
        """
        print(f"\nTraining {self.model_type}...")
        print(f"  Training samples: {len(X)}")
        print(f"  Features: {X.shape[1]}")
        
        start_time = time.time()
        
        self.model.fit(X, y)
        
        self.training_time_ = time.time() - start_time
        self.is_fitted = True
        self.classes_ = self.model.classes_
        
        print(f"✓ Training completed in {self.training_time_:.2f}s")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for samples.
        
        Args:
            X: Feature matrix [n_samples, n_features]
            
        Returns:
            Predicted labels [n_samples]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix [n_samples, n_features]
            
        Returns:
            Class probabilities [n_samples, n_classes]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict_proba(X)
    
    def evaluate(self, 
                 X: np.ndarray, 
                 y: np.ndarray,
                 verbose: bool = True) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Feature matrix
            y: True labels
            verbose: Whether to print results
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        # Predictions
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y, y_pred, average='weighted', zero_division=0)
        }
        
        # Add AUC if binary classification
        if len(self.classes_) == 2:
            metrics['roc_auc'] = roc_auc_score(y, y_proba[:, 1])
        
        if verbose:
            print(f"\n{self.model_type.upper()} Evaluation:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1_score']:.4f}")
            if 'roc_auc' in metrics:
                print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def get_classification_report(self, 
                                  X: np.ndarray, 
                                  y: np.ndarray) -> str:
        """
        Get detailed classification report.
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Classification report string
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        y_pred = self.predict(X)
        return classification_report(y, y_pred, zero_division=0)
    
    def get_confusion_matrix(self, 
                            X: np.ndarray, 
                            y: np.ndarray) -> np.ndarray:
        """
        Get confusion matrix.
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Confusion matrix
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        y_pred = self.predict(X)
        return confusion_matrix(y, y_pred)
    
    def save(self, filepath: str) -> None:
        """
        Save trained model to disk.
        
        Args:
            filepath: Path to save file
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        save_data = {
            'model': self.model,
            'model_type': self.model_type,
            'model_params': self.model_params,
            'classes_': self.classes_,
            'training_time_': self.training_time_
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"✓ Model saved to {filepath}")
    
    def load(self, filepath: str) -> 'SentimentClassifier':
        """
        Load trained model from disk.
        
        Args:
            filepath: Path to saved file
            
        Returns:
            self
        """
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.model = save_data['model']
        self.model_type = save_data['model_type']
        self.model_params = save_data['model_params']
        self.classes_ = save_data['classes_']
        self.training_time_ = save_data['training_time_']
        
        self.is_fitted = True
        
        print(f"✓ Model loaded from {filepath}")
        
        return self


def cross_validate_model(model: SentimentClassifier,
                        X: np.ndarray,
                        y: np.ndarray,
                        cv: int = 5,
                        verbose: bool = True) -> Dict[str, Any]:
    """
    Perform k-fold cross-validation on a model.
    
    Args:
        model: Initialized SentimentClassifier
        X: Feature matrix
        y: Labels
        cv: Number of folds (default: 5)
        verbose: Whether to print results
        
    Returns:
        Dictionary of cross-validation results
    """
    if verbose:
        print(f"\nPerforming {cv}-fold cross-validation...")
        print(f"  Model: {model.model_type}")
        print(f"  Samples: {len(X)}")
    
    # Define scoring metrics
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision_weighted',
        'recall': 'recall_weighted',
        'f1': 'f1_weighted'
    }
    
    # Perform cross-validation
    cv_results = cross_validate(
        model.model,
        X, y,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring=scoring,
        n_jobs=-1,
        return_train_score=True
    )
    
    # Calculate mean and std for each metric
    results = {}
    for metric in scoring.keys():
        test_scores = cv_results[f'test_{metric}']
        train_scores = cv_results[f'train_{metric}']
        
        results[f'{metric}_mean'] = np.mean(test_scores)
        results[f'{metric}_std'] = np.std(test_scores)
        results[f'{metric}_train_mean'] = np.mean(train_scores)
        
        if verbose:
            print(f"  {metric.capitalize():10s}: {results[f'{metric}_mean']:.4f} (+/- {results[f'{metric}_std']:.4f})")
    
    # Store raw results
    results['raw_results'] = cv_results
    results['n_folds'] = cv
    
    return results


def compare_models(models: Dict[str, SentimentClassifier],
                  X_train: np.ndarray,
                  y_train: np.ndarray,
                  X_test: np.ndarray,
                  y_test: np.ndarray,
                  cv: int = 5) -> pd.DataFrame:
    """
    Train and compare multiple models.
    
    Args:
        models: Dictionary of {name: model} to compare
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        cv: Number of cross-validation folds
        
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training and evaluating: {name}")
        print(f"{'='*60}")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate on test set
        test_metrics = model.evaluate(X_test, y_test, verbose=True)
        
        # Cross-validation
        cv_results = cross_validate_model(model, X_train, y_train, cv=cv, verbose=True)
        
        # Combine results
        result = {
            'model': name,
            'test_accuracy': test_metrics['accuracy'],
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall'],
            'test_f1': test_metrics['f1_score'],
            'cv_accuracy_mean': cv_results['accuracy_mean'],
            'cv_accuracy_std': cv_results['accuracy_std'],
            'cv_f1_mean': cv_results['f1_mean'],
            'cv_f1_std': cv_results['f1_std'],
            'training_time': model.training_time_
        }
        
        if 'roc_auc' in test_metrics:
            result['test_roc_auc'] = test_metrics['roc_auc']
        
        results.append(result)
    
    # Create comparison DataFrame
    df = pd.DataFrame(results)
    
    print(f"\n{'='*60}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(df.to_string(index=False))
    
    return df
