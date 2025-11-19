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
from sklearn.ensemble import RandomForestClassifier
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
            model_type: Type of model ('naive_bayes', 'logistic_regression', or 'random_forest')
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
        elif self.model_type == 'random_forest':
            # Random Forest Classifier
            self.model = RandomForestClassifier(
                n_estimators=self.model_params.get('n_estimators', 100),
                max_depth=self.model_params.get('max_depth', None),
                min_samples_split=self.model_params.get('min_samples_split', 2),
                min_samples_leaf=self.model_params.get('min_samples_leaf', 1),
                max_features=self.model_params.get('max_features', 'sqrt'),
                random_state=42,
                n_jobs=-1,
                verbose=0
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
        print(f"  Training samples: {X.shape[0]}")
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
        print(f"  Samples: {X.shape[0]}")
    
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


class FeatureAblationStudy:
    """
    Framework for testing individual feature types through ablation studies.
    
    Tests model performance with:
    - Each feature type individually
    - All features combined
    - All features minus one (leave-one-out ablation)
    """
    
    def __init__(self, model_type: str = 'random_forest', **model_params):
        """
        Initialize ablation study.
        
        Args:
            model_type: Type of model to use for ablation tests
            **model_params: Parameters for the model
        """
        self.model_type = model_type
        self.model_params = model_params
        self.results = []
        self.feature_types = []
        
    def run_ablation_study(self,
                          feature_dict: Dict[str, np.ndarray],
                          y_train: np.ndarray,
                          y_test: np.ndarray,
                          cv: int = 5,
                          test_individual: bool = True,
                          test_combined: bool = True,
                          test_leave_one_out: bool = True) -> pd.DataFrame:
        """
        Run complete feature ablation study.
        
        Args:
            feature_dict: Dictionary of {feature_type: feature_array}
                         Must contain both train and test splits
                         Format: {'train': {type: array}, 'test': {type: array}}
            y_train: Training labels
            y_test: Test labels
            cv: Number of cross-validation folds
            test_individual: Test each feature type individually
            test_combined: Test all features combined
            test_leave_one_out: Test with each feature type removed
            
        Returns:
            DataFrame with ablation study results
        """
        print("\n" + "="*70)
        print("FEATURE ABLATION STUDY")
        print("="*70)
        print(f"Model: {self.model_type}")
        print(f"Cross-validation folds: {cv}")
        print(f"Training samples: {len(y_train)}")
        print(f"Test samples: {len(y_test)}")
        print("="*70)
        
        # Extract train and test features
        if 'train' in feature_dict and 'test' in feature_dict:
            train_features = feature_dict['train']
            test_features = feature_dict['test']
        else:
            # Assume feature_dict contains the train features directly
            train_features = feature_dict
            test_features = feature_dict  # Will need separate test dict
            
        self.feature_types = list(train_features.keys())
        
        # Test individual feature types
        if test_individual:
            print("\n" + "-"*70)
            print("TESTING INDIVIDUAL FEATURE TYPES")
            print("-"*70)
            for feature_type in self.feature_types:
                self._test_feature_combination(
                    feature_types=[feature_type],
                    train_features=train_features,
                    test_features=test_features,
                    y_train=y_train,
                    y_test=y_test,
                    cv=cv,
                    experiment_name=f"Individual: {feature_type}"
                )
        
        # Test all features combined
        if test_combined:
            print("\n" + "-"*70)
            print("TESTING ALL FEATURES COMBINED")
            print("-"*70)
            self._test_feature_combination(
                feature_types=self.feature_types,
                train_features=train_features,
                test_features=test_features,
                y_train=y_train,
                y_test=y_test,
                cv=cv,
                experiment_name="All Features"
            )
        
        # Test leave-one-out (all except one feature type)
        if test_leave_one_out and len(self.feature_types) > 1:
            print("\n" + "-"*70)
            print("TESTING LEAVE-ONE-OUT ABLATION")
            print("-"*70)
            for excluded_type in self.feature_types:
                remaining_types = [ft for ft in self.feature_types if ft != excluded_type]
                self._test_feature_combination(
                    feature_types=remaining_types,
                    train_features=train_features,
                    test_features=test_features,
                    y_train=y_train,
                    y_test=y_test,
                    cv=cv,
                    experiment_name=f"All except {excluded_type}"
                )
        
        # Create results DataFrame
        results_df = pd.DataFrame(self.results)
        
        # Print summary
        self._print_summary(results_df)
        
        return results_df
    
    def _test_feature_combination(self,
                                 feature_types: List[str],
                                 train_features: Dict[str, np.ndarray],
                                 test_features: Dict[str, np.ndarray],
                                 y_train: np.ndarray,
                                 y_test: np.ndarray,
                                 cv: int,
                                 experiment_name: str) -> None:
        """
        Test a specific combination of feature types.
        
        Args:
            feature_types: List of feature types to combine
            train_features: Dictionary of training features
            test_features: Dictionary of test features
            y_train: Training labels
            y_test: Test labels
            cv: Number of CV folds
            experiment_name: Name for this experiment
        """
        print(f"\n{experiment_name}")
        print(f"  Features: {', '.join(feature_types)}")
        
        # Combine selected features
        X_train = self._combine_features(train_features, feature_types)
        X_test = self._combine_features(test_features, feature_types)
        
        print(f"  Feature dimensions: {X_train.shape[1]}")
        
        # Initialize and train model
        model = SentimentClassifier(
            model_type=self.model_type,
            **self.model_params
        )
        
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Evaluate on test set
        test_metrics = model.evaluate(X_test, y_test, verbose=False)
        
        # Cross-validation on training set
        cv_results = cross_validate_model(
            model, X_train, y_train, cv=cv, verbose=False
        )
        
        # Store results
        result = {
            'experiment': experiment_name,
            'feature_types': ', '.join(feature_types),
            'n_features': X_train.shape[1],
            'test_accuracy': test_metrics['accuracy'],
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall'],
            'test_f1': test_metrics['f1_score'],
            'cv_accuracy_mean': cv_results['accuracy_mean'],
            'cv_accuracy_std': cv_results['accuracy_std'],
            'cv_f1_mean': cv_results['f1_mean'],
            'cv_f1_std': cv_results['f1_std'],
            'training_time': training_time
        }
        
        if 'roc_auc' in test_metrics:
            result['test_roc_auc'] = test_metrics['roc_auc']
        
        self.results.append(result)
        
        # Print brief summary
        print(f"  ✓ Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  ✓ Test F1-Score: {test_metrics['f1_score']:.4f}")
        print(f"  ✓ CV Accuracy: {cv_results['accuracy_mean']:.4f} (+/- {cv_results['accuracy_std']:.4f})")
        print(f"  ✓ Training time: {training_time:.2f}s")
    
    def _combine_features(self,
                         features_dict: Dict[str, np.ndarray],
                         feature_types: List[str]) -> np.ndarray:
        """
        Combine multiple feature types into a single array.
        
        Args:
            features_dict: Dictionary of feature arrays
            feature_types: List of feature types to combine
            
        Returns:
            Combined feature array
        """
        feature_arrays = []
        
        for feature_type in feature_types:
            if feature_type in features_dict:
                feature_array = features_dict[feature_type]
                if feature_array.size > 0:
                    # Ensure 2D array
                    if len(feature_array.shape) == 1:
                        feature_array = feature_array.reshape(-1, 1)
                    feature_arrays.append(feature_array)
        
        if not feature_arrays:
            raise ValueError(f"No valid features found for types: {feature_types}")
        
        return np.hstack(feature_arrays)
    
    def _print_summary(self, results_df: pd.DataFrame) -> None:
        """Print summary of ablation study results."""
        print("\n" + "="*70)
        print("ABLATION STUDY SUMMARY")
        print("="*70)
        
        # Sort by test F1 score
        sorted_df = results_df.sort_values('test_f1', ascending=False)
        
        print("\nTop 5 Feature Combinations by Test F1-Score:")
        print("-"*70)
        
        display_cols = ['experiment', 'n_features', 'test_accuracy', 'test_f1', 'cv_f1_mean']
        print(sorted_df[display_cols].head().to_string(index=False))
        
        # Find best individual feature
        individual_results = results_df[results_df['experiment'].str.startswith('Individual:')]
        if not individual_results.empty:
            best_individual = individual_results.loc[individual_results['test_f1'].idxmax()]
            print(f"\nBest Individual Feature Type:")
            print(f"  {best_individual['experiment']}")
            print(f"  Test F1: {best_individual['test_f1']:.4f}")
            print(f"  Test Accuracy: {best_individual['test_accuracy']:.4f}")
        
        # Compare all features vs best individual
        all_features = results_df[results_df['experiment'] == 'All Features']
        if not all_features.empty and not individual_results.empty:
            all_f1 = all_features.iloc[0]['test_f1']
            best_ind_f1 = individual_results['test_f1'].max()
            improvement = ((all_f1 - best_ind_f1) / best_ind_f1) * 100
            
            print(f"\nImprovement from combining all features:")
            print(f"  All Features F1: {all_f1:.4f}")
            print(f"  Best Individual F1: {best_ind_f1:.4f}")
            print(f"  Improvement: {improvement:+.2f}%")
        
        print("="*70)
    
    def save_results(self, filepath: str) -> None:
        """
        Save ablation study results to CSV.
        
        Args:
            filepath: Path to save results
        """
        if not self.results:
            print("No results to save!")
            return
        
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(filepath, index=False)
        print(f"✓ Ablation study results saved to {filepath}")
    
    def plot_results(self, save_path: Optional[str] = None) -> None:
        """
        Create visualization of ablation study results.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.results:
            print("No results to plot!")
            return
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            results_df = pd.DataFrame(self.results)
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Feature Ablation Study Results', fontsize=16, fontweight='bold')
            
            # Plot 1: Test F1-Score comparison
            ax1 = axes[0, 0]
            sorted_df = results_df.sort_values('test_f1', ascending=True)
            ax1.barh(range(len(sorted_df)), sorted_df['test_f1'])
            ax1.set_yticks(range(len(sorted_df)))
            ax1.set_yticklabels(sorted_df['experiment'], fontsize=8)
            ax1.set_xlabel('Test F1-Score')
            ax1.set_title('Test F1-Score by Feature Combination')
            ax1.grid(axis='x', alpha=0.3)
            
            # Plot 2: Test Accuracy comparison
            ax2 = axes[0, 1]
            sorted_df = results_df.sort_values('test_accuracy', ascending=True)
            ax2.barh(range(len(sorted_df)), sorted_df['test_accuracy'])
            ax2.set_yticks(range(len(sorted_df)))
            ax2.set_yticklabels(sorted_df['experiment'], fontsize=8)
            ax2.set_xlabel('Test Accuracy')
            ax2.set_title('Test Accuracy by Feature Combination')
            ax2.grid(axis='x', alpha=0.3)
            
            # Plot 3: Number of features vs performance
            ax3 = axes[1, 0]
            ax3.scatter(results_df['n_features'], results_df['test_f1'], s=100, alpha=0.6)
            ax3.set_xlabel('Number of Features')
            ax3.set_ylabel('Test F1-Score')
            ax3.set_title('Feature Count vs Performance')
            ax3.grid(alpha=0.3)
            
            # Add labels
            for idx, row in results_df.iterrows():
                ax3.annotate(row['experiment'], 
                           (row['n_features'], row['test_f1']),
                           fontsize=6, alpha=0.7)
            
            # Plot 4: Cross-validation scores
            ax4 = axes[1, 1]
            x_pos = range(len(results_df))
            ax4.errorbar(x_pos, results_df['cv_f1_mean'], 
                        yerr=results_df['cv_f1_std'],
                        fmt='o', capsize=5, capthick=2)
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(results_df['experiment'], rotation=45, ha='right', fontsize=7)
            ax4.set_ylabel('CV F1-Score (mean ± std)')
            ax4.set_title('Cross-Validation F1-Scores')
            ax4.grid(alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✓ Plot saved to {save_path}")
            
            plt.show()
            
        except ImportError:
            print("Matplotlib not available. Install it to create visualizations.")
