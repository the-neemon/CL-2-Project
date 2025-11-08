"""
Traditional feature extraction for sentiment analysis.

Implements N-gram features (unigrams, bigrams) and POS tagging
for sentiment-bearing words.

Author: Naman
Phase: 2 - Traditional Feature Engineering
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union
from collections import Counter
import pickle
from pathlib import Path

import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


class TraditionalFeatureExtractor:
    """
    Extract traditional features for sentiment analysis.
    
    Features include:
    - N-grams (unigrams and bigrams)
    - POS tags (focusing on sentiment-bearing words)
    - TF-IDF and Bag-of-Words representations
    """
    
    def __init__(self,
                 ngram_range: Tuple[int, int] = (1, 2),
                 max_features: int = 5000,
                 min_df: int = 2,
                 max_df: float = 0.95,
                 use_idf: bool = True):
        """
        Initialize feature extractor.
        
        Args:
            ngram_range: Range of n-grams (default: unigrams + bigrams)
            max_features: Maximum number of features to extract
            min_df: Minimum document frequency
            max_df: Maximum document frequency (proportion)
            use_idf: Whether to use TF-IDF (vs simple counts)
        """
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.use_idf = use_idf
        
        # Initialize vectorizers
        if use_idf:
            self.vectorizer = TfidfVectorizer(
                ngram_range=ngram_range,
                max_features=max_features,
                min_df=min_df,
                max_df=max_df,
                lowercase=True,
                strip_accents='unicode'
            )
        else:
            self.vectorizer = CountVectorizer(
                ngram_range=ngram_range,
                max_features=max_features,
                min_df=min_df,
                max_df=max_df,
                lowercase=True,
                strip_accents='unicode'
            )
        
        # POS tags that often carry sentiment
        self.sentiment_pos_tags = {
            'JJ', 'JJR', 'JJS',    # Adjectives
            'RB', 'RBR', 'RBS',    # Adverbs
            'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',  # Verbs
            'NN', 'NNS', 'NNP', 'NNPS'  # Nouns
        }
        
        # Download required NLTK data
        self._setup_nltk()
        
        # Storage for fitted data
        self.is_fitted = False
        self.feature_names_ = None
        self.vocabulary_ = None
    
    def _setup_nltk(self) -> None:
        """Download required NLTK resources."""
        required_data = [
            'averaged_perceptron_tagger',
            'universal_tagset',
            'punkt'
        ]
        
        for data_name in required_data:
            try:
                nltk.data.find(f'taggers/{data_name}')
            except LookupError:
                try:
                    nltk.data.find(f'tokenizers/{data_name}')
                except LookupError:
                    try:
                        nltk.download(data_name, quiet=True)
                    except Exception:
                        pass
    
    def extract_pos_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract POS tag features from texts.
        
        Focuses on sentiment-bearing POS categories:
        - Adjectives (positive/negative descriptors)
        - Adverbs (intensifiers/modifiers)
        - Verbs (action/state)
        - Nouns (entities/topics)
        
        Args:
            texts: List of preprocessed texts
            
        Returns:
            Array of POS feature counts (shape: [n_texts, n_pos_tags])
        """
        pos_features = []
        
        for text in texts:
            # Tokenize and tag
            tokens = word_tokenize(text.lower())
            tagged = pos_tag(tokens)
            
            # Count sentiment-bearing POS tags
            pos_counts = Counter()
            for word, tag in tagged:
                if tag in self.sentiment_pos_tags:
                    pos_counts[tag] += 1
            
            # Create feature vector (normalized by text length)
            total_words = len(tokens) if len(tokens) > 0 else 1
            feature_vector = [
                pos_counts.get(tag, 0) / total_words
                for tag in sorted(self.sentiment_pos_tags)
            ]
            
            pos_features.append(feature_vector)
        
        return np.array(pos_features)
    
    def fit(self, texts: List[str]) -> 'TraditionalFeatureExtractor':
        """
        Fit the feature extractor on training texts.
        
        Args:
            texts: List of preprocessed training texts
            
        Returns:
            self
        """
        print(f"Fitting {self.__class__.__name__}...")
        print(f"  N-gram range: {self.ngram_range}")
        print(f"  Max features: {self.max_features}")
        
        # Fit vectorizer
        self.vectorizer.fit(texts)
        
        # Store feature names and vocabulary
        self.feature_names_ = self.vectorizer.get_feature_names_out()
        self.vocabulary_ = self.vectorizer.vocabulary_
        
        self.is_fitted = True
        
        print(f"✓ Fitted on {len(texts)} texts")
        print(f"✓ Vocabulary size: {len(self.vocabulary_)}")
        print(f"✓ Feature vector size: {len(self.feature_names_)}")
        
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to feature vectors.
        
        Args:
            texts: List of preprocessed texts
            
        Returns:
            Combined feature matrix [n_texts, n_features]
        """
        if not self.is_fitted:
            raise ValueError("Extractor must be fitted before transform. Call fit() first.")
        
        # Extract N-gram features
        ngram_features = self.vectorizer.transform(texts).toarray()
        
        # Extract POS features
        pos_features = self.extract_pos_features(texts)
        
        # Combine features
        combined_features = np.hstack([ngram_features, pos_features])
        
        return combined_features
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit extractor and transform texts in one step.
        
        Args:
            texts: List of preprocessed texts
            
        Returns:
            Feature matrix [n_texts, n_features]
        """
        self.fit(texts)
        return self.transform(texts)
    
    def get_feature_names(self) -> List[str]:
        """
        Get names of all features.
        
        Returns:
            List of feature names
        """
        if not self.is_fitted:
            return []
        
        # N-gram feature names
        ngram_names = list(self.feature_names_)
        
        # POS feature names
        pos_names = [f'pos_{tag}' for tag in sorted(self.sentiment_pos_tags)]
        
        return ngram_names + pos_names
    
    def get_top_features(self, 
                        n: int = 20,
                        feature_importances: Optional[np.ndarray] = None) -> List[Tuple[str, float]]:
        """
        Get top N most important features.
        
        Args:
            n: Number of top features to return
            feature_importances: Optional array of feature importances
                                (e.g., from model coefficients)
            
        Returns:
            List of (feature_name, importance) tuples
        """
        if not self.is_fitted:
            return []
        
        feature_names = self.get_feature_names()
        
        if feature_importances is None:
            # If no importances provided, return first N features
            return [(name, 1.0) for name in feature_names[:n]]
        
        # Get top N by absolute importance
        indices = np.argsort(np.abs(feature_importances))[-n:][::-1]
        
        return [
            (feature_names[i], feature_importances[i])
            for i in indices
        ]
    
    def save(self, filepath: str) -> None:
        """
        Save fitted extractor to disk.
        
        Args:
            filepath: Path to save file
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted extractor")
        
        save_data = {
            'vectorizer': self.vectorizer,
            'ngram_range': self.ngram_range,
            'max_features': self.max_features,
            'min_df': self.min_df,
            'max_df': self.max_df,
            'use_idf': self.use_idf,
            'sentiment_pos_tags': self.sentiment_pos_tags,
            'feature_names_': self.feature_names_,
            'vocabulary_': self.vocabulary_
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"✓ Feature extractor saved to {filepath}")
    
    def load(self, filepath: str) -> 'TraditionalFeatureExtractor':
        """
        Load fitted extractor from disk.
        
        Args:
            filepath: Path to saved file
            
        Returns:
            self
        """
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.vectorizer = save_data['vectorizer']
        self.ngram_range = save_data['ngram_range']
        self.max_features = save_data['max_features']
        self.min_df = save_data['min_df']
        self.max_df = save_data['max_df']
        self.use_idf = save_data['use_idf']
        self.sentiment_pos_tags = save_data['sentiment_pos_tags']
        self.feature_names_ = save_data['feature_names_']
        self.vocabulary_ = save_data['vocabulary_']
        
        self.is_fitted = True
        
        print(f"✓ Feature extractor loaded from {filepath}")
        
        return self
