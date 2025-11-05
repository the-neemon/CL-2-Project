import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import pickle
import os
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords

# Import our custom feature modules
from .contextual_features import ContextualFeatures
from .semantic_embeddings import SemanticEmbeddings
from .lexicon_scoring import LexiconBasedScoring

class FeatureExtractionPipeline:
    """
    Unified feature extraction pipeline that combines all feature types.
    """
    
    def __init__(self, 
                 config: Optional[Dict] = None,
                 cache_dir: str = './features_cache'):
        
        self.config = config or self._get_default_config()
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize feature extractors
        self.contextual_extractor = None
        self.semantic_extractor = None
        self.lexicon_extractor = None
        
        # Initialize traditional feature extractors
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        self.pos_tagger = None
        
        # Feature scalers
        self.scalers = {}
        
        # Feature names and dimensions
        self.feature_names = []
        self.feature_dimensions = {}
        
        # Setup NLTK
        self._setup_nltk()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for feature extraction."""
        return {
            'contextual_features': {
                'enabled': True,
                'include_negation': True,
                'include_intensifiers': True,
                'include_emphasis': True
            },
            'semantic_features': {
                'enabled': True,
                'use_word2vec': True,
                'use_glove': True,
                'embedding_dim_reduction': 10,
                'include_raw_embeddings': False
            },
            'lexicon_features': {
                'enabled': True,
                'use_vader': True,
                'use_nrc': True,
                'use_custom': True
            },
            'traditional_features': {
                'enabled': True,
                'use_tfidf': True,
                'use_bow': True,
                'use_pos_tags': True,
                'ngram_range': (1, 2),
                'max_features': 5000,
                'min_df': 2,
                'max_df': 0.95
            },
            'preprocessing': {
                'lowercase': True,
                'remove_stopwords': False,  # Keep for sentiment analysis
                'lemmatize': True,
                'remove_punctuation': False  # Keep for sentiment analysis
            },
            'scaling': {
                'method': 'standard',  # 'standard', 'minmax', 'none'
                'per_feature_type': True
            }
        }
    
    def _setup_nltk(self):
        """Setup NLTK components."""
        required_data = ['punkt', 'averaged_perceptron_tagger', 'stopwords', 'wordnet']
        
        for data_name in required_data:
            try:
                nltk.data.find(f'tokenizers/{data_name}')
            except LookupError:
                try:
                    nltk.data.find(f'taggers/{data_name}')
                except LookupError:
                    try:
                        nltk.data.find(f'corpora/{data_name}')
                    except LookupError:
                        try:
                            nltk.download(data_name, quiet=True)
                        except:
                            pass
    
    def initialize_extractors(self, 
                            init_contextual: bool = True,
                            init_semantic: bool = True, 
                            init_lexicon: bool = True):
        print("Initializing Feature Extraction Pipeline...")
        print("=" * 50)
        
        if init_contextual and self.config['contextual_features']['enabled']:
            print("\n1. Initializing Contextual Features Extractor...")
            self.contextual_extractor = ContextualFeatures()
            print("✓ Contextual features extractor ready")
        
        if init_semantic and self.config['semantic_features']['enabled']:
            print("\n2. Initializing Semantic Features Extractor...")
            self.semantic_extractor = SemanticEmbeddings()
            
            # Initialize embedding models based on config
            load_w2v = self.config['semantic_features']['use_word2vec']
            load_glove = self.config['semantic_features']['use_glove']
            
            if load_w2v or load_glove:
                self.semantic_extractor.initialize_models(load_w2v, load_glove)
            print("✓ Semantic features extractor ready")
        
        if init_lexicon and self.config['lexicon_features']['enabled']:
            print("\n3. Initializing Lexicon-Based Extractor...")
            self.lexicon_extractor = LexiconBasedScoring()
            self.lexicon_extractor.initialize_lexicons()
            print("✓ Lexicon-based extractor ready")
        
        # Initialize traditional feature extractors
        if self.config['traditional_features']['enabled']:
            print("\n4. Initializing Traditional Features...")
            self._initialize_traditional_extractors()
            print("✓ Traditional features ready")
        
        print("\n" + "=" * 50)
        print("✓ Feature Extraction Pipeline Initialized Successfully!")
        print("=" * 50)
    
    def _initialize_traditional_extractors(self):
        """Initialize traditional feature extractors."""
        config = self.config['traditional_features']
        
        if config['use_tfidf']:
            self.tfidf_vectorizer = TfidfVectorizer(
                ngram_range=config['ngram_range'],
                max_features=config['max_features'],
                min_df=config['min_df'],
                max_df=config['max_df'],
                stop_words='english' if self.config['preprocessing']['remove_stopwords'] else None
            )
        
        if config['use_bow']:
            self.count_vectorizer = CountVectorizer(
                ngram_range=config['ngram_range'],
                max_features=config['max_features'],
                min_df=config['min_df'],
                max_df=config['max_df'],
                stop_words='english' if self.config['preprocessing']['remove_stopwords'] else None
            )
    
    def extract_contextual_features(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Extract contextual features from texts."""
        if isinstance(texts, str):
            texts = [texts]
        
        if self.contextual_extractor is None:
            return np.array([])
        
        features = []
        for text in texts:
            feature_vector = self.contextual_extractor.get_feature_vector(text)
            features.append(feature_vector)
        
        return np.array(features)
    
    def extract_semantic_features(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Extract semantic features from texts."""
        if isinstance(texts, str):
            texts = [texts]
        
        if self.semantic_extractor is None:
            return np.array([])
        
        features = []
        for text in texts:
            feature_vector = self.semantic_extractor.get_feature_vector(
                text, 
                include_embeddings=self.config['semantic_features']['include_raw_embeddings']
            )
            features.append(feature_vector)
        
        return np.array(features)
    
    def extract_lexicon_features(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Extract lexicon-based features from texts."""
        if isinstance(texts, str):
            texts = [texts]
        
        if self.lexicon_extractor is None:
            return np.array([])
        
        features = []
        for text in texts:
            feature_vector = self.lexicon_extractor.get_feature_vector(text)
            features.append(feature_vector)
        
        return np.array(features)
    
    def extract_traditional_features(self, texts: Union[str, List[str]], 
                                   fit_vectorizers: bool = False) -> Dict[str, np.ndarray]:
        """Extract traditional features from texts."""
        if isinstance(texts, str):
            texts = [texts]
        
        features = {}
        
        # TF-IDF features
        if self.tfidf_vectorizer is not None:
            if fit_vectorizers:
                tfidf_features = self.tfidf_vectorizer.fit_transform(texts)
            else:
                tfidf_features = self.tfidf_vectorizer.transform(texts)
            features['tfidf'] = tfidf_features.toarray()
        
        # Bag of words features
        if self.count_vectorizer is not None:
            if fit_vectorizers:
                bow_features = self.count_vectorizer.fit_transform(texts)
            else:
                bow_features = self.count_vectorizer.transform(texts)
            features['bow'] = bow_features.toarray()
        
        # POS tag features
        if self.config['traditional_features']['use_pos_tags']:
            pos_features = self._extract_pos_features(texts)
            features['pos'] = pos_features
        
        return features
    
    def _extract_pos_features(self, texts: List[str]) -> np.ndarray:
        """Extract POS tag features."""
        pos_features = []
        
        # Common POS tags to track
        pos_tags_to_track = ['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
                            'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'IN', 'DT', 'CC', 'UH']
        
        for text in texts:
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            
            # Count POS tags
            pos_counts = defaultdict(int)
            for _, pos in pos_tags:
                if pos in pos_tags_to_track:
                    pos_counts[pos] += 1
            
            # Create feature vector
            feature_vector = [pos_counts[pos] / len(tokens) if len(tokens) > 0 else 0 
                            for pos in pos_tags_to_track]
            pos_features.append(feature_vector)
        
        return np.array(pos_features)
    
    def extract_all_features(self, texts: Union[str, List[str]], 
                           fit_vectorizers: bool = False,
                           return_dict: bool = False) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Extract all enabled features from texts.
        
        Args:
            texts: Input text(s)
            fit_vectorizers: Whether to fit traditional vectorizers
            return_dict: Whether to return dict of feature types or combined array
            
        Returns:
            Combined feature array or dict of feature arrays
        """
        if isinstance(texts, str):
            texts = [texts]
        
        all_features = {}
        
        # Extract contextual features
        if self.config['contextual_features']['enabled']:
            contextual_features = self.extract_contextual_features(texts)
            if contextual_features.size > 0:
                all_features['contextual'] = contextual_features
        
        # Extract semantic features
        if self.config['semantic_features']['enabled']:
            semantic_features = self.extract_semantic_features(texts)
            if semantic_features.size > 0:
                all_features['semantic'] = semantic_features
        
        # Extract lexicon features
        if self.config['lexicon_features']['enabled']:
            lexicon_features = self.extract_lexicon_features(texts)
            if lexicon_features.size > 0:
                all_features['lexicon'] = lexicon_features
        
        # Extract traditional features
        if self.config['traditional_features']['enabled']:
            traditional_features = self.extract_traditional_features(texts, fit_vectorizers)
            all_features.update(traditional_features)
        
        if return_dict:
            return all_features
        
        # Combine all features
        feature_arrays = []
        feature_names = []
        
        for feature_type, features in all_features.items():
            if features.size > 0:
                feature_arrays.append(features)
                
                # Add feature names
                if feature_type == 'contextual':
                    feature_names.extend([f'contextual_{name}' for name in 
                                        self.contextual_extractor.get_feature_names()])
                elif feature_type == 'lexicon':
                    feature_names.extend([f'lexicon_{name}' for name in 
                                        self.lexicon_extractor.get_feature_names()])
                elif feature_type == 'semantic':
                    # Add semantic feature names
                    n_semantic_features = features.shape[1]
                    feature_names.extend([f'semantic_{i}' for i in range(n_semantic_features)])
                elif feature_type in ['tfidf', 'bow']:
                    vectorizer = self.tfidf_vectorizer if feature_type == 'tfidf' else self.count_vectorizer
                    if vectorizer:
                        feature_names.extend([f'{feature_type}_{name}' for name in 
                                            vectorizer.get_feature_names_out()])
                elif feature_type == 'pos':
                    pos_tags = ['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
                               'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'IN', 'DT', 'CC', 'UH']
                    feature_names.extend([f'pos_{tag}' for tag in pos_tags])
        
        # Store feature names
        self.feature_names = feature_names
        
        if feature_arrays:
            return np.hstack(feature_arrays)
        else:
            return np.array([])
    
    def fit_scalers(self, features: Union[np.ndarray, Dict[str, np.ndarray]]):
        """Fit feature scalers on training data."""
        if self.config['scaling']['method'] == 'none':
            return
        
        scaler_class = StandardScaler if self.config['scaling']['method'] == 'standard' else MinMaxScaler
        
        if isinstance(features, dict):
            # Scale each feature type separately
            for feature_type, feature_array in features.items():
                if feature_array.size > 0:
                    scaler = scaler_class()
                    scaler.fit(feature_array)
                    self.scalers[feature_type] = scaler
        else:
            # Scale all features together
            if features.size > 0:
                scaler = scaler_class()
                scaler.fit(features)
                self.scalers['all'] = scaler
    
    def transform_features(self, features: Union[np.ndarray, Dict[str, np.ndarray]]) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Transform features using fitted scalers."""
        if self.config['scaling']['method'] == 'none' or not self.scalers:
            return features
        
        if isinstance(features, dict):
            scaled_features = {}
            for feature_type, feature_array in features.items():
                if feature_type in self.scalers and feature_array.size > 0:
                    scaled_features[feature_type] = self.scalers[feature_type].transform(feature_array)
                else:
                    scaled_features[feature_type] = feature_array
            return scaled_features
        else:
            if 'all' in self.scalers and features.size > 0:
                return self.scalers['all'].transform(features)
            return features
    
    def save_pipeline(self, filepath: str):
        """Save the feature extraction pipeline."""
        pipeline_data = {
            'config': self.config,
            'feature_names': self.feature_names,
            'scalers': self.scalers,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'count_vectorizer': self.count_vectorizer
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(pipeline_data, f)
        
        print(f"✓ Pipeline saved to {filepath}")
    
    def load_pipeline(self, filepath: str):
        """Load a saved feature extraction pipeline."""
        with open(filepath, 'rb') as f:
            pipeline_data = pickle.load(f)
        
        self.config = pipeline_data['config']
        self.feature_names = pipeline_data['feature_names']
        self.scalers = pipeline_data['scalers']
        self.tfidf_vectorizer = pipeline_data['tfidf_vectorizer']
        self.count_vectorizer = pipeline_data['count_vectorizer']
        
        print(f"✓ Pipeline loaded from {filepath}")
    
    def get_feature_info(self) -> Dict[str, any]:
        """Get information about extracted features."""
        info = {
            'total_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'config': self.config,
            'extractors_initialized': {
                'contextual': self.contextual_extractor is not None,
                'semantic': self.semantic_extractor is not None,
                'lexicon': self.lexicon_extractor is not None,
                'traditional': any([
                    self.tfidf_vectorizer is not None,
                    self.count_vectorizer is not None
                ])
            }
        }
        
        return info
