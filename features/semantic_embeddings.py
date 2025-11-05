import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gensim.downloader as api
import numpy as np
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# Suppress gensim logging
logging.getLogger('gensim').setLevel(logging.WARNING)


class SemanticEmbeddings:
    def __init__(self,
                 word2vec_model: str = 'word2vec-google-news-300',
                 glove_model: str = 'glove-wiki-gigaword-300',
                 cache_dir: str = './embeddings_cache'):
        self.word2vec_model_name = word2vec_model
        self.glove_model_name = glove_model
        self.cache_dir = cache_dir
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Model storage
        self.word2vec_model = None
        self.glove_model = None
        
        # Common sentiment-related words for reference
        self.sentiment_words = {
            'positive': [
                'good', 'great', 'excellent', 'amazing', 'wonderful',
                'fantastic', 'awesome', 'brilliant', 'perfect',
                'outstanding', 'superb', 'magnificent'
            ],
            'negative': [
                'bad', 'terrible', 'awful', 'horrible', 'disgusting',
                'disappointing', 'pathetic', 'dreadful', 'appalling',
                'abysmal', 'atrocious', 'deplorable'
            ],
            'neutral': [
                'okay', 'fine', 'average', 'normal', 'standard',
                'typical', 'ordinary'
            ]
        }
        
    def load_word2vec(self, force_download: bool = False) -> bool:
        cache_path = os.path.join(
            self.cache_dir,
            f"{self.word2vec_model_name.replace('-', '_')}.pkl"
        )
        
        try:
            # Try to load from cache first
            if not force_download and os.path.exists(cache_path):
                print(f"Loading Word2Vec from cache: {cache_path}")
                with open(cache_path, 'rb') as f:
                    self.word2vec_model = pickle.load(f)
                print("✓ Word2Vec loaded from cache")
                return True
            
            # Download and load model
            print(f"Downloading Word2Vec model: {self.word2vec_model_name}")
            print("This may take a while for the first time...")
            
            self.word2vec_model = api.load(self.word2vec_model_name)
            
            # Cache the model
            print(f"Caching Word2Vec model to: {cache_path}")
            with open(cache_path, 'wb') as f:
                pickle.dump(self.word2vec_model, f)
            
            print("✓ Word2Vec model loaded and cached successfully")
            return True
            
        except Exception as e:
            print(f"✗ Failed to load Word2Vec model: {str(e)}")
            print("Using a smaller model as fallback...")
            try:
                # Try a smaller model as fallback
                self.word2vec_model = api.load('word2vec-google-news-300')
                return True
            except Exception:
                print("✗ Could not load any Word2Vec model")
                return False
    
    def load_glove(self, force_download: bool = False) -> bool:
        cache_path = os.path.join(
            self.cache_dir,
            f"{self.glove_model_name.replace('-', '_')}.pkl"
        )
        
        try:
            # Try to load from cache first
            if not force_download and os.path.exists(cache_path):
                print(f"Loading GloVe from cache: {cache_path}")
                with open(cache_path, 'rb') as f:
                    self.glove_model = pickle.load(f)
                print("✓ GloVe loaded from cache")
                return True
            
            # Download and load model
            print(f"Downloading GloVe model: {self.glove_model_name}")
            print("This may take a while for the first time...")
            
            self.glove_model = api.load(self.glove_model_name)
            
            # Cache the model
            print(f"Caching GloVe model to: {cache_path}")
            with open(cache_path, 'wb') as f:
                pickle.dump(self.glove_model, f)
            
            print("✓ GloVe model loaded and cached successfully")
            return True
            
        except Exception as e:
            print(f"✗ Failed to load GloVe model: {str(e)}")
            print("Using a smaller model as fallback...")
            try:
                # Try a smaller model as fallback
                self.glove_model = api.load('glove-wiki-gigaword-100')
                return True
            except Exception:
                print("✗ Could not load any GloVe model")
                return False
    
    def get_word_embedding(self, word: str,
                          model_type: str = 'word2vec') -> Optional[np.ndarray]:
        model = (self.word2vec_model if model_type == 'word2vec'
                 else self.glove_model)
        
        if model is None:
            return None
        
        try:
            return model[word.lower()]
        except KeyError:
            return None
    
    def get_sentence_embedding(self, text: str,
                              model_type: str = 'word2vec',
                              method: str = 'mean') -> Optional[np.ndarray]:
        tokens = word_tokenize(text.lower())
        embeddings = []
        
        for token in tokens:
            emb = self.get_word_embedding(token, model_type)
            if emb is not None:
                embeddings.append(emb)
        
        if not embeddings:
            return None
        
        embeddings = np.array(embeddings)
        
        if method == 'mean':
            return np.mean(embeddings, axis=0)
        elif method == 'max':
            return np.max(embeddings, axis=0)
        elif method == 'sum':
            return np.sum(embeddings, axis=0)
        else:
            return np.mean(embeddings, axis=0)
    
    def calculate_semantic_similarity(self, text1: str, text2: str,
                                     model_type: str = 'word2vec') -> float:
        emb1 = self.get_sentence_embedding(text1, model_type)
        emb2 = self.get_sentence_embedding(text2, model_type)
        
        if emb1 is None or emb2 is None:
            return 0.0
        
        # Calculate cosine similarity
        similarity = cosine_similarity([emb1], [emb2])[0][0]
        return max(0, similarity)  # Ensure non-negative
    
    def get_sentiment_similarity_scores(self, text: str,
                                        model_type: str = 'word2vec'
                                        ) -> Dict[str, float]:
        text_embedding = self.get_sentence_embedding(text, model_type)
        
        if text_embedding is None:
            return {'positive_sim': 0.0, 'negative_sim': 0.0, 'neutral_sim': 0.0}
        
        sentiment_scores = {}
        
        for sentiment, words in self.sentiment_words.items():
            similarities = []
            
            for word in words:
                word_emb = self.get_word_embedding(word, model_type)
                if word_emb is not None:
                    sim = cosine_similarity([text_embedding], [word_emb])[0][0]
                    similarities.append(max(0, sim))
            
            # Average similarity to sentiment category
            avg_sim = np.mean(similarities) if similarities else 0.0
            sentiment_scores[f'{sentiment}_sim'] = avg_sim
        
        return sentiment_scores
    
    def extract_embedding_features(self, text: str) -> Dict[str, any]:
        features = {}
        
        # Try both models if available
        for model_type in ['word2vec', 'glove']:
            model = self.word2vec_model if model_type == 'word2vec' else self.glove_model
            
            if model is None:
                continue
            
            # Sentence embeddings with different aggregation methods
            mean_emb = self.get_sentence_embedding(text, model_type, 'mean')
            max_emb = self.get_sentence_embedding(text, model_type, 'max')
            
            if mean_emb is not None:
                # Basic embedding statistics
                features[f'{model_type}_mean_norm'] = np.linalg.norm(mean_emb)
                features[f'{model_type}_mean_sum'] = np.sum(mean_emb)
                features[f'{model_type}_mean_std'] = np.std(mean_emb)
                
                # Dimensionality reduction features (first 10 components)
                if len(mean_emb) > 10:
                    pca = PCA(n_components=10)
                    reduced_emb = pca.fit_transform([mean_emb])[0]
                    for i, val in enumerate(reduced_emb):
                        features[f'{model_type}_pca_{i}'] = val
            
            if max_emb is not None:
                features[f'{model_type}_max_norm'] = np.linalg.norm(max_emb)
            
            # Sentiment similarity scores
            sentiment_sims = self.get_sentiment_similarity_scores(text, model_type)
            for key, value in sentiment_sims.items():
                features[f'{model_type}_{key}'] = value
            
            # Calculate sentiment polarity from similarities
            pos_sim = sentiment_sims.get('positive_sim', 0)
            neg_sim = sentiment_sims.get('negative_sim', 0)
            neu_sim = sentiment_sims.get('neutral_sim', 0)
            
            # Semantic polarity score
            if pos_sim + neg_sim > 0:
                polarity = (pos_sim - neg_sim) / (pos_sim + neg_sim)
                features[f'{model_type}_semantic_polarity'] = polarity
            else:
                features[f'{model_type}_semantic_polarity'] = 0.0
            
            # Semantic intensity (distance from neutral)
            features[f'{model_type}_semantic_intensity'] = max(pos_sim, neg_sim) - neu_sim
        
        return features
    
    def get_feature_vector(self, text: str, include_embeddings: bool = False) -> np.ndarray:
        features = self.extract_embedding_features(text)
        
        # Select key features for ML
        key_features = []
        
        for model_type in ['word2vec', 'glove']:
            if f'{model_type}_mean_norm' in features:
                key_features.extend([
                    features.get(f'{model_type}_mean_norm', 0),
                    features.get(f'{model_type}_mean_std', 0),
                    features.get(f'{model_type}_positive_sim', 0),
                    features.get(f'{model_type}_negative_sim', 0),
                    features.get(f'{model_type}_neutral_sim', 0),
                    features.get(f'{model_type}_semantic_polarity', 0),
                    features.get(f'{model_type}_semantic_intensity', 0)
                ])
                
                # Add PCA components if available
                for i in range(10):
                    key_features.append(features.get(f'{model_type}_pca_{i}', 0))
        
        return np.array(key_features)
    
    def initialize_models(self, load_word2vec: bool = True, load_glove: bool = True):
        """
        Initialize and load embedding models.
        
        Args:
            load_word2vec: Whether to load Word2Vec
            load_glove: Whether to load GloVe
        """
        print("Initializing semantic embedding models...")
        
        if load_word2vec:
            print("\n1. Loading Word2Vec model...")
            self.load_word2vec()
        
        if load_glove:
            print("\n2. Loading GloVe model...")
            self.load_glove()
        