import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
import json
import re
from collections import defaultdict, Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import requests
import os

class LexiconBasedScoring:
    def __init__(self, cache_dir: str = './lexicons_cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize VADER
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Initialize NLTK components
        self._setup_nltk()
        
        # Lexicon storage
        self.nrc_lexicon = {}
        self.custom_lexicons = {}
        
        # Additional sentiment words
        self.positive_words = set()
        self.negative_words = set()
        
        # Emotion categories from NRC
        self.nrc_emotions = [
            'anger', 'anticipation', 'disgust', 'fear', 'joy',
            'negative', 'positive', 'sadness', 'surprise', 'trust'
        ]
        
        # Sentiment modifiers
        self.intensifiers = {
            'very': 1.5, 'really': 1.4, 'extremely': 2.0, 'quite': 1.3,
            'rather': 1.2, 'pretty': 1.2, 'fairly': 1.1, 'incredibly': 1.8,
            'amazingly': 1.7, 'absolutely': 1.9, 'totally': 1.6, 'completely': 1.8
        }
        
        self.diminishers = {
            'slightly': 0.8, 'somewhat': 0.7, 'barely': 0.6, 'hardly': 0.5,
            'a little': 0.8, 'kind of': 0.7, 'sort of': 0.7, 'rarely': 0.4
        }
    
    def _setup_nltk(self):
        """Setup required NLTK components."""
        required_nltk_data = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
        
        for data_name in required_nltk_data:
            try:
                nltk.data.find(f'tokenizers/{data_name}')
            except LookupError:
                try:
                    nltk.data.find(f'corpora/{data_name}')
                except LookupError:
                    try:
                        nltk.download(data_name, quiet=True)
                    except:
                        pass
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def load_nrc_lexicon(self) -> bool:
        nrc_file = os.path.join(self.cache_dir, 'nrc_lexicon.json')
        
        # Try to load from cache
        if os.path.exists(nrc_file):
            try:
                with open(nrc_file, 'r', encoding='utf-8') as f:
                    self.nrc_lexicon = json.load(f)
                print("✓ NRC Lexicon loaded from cache")
                return True
            except:
                pass
        
        # Create a basic NRC-style lexicon if download fails
        print("Creating basic emotion lexicon...")
        
        # Basic emotion words (subset of NRC for demonstration)
        basic_emotions = {
            'anger': ['angry', 'furious', 'mad', 'irritated', 'annoyed', 'outraged', 'livid', 'irate'],
            'anticipation': ['excited', 'eager', 'hopeful', 'expecting', 'anticipating', 'looking', 'forward'],
            'disgust': ['disgusted', 'revolted', 'sick', 'nauseated', 'repulsed', 'appalled'],
            'fear': ['afraid', 'scared', 'terrified', 'frightened', 'worried', 'anxious', 'nervous'],
            'joy': ['happy', 'joyful', 'delighted', 'pleased', 'cheerful', 'glad', 'ecstatic', 'elated'],
            'negative': ['bad', 'terrible', 'awful', 'horrible', 'disappointing', 'sad', 'unhappy'],
            'positive': ['good', 'great', 'excellent', 'wonderful', 'amazing', 'fantastic', 'awesome'],
            'sadness': ['sad', 'depressed', 'melancholy', 'sorrowful', 'gloomy', 'dejected', 'downhearted'],
            'surprise': ['surprised', 'amazed', 'astonished', 'shocked', 'stunned', 'startled'],
            'trust': ['trust', 'confident', 'reliable', 'faithful', 'loyal', 'dependable', 'secure']
        }
        
        # Convert to word-emotion mapping
        self.nrc_lexicon = {}
        for emotion, words in basic_emotions.items():
            for word in words:
                if word not in self.nrc_lexicon:
                    self.nrc_lexicon[word] = {}
                self.nrc_lexicon[word][emotion] = 1
        
        # Cache the lexicon
        try:
            with open(nrc_file, 'w', encoding='utf-8') as f:
                json.dump(self.nrc_lexicon, f, indent=2)
        except:
            pass
        
        print("✓ Basic emotion lexicon created and cached")
        return True
    
    def load_additional_lexicons(self):
        """Load additional sentiment lexicons."""
        # Load positive and negative words
        positive_words_basic = [
            'excellent', 'amazing', 'wonderful', 'fantastic', 'great', 'good', 'nice',
            'beautiful', 'perfect', 'awesome', 'brilliant', 'outstanding', 'superb',
            'magnificent', 'marvelous', 'splendid', 'remarkable', 'impressive', 'incredible',
            'love', 'like', 'enjoy', 'appreciate', 'adore', 'cherish', 'treasure'
        ]
        
        negative_words_basic = [
            'terrible', 'awful', 'horrible', 'disgusting', 'bad', 'disappointing',
            'pathetic', 'dreadful', 'appalling', 'abysmal', 'atrocious', 'deplorable',
            'miserable', 'wretched', 'lousy', 'rubbish', 'trash', 'garbage',
            'hate', 'dislike', 'despise', 'detest', 'loathe', 'abhor'
        ]
        
        self.positive_words = set(positive_words_basic)
        self.negative_words = set(negative_words_basic)
        
        print("✓ Additional lexicons loaded")
    
    def analyze_vader(self, text: str) -> Dict[str, float]:
        scores = self.vader_analyzer.polarity_scores(text)
        return {
            'vader_negative': scores['neg'],
            'vader_neutral': scores['neu'],
            'vader_positive': scores['pos'],
            'vader_compound': scores['compound']
        }
    
    def analyze_nrc_emotions(self, text: str) -> Dict[str, float]:
        tokens = word_tokenize(text.lower())
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and token.isalpha()]
        
        emotion_scores = {emotion: 0 for emotion in self.nrc_emotions}
        word_count = 0
        
        for token in tokens:
            if token in self.nrc_lexicon:
                word_count += 1
                for emotion in self.nrc_emotions:
                    if emotion in self.nrc_lexicon[token]:
                        emotion_scores[emotion] += self.nrc_lexicon[token][emotion]
        
        # Normalize by word count
        if word_count > 0:
            for emotion in emotion_scores:
                emotion_scores[emotion] /= word_count
        
        # Add NRC prefix to keys
        return {f'nrc_{emotion}': score for emotion, score in emotion_scores.items()}
    
    def analyze_custom_polarity(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using custom lexicons.
        
        Args:
            text: Input text
            
        Returns:
            Dict with custom polarity scores
        """
        tokens = word_tokenize(text.lower())
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token.isalpha()]
        
        positive_count = sum(1 for token in tokens if token in self.positive_words)
        negative_count = sum(1 for token in tokens if token in self.negative_words)
        total_sentiment_words = positive_count + negative_count
        
        # Calculate scores
        if len(tokens) > 0:
            positive_ratio = positive_count / len(tokens)
            negative_ratio = negative_count / len(tokens)
            sentiment_ratio = total_sentiment_words / len(tokens)
        else:
            positive_ratio = negative_ratio = sentiment_ratio = 0
        
        # Calculate polarity
        if total_sentiment_words > 0:
            polarity = (positive_count - negative_count) / total_sentiment_words
        else:
            polarity = 0
        
        return {
            'custom_positive_ratio': positive_ratio,
            'custom_negative_ratio': negative_ratio,
            'custom_sentiment_ratio': sentiment_ratio,
            'custom_polarity': polarity,
            'custom_positive_count': positive_count,
            'custom_negative_count': negative_count
        }
    
    def calculate_sentiment_modifiers(self, text: str) -> Dict[str, float]:
        tokens = word_tokenize(text.lower())
        
        intensifier_score = 1.0
        diminisher_score = 1.0
        modifier_count = 0
        
        for i, token in enumerate(tokens):
            if token in self.intensifiers:
                intensifier_score *= self.intensifiers[token]
                modifier_count += 1
            elif token in self.diminishers:
                diminisher_score *= self.diminishers[token]
                modifier_count += 1
        
        # Combined modifier effect
        total_modifier = intensifier_score * diminisher_score
        
        return {
            'intensifier_effect': intensifier_score,
            'diminisher_effect': diminisher_score,
            'total_modifier_effect': total_modifier,
            'modifier_count': modifier_count,
            'has_modifiers': modifier_count > 0
        }
    
    def extract_lexicon_features(self, text: str) -> Dict[str, any]:
        features = {}
        
        # VADER analysis
        vader_features = self.analyze_vader(text)
        features.update(vader_features)
        
        # NRC emotion analysis
        if self.nrc_lexicon:
            nrc_features = self.analyze_nrc_emotions(text)
            features.update(nrc_features)
        
        # Custom polarity analysis
        custom_features = self.analyze_custom_polarity(text)
        features.update(custom_features)
        
        # Sentiment modifiers
        modifier_features = self.calculate_sentiment_modifiers(text)
        features.update(modifier_features)
        
        # Composite scores
        # Overall polarity combining different methods
        polarity_scores = [
            features.get('vader_compound', 0),
            features.get('custom_polarity', 0)
        ]
        
        if features.get('nrc_positive', 0) + features.get('nrc_negative', 0) > 0:
            nrc_polarity = (features.get('nrc_positive', 0) - features.get('nrc_negative', 0)) / \
                          (features.get('nrc_positive', 0) + features.get('nrc_negative', 0))
            polarity_scores.append(nrc_polarity)
        
        features['combined_polarity'] = np.mean(polarity_scores)
        features['polarity_agreement'] = np.std(polarity_scores)  # Lower std = more agreement
        
        # Emotional intensity
        emotion_scores = [features.get(f'nrc_{emotion}', 0) for emotion in self.nrc_emotions]
        features['emotional_intensity'] = np.sum(emotion_scores)
        features['emotional_diversity'] = np.count_nonzero(emotion_scores)
        
        return features
    
    def get_feature_vector(self, text: str) -> np.ndarray:
        """
        Get numerical feature vector for machine learning.
        
        Args:
            text: Input text
            
        Returns:
            numpy array of lexicon-based features
        """
        features = self.extract_lexicon_features(text)
        
        # Select key features for ML
        feature_vector = [
            # VADER features
            features.get('vader_negative', 0),
            features.get('vader_neutral', 0),
            features.get('vader_positive', 0),
            features.get('vader_compound', 0),
            
            # Custom features
            features.get('custom_positive_ratio', 0),
            features.get('custom_negative_ratio', 0),
            features.get('custom_polarity', 0),
            features.get('custom_sentiment_ratio', 0),
            
            # NRC emotion features (key emotions)
            features.get('nrc_positive', 0),
            features.get('nrc_negative', 0),
            features.get('nrc_joy', 0),
            features.get('nrc_anger', 0),
            features.get('nrc_sadness', 0),
            features.get('nrc_fear', 0),
            
            # Modifier features
            features.get('intensifier_effect', 1.0),
            features.get('diminisher_effect', 1.0),
            features.get('total_modifier_effect', 1.0),
            features.get('modifier_count', 0),
            
            # Composite features
            features.get('combined_polarity', 0),
            features.get('polarity_agreement', 0),
            features.get('emotional_intensity', 0),
            features.get('emotional_diversity', 0)
        ]
        
        return np.array(feature_vector)
    
    def get_feature_names(self) -> List[str]:
        return [
            'vader_negative', 'vader_neutral', 'vader_positive', 'vader_compound',
            'custom_positive_ratio', 'custom_negative_ratio', 'custom_polarity', 
            'custom_sentiment_ratio', 'nrc_positive', 'nrc_negative', 'nrc_joy',
            'nrc_anger', 'nrc_sadness', 'nrc_fear', 'intensifier_effect',
            'diminisher_effect', 'total_modifier_effect', 'modifier_count',
            'combined_polarity', 'polarity_agreement', 'emotional_intensity',
            'emotional_diversity'
        ]
    
    def initialize_lexicons(self):
        """Initialize all lexicons."""
        print("Initializing lexicon-based scoring system...")
        
        print("\n1. Loading NRC Emotion Lexicon...")
        self.load_nrc_lexicon()
        
        print("\n2. Loading additional lexicons...")
        self.load_additional_lexicons()
        
        print("\n✓ Lexicon-based scoring system initialized!")
