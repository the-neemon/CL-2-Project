
import re
import nltk
from typing import List, Dict, Tuple, Set
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np

class ContextualFeatures:
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        # Negation words and patterns
        self.negation_words = {
            'not', 'no', 'never', 'nothing', 'nowhere', 'nobody', 'none', 
            'neither', 'nor', 'cannot', 'cant', 'couldn\'t', 'shouldn\'t', 
            'wouldn\'t', 'don\'t', 'doesn\'t', 'didn\'t', 'isn\'t', 'aren\'t', 
            'wasn\'t', 'weren\'t', 'haven\'t', 'hasn\'t', 'hadn\'t', 'won\'t',
            'nt', 'n\'t', 'barely', 'hardly', 'scarcely', 'seldom', 'rarely'
        }
        
        # Negation patterns (regex)
        self.negation_patterns = [
            r'\b(not|never|no)\s+\w+',
            r'\b\w+n\'t\b',
            r'\b\w+nt\b',
            r'\bno\s+\w+',
            r'\bnever\s+\w+',
            r'\bnothing\s+\w*',
            r'\bnobody\s+\w*',
            r'\bnowhere\s+\w*'
        ]
        
        # Intensifiers and their strength scores
        self.intensifiers = {
            # Strong intensifiers (2.0 multiplier)
            'extremely': 2.0, 'incredibly': 2.0, 'amazingly': 2.0, 'utterly': 2.0,
            'absolutely': 2.0, 'completely': 2.0, 'totally': 2.0, 'perfectly': 2.0,
            'tremendously': 2.0, 'enormously': 2.0, 'exceptionally': 2.0,
            
            # Medium intensifiers (1.5 multiplier)
            'very': 1.5, 'really': 1.5, 'quite': 1.5, 'pretty': 1.5, 'rather': 1.5,
            'fairly': 1.5, 'considerably': 1.5, 'significantly': 1.5, 'highly': 1.5,
            'deeply': 1.5, 'strongly': 1.5, 'seriously': 1.5, 'truly': 1.5,
            
            # Weak intensifiers (1.2 multiplier)
            'somewhat': 1.2, 'slightly': 1.2, 'a bit': 1.2, 'a little': 1.2,
            'kind of': 1.2, 'sort of': 1.2, 'moderately': 1.2, 'relatively': 1.2,
            
            # Diminishers (0.8 multiplier)
            'barely': 0.8, 'hardly': 0.8, 'scarcely': 0.8, 'rarely': 0.8,
            'seldom': 0.8, 'occasionally': 0.8, 'sometimes': 0.8, 'partly': 0.8
        }
        
        # Exclamation and emphasis patterns
        self.emphasis_patterns = {
            'exclamation_marks': r'!+',
            'question_marks': r'\?+',
            'caps_words': r'\b[A-Z]{2,}\b',
            'repeated_chars': r'(\w)\1{2,}',
            'multiple_punctuation': r'[.!?]{2,}'
        }
        
        # Twitter-specific patterns
        self.twitter_patterns = {
            'mentions': r'@\w+',
            'hashtags': r'#\w+',
            'urls': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            'retweets': r'\bRT\b',
            'emoticons': r'[:;=]-?[)(\]\[dDoOpP/\\|*$]|[)(\]\[dDoOpP/\\|*$]-?[:;=]'
        }
        
    def detect_negation(self, text: str) -> Dict[str, any]:
        text_lower = text.lower()
        tokens = word_tokenize(text_lower)
        
        features = {
            'negation_count': 0,
            'negation_positions': [],
            'negated_words': [],
            'negation_scope': 0,
            'has_negation': False
        }
        
        # Count direct negation words
        for i, token in enumerate(tokens):
            if token in self.negation_words:
                features['negation_count'] += 1
                features['negation_positions'].append(i)
                features['has_negation'] = True
                
                # Identify words in negation scope (next 3-5 words)
                scope_end = min(i + 5, len(tokens))
                negated_scope = tokens[i+1:scope_end]
                features['negated_words'].extend(negated_scope)
        
        # Count pattern-based negations
        for pattern in self.negation_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                features['negation_count'] += len(matches)
                features['has_negation'] = True
        
        # Calculate average negation scope
        if features['negation_positions']:
            total_scope = 0
            for pos in features['negation_positions']:
                # Scope typically extends 3-5 words after negation
                scope = min(5, len(tokens) - pos - 1)
                total_scope += scope
            features['negation_scope'] = total_scope / len(features['negation_positions'])
        
        return features
    
    def detect_intensifiers(self, text: str) -> Dict[str, any]:
    
        text_lower = text.lower()
        tokens = word_tokenize(text_lower)
        
        features = {
            'intensifier_count': 0,
            'intensifier_score': 1.0,  # Base score
            'intensifier_words': [],
            'max_intensity': 1.0,
            'avg_intensity': 1.0
        }
        
        intensities = []
        
        # Check for multi-word intensifiers first
        text_joined = ' '.join(tokens)
        for intensifier, score in self.intensifiers.items():
            if ' ' in intensifier and intensifier in text_joined:
                features['intensifier_count'] += 1
                features['intensifier_words'].append(intensifier)
                intensities.append(score)
        
        # Check for single-word intensifiers
        for token in tokens:
            if token in self.intensifiers:
                score = self.intensifiers[token]
                features['intensifier_count'] += 1
                features['intensifier_words'].append(token)
                intensities.append(score)
        
        # Calculate intensity scores
        if intensities:
            features['max_intensity'] = max(intensities)
            features['avg_intensity'] = np.mean(intensities)
            # Combined score considers both average and maximum
            features['intensifier_score'] = (features['avg_intensity'] + features['max_intensity']) / 2
        
        return features
    
    def detect_emphasis_patterns(self, text: str) -> Dict[str, any]:
        features = {
            'exclamation_count': 0,
            'question_count': 0,
            'caps_words_count': 0,
            'repeated_chars_count': 0,
            'multiple_punct_count': 0,
            'emphasis_score': 0
        }
        
        # Count different emphasis patterns
        for pattern_name, pattern in self.emphasis_patterns.items():
            matches = re.findall(pattern, text)
            if pattern_name == 'exclamation_marks':
                features['exclamation_count'] = len(matches)
            elif pattern_name == 'question_marks':
                features['question_count'] = len(matches)
            elif pattern_name == 'caps_words':
                features['caps_words_count'] = len(matches)
            elif pattern_name == 'repeated_chars':
                features['repeated_chars_count'] = len(matches)
            elif pattern_name == 'multiple_punctuation':
                features['multiple_punct_count'] = len(matches)
        
        # Calculate overall emphasis score
        features['emphasis_score'] = (
            features['exclamation_count'] * 0.3 +
            features['caps_words_count'] * 0.2 +
            features['repeated_chars_count'] * 0.2 +
            features['question_count'] * 0.1 +
            features['multiple_punct_count'] * 0.2
        )
        
        return features
    
    def detect_twitter_features(self, text: str) -> Dict[str, any]:
        features = {
            'mention_count': 0,
            'hashtag_count': 0,
            'url_count': 0,
            'retweet_count': 0,
            'emoticon_count': 0,
            'mentions': [],
            'hashtags': [],
            'has_mention': False,
            'has_hashtag': False,
            'has_url': False,
            'is_retweet': False,
            'has_emoticon': False
        }
        
        # Detect each Twitter pattern
        for pattern_name, pattern in self.twitter_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            
            if pattern_name == 'mentions':
                features['mention_count'] = len(matches)
                features['mentions'] = matches
                features['has_mention'] = len(matches) > 0
            elif pattern_name == 'hashtags':
                features['hashtag_count'] = len(matches)
                features['hashtags'] = matches
                features['has_hashtag'] = len(matches) > 0
            elif pattern_name == 'urls':
                features['url_count'] = len(matches)
                features['has_url'] = len(matches) > 0
            elif pattern_name == 'retweets':
                features['retweet_count'] = len(matches)
                features['is_retweet'] = len(matches) > 0
            elif pattern_name == 'emoticons':
                features['emoticon_count'] = len(matches)
                features['has_emoticon'] = len(matches) > 0
        
        return features
    
    def extract_contextual_features(self, text: str) -> Dict[str, any]:
        # Get individual feature sets
        negation_features = self.detect_negation(text)
        intensifier_features = self.detect_intensifiers(text)
        emphasis_features = self.detect_emphasis_patterns(text)
        twitter_features = self.detect_twitter_features(text)
        
        # Combine all features
        contextual_features = {
            **negation_features,
            **intensifier_features,
            **emphasis_features,
            **twitter_features
        }
        
        # Add composite features
        contextual_features['context_complexity'] = (
            contextual_features['negation_count'] +
            contextual_features['intensifier_count'] +
            contextual_features['emphasis_score']
        )
        
        # Calculate sentiment modification factor
        modification_factor = 1.0
        if contextual_features['has_negation']:
            modification_factor *= -1  # Flip sentiment
        
        if contextual_features['intensifier_score'] != 1.0:
            modification_factor *= contextual_features['intensifier_score']
        
        contextual_features['sentiment_modification_factor'] = modification_factor
        
        return contextual_features
    
    def get_feature_vector(self, text: str) -> np.ndarray:
        features = self.extract_contextual_features(text)
        
        # Create feature vector with key numerical features
        feature_vector = np.array([
            features['negation_count'],
            features['negation_scope'],
            int(features['has_negation']),
            features['intensifier_count'],
            features['intensifier_score'],
            features['max_intensity'],
            features['exclamation_count'],
            features['caps_words_count'],
            features['repeated_chars_count'],
            features['emphasis_score'],
            features['context_complexity'],
            features['sentiment_modification_factor']
        ])
        
        return feature_vector
    
    def get_feature_names(self) -> List[str]:
        """
        Get names of features in the feature vector.
        
        Returns:
            List of feature names
        """
        return [
            'negation_count',
            'negation_scope',
            'has_negation',
            'intensifier_count',
            'intensifier_score',
            'max_intensity',
            'exclamation_count',
            'caps_words_count',
            'repeated_chars_count',
            'emphasis_score',
            'context_complexity',
            'sentiment_modification_factor'
        ]
