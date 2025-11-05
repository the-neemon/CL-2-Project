"""
Data preprocessing pipeline for Twitter sentiment analysis.

This module implements comprehensive text preprocessing including:
- Text cleaning and normalization
- URL and mention removal with emoji preservation
- Tokenization using NLTK's TweetTokenizer
- Stopword removal and lemmatization
- HTML entity decoding

Author: Naman
Phase: 1 - Data Preparation
"""

import re
import html
import string
from typing import List

import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class TweetPreprocessor:
    """
    Preprocessor for Twitter data with emoji preservation.
    
    Handles comprehensive text cleaning while preserving emotionally
    relevant elements like emojis and emoticons for sentiment analysis.
    """
    
    def __init__(self, preserve_case: bool = False, reduce_len: bool = True):
        """
        Initialize the tweet preprocessor.
        
        Args:
            preserve_case: If True, preserve original casing. Default False.
            reduce_len: If True, reduce repeated characters. Default True.
        """
        self.preserve_case = preserve_case
        self.reduce_len = reduce_len
        
        # Initialize NLTK components
        self._download_nltk_resources()
        
        # TweetTokenizer handles emojis, emoticons, and reduces elongation
        self.tokenizer = TweetTokenizer(
            preserve_case=preserve_case,
            reduce_len=reduce_len,
            strip_handles=False  # We'll remove mentions manually
        )
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Common emoticons to preserve (positive and negative)
        self.emoticons = {
            ':)', ':-)', ':D', ':-D', ':P', ':-P', ';)', ';-)',
            ':(', ':-(', ":'(", ':/', ':-/', ':@', '>:(', 'D:',
            ':3', '<3', '=)', '=D', 'xD', 'XD', 'o_o', 'O_O',
            '^_^', '^.^', '-_-', '¯\\_(ツ)_/¯'
        }
        
        # Compile regex patterns for efficiency
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
            r'[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#(\w+)')
        self.number_pattern = re.compile(r'\d+')
        
    def _download_nltk_resources(self) -> None:
        """Download required NLTK resources if not available."""
        resources = [
            'punkt',
            'stopwords',
            'wordnet',
            'averaged_perceptron_tagger',
            'omw-1.4'
        ]
        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                try:
                    nltk.data.find(f'corpora/{resource}')
                except LookupError:
                    try:
                        nltk.download(resource, quiet=True)
                    except Exception:
                        pass  # Resource might not exist in all categories
    
    def clean_text(self, text: str) -> str:
        """
        Clean tweet text while preserving emojis and emoticons.
        
        Args:
            text: Raw tweet text
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""
        
        # Decode HTML entities (e.g., &amp; -> &, &lt; -> <)
        text = html.unescape(text)
        
        # Remove URLs
        text = self.url_pattern.sub('', text)
        
        # Remove mentions but keep the rest of the text
        text = self.mention_pattern.sub('', text)
        
        # Extract hashtag content (remove # but keep the word)
        text = self.hashtag_pattern.sub(r'\1', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using NLTK's TweetTokenizer.
        
        This tokenizer is specifically designed for tweets and handles
        emojis, emoticons, and Twitter-specific patterns correctly.
        
        Args:
            text: Cleaned text string
            
        Returns:
            List of tokens
        """
        return self.tokenizer.tokenize(text)
    
    def normalize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Normalize tokens: remove stopwords, punctuation, and lemmatize.
        
        Preserves emoticons and emojis for sentiment context.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of normalized tokens
        """
        normalized = []
        
        for token in tokens:
            # Preserve emoticons
            if token in self.emoticons:
                normalized.append(token)
                continue
            
            # Preserve emojis (Unicode characters outside ASCII range)
            if any(ord(char) > 127 for char in token):
                normalized.append(token)
                continue
            
            # Convert to lowercase for processing
            token_lower = token.lower()
            
            # Skip stopwords
            if token_lower in self.stop_words:
                continue
            
            # Skip pure punctuation
            if all(char in string.punctuation for char in token):
                continue
            
            # Skip tokens that are just numbers
            if self.number_pattern.fullmatch(token):
                continue
            
            # Lemmatize
            lemmatized = self.lemmatizer.lemmatize(token_lower)
            
            # Only add non-empty tokens
            if lemmatized:
                normalized.append(lemmatized)
        
        return normalized
    
    def preprocess(self, text: str, return_string: bool = True) -> str:
        """
        Complete preprocessing pipeline.
        
        Args:
            text: Raw tweet text
            return_string: If True, return joined string. If False, return
                          list of tokens.
            
        Returns:
            Preprocessed text (string) or tokens (list)
        """
        # Clean the text
        cleaned = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(cleaned)
        
        # Normalize
        normalized = self.normalize_tokens(tokens)
        
        if return_string:
            return ' '.join(normalized)
        return normalized
    
    def preprocess_batch(self, texts: List[str], 
                        return_string: bool = True) -> List[str]:
        """
        Preprocess multiple texts efficiently.
        
        Args:
            texts: List of raw tweet texts
            return_string: If True, return joined strings. If False, return
                          list of token lists.
            
        Returns:
            List of preprocessed texts or token lists
        """
        return [self.preprocess(text, return_string) for text in texts]


def create_preprocessor(preserve_case: bool = False, 
                       reduce_len: bool = True) -> TweetPreprocessor:
    """
    Factory function to create a preprocessor instance.
    
    Args:
        preserve_case: If True, preserve original casing. Default False.
        reduce_len: If True, reduce repeated characters. Default True.
        
    Returns:
        Configured TweetPreprocessor instance
    """
    return TweetPreprocessor(preserve_case=preserve_case, reduce_len=reduce_len)
