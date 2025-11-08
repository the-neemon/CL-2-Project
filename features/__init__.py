"""
Feature extraction module for Twitter sentiment analysis.

This module contains semantic, contextual, lexicon-based, and traditional feature extractors.
"""

from .contextual_features import ContextualFeatures
from .semantic_embeddings import SemanticEmbeddings
from .lexicon_scoring import LexiconBasedScoring
from .feature_pipeline import FeatureExtractionPipeline
from .traditional_features import TraditionalFeatureExtractor

__all__ = [
    'ContextualFeatures',
    'SemanticEmbeddings',
    'LexiconBasedScoring',
    'FeatureExtractionPipeline',
    'TraditionalFeatureExtractor'
]
