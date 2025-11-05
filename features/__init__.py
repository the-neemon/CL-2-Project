"""
Feature extraction module for Twitter sentiment analysis.

This module contains semantic, contextual, and lexicon-based feature extractors.
"""

from .contextual_features import ContextualFeatures
from .semantic_embeddings import SemanticEmbeddings
from .lexicon_scoring import LexiconBasedScoring
from .feature_pipeline import FeatureExtractionPipeline

__all__ = [
    'ContextualFeatures',
    'SemanticEmbeddings',
    'LexiconBasedScoring',
    'FeatureExtractionPipeline'
]
