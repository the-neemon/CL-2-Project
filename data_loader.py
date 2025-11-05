"""
Data loading utilities for Twitter sentiment datasets.

Handles loading, preprocessing, and train-test splitting for:
- Sentiment140 dataset (multiple CSV files)
- Twitter US Airline Sentiment dataset (cross-validation)

Ensures reproducibility through random seed control and implements
stratified splitting for class balance.

Author: Naman
Phase: 1 - Data Preparation
"""

import os
import random
from pathlib import Path
from typing import Tuple, List, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from preprocessing import TweetPreprocessor


# Set random seeds for reproducibility
RANDOM_SEED = 42


def set_random_seeds(seed: int = RANDOM_SEED) -> None:
    """
    Set random seeds for reproducibility across libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


class SentimentDataLoader:
    """
    Unified data loader for Twitter sentiment datasets.
    
    Supports loading from multiple sources and formats with consistent
    output format for downstream processing.
    """
    
    def __init__(self, dataset_dir: str, random_seed: int = RANDOM_SEED):
        """
        Initialize data loader.
        
        Args:
            dataset_dir: Root directory containing datasets
            random_seed: Random seed for reproducibility
        """
        self.dataset_dir = Path(dataset_dir)
        self.random_seed = random_seed
        set_random_seeds(random_seed)
        
        self.sentiment140_dir = self.dataset_dir / 'Sentiment140_dataset'
        self.airline_dir = self.dataset_dir / 'cross_validation_dataset'
    
    def load_sentiment140(self, 
                         sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Load Sentiment140 dataset from multiple CSV files.
        
        The Sentiment140 dataset format:
        Column 0: Polarity (0=negative, 2=neutral, 4=positive)
        Column 1: Tweet ID
        Column 2: Date
        Column 3: Query
        Column 4: User
        Column 5: Text
        
        Args:
            sample_size: If specified, randomly sample this many tweets
            
        Returns:
            DataFrame with columns: text, sentiment (0=negative, 1=positive)
        """
        # Find all training CSV files
        csv_files = sorted(self.sentiment140_dir.glob('training_part_*.csv'))
        
        if not csv_files:
            raise FileNotFoundError(
                f"No training files found in {self.sentiment140_dir}"
            )
        
        print(f"Loading {len(csv_files)} Sentiment140 CSV files...")
        
        # Column names for Sentiment140
        column_names = ['polarity', 'tweet_id', 'date', 'query', 'user', 'text']
        
        # Load and concatenate all files
        dataframes = []
        for csv_file in csv_files:
            df = pd.read_csv(
                csv_file,
                encoding='latin-1',
                names=column_names,
                header=None
            )
            dataframes.append(df)
        
        # Combine all dataframes
        combined_df = pd.concat(dataframes, ignore_index=True)
        
        print(f"Total tweets loaded: {len(combined_df):,}")
        
        # Convert polarity to binary sentiment (0=negative, 4=positive)
        # Map: 0 -> 0 (negative), 4 -> 1 (positive), drop neutral (2)
        combined_df = combined_df[combined_df['polarity'].isin([0, 4])]
        combined_df['sentiment'] = (combined_df['polarity'] == 4).astype(int)
        
        # Select only text and sentiment
        result_df = combined_df[['text', 'sentiment']].copy()
        
        # Remove duplicates and NaN values
        result_df = result_df.drop_duplicates(subset=['text'])
        result_df = result_df.dropna()
        
        # Sample if requested
        if sample_size and sample_size < len(result_df):
            result_df = result_df.sample(
                n=sample_size,
                random_state=self.random_seed
            ).reset_index(drop=True)
            print(f"Sampled {sample_size:,} tweets")
        
        print(f"Final dataset size: {len(result_df):,}")
        print(f"Sentiment distribution:\n{result_df['sentiment'].value_counts()}")
        
        return result_df
    
    def load_airline_sentiment(self) -> pd.DataFrame:
        """
        Load Twitter US Airline Sentiment dataset.
        
        Format includes sentiment labels: negative, neutral, positive
        
        Returns:
            DataFrame with columns: text, sentiment (0=negative, 1=neutral, 
                                                     2=positive)
        """
        csv_file = self.airline_dir / 'Tweets.csv'
        
        if not csv_file.exists():
            raise FileNotFoundError(
                f"Airline sentiment file not found: {csv_file}"
            )
        
        print(f"Loading airline sentiment dataset from {csv_file}...")
        
        df = pd.read_csv(csv_file)
        
        # Map sentiment labels to numeric values
        sentiment_mapping = {
            'negative': 0,
            'neutral': 1,
            'positive': 2
        }
        
        df['sentiment'] = df['airline_sentiment'].map(sentiment_mapping)
        
        # Select text and sentiment columns
        result_df = df[['text', 'sentiment']].copy()
        
        # Remove duplicates and NaN values
        result_df = result_df.drop_duplicates(subset=['text'])
        result_df = result_df.dropna()
        
        print(f"Total tweets loaded: {len(result_df):,}")
        print(f"Sentiment distribution:\n{result_df['sentiment'].value_counts()}")
        
        return result_df
    
    def create_train_test_split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        stratify: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create stratified train-test split with class balancing.
        
        Args:
            df: DataFrame with 'text' and 'sentiment' columns
            test_size: Proportion of data for testing (default 0.2 for 80-20)
            stratify: If True, maintain class distribution in splits
            
        Returns:
            Tuple of (train_df, test_df)
        """
        print(f"\nCreating train-test split (test_size={test_size})...")
        
        if stratify:
            stratify_column = df['sentiment']
        else:
            stratify_column = None
        
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=self.random_seed,
            stratify=stratify_column
        )
        
        print(f"Training set size: {len(train_df):,}")
        print(f"Test set size: {len(test_df):,}")
        print(f"\nTraining set sentiment distribution:")
        print(train_df['sentiment'].value_counts())
        print(f"\nTest set sentiment distribution:")
        print(test_df['sentiment'].value_counts())
        
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
    
    def preprocess_dataframe(
        self,
        df: pd.DataFrame,
        preprocessor: TweetPreprocessor,
        text_column: str = 'text',
        output_column: str = 'processed_text'
    ) -> pd.DataFrame:
        """
        Apply preprocessing to a dataframe.
        
        Args:
            df: DataFrame containing tweets
            preprocessor: TweetPreprocessor instance
            text_column: Name of column containing raw text
            output_column: Name of column for processed text
            
        Returns:
            DataFrame with additional processed_text column
        """
        print(f"\nPreprocessing {len(df):,} tweets...")
        
        df = df.copy()
        df[output_column] = preprocessor.preprocess_batch(
            df[text_column].tolist()
        )
        
        # Remove entries where preprocessing resulted in empty strings
        original_size = len(df)
        df = df[df[output_column].str.strip() != '']
        removed = original_size - len(df)
        
        if removed > 0:
            print(f"Removed {removed} tweets that became empty after preprocessing")
        
        print(f"Preprocessing complete. Final size: {len(df):,}")
        
        return df.reset_index(drop=True)
    
    def save_processed_data(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_dir: str,
        dataset_name: str = 'sentiment140'
    ) -> Dict[str, Path]:
        """
        Save preprocessed train and test data to CSV files.
        
        Args:
            train_df: Training dataframe
            test_df: Test dataframe
            output_dir: Directory to save files
            dataset_name: Name prefix for saved files
            
        Returns:
            Dictionary with paths to saved files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        train_file = output_path / f'{dataset_name}_train.csv'
        test_file = output_path / f'{dataset_name}_test.csv'
        
        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)
        
        print(f"\nSaved training data to: {train_file}")
        print(f"Saved test data to: {test_file}")
        
        return {
            'train': train_file,
            'test': test_file
        }
    
    def load_processed_data(
        self,
        input_dir: str,
        dataset_name: str = 'sentiment140'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load previously saved processed data.
        
        Args:
            input_dir: Directory containing processed files
            dataset_name: Name prefix of saved files
            
        Returns:
            Tuple of (train_df, test_df)
        """
        input_path = Path(input_dir)
        
        train_file = input_path / f'{dataset_name}_train.csv'
        test_file = input_path / f'{dataset_name}_test.csv'
        
        if not train_file.exists() or not test_file.exists():
            raise FileNotFoundError(
                f"Processed data files not found in {input_dir}"
            )
        
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
        
        print(f"Loaded training data: {len(train_df):,} samples")
        print(f"Loaded test data: {len(test_df):,} samples")
        
        return train_df, test_df


def prepare_sentiment140_data(
    dataset_dir: str,
    output_dir: str,
    sample_size: Optional[int] = None,
    test_size: float = 0.2,
    random_seed: int = RANDOM_SEED
) -> Dict[str, Path]:
    """
    Complete pipeline: load, preprocess, split, and save Sentiment140 data.
    
    Args:
        dataset_dir: Directory containing raw datasets
        output_dir: Directory to save processed data
        sample_size: Optional sample size (for testing pipeline)
        test_size: Test set proportion (default 0.2)
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with paths to saved files
    """
    # Initialize components
    loader = SentimentDataLoader(dataset_dir, random_seed)
    preprocessor = TweetPreprocessor(preserve_case=False, reduce_len=True)
    
    # Load data
    df = loader.load_sentiment140(sample_size=sample_size)
    
    # Create train-test split
    train_df, test_df = loader.create_train_test_split(
        df,
        test_size=test_size,
        stratify=True
    )
    
    # Preprocess both sets
    train_df = loader.preprocess_dataframe(train_df, preprocessor)
    test_df = loader.preprocess_dataframe(test_df, preprocessor)
    
    # Save processed data
    saved_files = loader.save_processed_data(
        train_df,
        test_df,
        output_dir,
        dataset_name='sentiment140'
    )
    
    return saved_files


def prepare_airline_data(
    dataset_dir: str,
    output_dir: str,
    test_size: float = 0.2,
    random_seed: int = RANDOM_SEED
) -> Dict[str, Path]:
    """
    Complete pipeline: load, preprocess, split, and save airline data.
    
    Args:
        dataset_dir: Directory containing raw datasets
        output_dir: Directory to save processed data
        test_size: Test set proportion (default 0.2)
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with paths to saved files
    """
    # Initialize components
    loader = SentimentDataLoader(dataset_dir, random_seed)
    preprocessor = TweetPreprocessor(preserve_case=False, reduce_len=True)
    
    # Load data
    df = loader.load_airline_sentiment()
    
    # Create train-test split
    train_df, test_df = loader.create_train_test_split(
        df,
        test_size=test_size,
        stratify=True
    )
    
    # Preprocess both sets
    train_df = loader.preprocess_dataframe(train_df, preprocessor)
    test_df = loader.preprocess_dataframe(test_df, preprocessor)
    
    # Save processed data
    saved_files = loader.save_processed_data(
        train_df,
        test_df,
        output_dir,
        dataset_name='airline_sentiment'
    )
    
    return saved_files
