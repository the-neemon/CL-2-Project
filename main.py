"""
Main script to run the complete data preprocessing pipeline.

This script demonstrates how to load, preprocess, and prepare the
Sentiment140 and Airline datasets for sentiment analysis.

Usage:
    python main.py

Author: Naman
Phase: 1 - Data Preparation
"""

import argparse
from pathlib import Path

from data_loader import (
    prepare_sentiment140_data,
    prepare_airline_data,
    RANDOM_SEED
)


def main():
    """Run the complete preprocessing pipeline."""
    parser = argparse.ArgumentParser(
        description='Preprocess Twitter sentiment datasets'
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='datasets',
        help='Directory containing raw datasets'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='processed_data',
        help='Directory to save processed data'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=None,
        help='Sample size for Sentiment140 (for testing). Use full dataset if not specified'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of data for test set (default: 0.2)'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=RANDOM_SEED,
        help=f'Random seed for reproducibility (default: {RANDOM_SEED})'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['sentiment140', 'airline', 'both'],
        default='both',
        help='Which dataset to process (default: both)'
    )
    
    args = parser.parse_args()
    
    # Verify dataset directory exists
    dataset_path = Path(args.dataset_dir)
    if not dataset_path.exists():
        print(f"Error: Dataset directory '{args.dataset_dir}' not found!")
        return
    
    print("=" * 70)
    print("Twitter Sentiment Analysis - Data Preprocessing Pipeline")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Dataset directory: {args.dataset_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Test size: {args.test_size}")
    print(f"  Random seed: {args.random_seed}")
    if args.sample_size:
        print(f"  Sample size: {args.sample_size:,}")
    print()
    
    # Process Sentiment140 dataset
    if args.dataset in ['sentiment140', 'both']:
        print("\n" + "=" * 70)
        print("Processing Sentiment140 Dataset")
        print("=" * 70)
        try:
            saved_files = prepare_sentiment140_data(
                dataset_dir=args.dataset_dir,
                output_dir=args.output_dir,
                sample_size=args.sample_size,
                test_size=args.test_size,
                random_seed=args.random_seed
            )
            print("\n✓ Sentiment140 dataset processed successfully!")
        except Exception as e:
            print(f"\n✗ Error processing Sentiment140 dataset: {e}")
    
    # Process Airline dataset
    if args.dataset in ['airline', 'both']:
        print("\n" + "=" * 70)
        print("Processing Twitter US Airline Sentiment Dataset")
        print("=" * 70)
        try:
            saved_files = prepare_airline_data(
                dataset_dir=args.dataset_dir,
                output_dir=args.output_dir,
                test_size=args.test_size,
                random_seed=args.random_seed
            )
            print("\n✓ Airline sentiment dataset processed successfully!")
        except Exception as e:
            print(f"\n✗ Error processing airline dataset: {e}")
    
    print("\n" + "=" * 70)
    print("Preprocessing Pipeline Complete!")
    print("=" * 70)
    print(f"\nProcessed data saved to: {args.output_dir}")
    print("\nNext steps:")
    print("  1. Verify the processed data files")
    print("  2. Proceed to Phase 2: Feature Engineering & Model Training")
