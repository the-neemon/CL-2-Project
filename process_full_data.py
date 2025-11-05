"""
Process the full Sentiment140 and Airline datasets.

This script processes the complete datasets and saves them for model training.
WARNING: Processing 1.6M tweets may take some time (5-15 minutes).

Author: Naman
Phase: 1 - Data Preparation
"""

from data_loader import prepare_sentiment140_data, prepare_airline_data


def main():
    """Process full datasets."""
    print("=" * 70)
    print("PROCESSING FULL DATASETS")
    print("=" * 70)
    print("\nThis will process:")
    print("  - Sentiment140: ~1.6M tweets")
    print("  - Airline Sentiment: ~14.6K tweets")
    print("\nEstimated time: 5-15 minutes depending on your system")
    
    response = input("\nContinue? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Aborted.")
        return
    
    # Process Sentiment140
    print("\n" + "=" * 70)
    print("PROCESSING SENTIMENT140 (Full Dataset)")
    print("=" * 70)
    
    try:
        files = prepare_sentiment140_data(
            dataset_dir='datasets',
            output_dir='processed_data',
            sample_size=None,  # Use full dataset
            test_size=0.2,
            random_seed=42
        )
        print("\n✓ Sentiment140 processing complete!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
    
    # Process Airline
    print("\n" + "=" * 70)
    print("PROCESSING AIRLINE SENTIMENT")
    print("=" * 70)
    
    try:
        files = prepare_airline_data(
            dataset_dir='datasets',
            output_dir='processed_data',
            test_size=0.2,
            random_seed=42
        )
        print("\n✓ Airline sentiment processing complete!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
    
    print("\n" + "=" * 70)
    print("ALL DATASETS PROCESSED!")
    print("=" * 70)
    print("\nProcessed data saved to: processed_data/")
    print("\nFiles created:")
    print("  - sentiment140_train.csv")
    print("  - sentiment140_test.csv")
    print("  - airline_sentiment_train.csv")
    print("  - airline_sentiment_test.csv")
    print("\nYou can now proceed to Phase 2: Feature Engineering!")
