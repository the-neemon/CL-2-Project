# Phase 3 Quick Reference Guide

## Overview
Phase 3 implements cross-domain validation and qualitative analysis with semantic interpretation.

## Quick Start

### Complete Pipeline (Recommended)
```bash
python scripts/run_phase3.py
```

This runs both cross-domain validation AND qualitative analysis in one go.

### Individual Components

#### 1. Cross-Domain Validation Only
```bash
python scripts/cross_domain_validation.py
```

**What it does:**
- Trains models on Sentiment140 (source domain)
- Tests on Twitter US Airline Sentiment (target domain)
- Compares multiple models (Logistic Regression, Naive Bayes, Random Forest)
- Measures domain gap (in-domain vs cross-domain performance)

**Output:** `analysis_results/cross_domain_validation.json`

#### 2. Qualitative Analysis Only
```bash
python scripts/qualitative_analysis.py --n-samples 15
```

**What it does:**
- Samples representative tweets from each prediction category
- Extracts semantic features (negations, intensifiers, VADER scores)
- Compares patterns in correct vs incorrect predictions
- Generates actionable insights

**Output:** `analysis_results/qualitative_analysis.json`

## Command Line Options

### run_phase3.py (Complete Pipeline)
```bash
python scripts/run_phase3.py \
    --dataset-dir datasets \
    --source-sample-size 50000 \
    --max-features 5000 \
    --n-qualitative-samples 15 \
    --output-dir analysis_results
```

**Parameters:**
- `--dataset-dir`: Directory containing datasets (default: `datasets`)
- `--source-sample-size`: Samples from Sentiment140 (default: `50000`)
- `--max-features`: Max TF-IDF features (default: `5000`)
- `--n-qualitative-samples`: Samples per category for qualitative analysis (default: `15`)
- `--output-dir`: Output directory (default: `analysis_results`)

### cross_domain_validation.py
```bash
python scripts/cross_domain_validation.py \
    --source-sample-size 100000 \
    --max-features 10000
```

### qualitative_analysis.py
```bash
python scripts/qualitative_analysis.py \
    --n-samples 20 \
    --output-dir analysis_results
```

## Expected Outputs

### 1. cross_domain_validation.json
Contains:
- Performance metrics for each model (source and target domains)
- Domain gap measurements
- Best model identification

**Key Fields:**
```json
{
  "source_domain": "Sentiment140",
  "target_domain": "Twitter US Airline Sentiment",
  "best_model": "Logistic Regression",
  "best_f1_score": 0.7234,
  "cross_domain_results": [...]
}
```

### 2. qualitative_analysis.json
Contains:
- Sample analyses with semantic features
- Pattern comparisons (correct vs incorrect)
- Generated insights

**Key Fields:**
```json
{
  "sample_analysis": {
    "accuracy": 0.72,
    "sample_analyses": [
      {
        "category": "True Positive - Correct",
        "samples": [...]
      }
    ]
  },
  "pattern_analysis": {
    "correct_predictions": {...},
    "incorrect_predictions": {...},
    "comparison": {...}
  },
  "insights": [...]
}
```

### 3. phase3_comprehensive_report.json
Combines both analyses with:
- Cross-domain validation summary
- Qualitative analysis summary
- Key findings
- Recommendations

### 4. PHASE3_REPORT.md
Human-readable markdown report with:
- Overview and configuration
- Cross-domain results table
- Semantic pattern comparison table
- Key findings and insights
- Recommendations

## Understanding the Results

### Cross-Domain Metrics
- **Source F1**: Performance on training domain (Sentiment140)
- **Target F1**: Performance on test domain (Airline)
- **Domain Gap**: Difference between source and target F1 (lower is better)

### Semantic Features Analyzed
1. **Negations**: "not", "n't", "never", "no", etc.
2. **Intensifiers**: "very", "really", "extremely", etc.
3. **VADER Scores**: Lexicon-based sentiment (-1 to +1)
4. **Emphasis**: ALL CAPS, exclamation marks (!)
5. **Questions**: Presence of question marks (?)

### Sample Categories
- **True Negative (Correct)**: Correctly predicted negative sentiment
- **True Positive (Correct)**: Correctly predicted positive sentiment
- **False Positive (Error)**: Predicted positive, actually negative
- **False Negative (Error)**: Predicted negative, actually positive

## Interpretation Guide

### Good Cross-Domain Performance
- **Small domain gap** (< 0.05): Model generalizes well
- **High target F1** (> 0.70): Effective on new domain

### Semantic Pattern Insights
- **More negations in correct predictions**: Model handles negation well
- **More intensifiers in incorrect predictions**: Over-emphasis causes confusion
- **Higher VADER scores in correct predictions**: Lexicon features help

## Troubleshooting

### Issue: Out of Memory
**Solution:** Reduce sample size
```bash
python scripts/run_phase3.py --source-sample-size 10000
```

### Issue: Takes too long
**Solution:** Reduce features and samples
```bash
python scripts/run_phase3.py --source-sample-size 20000 --max-features 3000
```

### Issue: Need more qualitative samples
**Solution:** Increase n-qualitative-samples
```bash
python scripts/run_phase3.py --n-qualitative-samples 30
```

## Example Workflow

```bash
# 1. Quick test with small samples
python scripts/run_phase3.py --source-sample-size 5000 --n-qualitative-samples 5

# 2. Standard run (recommended)
python scripts/run_phase3.py

# 3. Full analysis with more data
python scripts/run_phase3.py --source-sample-size 100000 --max-features 10000 --n-qualitative-samples 20

# 4. Check results
cat analysis_results/PHASE3_REPORT.md
```

## Key Files to Review

1. **PHASE3_REPORT.md** - Start here for high-level insights
2. **phase3_comprehensive_report.json** - Complete structured results
3. **cross_domain_validation.json** - Detailed model metrics
4. **qualitative_analysis.json** - Full sample analyses

## Python API Usage

```python
from scripts.run_phase3 import Phase3Pipeline

# Initialize pipeline
pipeline = Phase3Pipeline(
    dataset_dir='datasets',
    output_dir='analysis_results'
)

# Run complete analysis
results = pipeline.run_complete_pipeline(
    source_sample_size=50000,
    max_features=5000,
    n_qualitative_samples=15
)

# Access results
print(f"Best Model: {results['cross_domain_validation']['best_model']}")
print(f"Cross-Domain F1: {results['cross_domain_validation']['best_f1_score']:.4f}")
print(f"Insights: {results['qualitative_analysis']['insights']}")
```

## Next Steps After Phase 3

1. **Review PHASE3_REPORT.md** for insights
2. **Identify improvement areas** from qualitative analysis
3. **Consider domain adaptation** techniques if domain gap is large
4. **Refine feature engineering** based on semantic patterns
5. **Experiment with ensemble methods** if single models show weaknesses

---

**Authors:** Phase 3 Implementation Team  
**Date:** November 2025  
**Phase:** Cross-Domain Validation & Qualitative Analysis
