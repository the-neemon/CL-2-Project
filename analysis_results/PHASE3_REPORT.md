# Phase 3: Cross-Domain Validation & Qualitative Analysis

**Generated:** 2025-12-02T00:58:30.446109

---

## Overview

- **Source Domain:** Sentiment140
- **Target Domain:** Twitter US Airline Sentiment
- **Best Model:** Naive Bayes
- **Cross-Domain F1-Score:** 0.7126

## Cross-Domain Validation Results

| Model | Source F1 | Target F1 | Domain Gap |
|-------|-----------|-----------|------------|
| Logistic Regression | 0.5564 | 0.6595 | -0.1031 |
| Logistic Regression (L1) | 0.5564 | 0.6589 | -0.1024 |
| Naive Bayes | 0.5486 | 0.7126 | -0.1640 |
| Random Forest | 0.7514 | 0.6397 | 0.1117 |

## Qualitative Analysis Summary

- **Total Samples Analyzed:** 11361
- **Correctly Classified:** 7077
- **Accuracy:** 62.29%

### Semantic Pattern Comparison

| Metric | Correct | Incorrect | Difference |
|--------|---------|-----------|------------|
| avg_negations | 1.258 | 1.077 | +0.181 |
| avg_intensifiers | 0.085 | 0.089 | -0.004 |
| avg_vader_score | 0.247 | 0.285 | -0.038 |
| avg_text_length | 9.708 | 9.125 | +0.583 |

## Key Findings

1. Best cross-domain model: Naive Bayes with F1-score of 0.7126
2. Domain gap: -0.1031 (source: 0.5564, target: 0.6595)
3. Overall Accuracy: 62.29% (7077/11361)
4. Correctly classified tweets have 0.18 more negations on average, suggesting the model handles negation well.
5. VADER sentiment scores are lower for correctly classified tweets (diff: -0.038).
6. Negation handling: Good (diff: +0.18)

## Detailed Insights

**Insight 1:** Overall Accuracy: 62.29% (7077/11361)

**Insight 2:** Correctly classified tweets have 0.18 more negations on average, suggesting the model handles negation well.

**Insight 3:** VADER sentiment scores are lower for correctly classified tweets (diff: -0.038).

## Recommendations

Based on the analysis, we recommend:

1. **Model Selection:** Use Logistic Regression for cross-domain sentiment analysis
2. **Feature Engineering:** Focus on robust handling of negations and intensifiers
3. **Domain Adaptation:** Consider domain-specific fine-tuning for airline sentiment
4. **Error Analysis:** Pay special attention to tweets with mixed signals

---

*Report generated automatically by Phase 3 pipeline*
