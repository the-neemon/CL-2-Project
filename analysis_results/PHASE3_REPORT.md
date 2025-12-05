# Phase 3: Cross-Domain Validation & Qualitative Analysis

**Generated:** 2025-12-05T14:39:12.368738

---

## Overview

- **Source Domain:** Sentiment140
- **Target Domain:** Twitter US Airline Sentiment
- **Best Model:** Naive Bayes
- **Cross-Domain F1-Score:** 0.7299

## Cross-Domain Validation Results

| Model | Source F1 | Target F1 | Domain Gap |
|-------|-----------|-----------|------------|
| Logistic Regression | 0.5492 | 0.6624 | -0.1132 |
| Logistic Regression (L1) | 0.5497 | 0.6592 | -0.1095 |
| Naive Bayes | 0.5315 | 0.7299 | -0.1985 |
| Random Forest (Original) | 0.7901 | 0.6227 | 0.1675 |
| Random Forest (Regularized) | 0.6106 | 0.6780 | -0.0674 |

## Qualitative Analysis Summary

- **Total Samples Analyzed:** 11361
- **Correctly Classified:** 7117
- **Accuracy:** 62.64%

### Semantic Pattern Comparison

| Metric | Correct | Incorrect | Difference |
|--------|---------|-----------|------------|
| avg_negations | 1.250 | 1.072 | +0.178 |
| avg_intensifiers | 0.084 | 0.088 | -0.004 |
| avg_vader_score | 0.245 | 0.303 | -0.057 |
| avg_text_length | 9.750 | 9.025 | +0.725 |

## Key Findings

1. Best cross-domain model: Naive Bayes with F1-score of 0.7299
2. Domain gap: -0.1132 (source: 0.5492, target: 0.6624)
3. Overall Accuracy: 62.64% (7117/11361)
4. Correctly classified tweets have 0.18 more negations on average, suggesting the model handles negation well.
5. VADER sentiment scores are lower for correctly classified tweets (diff: -0.057).
6. Negation handling: Good (diff: +0.18)

## Detailed Insights

**Insight 1:** Overall Accuracy: 62.64% (7117/11361)

**Insight 2:** Correctly classified tweets have 0.18 more negations on average, suggesting the model handles negation well.

**Insight 3:** VADER sentiment scores are lower for correctly classified tweets (diff: -0.057).

## Recommendations

Based on the analysis, we recommend:

1. **Model Selection:** Use Logistic Regression for cross-domain sentiment analysis
2. **Feature Engineering:** Focus on robust handling of negations and intensifiers
3. **Domain Adaptation:** Consider domain-specific fine-tuning for airline sentiment
4. **Error Analysis:** Pay special attention to tweets with mixed signals

---

*Report generated automatically by Phase 3 pipeline*
