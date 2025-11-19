"""
Phase 3: Complete Pipeline - Cross-Domain Validation & Qualitative Analysis

This script runs the complete Phase 3 pipeline:
1. Cross-domain validation (Sentiment140 -> Airline)
2. Qualitative analysis with semantic interpretation
3. Comprehensive reporting

Author: Phase 3 Implementation
Date: November 2025
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from preprocessing.data_loader import SentimentDataLoader
from preprocessing.preprocessing import TweetPreprocessor
from features.feature_pipeline import FeatureExtractionPipeline
from models.traditional_models import SentimentClassifier
from scripts.cross_domain_validation import CrossDomainValidator
from scripts.qualitative_analysis import QualitativeAnalyzer


class Phase3Pipeline:
    """
    Complete Phase 3 pipeline integrating cross-domain validation
    and qualitative analysis.
    """
    
    def __init__(
        self,
        dataset_dir: str = "datasets",
        output_dir: str = "analysis_results"
    ):
        """
        Initialize Phase 3 pipeline.
        
        Args:
            dataset_dir: Directory containing datasets
            output_dir: Directory for saving results
        """
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.cross_domain_validator = CrossDomainValidator(
            dataset_dir=str(self.dataset_dir),
            output_dir=str(self.output_dir)
        )
        
        self.qualitative_analyzer = QualitativeAnalyzer(
            output_dir=str(self.output_dir)
        )
        
        self.results = {}
    
    def run_complete_pipeline(
        self,
        source_sample_size: int = 50000,
        max_features: int = 5000,
        n_qualitative_samples: int = 15
    ) -> Dict[str, Any]:
        """
        Run complete Phase 3 pipeline.
        
        Args:
            source_sample_size: Number of samples from Sentiment140
            max_features: Maximum TF-IDF features
            n_qualitative_samples: Samples per category for qualitative analysis
            
        Returns:
            Dictionary with all results
        """
        print("="*80)
        print("PHASE 3: COMPLETE PIPELINE")
        print("Cross-Domain Validation & Qualitative Analysis")
        print("="*80)
        print(f"\nConfiguration:")
        print(f"  Source Domain: Sentiment140 (sample: {source_sample_size:,})")
        print(f"  Target Domain: Twitter US Airline Sentiment")
        print(f"  Max Features: {max_features:,}")
        print(f"  Qualitative Samples: {n_qualitative_samples} per category")
        
        # Step 1: Cross-Domain Validation
        print("\n" + "="*80)
        print("STEP 1: CROSS-DOMAIN VALIDATION")
        print("="*80)
        
        cross_domain_results = self.cross_domain_validator.run_cross_domain_validation(
            source_sample_size=source_sample_size,
            max_features=max_features
        )
        
        # Step 2: Get best model for qualitative analysis
        print("\n" + "="*80)
        print("STEP 2: TRAINING BEST MODEL FOR QUALITATIVE ANALYSIS")
        print("="*80)
        
        best_model_info = cross_domain_results['cross_domain_results'][0]  # First model (usually best)
        print(f"\nUsing best model: {best_model_info['model_name']}")
        print(f"Cross-domain F1-score: {best_model_info['target_metrics']['f1_score']:.4f}")
        
        # Train best model on source, predict on target
        source_df = self.cross_domain_validator.source_df
        target_df = self.cross_domain_validator.target_df
        
        # Recreate features
        feature_pipeline = FeatureExtractionPipeline()
        X_source = feature_pipeline.fit_transform(
            source_df['processed_text'].tolist(),
            max_features=max_features
        )
        y_source = source_df['sentiment'].values
        
        X_target = feature_pipeline.transform(
            target_df['processed_text'].tolist()
        )
        y_target = target_df['sentiment'].values
        
        # Train best model
        best_model = SentimentClassifier(
            model_type=best_model_info['model_type'],
            **best_model_info['model_params']
        )
        best_model.fit(X_source, y_source)
        
        # Predict on target
        y_pred = best_model.predict(X_target)
        y_proba = best_model.predict_proba(X_target)
        
        # Step 3: Qualitative Analysis
        print("\n" + "="*80)
        print("STEP 3: QUALITATIVE ANALYSIS WITH SEMANTIC INTERPRETATION")
        print("="*80)
        
        qualitative_results = self.qualitative_analyzer.run_full_analysis(
            df=target_df,
            y_true=y_target,
            y_pred=y_pred,
            y_proba=y_proba,
            n_samples=n_qualitative_samples
        )
        
        # Step 4: Generate comprehensive report
        print("\n" + "="*80)
        print("STEP 4: GENERATING COMPREHENSIVE REPORT")
        print("="*80)
        
        comprehensive_report = self.generate_comprehensive_report(
            cross_domain_results,
            qualitative_results
        )
        
        # Save comprehensive report
        report_path = self.output_dir / "phase3_comprehensive_report.json"
        with open(report_path, 'w') as f:
            json.dump(comprehensive_report, f, indent=2)
        
        print(f"\nComprehensive report saved to: {report_path}")
        
        # Generate markdown report
        self.generate_markdown_report(comprehensive_report)
        
        return comprehensive_report
    
    def generate_comprehensive_report(
        self,
        cross_domain_results: Dict[str, Any],
        qualitative_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive report combining all analyses.
        
        Args:
            cross_domain_results: Results from cross-domain validation
            qualitative_results: Results from qualitative analysis
            
        Returns:
            Comprehensive report dictionary
        """
        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'phase': 'Phase 3: Validation & Qualitative Analysis',
                'source_domain': 'Sentiment140',
                'target_domain': 'Twitter US Airline Sentiment'
            },
            
            'cross_domain_validation': {
                'best_model': cross_domain_results['best_model'],
                'best_f1_score': cross_domain_results['best_f1_score'],
                'model_comparison': []
            },
            
            'qualitative_analysis': {
                'accuracy': qualitative_results['sample_analysis']['accuracy'],
                'total_samples': qualitative_results['sample_analysis']['total_samples'],
                'correct_predictions': qualitative_results['sample_analysis']['correct_predictions'],
                'insights': qualitative_results['insights'],
                'semantic_patterns': qualitative_results['pattern_analysis']
            },
            
            'key_findings': []
        }
        
        # Add model comparison
        for model_result in cross_domain_results['cross_domain_results']:
            report['cross_domain_validation']['model_comparison'].append({
                'model_name': model_result['model_name'],
                'source_f1': model_result['source_metrics']['f1_score'],
                'target_f1': model_result['target_metrics']['f1_score'],
                'domain_gap': model_result['domain_gap_f1']
            })
        
        # Generate key findings
        report['key_findings'] = self.extract_key_findings(
            cross_domain_results,
            qualitative_results
        )
        
        return report
    
    def extract_key_findings(
        self,
        cross_domain_results: Dict[str, Any],
        qualitative_results: Dict[str, Any]
    ) -> List[str]:
        """Extract key findings from all analyses."""
        findings = []
        
        # Cross-domain performance
        best_model = cross_domain_results['best_model']
        best_f1 = cross_domain_results['best_f1_score']
        findings.append(
            f"Best cross-domain model: {best_model} with F1-score of {best_f1:.4f}"
        )
        
        # Domain gap analysis
        best_result = cross_domain_results['cross_domain_results'][0]
        domain_gap = best_result['domain_gap_f1']
        findings.append(
            f"Domain gap: {domain_gap:.4f} (source: {best_result['source_metrics']['f1_score']:.4f}, "
            f"target: {best_result['target_metrics']['f1_score']:.4f})"
        )
        
        # Semantic insights
        findings.extend(qualitative_results['insights'])
        
        # Pattern insights
        patterns = qualitative_results['pattern_analysis']
        neg_diff = patterns['comparison']['avg_negations']['difference']
        if abs(neg_diff) > 0.1:
            findings.append(
                f"Negation handling: {'Good' if neg_diff > 0 else 'Needs improvement'} "
                f"(diff: {neg_diff:+.2f})"
            )
        
        return findings
    
    def generate_markdown_report(self, report: Dict[str, Any]):
        """Generate human-readable markdown report."""
        report_path = self.output_dir / "PHASE3_REPORT.md"
        
        with open(report_path, 'w') as f:
            f.write("# Phase 3: Cross-Domain Validation & Qualitative Analysis\n\n")
            f.write(f"**Generated:** {report['metadata']['timestamp']}\n\n")
            f.write("---\n\n")
            
            # Overview
            f.write("## Overview\n\n")
            f.write(f"- **Source Domain:** {report['metadata']['source_domain']}\n")
            f.write(f"- **Target Domain:** {report['metadata']['target_domain']}\n")
            f.write(f"- **Best Model:** {report['cross_domain_validation']['best_model']}\n")
            f.write(f"- **Cross-Domain F1-Score:** {report['cross_domain_validation']['best_f1_score']:.4f}\n\n")
            
            # Cross-Domain Results
            f.write("## Cross-Domain Validation Results\n\n")
            f.write("| Model | Source F1 | Target F1 | Domain Gap |\n")
            f.write("|-------|-----------|-----------|------------|\n")
            for model in report['cross_domain_validation']['model_comparison']:
                f.write(f"| {model['model_name']} | {model['source_f1']:.4f} | "
                       f"{model['target_f1']:.4f} | {model['domain_gap']:.4f} |\n")
            f.write("\n")
            
            # Qualitative Analysis
            f.write("## Qualitative Analysis Summary\n\n")
            f.write(f"- **Total Samples Analyzed:** {report['qualitative_analysis']['total_samples']}\n")
            f.write(f"- **Correctly Classified:** {report['qualitative_analysis']['correct_predictions']}\n")
            f.write(f"- **Accuracy:** {report['qualitative_analysis']['accuracy']:.2%}\n\n")
            
            # Semantic Patterns
            f.write("### Semantic Pattern Comparison\n\n")
            patterns = report['qualitative_analysis']['semantic_patterns']['comparison']
            f.write("| Metric | Correct | Incorrect | Difference |\n")
            f.write("|--------|---------|-----------|------------|\n")
            for metric, values in patterns.items():
                f.write(f"| {metric} | {values['correct']:.3f} | "
                       f"{values['incorrect']:.3f} | {values['difference']:+.3f} |\n")
            f.write("\n")
            
            # Key Findings
            f.write("## Key Findings\n\n")
            for i, finding in enumerate(report['key_findings'], 1):
                f.write(f"{i}. {finding}\n")
            f.write("\n")
            
            # Insights
            f.write("## Detailed Insights\n\n")
            for i, insight in enumerate(report['qualitative_analysis']['insights'], 1):
                f.write(f"**Insight {i}:** {insight}\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("Based on the analysis, we recommend:\n\n")
            f.write("1. **Model Selection:** Use Logistic Regression for cross-domain sentiment analysis\n")
            f.write("2. **Feature Engineering:** Focus on robust handling of negations and intensifiers\n")
            f.write("3. **Domain Adaptation:** Consider domain-specific fine-tuning for airline sentiment\n")
            f.write("4. **Error Analysis:** Pay special attention to tweets with mixed signals\n\n")
            
            f.write("---\n\n")
            f.write("*Report generated automatically by Phase 3 pipeline*\n")
        
        print(f"\nMarkdown report saved to: {report_path}")


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Phase 3: Complete Pipeline - Cross-Domain Validation & Qualitative Analysis'
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='datasets',
        help='Directory containing datasets'
    )
    parser.add_argument(
        '--source-sample-size',
        type=int,
        default=50000,
        help='Number of samples from Sentiment140'
    )
    parser.add_argument(
        '--max-features',
        type=int,
        default=5000,
        help='Maximum TF-IDF features'
    )
    parser.add_argument(
        '--n-qualitative-samples',
        type=int,
        default=15,
        help='Number of samples per category for qualitative analysis'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='analysis_results',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Initialize and run pipeline
    pipeline = Phase3Pipeline(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir
    )
    
    results = pipeline.run_complete_pipeline(
        source_sample_size=args.source_sample_size,
        max_features=args.max_features,
        n_qualitative_samples=args.n_qualitative_samples
    )
    
    print("\n" + "="*80)
    print("PHASE 3 COMPLETE")
    print("="*80)
    print(f"\nKey Results:")
    print(f"  Best Model: {results['cross_domain_validation']['best_model']}")
    print(f"  Cross-Domain F1: {results['cross_domain_validation']['best_f1_score']:.4f}")
    print(f"  Qualitative Accuracy: {results['qualitative_analysis']['accuracy']:.2%}")
    print(f"\nReports saved to: {args.output_dir}/")
    print(f"  - phase3_comprehensive_report.json")
    print(f"  - PHASE3_REPORT.md")
    print(f"  - cross_domain_validation.json")
    print(f"  - qualitative_analysis.json")


if __name__ == "__main__":
    main()
