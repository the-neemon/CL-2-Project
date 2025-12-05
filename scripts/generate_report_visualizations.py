"""
Generate Comprehensive Visualizations for Research Report

Creates figures, tables, and charts including:
- Model performance comparison charts
- Cross-domain validation results
- Feature importance analysis
- Statistical summaries
- Publication-ready tables
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Set publication-ready style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'


class ReportVisualizer:
    """Generate comprehensive visualizations for research report."""
    
    def __init__(self, results_dir: str = "analysis_results"):
        """Initialize visualizer with results directory."""
        self.results_dir = Path(results_dir)
        self.output_dir = self.results_dir / "report_figures"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Load results
        self.cross_domain_results = self._load_json("cross_domain_validation.json")
        self.qualitative_results = self._load_json("qualitative_analysis.json")
        
    def _load_json(self, filename: str) -> Dict:
        """Load JSON results file."""
        filepath = self.results_dir / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                return json.load(f)
        return {}
    
    def generate_all_visualizations(self):
        """Generate all figures and tables for the report."""
        print("="*70)
        print("GENERATING REPORT VISUALIZATIONS")
        print("="*70)
        
        # Figure 1: Model Performance Comparison
        print("\n[1/8] Generating model performance comparison...")
        self.plot_model_comparison()
        
        # Figure 2: Domain Transfer Analysis
        print("[2/8] Generating domain transfer analysis...")
        self.plot_domain_transfer()
        
        # Figure 3: Metric Comparison Radar Chart
        print("[3/8] Generating radar chart...")
        self.plot_radar_chart()
        
        # Figure 4: Performance Heatmap
        print("[4/8] Generating performance heatmap...")
        self.plot_performance_heatmap()
        
        # Figure 5: Domain Gap Analysis
        print("[5/8] Generating domain gap analysis...")
        self.plot_domain_gap()
        
        # Figure 6: Semantic Feature Analysis
        print("[6/8] Generating semantic feature analysis...")
        self.plot_semantic_features()
        
        # Table 1: Statistical Summary
        print("[7/8] Generating statistical summary table...")
        self.generate_summary_table()
        
        # Table 2: Model Comparison Table
        print("[8/9] Generating model comparison table...")
        self.generate_comparison_table()
        
        # Figure 7: Comprehensive overview (2x2 subplots)
        print("[9/9] Generating comprehensive overview...")
        self.plot_comprehensive_overview()
        
        print("\n" + "="*70)
        print("✅ ALL VISUALIZATIONS GENERATED")
        print("="*70)
        print(f"\nOutput directory: {self.output_dir}")
        print("\nGenerated files:")
        for file in sorted(self.output_dir.glob("*")):
            print(f"  - {file.name}")
    
    def plot_model_comparison(self):
        """Figure 1: Bar chart comparing model performance across metrics."""
        if not self.cross_domain_results:
            return
        
        results = self.cross_domain_results['cross_domain_results']
        
        # Extract data
        models = [r['model_name'] for r in results]
        metrics_data = {
            'Accuracy': [r['target_metrics']['accuracy'] for r in results],
            'Precision': [r['target_metrics']['precision'] for r in results],
            'Recall': [r['target_metrics']['recall'] for r in results],
            'F1-Score': [r['target_metrics']['f1_score'] for r in results]
        }
        
        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(models))
        width = 0.2
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        
        for i, (metric, values) in enumerate(metrics_data.items()):
            offset = width * (i - 1.5)
            bars = ax.bar(x + offset, values, width, label=metric, 
                         color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=8, rotation=0)
        
        ax.set_xlabel('Model', fontweight='bold', fontsize=12)
        ax.set_ylabel('Score', fontweight='bold', fontsize=12)
        ax.set_title('Model Performance Comparison (Cross-Domain Validation)', 
                    fontweight='bold', fontsize=14, pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha='right')
        ax.legend(loc='lower right', frameon=True, shadow=True)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim([0, 1.0])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure1_model_comparison.png', 
                   bbox_inches='tight', facecolor='white')
        plt.close()
    
    def plot_domain_transfer(self):
        """Figure 2: Line plot showing in-domain vs cross-domain performance."""
        if not self.cross_domain_results:
            return
        
        results = self.cross_domain_results['cross_domain_results']
        
        models = [r['model_name'] for r in results]
        source_f1 = [r['source_metrics']['f1_score'] for r in results]
        target_f1 = [r['target_metrics']['f1_score'] for r in results]
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        x = np.arange(len(models))
        
        # Plot lines
        ax.plot(x, source_f1, marker='o', linewidth=2.5, markersize=10,
               label='Source Domain (Sentiment140)', color='#3498db', 
               markeredgecolor='black', markeredgewidth=1)
        ax.plot(x, target_f1, marker='s', linewidth=2.5, markersize=10,
               label='Target Domain (Airline)', color='#e74c3c',
               markeredgecolor='black', markeredgewidth=1)
        
        # Add value labels
        for i, (s, t) in enumerate(zip(source_f1, target_f1)):
            ax.text(i, s + 0.02, f'{s:.3f}', ha='center', fontsize=9, 
                   fontweight='bold', color='#2980b9')
            ax.text(i, t - 0.04, f'{t:.3f}', ha='center', fontsize=9,
                   fontweight='bold', color='#c0392b')
        
        # Add shaded region between lines
        ax.fill_between(x, source_f1, target_f1, alpha=0.2, color='gray')
        
        ax.set_xlabel('Model', fontweight='bold', fontsize=12)
        ax.set_ylabel('F1-Score', fontweight='bold', fontsize=12)
        ax.set_title('Domain Transfer Performance: In-Domain vs Cross-Domain', 
                    fontweight='bold', fontsize=14, pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha='right')
        ax.legend(loc='best', frameon=True, shadow=True, fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim([0.4, 0.85])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure2_domain_transfer.png',
                   bbox_inches='tight', facecolor='white')
        plt.close()
    
    def plot_radar_chart(self):
        """Figure 3: Radar chart comparing top 3 models across all metrics."""
        if not self.cross_domain_results:
            return
        
        results = self.cross_domain_results['cross_domain_results']
        
        # Select top 3 models by F1-score
        sorted_results = sorted(results, 
                               key=lambda x: x['target_metrics']['f1_score'], 
                               reverse=True)[:3]
        
        # Metrics to compare
        categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        num_vars = len(categories)
        
        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        for i, result in enumerate(sorted_results):
            values = [
                result['target_metrics']['accuracy'],
                result['target_metrics']['precision'],
                result['target_metrics']['recall'],
                result['target_metrics']['f1_score']
            ]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=result['model_name'],
                   color=colors[i], markersize=8)
            ax.fill(angles, values, alpha=0.15, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        ax.set_title('Top 3 Models: Performance Metrics Comparison',
                    fontweight='bold', fontsize=14, pad=30)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), 
                 frameon=True, shadow=True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure3_radar_chart.png',
                   bbox_inches='tight', facecolor='white')
        plt.close()
    
    def plot_performance_heatmap(self):
        """Figure 4: Heatmap showing all metrics for all models."""
        if not self.cross_domain_results:
            return
        
        results = self.cross_domain_results['cross_domain_results']
        
        # Create data matrix
        models = [r['model_name'] for r in results]
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        data = []
        for result in results:
            row = [
                result['target_metrics']['accuracy'],
                result['target_metrics']['precision'],
                result['target_metrics']['recall'],
                result['target_metrics']['f1_score']
            ]
            data.append(row)
        
        df = pd.DataFrame(data, index=models, columns=metrics)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(df, annot=True, fmt='.4f', cmap='RdYlGn', 
                   cbar_kws={'label': 'Score'}, linewidths=1, linecolor='black',
                   vmin=0.5, vmax=0.8, ax=ax)
        
        ax.set_title('Performance Heatmap: Cross-Domain Validation Results',
                    fontweight='bold', fontsize=14, pad=20)
        ax.set_xlabel('Metrics', fontweight='bold', fontsize=12)
        ax.set_ylabel('Models', fontweight='bold', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure4_performance_heatmap.png',
                   bbox_inches='tight', facecolor='white')
        plt.close()
    
    def plot_domain_gap(self):
        """Figure 5: Bar chart showing domain gap (F1-score difference)."""
        if not self.cross_domain_results:
            return
        
        results = self.cross_domain_results['cross_domain_results']
        
        models = [r['model_name'] for r in results]
        gaps = [r['domain_gap_f1'] for r in results]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Color bars based on positive/negative gap
        colors = ['#e74c3c' if g < 0 else '#2ecc71' for g in gaps]
        
        bars = ax.barh(models, gaps, color=colors, alpha=0.7, 
                      edgecolor='black', linewidth=1)
        
        # Add value labels
        for i, (bar, gap) in enumerate(zip(bars, gaps)):
            width = bar.get_width()
            label_x = width + (0.01 if width > 0 else -0.01)
            ax.text(label_x, bar.get_y() + bar.get_height()/2,
                   f'{gap:.4f}',
                   ha='left' if width > 0 else 'right',
                   va='center', fontweight='bold', fontsize=10)
        
        ax.axvline(x=0, color='black', linewidth=2, linestyle='-')
        ax.set_xlabel('Domain Gap (Source F1 - Target F1)', 
                     fontweight='bold', fontsize=12)
        ax.set_ylabel('Model', fontweight='bold', fontsize=12)
        ax.set_title('Domain Gap Analysis: Performance Degradation Across Domains',
                    fontweight='bold', fontsize=14, pad=20)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#e74c3c', alpha=0.7, label='Negative Gap (Target < Source)'),
            Patch(facecolor='#2ecc71', alpha=0.7, label='Positive Gap (Target > Source)')
        ]
        ax.legend(handles=legend_elements, loc='lower right', frameon=True, shadow=True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure5_domain_gap.png',
                   bbox_inches='tight', facecolor='white')
        plt.close()
    
    def plot_semantic_features(self):
        """Figure 6: Semantic feature analysis (correct vs incorrect predictions)."""
        if not self.qualitative_results:
            return
        
        # Extract pattern analysis data
        pattern_data = self.qualitative_results.get('pattern_analysis', {})
        
        if not pattern_data:
            print("  Note: No semantic pattern data available")
            return
        
        correct_data = pattern_data.get('correct_predictions', {})
        incorrect_data = pattern_data.get('incorrect_predictions', {})
        
        # Prepare data
        features = ['Negations', 'Intensifiers', 'VADER Score', 'Text Length']
        correct = [
            correct_data.get('avg_negations', 0),
            correct_data.get('avg_intensifiers', 0),
            correct_data.get('avg_vader_score', 0),
            correct_data.get('avg_text_length', 0) / 10  # Normalize
        ]
        incorrect = [
            incorrect_data.get('avg_negations', 0),
            incorrect_data.get('avg_intensifiers', 0),
            incorrect_data.get('avg_vader_score', 0),
            incorrect_data.get('avg_text_length', 0) / 10  # Normalize
        ]
        
        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(features))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, correct, width, label='Correct Predictions',
                      color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1)
        bars2 = ax.bar(x + width/2, incorrect, width, label='Incorrect Predictions',
                      color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Semantic Features', fontweight='bold', fontsize=12)
        ax.set_ylabel('Average Count/Score', fontweight='bold', fontsize=12)
        ax.set_title('Semantic Feature Analysis: Correct vs Incorrect Predictions',
                    fontweight='bold', fontsize=14, pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(features)
        ax.legend(frameon=True, shadow=True, fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure6_semantic_features.png',
                   bbox_inches='tight', facecolor='white')
        plt.close()
    
    def generate_summary_table(self):
        """Table 1: Statistical summary of results."""
        if not self.cross_domain_results:
            return
        
        results = self.cross_domain_results['cross_domain_results']
        
        # Create summary dataframe
        data = []
        for result in results:
            data.append({
                'Model': result['model_name'],
                'Source Accuracy': f"{result['source_metrics']['accuracy']:.4f}",
                'Source F1': f"{result['source_metrics']['f1_score']:.4f}",
                'Target Accuracy': f"{result['target_metrics']['accuracy']:.4f}",
                'Target F1': f"{result['target_metrics']['f1_score']:.4f}",
                'Domain Gap (F1)': f"{result['domain_gap_f1']:.4f}"
            })
        
        df = pd.DataFrame(data)
        
        # Create figure with table
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=df.values, colLabels=df.columns,
                        cellLoc='center', loc='center',
                        colColours=['#3498db']*len(df.columns))
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#2c3e50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(df) + 1):
            for j in range(len(df.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#ecf0f1')
                else:
                    table[(i, j)].set_facecolor('white')
        
        plt.title('Table 1: Statistical Summary of Cross-Domain Validation Results',
                 fontweight='bold', fontsize=14, pad=20)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'table1_summary.png',
                   bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Also save as CSV
        df.to_csv(self.output_dir / 'table1_summary.csv', index=False)
    
    def generate_comparison_table(self):
        """Table 2: Detailed model comparison with all metrics."""
        if not self.cross_domain_results:
            return
        
        results = self.cross_domain_results['cross_domain_results']
        
        # Create detailed comparison
        data = []
        for result in results:
            data.append({
                'Model': result['model_name'],
                'Accuracy': f"{result['target_metrics']['accuracy']:.4f}",
                'Precision': f"{result['target_metrics']['precision']:.4f}",
                'Recall': f"{result['target_metrics']['recall']:.4f}",
                'F1-Score': f"{result['target_metrics']['f1_score']:.4f}",
                'Rank': ''  # Will fill later
            })
        
        df = pd.DataFrame(data)
        
        # Add ranking based on F1-score
        f1_scores = [float(d['F1-Score']) for d in data]
        ranks = pd.Series(f1_scores).rank(ascending=False, method='min').astype(int)
        df['Rank'] = ranks.values
        df = df.sort_values('Rank')
        
        # Create figure with table
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=df.values, colLabels=df.columns,
                        cellLoc='center', loc='center',
                        colColours=['#e74c3c']*len(df.columns))
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#c0392b')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Highlight best model
        for j in range(len(df.columns)):
            table[(1, j)].set_facecolor('#f1c40f')
            table[(1, j)].set_text_props(weight='bold')
        
        # Alternate row colors for rest
        for i in range(2, len(df) + 1):
            for j in range(len(df.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#ecf0f1')
                else:
                    table[(i, j)].set_facecolor('white')
        
        plt.title('Table 2: Model Performance Comparison (Ranked by F1-Score)',
                 fontweight='bold', fontsize=14, pad=20)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'table2_comparison.png',
                   bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Also save as CSV
        df.to_csv(self.output_dir / 'table2_comparison.csv', index=False)
    
    def plot_comprehensive_overview(self):
        """Figure 7: Comprehensive 2x2 overview with multiple visualizations."""
        if not self.cross_domain_results:
            return
        
        results = self.cross_domain_results['cross_domain_results']
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Subplot 1: F1-Score comparison
        ax1 = fig.add_subplot(gs[0, 0])
        models = [r['model_name'] for r in results]
        f1_scores = [r['target_metrics']['f1_score'] for r in results]
        colors_f1 = ['#2ecc71' if f1 > 0.7 else '#e74c3c' if f1 < 0.6 else '#f39c12' 
                     for f1 in f1_scores]
        
        bars = ax1.barh(models, f1_scores, color=colors_f1, alpha=0.8, 
                        edgecolor='black', linewidth=1)
        
        for i, (bar, score) in enumerate(zip(bars, f1_scores)):
            ax1.text(score + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{score:.4f}', va='center', fontweight='bold', fontsize=9)
        
        ax1.set_xlabel('F1-Score', fontweight='bold')
        ax1.set_title('(A) Model F1-Score Comparison', fontweight='bold', fontsize=12)
        ax1.grid(axis='x', alpha=0.3, linestyle='--')
        ax1.set_xlim([0.5, 0.8])
        
        # Subplot 2: Accuracy vs F1-Score scatter
        ax2 = fig.add_subplot(gs[0, 1])
        accuracies = [r['target_metrics']['accuracy'] for r in results]
        
        scatter = ax2.scatter(accuracies, f1_scores, s=300, alpha=0.6,
                             c=range(len(results)), cmap='viridis',
                             edgecolors='black', linewidths=2)
        
        for i, model in enumerate(models):
            ax2.annotate(model, (accuracies[i], f1_scores[i]),
                        fontsize=8, ha='center', va='bottom',
                        fontweight='bold')
        
        ax2.set_xlabel('Accuracy', fontweight='bold')
        ax2.set_ylabel('F1-Score', fontweight='bold')
        ax2.set_title('(B) Accuracy vs F1-Score', fontweight='bold', fontsize=12)
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        # Add diagonal reference line
        lims = [0.55, 0.75]
        ax2.plot(lims, lims, 'k--', alpha=0.3, zorder=0, linewidth=2)
        
        # Subplot 3: Precision-Recall comparison
        ax3 = fig.add_subplot(gs[1, 0])
        precisions = [r['target_metrics']['precision'] for r in results]
        recalls = [r['target_metrics']['recall'] for r in results]
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, precisions, width, label='Precision',
                       color='#3498db', alpha=0.8, edgecolor='black', linewidth=1)
        bars2 = ax3.bar(x + width/2, recalls, width, label='Recall',
                       color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1)
        
        ax3.set_xlabel('Model', fontweight='bold')
        ax3.set_ylabel('Score', fontweight='bold')
        ax3.set_title('(C) Precision vs Recall', fontweight='bold', fontsize=12)
        ax3.set_xticks(x)
        ax3.set_xticklabels(models, rotation=15, ha='right', fontsize=8)
        ax3.legend(frameon=True, shadow=True)
        ax3.grid(axis='y', alpha=0.3, linestyle='--')
        ax3.set_ylim([0.5, 0.8])
        
        # Subplot 4: Domain gap distribution
        ax4 = fig.add_subplot(gs[1, 1])
        gaps = [r['domain_gap_f1'] for r in results]
        colors_gap = ['#e74c3c' if g < 0 else '#2ecc71' for g in gaps]
        
        bars = ax4.bar(range(len(models)), gaps, color=colors_gap, alpha=0.7,
                      edgecolor='black', linewidth=1)
        
        for bar, gap in zip(bars, gaps):
            height = bar.get_height()
            label_y = height + (0.01 if height > 0 else -0.02)
            ax4.text(bar.get_x() + bar.get_width()/2, label_y,
                    f'{gap:.3f}', ha='center',
                    va='bottom' if height > 0 else 'top',
                    fontweight='bold', fontsize=9)
        
        ax4.axhline(y=0, color='black', linewidth=2, linestyle='-')
        ax4.set_xlabel('Model', fontweight='bold')
        ax4.set_ylabel('Domain Gap (F1)', fontweight='bold')
        ax4.set_title('(D) Domain Transfer Gap', fontweight='bold', fontsize=12)
        ax4.set_xticks(range(len(models)))
        ax4.set_xticklabels(models, rotation=15, ha='right', fontsize=8)
        ax4.grid(axis='y', alpha=0.3, linestyle='--')
        
        fig.suptitle('Comprehensive Analysis Overview: Cross-Domain Validation Results',
                    fontweight='bold', fontsize=16, y=0.995)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure7_comprehensive_overview.png',
                   bbox_inches='tight', facecolor='white')
        plt.close()


def main():
    """Main execution function."""
    visualizer = ReportVisualizer()
    visualizer.generate_all_visualizations()
    
    print("\n" + "="*70)
    print("📊 REPORT READY!")
    print("="*70)
    print("\nGenerated visualizations:")
    print("  📈 7 Figures (charts and graphs)")
    print("  📋 2 Tables (statistical summaries)")
    print("  📄 2 CSV files (data tables)")
    print("\nAll files saved to: analysis_results/report_figures/")
    print("\nYou can now include these in your research report!")


if __name__ == '__main__':
    main()
