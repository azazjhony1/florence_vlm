#!/usr/bin/env python3
"""
Compare and visualize benchmark results
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_results(results_file="benchmark_results.json"):
    """Load benchmark results"""
    with open(results_file, 'r') as f:
        return json.load(f)


def plot_comparison(results):
    """Create comparison plots"""
    
    opt_names = list(results.keys())
    total_times = [results[name]['metrics']['total_time'] * 1000 for name in opt_names]
    prefill_times = [results[name]['metrics'].get('prefill_time', 0) * 1000 for name in opt_names]
    decode_times = [results[name]['metrics'].get('decode_time', 0) * 1000 for name in opt_names]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Total time comparison
    colors = plt.cm.viridis(np.linspace(0, 1, len(opt_names)))
    bars = ax1.barh(opt_names, total_times, color=colors)
    ax1.set_xlabel('Total Time (ms)')
    ax1.set_title('Total Inference Time Comparison')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add speedup annotations
    if total_times:
        baseline_time = total_times[0]  # Assume first is baseline
        for i, (bar, time) in enumerate(zip(bars, total_times)):
            speedup = baseline_time / time if time > 0 else 0
            ax1.text(time, bar.get_y() + bar.get_height()/2, 
                    f'  {time:.0f}ms ({speedup:.2f}x)',
                    va='center')
    
    # Plot 2: Prefill vs Decode breakdown
    x = np.arange(len(opt_names))
    width = 0.35
    
    ax2.bar(x, prefill_times, width, label='Prefill', color='steelblue')
    ax2.bar(x, decode_times, width, bottom=prefill_times, label='Decode', color='coral')
    
    ax2.set_ylabel('Time (ms)')
    ax2.set_title('Prefill vs Decode Time Breakdown')
    ax2.set_xticks(x)
    ax2.set_xticklabels(opt_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('benchmark_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved plot to: benchmark_comparison.png")
    plt.show()


def print_detailed_comparison(results):
    """Print detailed comparison table"""
    
    print("\n" + "="*90)
    print("DETAILED PERFORMANCE COMPARISON")
    print("="*90)
    
    baseline_name = list(results.keys())[0]
    baseline = results[baseline_name]['metrics']
    
    print(f"\n{'Metric':<30} {'Baseline':<15} {'Best Opt':<15} {'Improvement':<15}")
    print("-" * 90)
    
    metrics_to_compare = [
        ('total_time', 'Total Time', 'ms'),
        ('prefill_time', 'Prefill Time', 'ms'),
        ('decode_time', 'Decode Time', 'ms'),
        ('vision_encoder_time', 'Vision Encoding', 'ms'),
        ('projection_time', 'Projection', 'ms'),
        ('avg_token_time', 'Avg Token Time', 'ms'),
    ]
    
    for metric_key, metric_name, unit in metrics_to_compare:
        baseline_val = baseline.get(metric_key, 0) * 1000
        
        # Find best optimization for this metric
        best_val = baseline_val
        best_opt = baseline_name
        
        for opt_name, data in results.items():
            val = data['metrics'].get(metric_key, 0) * 1000
            if val > 0 and val < best_val:
                best_val = val
                best_opt = opt_name
        
        improvement = baseline_val / best_val if best_val > 0 else 0
        
        print(f"{metric_name:<30} {baseline_val:>10.2f}{unit:<4} {best_val:>10.2f}{unit:<4} {improvement:>8.2f}x ({best_opt})")
    
    print("="*90)


def main():
    results = load_results()
    
    print_detailed_comparison(results)
    
    try:
        plot_comparison(results)
    except Exception as e:
        print(f"Could not create plots: {e}")
        print("Install matplotlib to see visualizations: pip install matplotlib")


if __name__ == "__main__":
    main()