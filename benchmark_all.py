#!/usr/bin/env python3
"""
Benchmark all optimization techniques using the existing profiler
"""

import sys
import importlib
from pathlib import Path
from dataclasses import asdict
import json

# Import your existing profiler
from florence2_profiler import Florence2Profiler, ProfilingMetrics


class OptimizationBenchmarker:
    """Benchmark different optimization techniques"""
    
    def __init__(self, image_path: str, device: str = "cpu"):
        self.image_path = image_path
        self.device = device
        self.results = {}
    
    def benchmark_optimization(self, opt_name: str, opt_module):
        """Benchmark a single optimization"""
        print(f"\n{'='*70}")
        print(f"BENCHMARKING: {opt_name}")
        print(f"{'='*70}")
        
        try:
            # Load optimized model
            model, processor, description = opt_module.get_model_and_processor(self.device)
            print(f"Description: {description}\n")
            
            # Create a profiler instance with this model
            profiler = Florence2Profiler.__new__(Florence2Profiler)
            profiler.device = self.device
            profiler.model_name = opt_name
            profiler.model = model
            profiler.processor = processor
            profiler.hook_times = {}
            
            # Run profiling
            metrics = profiler.profile_end_to_end(
                self.image_path,
                task="<CAPTION>",
                max_new_tokens=100,
                verbose=True
            )
            
            # Store results
            self.results[opt_name] = {
                'description': description,
                'metrics': asdict(metrics)
            }
            
            return metrics
            
        except Exception as e:
            print(f"ERROR benchmarking {opt_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_all_benchmarks(self):
        """Run all optimization benchmarks"""
        
        optimizations = [
            ('baseline', 'optimizations.baseline'),
            ('opt1_quantization', 'optimizations.opt1_quantization'),
            # ('opt2_onnx', 'optimizations.opt2_onnx_runtime'),
            # ('opt3_resolution', 'optimizations.opt3_reduced_resolution'),
            # ('opt4_smaller', 'optimizations.opt4_smaller_model'),
            # ('opt5_combined', 'optimizations.opt5_combined'),
        ]
        
        for opt_name, module_name in optimizations:
            try:
                opt_module = importlib.import_module(module_name)
                self.benchmark_optimization(opt_name, opt_module)
            except ImportError as e:
                print(f"\nSkipping {opt_name}: {e}")
                continue
            except Exception as e:
                print(f"\nError with {opt_name}: {e}")
                continue
        
        # Save all results
        self.save_results()
    
    def save_results(self):
        """Save benchmark results"""
        output_file = Path("benchmark_results.json")
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"Results saved to: {output_file}")
        print(f"{'='*70}\n")
        
        # Print comparison table
        self.print_comparison()
    
    def print_comparison(self):
        """Print comparison table"""
        print("\n" + "="*70)
        print("PERFORMANCE COMPARISON")
        print("="*70)
        
        if not self.results:
            print("No results to compare")
            return
        
        # Get baseline time
        baseline_time = None
        if 'baseline' in self.results:
            baseline_time = self.results['baseline']['metrics']['total_time']
        
        print(f"\n{'Optimization':<25} {'Total Time':<15} {'Speedup':<10} {'Prefill':<12} {'Decode':<12}")
        print("-" * 70)
        
        for opt_name, data in self.results.items():
            metrics = data['metrics']
            total_time = metrics['total_time']
            prefill = metrics.get('prefill_time', 0)
            decode = metrics.get('decode_time', 0)
            
            speedup = f"{baseline_time/total_time:.2f}x" if baseline_time else "N/A"
            
            print(f"{opt_name:<25} {total_time*1000:>10.2f}ms   {speedup:<10} {prefill*1000:>8.2f}ms   {decode*1000:>8.2f}ms")
        
        print("="*70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark all Florence-2 optimizations")
    parser.add_argument('--image', type=str, required=True, help='Path to test image')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    
    args = parser.parse_args()
    
    benchmarker = OptimizationBenchmarker(args.image, args.device)
    benchmarker.run_all_benchmarks()


if __name__ == "__main__":
    main()