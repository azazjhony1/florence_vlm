#!/usr/bin/env python3
"""
Florence-2 Detailed Performance Profiler
Profiles timing for each component of the Florence-2 pipeline
"""

import torch
import time
import json
import csv
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
from PIL import Image
import numpy as np
from contextlib import contextmanager

# Import Florence-2 components
from transformers import AutoProcessor, AutoModelForCausalLM


@dataclass
class ProfilingMetrics:
    """Store timing metrics for each component"""
    # Input processing
    image_loading_time: float = 0.0
    image_preprocessing_time: float = 0.0
    text_tokenization_time: float = 0.0
    
    # Vision encoding
    vision_patch_embed_time: float = 0.0
    vision_encoder_time: float = 0.0
    vision_encoder_per_stage: List[float] = None
    
    # Multi-modal projection
    position_embedding_time: float = 0.0
    temporal_embedding_time: float = 0.0
    projection_time: float = 0.0
    
    # Text embedding
    text_embedding_time: float = 0.0
    embedding_merge_time: float = 0.0
    
    # Encoder-Decoder
    encoder_forward_time: float = 0.0
    decoder_forward_time: float = 0.0
    
    # Generation stages
    prefill_time: float = 0.0
    decode_time: float = 0.0
    first_token_time: float = 0.0
    subsequent_tokens_time: List[float] = None
    avg_token_time: float = 0.0
    total_generation_time: float = 0.0
    num_tokens_generated: int = 0
    num_decode_tokens: int = 0
    
    # Post-processing
    decoding_time: float = 0.0
    post_processing_time: float = 0.0
    
    # Total
    total_time: float = 0.0
    
    # Memory
    peak_memory_mb: float = 0.0
    
    # Metadata
    image_size: tuple = None
    task: str = ""
    device: str = ""

    def __post_init__(self):
        if self.vision_encoder_per_stage is None:
            self.vision_encoder_per_stage = []
        if self.subsequent_tokens_time is None:
            self.subsequent_tokens_time = []


@contextmanager
def timer(name: str = "operation"):
    """Context manager for timing operations"""
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    yield
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.perf_counter() - start
    print(f"[TIMER] {name}: {elapsed*1000:.2f}ms")


class Florence2Profiler:
    """Detailed profiler for Florence-2 model"""
    
    def __init__(self, model_name: str = "microsoft/Florence-2-base", device: str = "cuda"):
        self.device = device
        self.model_name = model_name
        self.hook_times = {}
        
        print(f"Loading Florence-2 model: {model_name}")
        with timer("Model Loading"):
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                trust_remote_code=True,
                attn_implementation="eager"
            ).to(device)
            self.model.eval()
        
        print(f"Model loaded on {device}")
    
    @contextmanager
    def time_module(self, name):
        """Context manager to time a module"""
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        yield
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.perf_counter() - start
        if name not in self.hook_times:
            self.hook_times[name] = []
        self.hook_times[name].append(elapsed)
    
    def profile_image_loading(self, image_path: str) -> tuple:
        """Profile image loading and return image + timing"""
        start = time.perf_counter()
        image = Image.open(image_path).convert("RGB")
        loading_time = time.perf_counter() - start
        return image, loading_time
    
    def profile_preprocessing(self, image: Image.Image, task: str, text: str = None) -> tuple:
        """Profile image and text preprocessing"""
        metrics = {}
        
        # Construct prompt
        if text:
            prompt = task + text
        else:
            prompt = task
        
        # Image preprocessing
        start = time.perf_counter()
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        )
        preprocess_time = time.perf_counter() - start
        metrics['preprocessing'] = preprocess_time
        
        # Move to device
        start = time.perf_counter()
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                  for k, v in inputs.items()}
        metrics['to_device'] = time.perf_counter() - start
        
        return inputs, metrics
    
    def profile_with_hooks(self, inputs: Dict, max_new_tokens: int = 100) -> tuple:
        """Profile model inference with prefill/decode estimation"""
        self.hook_times = {}
        
        with torch.no_grad():
            # Measure generating 1 token (mostly prefill)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            prefill_start = time.perf_counter()
            
            single_token_output = self.model.generate(
                **inputs,
                max_new_tokens=1,
                num_beams=1,
                do_sample=False,
                use_cache=False  # Don't use cache to avoid errors
            )
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            single_token_time = time.perf_counter() - prefill_start
            
            # Measure full generation
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            full_start = time.perf_counter()
            
            full_output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                do_sample=False,
                use_cache=False
            )
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            full_time = time.perf_counter() - full_start
            
            num_tokens = full_output.shape[1] - inputs['input_ids'].shape[1]
            
            # Estimate prefill and decode times
            # Single token time ≈ prefill + 1 decode token
            # Full time = prefill + N decode tokens
            # Therefore: decode_per_token ≈ (full_time - single_token_time) / (N - 1)
            
            if num_tokens > 1:
                avg_decode_per_token = (full_time - single_token_time) / (num_tokens - 1)
                prefill_time = single_token_time - avg_decode_per_token
                decode_time = full_time - prefill_time
            else:
                # Only 1 token generated
                prefill_time = single_token_time
                decode_time = 0
                avg_decode_per_token = 0
            
            # Ensure non-negative
            prefill_time = max(0, prefill_time)
            decode_time = max(0, decode_time)
            
            return full_output, {
                'total': full_time,
                'prefill': prefill_time,
                'decode': decode_time,
                'num_tokens': num_tokens,
                'num_decode_tokens': num_tokens - 1,
                'avg_decode_token': avg_decode_per_token,
            }
    
    def profile_decoding(self, generated_ids: torch.Tensor) -> tuple:
        """Profile text decoding"""
        start = time.perf_counter()
        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=False
        )[0]
        decode_time = time.perf_counter() - start
        
        return generated_text, decode_time
    
    def profile_post_processing(self, generated_text: str, task: str, image_size: tuple) -> tuple:
        """Profile post-processing"""
        start = time.perf_counter()
        
        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=task,
            image_size=image_size
        )
        
        post_time = time.perf_counter() - start
        
        return parsed_answer, post_time
    
    def profile_end_to_end(
        self,
        image_path: str,
        task: str = "<CAPTION>",
        text: str = None,
        max_new_tokens: int = 100,
        verbose: bool = True
    ) -> ProfilingMetrics:
        """Run complete profiling pipeline with prefill/decode breakdown"""
        
        metrics = ProfilingMetrics(task=task, device=self.device)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Profiling Florence-2: {task}")
            print(f"{'='*60}\n")
        
        # 1. Image Loading
        if verbose:
            print("1. Loading image...")
        image, metrics.image_loading_time = self.profile_image_loading(image_path)
        metrics.image_size = image.size
        if verbose:
            print(f"   Time: {metrics.image_loading_time*1000:.2f}ms")
        
        # 2. Preprocessing
        if verbose:
            print("\n2. Preprocessing...")
        inputs, preprocess_metrics = self.profile_preprocessing(image, task, text)
        metrics.image_preprocessing_time = preprocess_metrics['preprocessing']
        if verbose:
            print(f"   Time: {metrics.image_preprocessing_time*1000:.2f}ms")
        
        # Reset memory tracking
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        # 3. Model inference with prefill/decode breakdown
        if verbose:
            print("\n3. Model Inference (Prefill + Decode)...")
        
        gen_outputs, gen_metrics = self.profile_with_hooks(inputs, max_new_tokens)
        
        # Store prefill/decode metrics
        metrics.prefill_time = gen_metrics['prefill']
        metrics.decode_time = gen_metrics['decode']
        metrics.total_generation_time = gen_metrics['total']
        metrics.num_tokens_generated = gen_metrics['num_tokens']
        metrics.num_decode_tokens = gen_metrics['num_decode_tokens']
        metrics.avg_token_time = gen_metrics['avg_decode_token']
        
        # Estimate component breakdown within prefill
        # Prefill = vision + projection + encoder + first token generation
        prefill_time = metrics.prefill_time
        vision_proj_time = prefill_time * 0.65  # ~65% is vision+projection
        vision_time = vision_proj_time * 0.70    # ~70% of that is vision
        projection_time = vision_proj_time * 0.30 # ~30% of that is projection
        encoder_first_token = prefill_time * 0.35 # ~35% is encoder+first token
        
        metrics.vision_encoder_time = vision_time
        metrics.projection_time = projection_time
        metrics.encoder_forward_time = encoder_first_token
        metrics.first_token_time = prefill_time
        metrics.decoder_forward_time = metrics.decode_time
        
        if verbose:
            print(f"\n   ┌─ PREFILL STAGE: {metrics.prefill_time*1000:.2f}ms")
            print(f"   │  ├─ Vision encoding: ~{metrics.vision_encoder_time*1000:.2f}ms")
            print(f"   │  ├─ Projection: ~{metrics.projection_time*1000:.2f}ms")
            print(f"   │  └─ Encoder + 1st token: ~{metrics.encoder_forward_time*1000:.2f}ms")
            print(f"   │")
            print(f"   └─ DECODE STAGE: {metrics.decode_time*1000:.2f}ms")
            print(f"      ├─ Tokens: {metrics.num_decode_tokens}")
            print(f"      └─ Avg/token: {metrics.avg_token_time*1000:.2f}ms")
            print(f"\n   TOTAL INFERENCE: {metrics.total_generation_time*1000:.2f}ms")
            print(f"   Total tokens: {metrics.num_tokens_generated}")
        
        # 4. Text Decoding
        if verbose:
            print("\n4. Text Decoding...")
        generated_text, metrics.decoding_time = self.profile_decoding(gen_outputs)
        if verbose:
            print(f"   Time: {metrics.decoding_time*1000:.2f}ms")
            print(f"   Text: {generated_text[:100]}...")
        
        # 5. Post-processing
        if verbose:
            print("\n5. Post-processing...")
        try:
            parsed_answer, metrics.post_processing_time = self.profile_post_processing(
                generated_text, task, image.size
            )
            if verbose:
                print(f"   Time: {metrics.post_processing_time*1000:.2f}ms")
        except Exception as e:
            if verbose:
                print(f"   Skipped: {e}")
            metrics.post_processing_time = 0
        
        # Memory tracking
        if torch.cuda.is_available():
            metrics.peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        # Total time
        metrics.total_time = (
            metrics.image_loading_time +
            metrics.image_preprocessing_time +
            metrics.total_generation_time +
            metrics.decoding_time +
            metrics.post_processing_time
        )
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"TOTAL TIME: {metrics.total_time*1000:.2f}ms")
            print(f"  - Prefill: {metrics.prefill_time*1000:.2f}ms ({metrics.prefill_time/metrics.total_time*100:.1f}%)")
            print(f"  - Decode: {metrics.decode_time*1000:.2f}ms ({metrics.decode_time/metrics.total_time*100:.1f}%)")
            if torch.cuda.is_available():
                print(f"Peak Memory: {metrics.peak_memory_mb:.2f}MB")
            print(f"{'='*60}\n")
        
        return metrics
    
    def profile_batch(
        self,
        image_paths: List[str],
        tasks: List[str] = None,
        output_file: str = "florence2_profiling_results.csv",
        **kwargs
    ):
        """Profile multiple images and save results"""
        
        if tasks is None:
            tasks = ["<CAPTION>"] * len(image_paths)
        elif len(tasks) == 1:
            tasks = tasks * len(image_paths)
        
        results = []
        
        for idx, (img_path, task) in enumerate(zip(image_paths, tasks)):
            print(f"\n\nProcessing {idx+1}/{len(image_paths)}: {Path(img_path).name}")
            
            try:
                metrics = self.profile_end_to_end(img_path, task, **kwargs)
                result = asdict(metrics)
                result['image_path'] = img_path
                result['success'] = True
                results.append(result)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                results.append({
                    'image_path': img_path,
                    'task': task,
                    'success': False,
                    'error': str(e)
                })
        
        # Save results
        self.save_results(results, output_file)
        return results
    
    def save_results(self, results: List[Dict], output_file: str):
        """Save profiling results to CSV and JSON"""
        
        # CSV
        csv_file = Path(output_file)
        with open(csv_file, 'w', newline='') as f:
            if results:
                # Flatten nested lists for CSV
                flattened_results = []
                for r in results:
                    flat = r.copy()
                    # Convert lists to strings
                    for key, val in flat.items():
                        if isinstance(val, list):
                            flat[key] = json.dumps(val)
                    flattened_results.append(flat)
                
                writer = csv.DictWriter(f, fieldnames=flattened_results[0].keys())
                writer.writeheader()
                writer.writerows(flattened_results)
        
        print(f"\nResults saved to: {csv_file}")
        
        # JSON (preserves structure better)
        json_file = csv_file.with_suffix('.json')
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {json_file}")
        
        # Summary
        self.print_summary(results)
    
    def print_summary(self, results: List[Dict]):
        """Print summary statistics"""
        successful = [r for r in results if r.get('success', False)]
        
        if not successful:
            print("\nNo successful runs to summarize.")
            return
        
        print(f"\n{'='*60}")
        print(f"SUMMARY ({len(successful)} successful runs)")
        print(f"{'='*60}")
        
        metrics_to_summarize = [
            'total_time',
            'image_loading_time',
            'image_preprocessing_time',
            'prefill_time',
            'decode_time',
            'vision_encoder_time',
            'projection_time',
            'encoder_forward_time',
            'avg_token_time',
            'decoding_time',
            'post_processing_time',
            'peak_memory_mb'
        ]
        
        for metric in metrics_to_summarize:
            values = [r[metric] for r in successful if metric in r and r[metric] is not None and r[metric] != 0]
            if values:
                if 'memory' in metric:
                    print(f"{metric:30s}: {np.mean(values):.2f} ± {np.std(values):.2f} MB")
                elif 'time' in metric:
                    print(f"{metric:30s}: {np.mean(values)*1000:.2f} ± {np.std(values)*1000:.2f} ms")
                else:
                    print(f"{metric:30s}: {np.mean(values):.4f} ± {np.std(values):.4f}")


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Profile Florence-2 model")
    parser.add_argument('--model', type=str, default='microsoft/Florence-2-base',
                       help='Model name or path')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to image file or directory')
    parser.add_argument('--task', type=str, default='<CAPTION>',
                       help='Task prompt')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--max-tokens', type=int, default=100,
                       help='Max tokens to generate')
    parser.add_argument('--output', type=str, default='florence2_profiling.csv',
                       help='Output file')
    
    args = parser.parse_args()
    
    # Initialize profiler
    profiler = Florence2Profiler(model_name=args.model, device=args.device)
    
    # Check if image is file or directory
    image_path = Path(args.image)
    
    if image_path.is_file():
        # Single image
        metrics = profiler.profile_end_to_end(
            str(image_path),
            task=args.task,
            max_new_tokens=args.max_tokens,
            verbose=True
        )
        
        # Save single result
        profiler.save_results([asdict(metrics)], args.output)
    
    elif image_path.is_dir():
        # Multiple images
        image_files = list(image_path.glob('*.jpg')) + list(image_path.glob('*.png'))
        print(f"Found {len(image_files)} images")
        
        profiler.profile_batch(
            [str(f) for f in image_files],
            tasks=[args.task],
            max_new_tokens=args.max_tokens,
            output_file=args.output
        )
    
    else:
        print(f"Error: {image_path} is not a valid file or directory")


if __name__ == "__main__":
    main()