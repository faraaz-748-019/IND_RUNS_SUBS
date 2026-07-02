"""
Scalability benchmark script.
Measures FAISS indexing and search latency across dataset sizes.
Proves the system can handle large candidate pools.
"""

import time
import json
import os
import numpy as np
from typing import Optional


def run_benchmark(sizes: list[int] = None, dim: int = 384) -> dict:
    """
    Benchmark FAISS index build and search latency across dataset sizes.
    
    Args:
        sizes: List of dataset sizes to test (default: [100, 1000, 10000, 100000])
        dim: Embedding dimension
    
    Returns:
        Dict with benchmark results per size
    """
    import faiss
    
    if sizes is None:
        sizes = [100, 1_000, 10_000, 50_000, 100_000]
    
    results = {}
    
    for n in sizes:
        print(f"  Benchmarking N={n:,}...")
        
        # Generate random embeddings
        embeddings = np.random.randn(n, dim).astype(np.float32)
        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        
        query = np.random.randn(1, dim).astype(np.float32)
        query = query / np.linalg.norm(query)
        
        # Build index
        t0 = time.perf_counter()
        if n <= 50000:
            index = faiss.IndexFlatIP(dim)
        else:
            nlist = min(int(np.sqrt(n)), 256)
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(embeddings)
            index.nprobe = 20
        
        index.add(embeddings)
        build_time = time.perf_counter() - t0
        
        # Search (average over 10 queries)
        search_times = []
        for _ in range(10):
            q = np.random.randn(1, dim).astype(np.float32)
            q = q / np.linalg.norm(q)
            t1 = time.perf_counter()
            index.search(q, min(100, n))
            search_times.append(time.perf_counter() - t1)
        
        avg_search = sum(search_times) / len(search_times)
        
        results[str(n)] = {
            "dataset_size": n,
            "index_type": "FlatIP" if n <= 50000 else "IVFFlat",
            "build_time_seconds": round(build_time, 4),
            "avg_search_time_ms": round(avg_search * 1000, 2),
            "searches_per_second": round(1.0 / avg_search, 1) if avg_search > 0 else 0,
        }
        
        print(f"    Build: {build_time:.3f}s | Search: {avg_search*1000:.2f}ms | {1.0/avg_search:.0f} queries/s")
    
    return results


def format_benchmark_table(results: dict) -> str:
    """Format benchmark results as a readable table."""
    header = f"{'N':>10} {'Index Type':>12} {'Build (s)':>12} {'Search (ms)':>14} {'Queries/s':>12}"
    separator = "-" * len(header)
    lines = [header, separator]
    
    for key, r in sorted(results.items(), key=lambda x: x[1]["dataset_size"]):
        lines.append(
            f"{r['dataset_size']:>10,} "
            f"{r['index_type']:>12} "
            f"{r['build_time_seconds']:>12.3f} "
            f"{r['avg_search_time_ms']:>14.2f} "
            f"{r['searches_per_second']:>12.1f}"
        )
    
    return "\n".join(lines)


def save_benchmark_results(results: dict, filepath: str):
    """Save benchmark results to JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    print("=" * 60)
    print("FAISS Scalability Benchmark")
    print("=" * 60)
    
    results = run_benchmark()
    print("\n" + format_benchmark_table(results))
    
    save_benchmark_results(results, "output/benchmark_results.json")
    print("\nResults saved to output/benchmark_results.json")
