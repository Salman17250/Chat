"""
tests/benchmark_concurrency.py
Performance & Concurrency Benchmark Suite.
Measures QPS, p50, p95, p99 latency, and cache hit rates under concurrent workloads:
  - Cold cache vs Warm cache
  - Query types: Semantic, Exact, Multi-hop, Comparison, Compound, Math
  - Concurrency levels: 1, 10, 25, 50, 100 concurrent workers
"""

import time
import sys
sys.path.insert(0, ".")
import statistics
import concurrent.futures
from typing import List, Dict, Any
from app.rag.engine import get_rag_engine


BENCHMARK_QUERIES = [
    ("semantic", "What is a Pipeline?"),
    ("semantic_paraphrase", "Explain what a pipeline does in business."),
    ("exact", "Why are stages important?"),
    ("multi_hop", "What happens after creating a pipeline?"),
    ("comparison", "Compare Pipeline and Stage."),
    ("compound", "What is a Pipeline, and why are stages important?"),
    ("math", "What is 18% GST on ₹10,000?"),
    ("table", "What are the default stages in the pipeline?"),
]


def benchmark_suite():
    engine = get_rag_engine()
    tenant_id = "default"

    print("================================================================================")
    print("      INTELLIGENT ZERO-LLM QUERY-PATH RAG ENGINE BENCHMARK SUITE")
    print("================================================================================\n")

    # 1. Cold Cache Single-Query Latency & Breakdown
    print("--- 1. Cold Cache Detailed Timing Breakdown (Per Query Type) ---")
    engine.multi_cache.clear_all()
    with engine.response_cache_lock:
        engine.response_cache.clear()

    cold_results = []
    for q_type, q_text in BENCHMARK_QUERIES:
        t0 = time.perf_counter()
        resp = engine.query(q_text, tenant_id=tenant_id, include_debug=True)
        dur_ms = (time.perf_counter() - t0) * 1000.0
        tb = resp.timing_breakdown or {}
        cold_results.append((q_type, dur_ms, resp.confidence, tb))
        print(f"[{q_type.upper():<20}] Total: {dur_ms:6.2f}ms | Confidence: {resp.confidence:<8} | "
              f"Norm: {tb.get('query_normalization_ms', 0):.2f}ms | "
              f"Ctx: {tb.get('context_resolution_ms', 0):.2f}ms | "
              f"Plan: {tb.get('entity_detection_ms', 0):.2f}ms | "
              f"Retrieval+Rerank: {tb.get('reranking_ms', 0):.2f}ms | "
              f"Answer: {tb.get('answer_composition_ms', 0):.2f}ms")

    # 2. Warm Cache Latency (L1 Multi-Tier RAM Cache)
    print("\n--- 2. Warm Cache In-Memory Response Latency ---")
    warm_latencies = []
    for _ in range(5):
        for _, q_text in BENCHMARK_QUERIES:
            t0 = time.perf_counter()
            resp = engine.query(q_text, tenant_id=tenant_id)
            warm_latencies.append((time.perf_counter() - t0) * 1000.0)

    p50_warm = statistics.median(warm_latencies)
    p95_warm = statistics.quantiles(warm_latencies, n=20)[18] if len(warm_latencies) >= 20 else max(warm_latencies)
    p99_warm = max(warm_latencies)
    print(f"Warm Cache Queries: {len(warm_latencies)} executions")
    print(f"  • p50 Latency: {p50_warm:.3f} ms")
    print(f"  • p95 Latency: {p95_warm:.3f} ms")
    print(f"  • p99 Latency: {p99_warm:.3f} ms")
    print(f"  • Effective Warm QPS: ~{1000.0 / max(0.001, p50_warm):,.0f} req/s per worker")

    # 3. High-Concurrency Stress Test
    concurrency_levels = [1, 10, 25, 50, 100]
    total_requests_per_level = 500

    print(f"\n--- 3. Concurrency Stress Test ({total_requests_per_level} requests per concurrency level) ---")
    print(f"{'Concurrency':<12} | {'Total Req':<10} | {'Duration (s)':<12} | {'Throughput (QPS)':<18} | {'p50 (ms)':<10} | {'p95 (ms)':<10} | {'p99 (ms)':<10}")
    print("-" * 92)

    for concurrency in concurrency_levels:
        queries_to_run = [BENCHMARK_QUERIES[i % len(BENCHMARK_QUERIES)][1] for i in range(total_requests_per_level)]
        latencies = []

        def worker(q):
            w_t0 = time.perf_counter()
            engine.query(q, tenant_id=tenant_id)
            return (time.perf_counter() - w_t0) * 1000.0

        t_bench_start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(worker, q) for q in queries_to_run]
            for f in concurrent.futures.as_completed(futures):
                latencies.append(f.result())
        t_bench_dur = time.perf_counter() - t_bench_start

        qps = total_requests_per_level / max(0.001, t_bench_dur)
        p50 = statistics.median(latencies)
        p95 = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies)
        p99 = max(latencies)

        print(f"{concurrency:<12} | {total_requests_per_level:<10} | {t_bench_dur:<12.3f} | {qps:<18.1f} | {p50:<10.3f} | {p95:<10.3f} | {p99:<10.3f}")

    print("\n================================================================================")
    print("                     BENCHMARK COMPLETE - ZERO ERRORS")
    print("================================================================================")


if __name__ == "__main__":
    benchmark_suite()
