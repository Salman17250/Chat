import os
import sys
sys.path.insert(0, os.path.abspath("."))
import time
import statistics
import concurrent.futures
from typing import List, Dict, Any
from tests.test_dynamic_accuracy import TestDynamicAccuracy

def run_concurrency_benchmark(concurrency_levels: List[int] = [1, 10, 25, 50, 100], total_queries_per_level: int = 100):
    print("=" * 80)
    print("ULTRA-FAST SEMANTIC SEARCH + INTELLIGENT RAG CONCURRENCY BENCHMARK")
    print("=" * 80)

    # Initialize environment and tenant index
    TestDynamicAccuracy.setUpClass()
    test_suite = TestDynamicAccuracy()
    engine = test_suite.engine
    tenant_id = test_suite.tenant_id

    benchmark_queries = [
        ("Semantic Paraphrase", "How long do I have to cancel?"),
        ("Entity + Attribute", "How old is Ovesh?"),
        ("Attribute Paraphrase", "Tell me Ovesh's current age."),
        ("Novel Entity Query", "How old is Sarah Jenkins?"),
        ("Structured Table Query", "What is the price of the Pro plan?"),
        ("Multi-Hop Composite", "What did Ovesh study and what technologies does he use?"),
        ("Negative Out-Of-Domain", "What is the maternity leave allowance?")
    ]

    # Pre-warm query paths
    for _, q in benchmark_queries:
        engine.query(q, tenant_id=tenant_id)

    results_table = []

    for concurrency in concurrency_levels:
        latencies = []
        errors = 0

        # Create query workload
        workload = [benchmark_queries[i % len(benchmark_queries)][1] for i in range(total_queries_per_level)]

        def _worker(q_str: str) -> float:
            t0 = time.perf_counter()
            try:
                resp = engine.query(q_str, tenant_id=tenant_id)
                t1 = time.perf_counter()
                if not resp or not resp.answer:
                    return -1.0
                return (t1 - t0) * 1000.0  # ms
            except Exception:
                return -1.0

        wall_start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            raw_latencies = list(executor.map(_worker, workload))
        wall_end = time.perf_counter()

        wall_duration = wall_end - wall_start
        valid_latencies = [l for l in raw_latencies if l > 0]
        errors = len(raw_latencies) - len(valid_latencies)

        if valid_latencies:
            valid_latencies.sort()
            qps = len(valid_latencies) / wall_duration
            p50 = statistics.median(valid_latencies)
            p95 = valid_latencies[int(len(valid_latencies) * 0.95)] if len(valid_latencies) >= 20 else valid_latencies[-1]
            p99 = valid_latencies[int(len(valid_latencies) * 0.99)] if len(valid_latencies) >= 100 else valid_latencies[-1]
            avg = statistics.mean(valid_latencies)

            res = {
                "Concurrency": concurrency,
                "Total Queries": total_queries_per_level,
                "QPS": round(qps, 2),
                "p50 (ms)": round(p50, 2),
                "p95 (ms)": round(p95, 2),
                "p99 (ms)": round(p99, 2),
                "Avg (ms)": round(avg, 2),
                "Errors": errors,
                "Error Rate": f"{(errors / total_queries_per_level) * 100:.1f}%"
            }
            results_table.append(res)
            print(f"Workers: {concurrency:3d} | QPS: {qps:8.2f} | p50: {p50:6.2f}ms | p95: {p95:6.2f}ms | p99: {p99:6.2f}ms | Errors: {errors}")

    print("\n" + "=" * 80)
    print(f"{'Concurrency':<12} | {'QPS':<10} | {'p50 (ms)':<10} | {'p95 (ms)':<10} | {'p99 (ms)':<10} | {'Error Rate':<10}")
    print("-" * 80)
    for r in results_table:
        print(f"{r['Concurrency']:<12} | {r['QPS']:<10.2f} | {r['p50 (ms)']:<10.2f} | {r['p95 (ms)']:<10.2f} | {r['p99 (ms)']:<10.2f} | {r['Error Rate']:<10}")
    print("=" * 80)

if __name__ == "__main__":
    run_concurrency_benchmark()
