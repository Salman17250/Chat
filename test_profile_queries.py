import time
import sys
import os

# Set working directory to project root
sys.path.insert(0, os.path.abspath("."))

from app.storage_postgres import get_postgres_store
from app.rag.engine import get_rag_engine
from app.translator import get_translator
from app.main import chat, ChatRequest

print("Initializing components...", flush=True)
t0 = time.perf_counter()
pg = get_postgres_store()
t1 = time.perf_counter()
print(f"Postgres store init: {(t1 - t0)*1000:.2f} ms", flush=True)

t0 = time.perf_counter()
rag = get_rag_engine()
t1 = time.perf_counter()
print(f"RAG Engine init: {(t1 - t0)*1000:.2f} ms", flush=True)

test_queries = [
    "what is the leave policy",
    "who is Jon Due",
    "what are the office hours",
    "how to apply for loan",
    "what is the probation period",
    "tell me about wortal",
    "can I carry forward leaves",
    "what is the notice period",
    "working days of company",
    "maternity leave policy"
]

print("\n--- Testing Single Query Step-by-Step ---", flush=True)
q = test_queries[0]

# Profile translator
tr = get_translator()
t0 = time.perf_counter()
eng_q, lang, orig_q, form = tr.process_query(q)
t1 = time.perf_counter()
print(f"Translator process_query: {(t1 - t0)*1000:.2f} ms", flush=True)

# Profile embedding
t0 = time.perf_counter()
q_vec = rag.embedder.embed_text(q)
t1 = time.perf_counter()
print(f"Embed text: {(t1 - t0)*1000:.2f} ms", flush=True)

# Profile rag.query
t0 = time.perf_counter()
res = rag.query(q, tenant_id="default")
t1 = time.perf_counter()
print(f"RAG query: {(t1 - t0)*1000:.2f} ms (Answer len={len(res.answer)}, Conf={res.confidence})", flush=True)

# Profile chat endpoint directly
print("\n--- Testing 10 Queries via Chat Endpoint ---", flush=True)
latencies = []
for i, query_text in enumerate(test_queries):
    req = ChatRequest(message=query_text, tenant_id="default")
    t0 = time.perf_counter()
    resp = chat(req)
    t1 = time.perf_counter()
    dur_ms = (t1 - t0) * 1000
    latencies.append(dur_ms)
    print(f"Query {i+1} ('{query_text[:30]}'): {dur_ms:.2f} ms | Conf: {resp.confidence} | Ans: {resp.answer[:60]}...", flush=True)

print(f"\n--- Summary ---")
print(f"Total time for 10 queries: {sum(latencies):.2f} ms")
print(f"Average latency per query: {sum(latencies)/len(latencies):.2f} ms")
print(f"Effective QPS: {1000 / (sum(latencies)/len(latencies)):.1f} queries/sec")
