"""Python SDK benchmark — measures insert and query latency through ctypes FFI."""
import sys, os, time, random, shutil, math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
import edgevdb

DB_DIR = os.path.join(os.environ.get("TEMP", "/tmp"), "edgevdb_bench_py")

def l2_norm(v):
    n = math.sqrt(sum(x*x for x in v))
    return [x/n for x in v] if n > 0 else v

def run():
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
    os.makedirs(DB_DIR, exist_ok=True)

    db = edgevdb.EdgeVDB(DB_DIR)
    rng = random.Random(42)

    sizes = [100, 1000, 5000, 10000]
    print("=== Python SDK Insert Benchmark ===")
    for N in sizes:
        if os.path.exists(DB_DIR):
            shutil.rmtree(DB_DIR)
        os.makedirs(DB_DIR, exist_ok=True)
        db = edgevdb.EdgeVDB(DB_DIR)
        rng = random.Random(42)

        t0 = time.perf_counter()
        for i in range(N):
            emb = l2_norm([rng.gauss(0, 1) for _ in range(384)])
            db.insert_chunk(f"Chunk {i}: Lorem ipsum dolor sit amet.", emb, i // 100, i % 100)
        t1 = time.perf_counter()
        ms = (t1 - t0) * 1000
        rate = N / (t1 - t0)
        print(f"  N={N:>6}  Insert: {ms:.1f} ms  Rate: {rate:.0f} chunks/sec")

        if N == 10000:
            # Query benchmark on 10k
            print(f"\n=== Python SDK Query Benchmark (N={N}) ===")
            QUERIES = 100
            latencies = []
            for q in range(QUERIES):
                qemb = l2_norm([rng.gauss(0, 1) for _ in range(384)])
                t_q0 = time.perf_counter()
                results = db.query_vector(qemb, top_k=5)
                count = results.count
                results.free()
                t_q1 = time.perf_counter()
                latencies.append((t_q1 - t_q0) * 1000)

            avg = sum(latencies) / len(latencies)
            latencies.sort()
            p50 = latencies[len(latencies) // 2]
            p95 = latencies[int(len(latencies) * 0.95)]
            p99 = latencies[int(len(latencies) * 0.99)]
            print(f"  Queries: {QUERIES}")
            print(f"  Avg: {avg:.2f} ms")
            print(f"  Min: {min(latencies):.2f} ms")
            print(f"  Max: {max(latencies):.2f} ms")
            print(f"  P50: {p50:.2f} ms")
            print(f"  P95: {p95:.2f} ms")
            print(f"  P99: {p99:.2f} ms")

        db.close()

    # Cleanup
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)

if __name__ == "__main__":
    run()
