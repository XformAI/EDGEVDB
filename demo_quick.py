#!/usr/bin/env python3
"""EdgeVDB Quick Demo — run with: python3 demo_quick.py"""
import sys, os, shutil, hashlib, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "python"))
from edgevdb import EdgeVDB

try:
    from edgevdb import Embedder
    model_path = os.path.join(os.path.dirname(__file__), "models", "model.onnx")
    vocab_path = os.path.join(os.path.dirname(__file__), "models", "vocab.txt")
    if os.path.exists(model_path) and os.path.exists(vocab_path):
        real_embedder = Embedder(model_path, vocab_path)
    else:
        real_embedder = None
except Exception:
    real_embedder = None

def embed(text):
    if real_embedder:
        return real_embedder.embed(text)
    dim, vec = 384, [0.0]*384
    words = text.lower().split()
    for wi, w in enumerate(words):
        h = int(hashlib.sha256(w.encode()).hexdigest(), 16)
        for d in range(dim):
            seed = h ^ (d * 2654435761) ^ (wi * 40503)
            vec[d] += ((seed & 0xFFFF) - 32768) / 32768.0 / max(len(words), 1)
    norm = math.sqrt(sum(v*v for v in vec))
    if norm > 1e-8:
        vec = [v/norm for v in vec]
    return vec

# Setup
db_path = "/tmp/edgevdb_demo"
shutil.rmtree(db_path, ignore_errors=True)
os.makedirs(db_path, exist_ok=True)

print("=" * 60)
print("  EdgeVDB Vector Database Demo")
print("=" * 60)

db = EdgeVDB(db_path)
print(f"\n[OK] Database opened at {db_path}")
if real_embedder:
    print("[OK] Real ONNX Embedder loaded\n")
else:
    print("[WARN] Using pseudo-random character hash (ONNX not found). Semantics will be nonsense!\n")

# Insert documents
texts = [
    "Machine learning finds patterns in large datasets",
    "Neural networks are used for image classification",
    "Python is a popular language for data science",
    "Transformers revolutionized natural language processing",
    "Vector databases enable fast semantic search",
]

store = {}
print("--- INSERTING DOCUMENTS ---\n")
for i, t in enumerate(texts):
    e = embed(t)
    cid = db.insert_chunk(t, e, doc_id=i+1, page_number=0)
    store[cid] = (t, e)
    print(f"  [{i+1}] chunk_id={cid}")
    print(f"      text: \"{t}\"")
    print(f"      embedding: [{e[0]:.4f}, {e[1]:.4f}, {e[2]:.4f}, ...]")
    print()

print(f"  Total: {len(store)} chunks stored\n")

# Query
queries = ["machine learning AI", "image recognition", "search database"]
print("--- QUERYING ---\n")
for q in queries:
    qe = embed(q)
    print(f"  Query: \"{q}\"")
    print(f"  Embedding: [{qe[0]:.4f}, {qe[1]:.4f}, ...]")
    
    if real_embedder:
        results = db.query_text(real_embedder, q, top_k=3)
    else:
        results = db.query_vector(qe, query_text=q, top_k=3)

    for rank, r in enumerate(results):
        bar_len = min(30, max(0, int(r.score * 30)))
        bar = "#" * bar_len
        print(f"    #{rank+1} [{bar:30s}] {r.score:.4f}  \"{r.text}\"")
    results.free()
    print()

# Object store
print("--- OBJECT STORE ---\n")
oid = db.put_object("Document", {"title": "ML Guide", "author": "Alice", "pages": "42"})
print(f"  Stored object: id={oid} type=Document")
obj = db.get_object(oid)
print(f"  Retrieved:     {obj}")

# Relations
print("\n--- RELATIONS ---\n")
first_chunk = list(store.keys())[0]
db.add_relation("has_chunk", oid, first_chunk)
print(f"  Added: Document({oid}) --has_chunk--> Chunk({first_chunk})")

# Save
db.save()
db.close()
print(f"\n[OK] Database saved and closed.")
print("=" * 60)
