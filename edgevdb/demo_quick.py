#!/usr/bin/env python3
"""EdgeVDB Quick Demo — run with: python3 demo_quick.py"""
import sys, os, shutil, hashlib, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "python"))
from edgevdb import EdgeVDB

def embed(text):
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

def cosim(a, b):
    d = sum(x*y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x*x for x in a)), math.sqrt(sum(x*x for x in b))
    return d / (na * nb) if na > 1e-8 and nb > 1e-8 else 0

# Setup
db_path = "/tmp/edgevdb_demo"
shutil.rmtree(db_path, ignore_errors=True)
os.makedirs(db_path, exist_ok=True)

print("=" * 60)
print("  EdgeVDB Vector Database Demo")
print("=" * 60)

db = EdgeVDB(db_path)
print(f"\n[OK] Database opened at {db_path}\n")

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
    results = sorted([(cosim(qe, e), cid, t) for cid, (t, e) in store.items()], reverse=True)
    for rank, (score, cid, txt) in enumerate(results[:3]):
        bar = "#" * int(score * 30)
        print(f"    #{rank+1} [{bar:30s}] {score:.4f}  \"{txt}\"")
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
