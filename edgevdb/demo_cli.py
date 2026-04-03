#!/usr/bin/env python3
"""
EdgeVDB Interactive CLI Demo
-----------------------------
Takes text input, creates embeddings, stores in vector DB,
and lets you query — all visible in the terminal.

Usage:
    python3 demo_cli.py
"""

import sys, os, shutil, hashlib, struct, math

# Add the python package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))

from edgevdb import EdgeVDB

# ─── Simple Embedding Generator (no ONNX needed) ─────────────────────
def text_to_embedding(text: str) -> list:
    """Generate a deterministic 384-dim embedding from text using character hashing.
    Similar texts produce similar vectors — good enough for demo purposes."""
    dim = 384
    vec = [0.0] * dim
    words = text.lower().split()
    for wi, word in enumerate(words):
        h = int(hashlib.sha256(word.encode()).hexdigest(), 16)
        for d in range(dim):
            # Spread word contribution across dimensions
            seed = h ^ (d * 2654435761) ^ (wi * 40503)
            val = ((seed & 0xFFFF) - 32768) / 32768.0
            vec[d] += val / max(len(words), 1)
    # L2 normalize
    norm = math.sqrt(sum(v*v for v in vec))
    if norm > 1e-8:
        vec = [v / norm for v in vec]
    return vec

# ─── Cosine Similarity ───────────────────────────────────────────────
def cosine_sim(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    return dot / (na * nb) if na > 1e-8 and nb > 1e-8 else 0.0

# ─── Colors ──────────────────────────────────────────────────────────
CYAN    = "\033[96m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
MAGENTA = "\033[95m"
RED     = "\033[91m"
DIM     = "\033[2m"
BOLD    = "\033[1m"
RESET   = "\033[0m"

def banner():
    print(f"""
{CYAN}{BOLD}╔══════════════════════════════════════════════════════╗
║            EdgeVDB Interactive CLI Demo              ║
║        On-Device Vector Database SDK v1.0.0          ║
╚══════════════════════════════════════════════════════╝{RESET}
""")

def show_help():
    print(f"""
{BOLD}Commands:{RESET}
  {GREEN}add{RESET}     <text>         Insert text into the vector DB
  {GREEN}query{RESET}   <text>         Search for similar texts (top 5)
  {GREEN}list{RESET}                   Show all stored chunks
  {GREEN}object{RESET}  <type> <json>  Store a JSON object  (e.g. object Document {{"title":"AI"}})
  {GREEN}relate{RESET}  <from> <to>    Add a relation between two IDs
  {GREEN}stats{RESET}                  Show database statistics
  {GREEN}save{RESET}                   Save database to disk
  {GREEN}clear{RESET}                  Delete all data and start fresh
  {GREEN}help{RESET}                   Show this help
  {GREEN}quit{RESET}                   Exit
""")

def main():
    banner()

    db_path = "/tmp/edgevdb_demo"
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    os.makedirs(db_path, exist_ok=True)

    db = EdgeVDB(db_path)
    print(f"  {DIM}Database opened at: {db_path}{RESET}")
    print(f"  {DIM}Embedding model: deterministic hash (384-dim, no ONNX){RESET}")

    # Store chunks locally for listing
    chunks = {}  # chunk_id -> (text, embedding, doc_id, page)
    objects = {}  # obj_id -> (type, props)
    doc_counter = [1]

    show_help()

    while True:
        try:
            raw = input(f"\n{CYAN}edgevdb>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue

        parts = raw.split(None, 1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        # ── ADD ──────────────────────────────────────────────
        if cmd == "add":
            if not arg:
                print(f"  {RED}Usage: add <text>{RESET}")
                continue

            text = arg
            print(f"\n  {DIM}┌─ Processing ──────────────────────────{RESET}")
            print(f"  {DIM}│ Input text:  \"{text}\"{RESET}")

            # Generate embedding
            emb = text_to_embedding(text)
            top5 = sorted(range(len(emb)), key=lambda i: abs(emb[i]), reverse=True)[:5]
            print(f"  {DIM}│ Embedding:   [{emb[0]:.4f}, {emb[1]:.4f}, {emb[2]:.4f}, ... ] (384-dim){RESET}")
            print(f"  {DIM}│ Top dims:    {', '.join(f'd{i}={emb[i]:.3f}' for i in top5)}{RESET}")

            # Insert into DB
            doc_id = doc_counter[0]
            chunk_id = db.insert_chunk(text, emb, doc_id=doc_id, page_number=0)
            chunks[chunk_id] = (text, emb, doc_id, 0)
            doc_counter[0] += 1

            print(f"  {DIM}│{RESET}")
            print(f"  {DIM}│{RESET} {GREEN}✓ Stored!{RESET}  chunk_id={BOLD}{chunk_id}{RESET}  doc_id={doc_id}")
            print(f"  {DIM}└────────────────────────────────────────{RESET}")

        # ── QUERY ────────────────────────────────────────────
        elif cmd == "query":
            if not arg:
                print(f"  {RED}Usage: query <text>{RESET}")
                continue

            if not chunks:
                print(f"  {YELLOW}Database is empty. Add some text first!{RESET}")
                continue

            query_text = arg
            query_emb = text_to_embedding(query_text)

            print(f"\n  {DIM}┌─ Query ───────────────────────────────{RESET}")
            print(f"  {DIM}│ Query:     \"{query_text}\"{RESET}")
            print(f"  {DIM}│ Embedding: [{query_emb[0]:.4f}, {query_emb[1]:.4f}, ...]{RESET}")
            print(f"  {DIM}│{RESET}")

            # Compute similarities manually for display
            results = []
            for cid, (txt, emb, did, pg) in chunks.items():
                sim = cosine_sim(query_emb, emb)
                results.append((sim, cid, txt))
            results.sort(reverse=True)

            top_k = min(5, len(results))
            print(f"  {DIM}│ Results (top {top_k}):{RESET}")
            print(f"  {DIM}│{RESET}")

            for i, (score, cid, txt) in enumerate(results[:top_k]):
                bar_len = int(score * 20)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                color = GREEN if score > 0.5 else YELLOW if score > 0.2 else DIM
                print(f"  {DIM}│{RESET}  {BOLD}#{i+1}{RESET}  {color}[{bar}] {score:.4f}{RESET}")
                print(f"  {DIM}│{RESET}      chunk_id={cid}  \"{txt}\"")
                print(f"  {DIM}│{RESET}")

            print(f"  {DIM}└────────────────────────────────────────{RESET}")

        # ── LIST ─────────────────────────────────────────────
        elif cmd == "list":
            if not chunks:
                print(f"  {YELLOW}Database is empty.{RESET}")
                continue

            print(f"\n  {BOLD}Stored Chunks ({len(chunks)} total):{RESET}")
            print(f"  {'─' * 50}")
            for cid, (txt, emb, did, pg) in chunks.items():
                print(f"  {DIM}│{RESET} id={BOLD}{cid}{RESET}  doc={did}  \"{txt[:60]}{'...' if len(txt) > 60 else ''}\"")
            print(f"  {'─' * 50}")

            if objects:
                print(f"\n  {BOLD}Stored Objects ({len(objects)} total):{RESET}")
                print(f"  {'─' * 50}")
                for oid, (typ, props) in objects.items():
                    print(f"  {DIM}│{RESET} id={BOLD}{oid}{RESET}  type={typ}  {props}")
                print(f"  {'─' * 50}")

        # ── OBJECT ───────────────────────────────────────────
        elif cmd == "object":
            if not arg:
                print(f"  {RED}Usage: object <type> <json_props>{RESET}")
                print(f"  {DIM}Example: object Document {{\"title\":\"AI\",\"author\":\"Alice\"}}{RESET}")
                continue

            obj_parts = arg.split(None, 1)
            if len(obj_parts) < 2:
                print(f"  {RED}Need type AND properties. Example: object Document {{\"title\":\"AI\"}}{RESET}")
                continue

            obj_type = obj_parts[0]
            import json
            try:
                props = json.loads(obj_parts[1])
            except json.JSONDecodeError:
                print(f"  {RED}Invalid JSON. Use double quotes: {{\"key\":\"value\"}}{RESET}")
                continue

            obj_id = db.put_object(obj_type, props)
            objects[obj_id] = (obj_type, props)
            print(f"  {GREEN}✓ Object stored!{RESET}  id={BOLD}{obj_id}{RESET}  type={obj_type}  {props}")

        # ── RELATE ───────────────────────────────────────────
        elif cmd == "relate":
            if not arg:
                print(f"  {RED}Usage: relate <from_id> <to_id>{RESET}")
                continue
            ids = arg.split()
            if len(ids) != 2:
                print(f"  {RED}Need exactly 2 IDs. Usage: relate <from_id> <to_id>{RESET}")
                continue
            try:
                from_id, to_id = int(ids[0]), int(ids[1])
            except ValueError:
                print(f"  {RED}IDs must be numbers.{RESET}")
                continue
            db.add_relation("related_to", from_id, to_id)
            print(f"  {GREEN}✓ Relation added!{RESET}  {from_id} ──related_to──▸ {to_id}")

        # ── STATS ────────────────────────────────────────────
        elif cmd == "stats":
            print(f"\n  {BOLD}Database Statistics:{RESET}")
            print(f"  {'─' * 35}")
            print(f"  Chunks stored:    {BOLD}{len(chunks)}{RESET}")
            print(f"  Objects stored:   {BOLD}{len(objects)}{RESET}")
            print(f"  Embedding dim:    384")
            print(f"  Storage path:     {db_path}")
            print(f"  ONNX Runtime:     not linked (hash embeddings)")
            print(f"  {'─' * 35}")

        # ── SAVE ─────────────────────────────────────────────
        elif cmd == "save":
            db.save()
            print(f"  {GREEN}✓ Database saved to disk.{RESET}")

        # ── CLEAR ────────────────────────────────────────────
        elif cmd == "clear":
            db.close()
            shutil.rmtree(db_path, ignore_errors=True)
            os.makedirs(db_path, exist_ok=True)
            db = EdgeVDB(db_path)
            chunks.clear()
            objects.clear()
            doc_counter[0] = 1
            print(f"  {GREEN}✓ Database cleared.{RESET}")

        # ── HELP ─────────────────────────────────────────────
        elif cmd == "help":
            show_help()

        # ── QUIT ─────────────────────────────────────────────
        elif cmd in ("quit", "exit", "q"):
            break

        else:
            print(f"  {RED}Unknown command: {cmd}{RESET}. Type {GREEN}help{RESET} for commands.")

    # Cleanup
    db.save()
    db.close()
    print(f"\n  {DIM}Database saved and closed. Goodbye!{RESET}\n")


if __name__ == "__main__":
    main()
