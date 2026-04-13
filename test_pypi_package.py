# -*- coding: utf-8 -*-
"""
Test the edgevdb package installed from PyPI.
Run from any directory OUTSIDE python/ to test the installed package.
"""
import sys
import os
import tempfile
import shutil

PASS = 0
FAIL = 0

def report(name, passed, detail=""):
    global PASS, FAIL
    if passed:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} -- {detail}")


print("=" * 60)
print("EdgeVDB PyPI Package Verification")
print("=" * 60)

# --- Test 1: Import ---
print("\n1. Import tests")
try:
    import edgevdb
    report("import edgevdb", True)
except Exception as e:
    report("import edgevdb", False, str(e))
    print("\nCannot continue without import. Exiting.")
    sys.exit(1)

try:
    from edgevdb import EdgeVDB
    report("from edgevdb import EdgeVDB", True)
except Exception as e:
    report("from edgevdb import EdgeVDB", False, str(e))

# --- Test 2: Package metadata ---
print("\n2. Package metadata")
try:
    pkg_dir = os.path.dirname(edgevdb.__file__)
    report("package location: " + pkg_dir, True)
except Exception as e:
    report("package location", False, str(e))

try:
    ver = edgevdb.version()
    report(f"version() = {ver}", ver == "1.0.0")
except Exception as e:
    report("version()", False, str(e))

# --- Test 3: Library loading ---
print("\n3. Native library loading")
try:
    lib_path = edgevdb._find_library()
    report(f"_find_library() = {os.path.basename(lib_path)}", os.path.exists(lib_path))
except Exception as e:
    report("_find_library()", False, str(e))

try:
    import ctypes
    lib_path = edgevdb._find_library()
    lib = ctypes.CDLL(lib_path)
    report("ctypes.CDLL() load succeeded", lib is not None)
except Exception as e:
    report("ctypes.CDLL() load", False, str(e))
    print("\nNative library failed to load. Remaining tests will be skipped.")
    print(f"\nResults: {PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL > 0 else 0)

# --- Test 4: Open/Close ---
print("\n4. Database open/close")
tmp = tempfile.mkdtemp(prefix="edgevdb_pypi_test_")
try:
    db = EdgeVDB(tmp)
    report("EdgeVDB(tmpdir) open", True)
    db.save()
    report("db.save()", True)
    db.close()
    report("db.close()", True)
except Exception as e:
    report("open/save/close", False, str(e))

# --- Test 5: Context manager ---
print("\n5. Context manager")
try:
    with EdgeVDB(tmp) as db:
        report("with EdgeVDB() as db", True)
except Exception as e:
    report("context manager", False, str(e))

# --- Test 6: Insert chunk ---
print("\n6. Insert and query")
try:
    with EdgeVDB(tmp) as db:
        emb = [0.0] * 384
        emb[0] = 1.0
        chunk_id = db.insert_chunk("Neural networks process data", emb, doc_id=1, page_number=0)
        report(f"insert_chunk() -> chunk_id={chunk_id}", chunk_id > 0)

        emb2 = [0.0] * 384
        emb2[1] = 1.0
        chunk_id2 = db.insert_chunk("Vector databases enable search", emb2, doc_id=1, page_number=1)
        report(f"insert_chunk() -> chunk_id={chunk_id2}", chunk_id2 > 0)

        # Query
        results = db.query_vector(emb, query_text="neural networks", top_k=2)
        n = len(list(results))
        report(f"query_vector() returned {n} results", n > 0)
        results.free()
        report("results.free()", True)

        db.save()
        report("db.save() after inserts", True)
except Exception as e:
    report("insert/query", False, str(e))

# --- Test 7: Object store ---
print("\n7. Object store")
try:
    with EdgeVDB(tmp) as db:
        obj_id = db.put_object("Document", {"title": "Test Paper", "author": "Alice"})
        report(f"put_object() -> obj_id={obj_id}", obj_id > 0)

        obj = db.get_object(obj_id)
        report(f"get_object() -> {obj}", obj is not None and "title" in obj)
except Exception as e:
    report("object store", False, str(e))

# --- Test 8: File persistence ---
print("\n8. File persistence")
expected_files = ["chunks.bin", "hnsw.bin", "page.bin", "kg.bin", "objects.bin"]
for f in expected_files:
    path = os.path.join(tmp, f)
    exists = os.path.exists(path)
    if exists:
        size = os.path.getsize(path)
        report(f"{f} ({size} bytes)", True)
    else:
        report(f"{f}", False, "file not found")

# --- Cleanup ---
shutil.rmtree(tmp, ignore_errors=True)

# --- Summary ---
print("\n" + "=" * 60)
total = PASS + FAIL
print(f"Results: {PASS}/{total} passed, {FAIL} failed")
if FAIL == 0:
    print("All tests passed! Package is working correctly.")
else:
    print("Some tests failed. Review the output above.")
print("=" * 60)
sys.exit(1 if FAIL > 0 else 0)
