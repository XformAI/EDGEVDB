"""
EdgeVDB Python SDK Tests
"""

import os
import sys
import tempfile
import unittest

# Add parent to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestEdgeVDB(unittest.TestCase):
    """Test EdgeVDB Python SDK.
    
    NOTE: These tests require the compiled shared library.
    Run after building with: cmake --preset desktop-release && cmake --build --preset desktop-release
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="edgevdb_test_")

    def tearDown(self):
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @unittest.skipUnless(
        os.path.exists(os.path.join(os.path.dirname(__file__), "..", "edgevdb", "libedgevdb_shared.so")) or
        os.path.exists(os.path.join(os.path.dirname(__file__), "..", "edgevdb", "edgevdb_shared.dll")),
        "Shared library not found — build first"
    )
    def test_open_close(self):
        from edgevdb import EdgeVDB
        db = EdgeVDB(self.temp_dir)
        db.save()
        db.close()

    @unittest.skipUnless(
        os.path.exists(os.path.join(os.path.dirname(__file__), "..", "edgevdb", "libedgevdb_shared.so")) or
        os.path.exists(os.path.join(os.path.dirname(__file__), "..", "edgevdb", "edgevdb_shared.dll")),
        "Shared library not found — build first"
    )
    def test_object_store(self):
        from edgevdb import EdgeVDB
        with EdgeVDB(self.temp_dir) as db:
            obj_id = db.put_object("Document", {"title": "Test", "author": "Alice"})
            self.assertGreater(obj_id, 0)

            obj = db.get_object(obj_id)
            self.assertIsNotNone(obj)
            self.assertIn("title", obj)

    @unittest.skipUnless(
        os.path.exists(os.path.join(os.path.dirname(__file__), "..", "edgevdb", "libedgevdb_shared.so")) or
        os.path.exists(os.path.join(os.path.dirname(__file__), "..", "edgevdb", "edgevdb_shared.dll")),
        "Shared library not found — build first"
    )
    def test_version(self):
        from edgevdb import version
        ver = version()
        self.assertEqual(ver, "1.0.0")

    @unittest.skipUnless(
        os.path.exists(os.path.join(os.path.dirname(__file__), "..", "edgevdb", "libedgevdb_shared.so")) or
        os.path.exists(os.path.join(os.path.dirname(__file__), "..", "edgevdb", "edgevdb_shared.dll")),
        "Shared library not found — build first"
    )
    def test_insert_and_query(self):
        from edgevdb import EdgeVDB
        with EdgeVDB(self.temp_dir) as db:
            # Insert with pre-computed embedding
            emb = [0.0] * 384
            emb[0] = 1.0
            chunk_id = db.insert_chunk("Test chunk about AI", emb, doc_id=1, page_number=0)
            self.assertGreater(chunk_id, 0)


if __name__ == "__main__":
    unittest.main()
