# -*- coding: utf-8 -*-
"""
EdgeVDB Python SDK -- ctypes wrapper for desktop/Raspberry Pi.
"""

import ctypes
import os
import platform
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# Library loading
def _find_library() -> str:
    """Find the EdgeVDB shared library."""
    system = platform.system()
    if system == "Windows":
        names = ["libedgevdb_shared.dll", "edgevdb_shared.dll", "edgevdb.dll"]
    elif system == "Darwin":
        names = ["libedgevdb_shared.dylib", "libedgevdb.dylib"]
    else:
        names = ["libedgevdb_shared.so", "libedgevdb.so"]

    # Search paths — platform-specific lib/ subdirectory first
    pkg_dir = Path(__file__).parent
    plat_dir = {"Windows": "windows", "Darwin": "darwin", "Linux": "linux"}.get(system, "linux")
    search_dirs = [
        pkg_dir / "lib" / plat_dir,
        pkg_dir / "lib",
        pkg_dir,
        Path.cwd(),
        Path.cwd() / "build",
        Path.cwd() / "build" / "desktop-release" / "core",
        Path.cwd() / "build" / "desktop-debug" / "core",
    ]

    for dir_path in search_dirs:
        for name in names:
            lib_path = dir_path / name
            if lib_path.exists():
                return str(lib_path)

    raise FileNotFoundError(
        f"Could not find EdgeVDB library. Searched: {[str(d) for d in search_dirs]}"
    )


class _Lib:
    """Lazy-loaded library singleton."""
    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = ctypes.CDLL(_find_library())
            cls._setup_signatures(cls._instance)
        return cls._instance

    @staticmethod
    def _setup_signatures(lib):
        # evdb_default_config
        lib.evdb_default_config.argtypes = [ctypes.c_void_p]
        lib.evdb_default_config.restype = None

        # evdb_open
        lib.evdb_open.argtypes = [ctypes.c_void_p]
        lib.evdb_open.restype = ctypes.c_void_p

        # evdb_close
        lib.evdb_close.argtypes = [ctypes.c_void_p]
        lib.evdb_close.restype = None

        # evdb_save
        lib.evdb_save.argtypes = [ctypes.c_void_p]
        lib.evdb_save.restype = ctypes.c_int

        # evdb_embedder_create
        lib.evdb_embedder_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        lib.evdb_embedder_create.restype = ctypes.c_void_p

        # evdb_embedder_destroy
        lib.evdb_embedder_destroy.argtypes = [ctypes.c_void_p]
        lib.evdb_embedder_destroy.restype = None

        # evdb_embed_text
        lib.evdb_embed_text.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_float)]
        lib.evdb_embed_text.restype = ctypes.c_int

        # evdb_insert_text
        lib.evdb_insert_text.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64)
        ]
        lib.evdb_insert_text.restype = ctypes.c_int

        # evdb_insert_chunk
        lib.evdb_insert_chunk.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64)
        ]
        lib.evdb_insert_chunk.restype = ctypes.c_int

        # evdb_remove_chunk
        lib.evdb_remove_chunk.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        lib.evdb_remove_chunk.restype = ctypes.c_int

        # evdb_query_text
        lib.evdb_query_text.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p,
            ctypes.c_int, ctypes.c_int
        ]
        lib.evdb_query_text.restype = ctypes.c_void_p

        # evdb_query_vector
        lib.evdb_query_vector.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
            ctypes.c_char_p, ctypes.c_int
        ]
        lib.evdb_query_vector.restype = ctypes.c_void_p

        # evdb_query_lexical (BM25 — no embedder required)
        lib.evdb_query_lexical.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
        lib.evdb_query_lexical.restype = ctypes.c_void_p

        # evdb_result_count / text / score / chunk_id / page / context
        lib.evdb_result_count.argtypes = [ctypes.c_void_p]
        lib.evdb_result_count.restype = ctypes.c_int

        lib.evdb_result_text.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.evdb_result_text.restype = ctypes.c_char_p

        lib.evdb_result_score.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.evdb_result_score.restype = ctypes.c_float

        lib.evdb_result_chunk_id.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.evdb_result_chunk_id.restype = ctypes.c_uint64

        lib.evdb_result_page.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.evdb_result_page.restype = ctypes.c_uint32

        lib.evdb_result_context_string.argtypes = [ctypes.c_void_p]
        lib.evdb_result_context_string.restype = ctypes.c_char_p

        lib.evdb_query_free.argtypes = [ctypes.c_void_p]
        lib.evdb_query_free.restype = None

        # Object store
        lib.evdb_object_put.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_uint64)
        ]
        lib.evdb_object_put.restype = ctypes.c_int

        lib.evdb_object_get.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_char_p, ctypes.c_int
        ]
        lib.evdb_object_get.restype = ctypes.c_int

        lib.evdb_object_get_size.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        lib.evdb_object_get_size.restype = ctypes.c_int

        lib.evdb_object_query.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
            ctypes.c_char_p, ctypes.c_int, ctypes.c_int
        ]
        lib.evdb_object_query.restype = ctypes.c_int

        lib.evdb_embedder_is_semantic.argtypes = [ctypes.c_void_p]
        lib.evdb_embedder_is_semantic.restype = ctypes.c_int

        lib.evdb_relation_get_targets.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_int)
        ]
        lib.evdb_relation_get_targets.restype = ctypes.c_int

        # Sync
        lib.evdb_sync_create.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.evdb_sync_create.restype = ctypes.c_void_p
        lib.evdb_sync_destroy.argtypes = [ctypes.c_void_p]
        lib.evdb_sync_destroy.restype = None
        lib.evdb_sync_export_to_file.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint64
        ]
        lib.evdb_sync_export_to_file.restype = ctypes.c_int
        lib.evdb_sync_import_from_file.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.evdb_sync_import_from_file.restype = ctypes.c_int
        lib.evdb_sync_current_clock.argtypes = [ctypes.c_void_p]
        lib.evdb_sync_current_clock.restype = ctypes.c_uint64

        lib.evdb_object_remove.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        lib.evdb_object_remove.restype = ctypes.c_int

        # Relations
        lib.evdb_relation_add.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint64, ctypes.c_uint64
        ]
        lib.evdb_relation_add.restype = ctypes.c_int

        # Version
        lib.evdb_version_string.argtypes = []
        lib.evdb_version_string.restype = ctypes.c_char_p

        # Log level
        lib.evdb_set_log_level.argtypes = [ctypes.c_int]
        lib.evdb_set_log_level.restype = None


class EvdbConfig(ctypes.Structure):
    _fields_ = [
        ("storage_dir", ctypes.c_char_p),
        ("hnsw_M", ctypes.c_int),
        ("hnsw_ef_construction", ctypes.c_int),
        ("hnsw_ef_search", ctypes.c_int),
        ("ranker_alpha", ctypes.c_float),
        ("ranker_beta", ctypes.c_float),
        ("ranker_gamma", ctypes.c_float),
        ("token_budget", ctypes.c_int),
        ("embedding_threads", ctypes.c_int),
        ("enable_knowledge_graph", ctypes.c_int),
        ("enable_sync", ctypes.c_int),
        ("device_id", ctypes.c_char_p),
        # Optional algorithm modes (appended in 0.2.0; must match vectordb.h
        # exactly — the C side copies the whole struct).
        ("hnsw_use_heuristic_selection", ctypes.c_int),
        ("hnsw_adaptive_ef", ctypes.c_int),
        ("hnsw_quantized_search", ctypes.c_int),
        ("ranker_mode", ctypes.c_int),
        ("rrf_k", ctypes.c_float),
        ("enable_self_check", ctypes.c_int),
        # 0=hybrid (vector + BM25 union), 1=vector-only, 2=BM25-only
        ("retrieval_mode", ctypes.c_int),
        # 1 (default) = maintain page index and use page proximity in ranking;
        # 0 = skip it entirely (no page.bin)
        ("enable_page_index", ctypes.c_int),
    ]


class ChunkResult:
    """A single query result chunk."""
    def __init__(self, chunk_id: int, text: str, score: float,
                 page_number: int, doc_id: int = 0):
        self.chunk_id = chunk_id
        self.text = text
        self.score = score
        self.page_number = page_number
        self.doc_id = doc_id

    def __repr__(self):
        return f"ChunkResult(id={self.chunk_id}, score={self.score:.3f}, page={self.page_number})"


class QueryResults:
    """Query results with lazy access to individual chunks."""
    def __init__(self, handle):
        self._handle = handle
        self._lib = _Lib.get()

    @property
    def count(self) -> int:
        return self._lib.evdb_result_count(self._handle)

    def __getitem__(self, index: int) -> ChunkResult:
        return ChunkResult(
            chunk_id=self._lib.evdb_result_chunk_id(self._handle, index),
            text=self._lib.evdb_result_text(self._handle, index).decode("utf-8"),
            score=self._lib.evdb_result_score(self._handle, index),
            page_number=self._lib.evdb_result_page(self._handle, index),
        )

    def __len__(self) -> int:
        return self.count

    def __iter__(self):
        for i in range(self.count):
            yield self[i]

    @property
    def context_string(self) -> str:
        raw = self._lib.evdb_result_context_string(self._handle)
        return raw.decode("utf-8") if raw else ""

    def to_list(self) -> List[ChunkResult]:
        return list(self)

    def free(self):
        if hasattr(self, '_handle') and self._handle:
            self._lib.evdb_query_free(self._handle)
            self._handle = None

    def __del__(self):
        self.free()


class Embedder:
    """ONNX embedding model wrapper."""
    def __init__(self, model_path: str, vocab_path: str, threads: int = 2):
        self._lib = _Lib.get()
        self._handle = self._lib.evdb_embedder_create(
            model_path.encode(), vocab_path.encode(), threads)
        if not self._handle:
            raise RuntimeError("Failed to create embedder")

    @property
    def handle(self):
        return self._handle

    @property
    def is_semantic(self) -> bool:
        """True only when real ONNX inference is active.

        False means the deterministic hash fallback is in use: stable and
        fine for tests, but NOT semantic — don't use it for production
        similarity search.
        """
        return bool(self._lib.evdb_embedder_is_semantic(self._handle))

    def embed(self, text: str) -> List[float]:
        out = (ctypes.c_float * 384)()
        err = self._lib.evdb_embed_text(self._handle, text.encode(), out)
        if err != 0:
            raise RuntimeError(f"Embedding failed: {err}")
        return list(out)

    def destroy(self):
        if self._handle:
            self._lib.evdb_embedder_destroy(self._handle)
            self._handle = None

    def __del__(self):
        self.destroy()


class EdgeVDB:
    """Main EdgeVDB database class."""

    def __init__(self, storage_dir: str, **kwargs):
        self._lib = _Lib.get()

        config = EvdbConfig()
        self._lib.evdb_default_config(ctypes.byref(config))
        config.storage_dir = storage_dir.encode()

        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        os.makedirs(storage_dir, exist_ok=True)
        self._handle = self._lib.evdb_open(ctypes.byref(config))
        if not self._handle:
            raise RuntimeError(f"Failed to open EdgeVDB at {storage_dir}")

    def close(self):
        # getattr: __init__ may have failed before _handle was assigned.
        if getattr(self, "_handle", None):
            self._lib.evdb_close(self._handle)
            self._handle = None

    def save(self):
        err = self._lib.evdb_save(self._handle)
        if err != 0:
            raise RuntimeError(f"Save failed: {err}")

    # Vector store
    def insert_text(self, embedder: Embedder, text: str,
                    doc_id: int = 0, page_number: int = 0) -> int:
        chunk_id = ctypes.c_uint64(0)
        err = self._lib.evdb_insert_text(
            self._handle, embedder.handle, text.encode(),
            doc_id, page_number, ctypes.byref(chunk_id))
        if err != 0:
            raise RuntimeError(f"Insert failed: {err}")
        return chunk_id.value

    def insert_chunk(self, text: str, embedding: List[float],
                     doc_id: int = 0, page_number: int = 0) -> int:
        emb_arr = (ctypes.c_float * 384)(*embedding)
        chunk_id = ctypes.c_uint64(0)
        err = self._lib.evdb_insert_chunk(
            self._handle, text.encode(), emb_arr,
            doc_id, page_number, ctypes.byref(chunk_id))
        if err != 0:
            raise RuntimeError(f"Insert failed: {err}")
        return chunk_id.value

    def remove_chunk(self, chunk_id: int):
        err = self._lib.evdb_remove_chunk(self._handle, chunk_id)
        if err != 0:
            raise RuntimeError(f"Remove failed: {err}")

    # Query
    def query_text(self, embedder: Embedder, query: str,
                   top_k: int = 5, use_kg_expansion: bool = False) -> QueryResults:
        qh = self._lib.evdb_query_text(
            self._handle, embedder.handle, query.encode(),
            top_k, 1 if use_kg_expansion else 0)
        if not qh:
            raise RuntimeError("Query failed")
        return QueryResults(qh)

    def query_vector(self, embedding: List[float], query_text: str = "",
                     top_k: int = 5) -> QueryResults:
        emb_arr = (ctypes.c_float * 384)(*embedding)
        qh = self._lib.evdb_query_vector(
            self._handle, emb_arr, query_text.encode(), top_k)
        if not qh:
            raise RuntimeError("Query failed")
        return QueryResults(qh)

    def query_bm25(self, query: str, top_k: int = 5) -> QueryResults:
        """Pure lexical BM25 retrieval — needs no embedding model at all."""
        qh = self._lib.evdb_query_lexical(self._handle, query.encode(), top_k)
        if not qh:
            raise RuntimeError("BM25 query failed")
        return QueryResults(qh)

    # Object store
    def put_object(self, type_name: str, properties: Dict[str, Any]) -> int:
        json_str = json.dumps(properties)
        out_id = ctypes.c_uint64(0)
        err = self._lib.evdb_object_put(
            self._handle, type_name.encode(), json_str.encode(),
            ctypes.byref(out_id))
        if err != 0:
            raise RuntimeError(f"Object put failed: {err}")
        return out_id.value

    def get_object(self, object_id: int) -> Optional[Dict[str, Any]]:
        size = self._lib.evdb_object_get_size(self._handle, object_id)
        if size <= 0:
            return None
        buf = ctypes.create_string_buffer(size)
        err = self._lib.evdb_object_get(self._handle, object_id, buf, size)
        if err == 4:  # NOT_FOUND
            return None
        if err != 0:
            raise RuntimeError(f"Object get failed: {err}")
        return json.loads(buf.value.decode())

    def query_objects(self, type_name: str, filter_property: str = None,
                      filter_value: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        buf_size = 1 << 16
        while True:
            buf = ctypes.create_string_buffer(buf_size)
            err = self._lib.evdb_object_query(
                self._handle, type_name.encode(),
                filter_property.encode() if filter_property else None,
                filter_value.encode() if filter_value else None,
                buf, buf_size, limit)
            if err == 8:  # BUFFER_TOO_SMALL: retry doubled
                buf_size *= 2
                continue
            if err != 0:
                raise RuntimeError(f"Object query failed: {err}")
            return json.loads(buf.value.decode() or "[]")

    def remove_object(self, object_id: int):
        err = self._lib.evdb_object_remove(self._handle, object_id)
        if err != 0:
            raise RuntimeError(f"Object remove failed: {err}")

    # Relations
    def add_relation(self, name: str, from_id: int, to_id: int):
        err = self._lib.evdb_relation_add(
            self._handle, name.encode(), from_id, to_id)
        if err != 0:
            raise RuntimeError(f"Relation add failed: {err}")

    # Context manager
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.save()
        self.close()

    def __del__(self):
        self.close()


def version() -> str:
    lib = _Lib.get()
    return lib.evdb_version_string().decode()


def set_log_level(level: int):
    lib = _Lib.get()
    lib.evdb_set_log_level(level)
