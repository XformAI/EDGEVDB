"""
EdgeVDB Python object_store module — re-exported from __init__.py.
Provides convenience methods for the relational object store.
"""

from edgevdb import EdgeVDB

__all__ = ["ObjectStoreHelper"]


class ObjectStoreHelper:
    """Convenience wrapper for object store operations."""

    def __init__(self, db: EdgeVDB):
        self._db = db

    def put(self, type_name: str, properties: dict) -> int:
        return self._db.put_object(type_name, properties)

    def get(self, object_id: int) -> dict:
        return self._db.get_object(object_id)

    def remove(self, object_id: int):
        self._db.remove_object(object_id)

    def add_relation(self, name: str, from_id: int, to_id: int):
        self._db.add_relation(name, from_id, to_id)
