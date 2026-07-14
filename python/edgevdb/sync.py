"""
EdgeVDB Python sync module — file-based sync over the C sync engine.
"""

from edgevdb import EdgeVDB, _Lib


class SyncHelper:
    """File-based sync helper backed by the native sync engine.

    Usage:
        with EdgeVDB("./data") as db:
            sync = SyncHelper(db, device_id="laptop")
            sync.export_to_file("delta.json", since_clock=0)
            sync.import_from_file("delta_from_other_device.json")
            sync.close()

    The helper must be closed (or garbage-collected) before the database
    is closed.
    """

    def __init__(self, db: EdgeVDB, device_id: str = "python-desktop"):
        self._lib = _Lib.get()
        self._db = db
        self._device_id = device_id
        self._handle = self._lib.evdb_sync_create(db._handle, device_id.encode())
        if not self._handle:
            raise RuntimeError("Failed to create sync engine")

    @property
    def current_clock(self) -> int:
        """Current logical clock of this device."""
        return self._lib.evdb_sync_current_clock(self._handle)

    def export_to_file(self, path: str, since_clock: int = 0) -> bool:
        """Export a sync delta (everything after since_clock) to a JSON file."""
        err = self._lib.evdb_sync_export_to_file(
            self._handle, path.encode(), since_clock)
        return err == 0

    def import_from_file(self, path: str) -> bool:
        """Apply a sync delta from a JSON file produced by another device."""
        err = self._lib.evdb_sync_import_from_file(self._handle, path.encode())
        return err == 0

    def close(self):
        if self._handle:
            self._lib.evdb_sync_destroy(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


__all__ = ["SyncHelper"]
