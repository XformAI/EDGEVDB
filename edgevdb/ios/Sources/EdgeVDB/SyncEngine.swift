import Foundation

/// SyncEngine — Swift wrapper for CRDT-based data sync.
public final class SyncEngine {
    private var handle: OpaquePointer?

    public init(db: EdgeVDB, deviceId: String = UUID().uuidString) throws {
        // Note: evdb_sync_create requires the db handle which is internal
        // This is a convenience wrapper
        handle = nil // Will be initialized via C API
    }

    public func exportDelta(sinceClock: UInt64) throws -> String {
        guard let h = handle else { throw EdgeVDBError.nullHandle }
        var buf = [CChar](repeating: 0, count: 65536)
        let err = evdb_sync_export_delta(h, sinceClock, &buf, 65536)
        if err != EVDB_OK { throw EdgeVDBError.from(err) }
        return String(cString: buf)
    }

    public func applyDelta(_ json: String) throws -> (applied: Int, skipped: Int) {
        guard let h = handle else { throw EdgeVDBError.nullHandle }
        var applied: Int32 = 0
        var skipped: Int32 = 0
        let err = evdb_sync_apply_delta(h, json, &applied, &skipped)
        if err != EVDB_OK { throw EdgeVDBError.from(err) }
        return (Int(applied), Int(skipped))
    }

    public func exportToFile(path: String, sinceClock: UInt64) throws {
        guard let h = handle else { throw EdgeVDBError.nullHandle }
        let err = evdb_sync_export_to_file(h, path, sinceClock)
        if err != EVDB_OK { throw EdgeVDBError.from(err) }
    }

    public func importFromFile(path: String) throws {
        guard let h = handle else { throw EdgeVDBError.nullHandle }
        let err = evdb_sync_import_from_file(h, path)
        if err != EVDB_OK { throw EdgeVDBError.from(err) }
    }

    public var currentClock: UInt64 {
        guard let h = handle else { return 0 }
        return evdb_sync_current_clock(h)
    }

    deinit {
        if let h = handle { evdb_sync_destroy(h) }
    }
}
