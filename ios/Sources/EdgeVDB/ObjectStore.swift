import Foundation

/// ObjectStore — Swift wrapper for the relational object database.
public final class ObjectStore {
    private weak var db: EdgeVDB?

    internal init(db: EdgeVDB) {
        self.db = db
    }

    public func put(type: String, properties: [String: Any]) throws -> UInt64 {
        guard let db = db else { throw EdgeVDBError.nullHandle }
        return try db.putObject(type: type, properties: properties)
    }

    public func get(_ id: UInt64) throws -> [String: Any]? {
        guard let db = db else { throw EdgeVDBError.nullHandle }
        return try db.getObject(id)
    }

    public func remove(_ id: UInt64) throws {
        guard let db = db else { throw EdgeVDBError.nullHandle }
        try db.removeObject(id)
    }

    public func addRelation(_ name: String, from: UInt64, to: UInt64) throws {
        guard let db = db else { throw EdgeVDBError.nullHandle }
        try db.addRelation(name, from: from, to: to)
    }
}
