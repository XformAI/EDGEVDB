import Foundation

/// VectorStore — focused vector search operations.
public final class VectorStore {
    private weak var db: EdgeVDB?

    internal init(db: EdgeVDB) {
        self.db = db
    }

    public func insert(text: String, embedding: [Float], docId: UInt32, pageNumber: UInt32) throws -> UInt64 {
        guard let db = db else { throw EdgeVDBError.nullHandle }
        return try db.insertChunk(embedding: embedding, text: text, docId: docId, pageNumber: pageNumber)
    }

    public func insert(text: String, using embedder: Embedder, docId: UInt32, pageNumber: UInt32) throws -> UInt64 {
        guard let db = db else { throw EdgeVDBError.nullHandle }
        return try db.insertText(using: embedder, text: text, docId: docId, pageNumber: pageNumber)
    }

    public func remove(chunkId: UInt64) throws {
        guard let db = db else { throw EdgeVDBError.nullHandle }
        try db.removeChunk(chunkId)
    }

    public func query(embedding: [Float], text: String = "", topK: Int = 5) throws -> QueryResults {
        guard let db = db else { throw EdgeVDBError.nullHandle }
        var emb = embedding
        guard let handle = evdb_query_vector(nil, &emb, text, Int32(topK)) else {
            throw EdgeVDBError.queryFailed
        }
        return QueryResults(handle: handle)
    }
}
