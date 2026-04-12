import Foundation

/// EdgeVDB — Swift wrapper for the on-device vector database.
public final class EdgeVDB {
    private var handle: OpaquePointer?

    public init(storageDir: URL, config: EdgeVDBConfig = EdgeVDBConfig()) throws {
        var cConfig = EvdbConfig()
        evdb_default_config(&cConfig)
        let dirPath = storageDir.path
        cConfig.storage_dir = (dirPath as NSString).utf8String
        cConfig.hnsw_M = config.hnswM
        cConfig.hnsw_ef_construction = config.efConstruction
        cConfig.hnsw_ef_search = config.efSearch
        cConfig.ranker_alpha = config.rankerAlpha
        cConfig.ranker_beta = config.rankerBeta
        cConfig.ranker_gamma = config.rankerGamma
        cConfig.token_budget = config.tokenBudget

        handle = evdb_open(&cConfig)
        guard handle != nil else {
            throw EdgeVDBError.failedToOpen
        }
    }

    deinit { close() }

    public func close() {
        if let h = handle { evdb_close(h); handle = nil }
    }

    public func save() throws {
        guard let h = handle else { throw EdgeVDBError.nullHandle }
        let err = evdb_save(h)
        if err != EVDB_OK { throw EdgeVDBError.from(err) }
    }

    // Vector store
    public func insertText(using embedder: Embedder, text: String,
                           docId: UInt32, pageNumber: UInt32) throws -> UInt64 {
        guard let h = handle else { throw EdgeVDBError.nullHandle }
        var chunkId: UInt64 = 0
        let err = evdb_insert_text(h, embedder.handle, text, docId, pageNumber, &chunkId)
        if err != EVDB_OK { throw EdgeVDBError.from(err) }
        return chunkId
    }

    public func insertChunk(embedding: [Float], text: String,
                            docId: UInt32, pageNumber: UInt32) throws -> UInt64 {
        guard let h = handle else { throw EdgeVDBError.nullHandle }
        var chunkId: UInt64 = 0
        let err = embedding.withUnsafeBufferPointer { ptr in
            evdb_insert_chunk(h, text, ptr.baseAddress, docId, pageNumber, &chunkId)
        }
        if err != EVDB_OK { throw EdgeVDBError.from(err) }
        return chunkId
    }

    public func removeChunk(_ chunkId: UInt64) throws {
        guard let h = handle else { throw EdgeVDBError.nullHandle }
        let err = evdb_remove_chunk(h, chunkId)
        if err != EVDB_OK { throw EdgeVDBError.from(err) }
    }

    // Query
    public func queryText(using embedder: Embedder, query: String,
                          topK: Int = 5, useKgExpansion: Bool = false) throws -> QueryResults {
        guard let h = handle else { throw EdgeVDBError.nullHandle }
        guard let qh = evdb_query_text(h, embedder.handle, query,
                                        Int32(topK), useKgExpansion ? 1 : 0) else {
            throw EdgeVDBError.queryFailed
        }
        return QueryResults(handle: qh)
    }

    public func queryVector(embedding: [Float], queryText: String = "", 
                          topK: Int = 5) throws -> QueryResults {
        guard let h = handle else { throw EdgeVDBError.nullHandle }
        var emb = embedding
        guard let qh = evdb_query_vector(h, &emb, queryText, Int32(topK)) else {
            throw EdgeVDBError.queryFailed
        }
        return QueryResults(handle: qh)
    }

    // Object store
    public func putObject(type: String, properties: [String: Any]) throws -> UInt64 {
        guard let h = handle else { throw EdgeVDBError.nullHandle }
        let jsonData = try JSONSerialization.data(withJSONObject: properties)
        let jsonStr = String(data: jsonData, encoding: .utf8) ?? "{}"
        var outId: UInt64 = 0
        let err = evdb_object_put(h, type, jsonStr, &outId)
        if err != EVDB_OK { throw EdgeVDBError.from(err) }
        return outId
    }

    public func getObject(_ id: UInt64) throws -> [String: Any]? {
        guard let h = handle else { throw EdgeVDBError.nullHandle }
        var buf = [CChar](repeating: 0, count: 4096)
        let err = evdb_object_get(h, id, &buf, 4096)
        if err == EVDB_ERR_NOT_FOUND { return nil }
        if err != EVDB_OK { throw EdgeVDBError.from(err) }
        let str = String(cString: buf)
        guard let data = str.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        return obj
    }

    public func removeObject(_ id: UInt64) throws {
        guard let h = handle else { throw EdgeVDBError.nullHandle }
        let err = evdb_object_remove(h, id)
        if err != EVDB_OK { throw EdgeVDBError.from(err) }
    }

    // Relations
    public func addRelation(_ name: String, from fromId: UInt64, to toId: UInt64) throws {
        guard let h = handle else { throw EdgeVDBError.nullHandle }
        let err = evdb_relation_add(h, name, fromId, toId)
        if err != EVDB_OK { throw EdgeVDBError.from(err) }
    }
}

// MARK: - Config
public struct EdgeVDBConfig {
    public var hnswM: Int32 = 16
    public var efConstruction: Int32 = 200
    public var efSearch: Int32 = 64
    public var rankerAlpha: Float = 0.70
    public var rankerBeta: Float = 0.20
    public var rankerGamma: Float = 0.10
    public var tokenBudget: Int32 = 3200
    public var embeddingThreads: Int32 = 2
    public var enableKnowledgeGraph: Bool = true
    public var enableSync: Bool = false
    public var deviceId: String = UUID().uuidString

    public init() {}
}

// MARK: - Embedder
public final class Embedder {
    internal var handle: OpaquePointer?

    public init(modelURL: URL, vocabURL: URL, threads: Int = 2) throws {
        handle = evdb_embedder_create(modelURL.path, vocabURL.path, Int32(threads))
        guard handle != nil else { throw EdgeVDBError.failedToCreateEmbedder }
    }

    public func embed(_ text: String) throws -> [Float] {
        guard let h = handle else { throw EdgeVDBError.nullHandle }
        var result = [Float](repeating: 0, count: 384)
        let err = evdb_embed_text(h, text, &result)
        if err != EVDB_OK { throw EdgeVDBError.from(err) }
        return result
    }

    public func destroy() {
        if let h = handle { evdb_embedder_destroy(h); handle = nil }
    }

    deinit { destroy() }
}

// MARK: - QueryResults
public struct ChunkResult {
    public let chunkId: UInt64
    public let text: String
    public let score: Float
    public let pageNumber: UInt32
    public let docId: UInt32
}

public final class QueryResults {
    private var handle: OpaquePointer?

    init(handle: OpaquePointer) { self.handle = handle }

    deinit { close() }

    public var count: Int { Int(evdb_result_count(handle)) }

    public subscript(index: Int) -> ChunkResult {
        ChunkResult(
            chunkId: evdb_result_chunk_id(handle, Int32(index)),
            text: String(cString: evdb_result_text(handle, Int32(index))),
            score: evdb_result_score(handle, Int32(index)),
            pageNumber: evdb_result_page(handle, Int32(index)),
            docId: 0
        )
    }

    public var contextString: String {
        String(cString: evdb_result_context_string(handle))
    }

    public func toArray() -> [ChunkResult] {
        (0..<count).map { self[$0] }
    }
    
    /// Explicitly free the native query handle
    public func close() {
        if let h = handle { 
            evdb_query_free(h)
            handle = nil
        }
    }
}

// MARK: - Error
public enum EdgeVDBError: Error {
    case nullHandle
    case ioError
    case outOfMemory
    case notFound
    case invalidArgument
    case onnxError
    case syncError
    case failedToOpen
    case failedToCreateEmbedder
    case queryFailed

    static func from(_ err: EvdbError) -> EdgeVDBError {
        switch err {
        case EVDB_ERR_NULL_HANDLE: return .nullHandle
        case EVDB_ERR_IO: return .ioError
        case EVDB_ERR_OOM: return .outOfMemory
        case EVDB_ERR_NOT_FOUND: return .notFound
        case EVDB_ERR_INVALID_ARG: return .invalidArgument
        case EVDB_ERR_ONNX: return .onnxError
        case EVDB_ERR_SYNC: return .syncError
        default: return .ioError
        }
    }
}
