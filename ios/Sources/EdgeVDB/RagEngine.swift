import Foundation

/// RagEngine - High-level Retrieval-Augmented Generation engine
public final class RagEngine {
    private let db: EdgeVDB
    private let embedder: Embedder?
    private let textChunker: TextChunker
    private let contextMaxTokens: Int
    
    public init(db: EdgeVDB, embedder: Embedder? = nil, 
               textChunker: TextChunker = TextChunker(),
               contextMaxTokens: Int = 3200) {
        self.db = db
        self.embedder = embedder
        self.textChunker = textChunker
        self.contextMaxTokens = contextMaxTokens
    }
    
    /// Ingest a full document by chunking and embedding
    public func ingestDocument(text: String, docId: String, 
                              chunkSize: Int = 200, chunkOverlap: Int = 40) throws -> [UInt64] {
        let chunker = TextChunker(chunkSize: chunkSize, chunkOverlap: chunkOverlap)
        let chunks = chunker.chunk(text, docId: docId)
        var chunkIds: [UInt64] = []
        
        for chunk in chunks {
            let chunkId: UInt64
            if let embedder = embedder {
                chunkId = try db.insertText(using: embedder, text: chunk.text, 
                                           docId: UInt32(chunk.docId.hashValue), 
                                           pageNumber: UInt32(chunk.page))
            } else {
                throw EdgeVDBError.failedToCreateEmbedder
            }
            chunkIds.append(chunkId)
        }
        
        try db.save()
        return chunkIds
    }
    
    /// Query with RAG context assembly
    public func query(query: String, topK: Int = 5, 
                    useKgExpansion: Bool = false) throws -> RagResult {
        guard let embedder = embedder else {
            throw EdgeVDBError.failedToCreateEmbedder
        }
        
        let startTime = Date()
        let results = try db.queryText(using: embedder, query: query, 
                                      topK: topK, useKgExpansion: useKgExpansion)
        
        let resultsArray = results.toArray()
        let contextString = results.contextString
        let latencyMs = Date().timeIntervalSince(startTime) * 1000
        
        return RagResult(
            query: query,
            results: resultsArray,
            contextString: contextString,
            latencyMs: latencyMs
        )
    }
    
    /// Query with pre-computed embedding
    public func query(embedding: [Float], query: String, topK: Int = 5) throws -> RagResult {
        let startTime = Date()
        let results = try db.queryVector(embedding: embedding, queryText: query, topK: topK)
        
        let resultsArray = results.toArray()
        let contextString = results.contextString
        let latencyMs = Date().timeIntervalSince(startTime) * 1000
        
        return RagResult(
            query: query,
            results: resultsArray,
            contextString: contextString,
            latencyMs: latencyMs
        )
    }
}

/// RagResult - Result from RAG query
public struct RagResult {
    public let query: String
    public let results: [ChunkResult]
    public let contextString: String
    public let latencyMs: TimeInterval
    
    public init(query: String, results: [ChunkResult], 
                contextString: String, latencyMs: TimeInterval) {
        self.query = query
        self.results = results
        self.contextString = contextString
        self.latencyMs = latencyMs
    }
}
