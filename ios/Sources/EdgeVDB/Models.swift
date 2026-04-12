import Foundation

// MARK: - Data Models

/// DocumentChunk - Represents a chunk of text with metadata
public struct DocumentChunk {
    public let text: String
    public let docId: String
    public let page: Int
    public let embedding: [Float]?
    
    public init(text: String, docId: String, page: Int, embedding: [Float]? = nil) {
        self.text = text
        self.docId = docId
        self.page = page
        self.embedding = embedding
    }
}

/// QueryResult - Enhanced result with additional metadata
public struct QueryResult {
    public let chunkId: UInt64
    public let text: String
    public let score: Float
    public let page: Int
    public let docId: String
    
    public init(chunkId: UInt64, text: String, score: Float, page: Int, docId: String) {
        self.chunkId = chunkId
        self.text = text
        self.score = score
        self.page = page
        self.docId = docId
    }
}

/// ObjectRecord - Generic object with type and properties
public struct ObjectRecord {
    public let id: UInt64
    public let type: String
    public let properties: [String: Any]
    public let timestamp: UInt64
    
    public init(id: UInt64, type: String, properties: [String: Any], timestamp: UInt64 = 0) {
        self.id = id
        self.type = type
        self.properties = properties
        self.timestamp = timestamp
    }
    
    /// Helper to get typed property
    public func property<T>(_ key: String) -> T? {
        return properties[key] as? T
    }
}

/// ChunkResult - Simplified result for backward compatibility
public struct ChunkResult {
    public let chunkId: UInt64
    public let text: String
    public let score: Float
    public let page: UInt32
    public let docId: UInt32
    
    public init(chunkId: UInt64, text: String, score: Float, page: UInt32, docId: UInt32) {
        self.chunkId = chunkId
        self.text = text
        self.score = score
        self.page = page
        self.docId = docId
    }
}
