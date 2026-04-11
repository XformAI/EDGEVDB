import Foundation

/// iOS Embedder — re-exported from the main EdgeVDB.swift file.
/// This file provides convenience extensions and batch embedding.
extension Embedder {
    /// Create from bundle resources
    public convenience init(modelResource: String, vocabResource: String,
                            bundle: Bundle = .main, threads: Int = 2) throws {
        guard let modelURL = bundle.url(forResource: modelResource, withExtension: "onnx"),
              let vocabURL = bundle.url(forResource: vocabResource, withExtension: "txt") else {
            throw EdgeVDBError.failedToCreateEmbedder
        }
        try self.init(modelURL: modelURL, vocabURL: vocabURL, threads: threads)
    }

    /// Batch embed multiple texts
    public func embedBatch(_ texts: [String]) throws -> [[Float]] {
        return try texts.map { try embed($0) }
    }
}
