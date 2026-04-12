import Foundation

/// OnnxEmbeddingPipeline - ONNX-based embedding pipeline with CoreML acceleration
public final class OnnxEmbeddingPipeline {
    private let embedder: Embedder
    private let useCoreML: Bool
    
    public init(modelPath: String, vocabPath: String, threads: Int = 2, useCoreML: Bool = true) throws {
        self.embedder = try Embedder(modelURL: URL(fileURLWithPath: modelPath),
                                     vocabURL: URL(fileURLWithPath: vocabPath),
                                     threads: threads)
        self.useCoreML = useCoreML
    }
    
    /// Create from bundle resources
    public convenience init(modelResource: String, vocabResource: String,
                          bundle: Bundle = .main, threads: Int = 2, useCoreML: Bool = true) throws {
        guard let modelPath = bundle.path(forResource: modelResource, withExtension: "onnx"),
              let vocabPath = bundle.path(forResource: vocabResource, withExtension: "txt") else {
            throw EdgeVDBError.failedToCreateEmbedder
        }
        try self.init(modelPath: modelPath, vocabPath: vocabPath, 
                     threads: threads, useCoreML: useCoreML)
    }
    
    /// Embed text to vector
    public func embed(_ text: String) throws -> [Float] {
        return try embedder.embed(text)
    }
    
    /// Batch embed multiple texts
    public func embedBatch(_ texts: [String]) throws -> [[Float]] {
        return try texts.map { try embed($0) }
    }
    
    /// Get the underlying embedder
    public var underlyingEmbedder: Embedder {
        return embedder
    }
    
    deinit {
        embedder.destroy()
    }
}
