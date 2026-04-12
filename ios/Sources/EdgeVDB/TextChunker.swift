import Foundation

/// TextChunker - Splits text into chunks for indexing
public struct TextChunker {
    /// Chunk size in words
    public let chunkSize: Int
    /// Overlap between chunks in words
    public let chunkOverlap: Int
    /// Minimum chunk length in characters
    public let minChunkLength: Int
    
    public init(chunkSize: Int = 200, chunkOverlap: Int = 40, minChunkLength: Int = 20) {
        self.chunkSize = chunkSize
        self.chunkOverlap = chunkOverlap
        self.minChunkLength = minChunkLength
    }
    
    /// Split text into chunks
    public func chunk(_ text: String) -> [DocumentChunk] {
        let words = text.components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }
        var chunks: [DocumentChunk] = []
        var page = 0
        var i = 0
        
        while i < words.count {
            let end = min(i + chunkSize, words.count)
            let chunkWords = Array(words[i..<end])
            let chunkText = chunkWords.joined(separator: " ")
            
            // Skip trivially small final fragments
            if chunkText.count < minChunkLength {
                break
            }
            
            let chunk = DocumentChunk(
                text: chunkText,
                docId: "",
                page: page
            )
            chunks.append(chunk)
            
            page += 1
            i += chunkSize - chunkOverlap
        }
        
        return chunks
    }
    
    /// Split text into chunks with document ID
    public func chunk(_ text: String, docId: String) -> [DocumentChunk] {
        let chunks = chunk(text)
        return chunks.map { chunk in
            DocumentChunk(
                text: chunk.text,
                docId: docId,
                page: chunk.page
            )
        }
    }
}
