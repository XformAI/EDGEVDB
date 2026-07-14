import XCTest
@testable import EdgeVDB

final class EdgeVDBTests: XCTestCase {

    var tempDir: URL!

    override func setUp() {
        super.setUp()
        tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try? FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    }

    override func tearDown() {
        try? FileManager.default.removeItem(at: tempDir)
        super.tearDown()
    }

    func testOpenClose() throws {
        let db = try EdgeVDB(storageDir: tempDir)
        try db.save()
        db.close()
    }

    func testVersionString() {
        let version = evdb_version_string()
        XCTAssertNotNil(version)
        let str = String(cString: version!)
        XCTAssertEqual(str, "0.2.0")
    }

    func testInsertAndQueryChunks() throws {
        let db = try EdgeVDB(storageDir: tempDir)

        // Create a simple embedding (unit vector along dim 0)
        var embedding = [Float](repeating: 0, count: 384)
        embedding[0] = 1.0

        let chunkId = try db.insertChunk(
            embedding: embedding,
            text: "Test chunk about machine learning",
            docId: 1, pageNumber: 0
        )
        XCTAssertGreaterThan(chunkId, 0)

        try db.save()
        db.close()
    }

    func testQueryVector() throws {
        let db = try EdgeVDB(storageDir: tempDir)

        var embedding = [Float](repeating: 0, count: 384)
        embedding[0] = 1.0

        let chunkId = try db.insertChunk(
            embedding: embedding,
            text: "Test chunk about machine learning",
            docId: 1, pageNumber: 0
        )

        let results = try db.queryVector(embedding: embedding, queryText: "machine learning", topK: 5)
        XCTAssertGreaterThan(results.count, 0)
        results.close()

        try db.save()
        db.close()
    }

    func testObjectStore() throws {
        let db = try EdgeVDB(storageDir: tempDir)

        let id = try db.putObject(type: "Document", properties: [
            "title": "Test Doc",
            "author": "Alice"
        ])
        XCTAssertGreaterThan(id, 0)

        let obj = try db.getObject(id)
        XCTAssertNotNil(obj)
        XCTAssertEqual(obj?["title"] as? String, "Test Doc")

        try db.removeObject(id)
        let removed = try db.getObject(id)
        XCTAssertNil(removed)

        db.close()
    }

    func testRelations() throws {
        let db = try EdgeVDB(storageDir: tempDir)

        try db.addRelation("authored_by", from: 1, to: 100)
        // Verify no crash
        db.close()
    }

    func testTextChunker() {
        let chunker = TextChunker(chunkSize: 10, chunkOverlap: 2, minChunkLength: 5)
        let text = "This is a test text that should be split into multiple chunks for testing purposes."
        let chunks = chunker.chunk(text, docId: "test-doc")
        
        XCTAssertGreaterThan(chunks.count, 1)
        XCTAssertEqual(chunks.first?.docId, "test-doc")
    }

    func testModels() {
        let chunk = DocumentChunk(text: "Test", docId: "doc1", page: 0)
        XCTAssertEqual(chunk.text, "Test")
        XCTAssertEqual(chunk.docId, "doc1")
        
        let result = QueryResult(chunkId: 1, text: "Result", score: 0.9, page: 0, docId: "doc1")
        XCTAssertEqual(result.score, 0.9)
        
        let obj = ObjectRecord(id: 1, type: "Test", properties: ["key": "value"])
        XCTAssertEqual(obj.type, "Test")
        XCTAssertEqual(obj.property("key") as? String, "value")
    }

    func testQueryResultsClose() throws {
        let db = try EdgeVDB(storageDir: tempDir)
        
        var embedding = [Float](repeating: 0, count: 384)
        embedding[0] = 1.0
        
        let chunkId = try db.insertChunk(
            embedding: embedding,
            text: "Test chunk",
            docId: 1, pageNumber: 0
        )
        
        let results = try db.queryVector(embedding: embedding, queryText: "test", topK: 5)
        XCTAssertGreaterThan(results.count, 0)
        results.close() // Explicit close should work without crash
        
        db.close()
    }
}
