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
        XCTAssertEqual(str, "1.0.0")
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
}
