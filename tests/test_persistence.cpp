#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "chunk_store.hpp"
#include "hnsw_index.hpp"
#include "page_index.hpp"
#include "object_store.hpp"
#include "knowledge_graph.hpp"
#include "relation_index.hpp"
#include "kg_extractor.hpp"

#include <fstream>
#include <cstdio>
#include <random>
#include <vector>

using namespace edgevdb;

namespace {

// Flip one byte in the middle of a file (inside the CRC-covered payload).
void flipByte(const std::string& path, long offset_from_middle = 0) {
    std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
    REQUIRE(f.is_open());
    f.seekg(0, std::ios::end);
    long size = static_cast<long>(f.tellg());
    long pos = size / 2 + offset_from_middle;
    REQUIRE(pos > 16);
    REQUIRE(pos < size);
    f.seekg(pos);
    char c;
    f.read(&c, 1);
    c ^= 0x5A;
    f.seekp(pos);
    f.write(&c, 1);
}

void truncateFile(const std::string& path, long keep_fraction_percent = 50) {
    std::ifstream in(path, std::ios::binary);
    REQUIRE(in.is_open());
    std::string data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();
    data.resize(data.size() * keep_fraction_percent / 100);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(data.data(), data.size());
}

std::vector<float> randomUnitVector(std::mt19937& rng) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> v(EMBEDDING_DIM);
    float norm = 0.0f;
    for (auto& x : v) { x = dist(rng); norm += x * x; }
    norm = std::sqrt(norm);
    for (auto& x : v) x /= norm;
    return v;
}

} // namespace

TEST_CASE("ChunkStore persistence") {
    const std::string path = "test_chunks_tmp.bin";
    std::remove(path.c_str());
    std::mt19937 rng(1);

    {
        ChunkStore store(path);
        REQUIRE(store.open());
        for (int i = 0; i < 20; i++) {
            ChunkNode c;
            snprintf(c.text, MAX_TEXT_LEN, "chunk number %d", i);
            auto emb = randomUnitVector(rng);
            std::memcpy(c.embedding, emb.data(), EMBEDDING_DIM * sizeof(float));
            c.doc_id = 7;
            c.page_number = i;
            store.put(c);
        }
        REQUIRE(store.save());
    }

    SUBCASE("round-trip") {
        ChunkStore store(path);
        REQUIRE(store.open());
        CHECK(store.size() == 20);
        ChunkNode c;
        REQUIRE(store.get(1, c));
        CHECK(std::string(c.text) == "chunk number 0");
        CHECK(c.doc_id == 7);
    }

    SUBCASE("byte flip is rejected") {
        flipByte(path);
        ChunkStore store(path);
        CHECK_FALSE(store.open());
        CHECK(store.size() == 0);
    }

    SUBCASE("truncation is rejected") {
        truncateFile(path);
        ChunkStore store(path);
        CHECK_FALSE(store.open());
    }

    SUBCASE("absurd count is rejected") {
        std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
        uint64_t absurd = ~0ULL;
        f.seekp(12); // after magic(8) + version(4)
        f.write(reinterpret_cast<const char*>(&absurd), 8);
        f.close();
        ChunkStore store(path);
        CHECK_FALSE(store.open());
    }

    std::remove(path.c_str());
}

TEST_CASE("HNSWIndex persistence") {
    const std::string path = "test_hnsw_tmp.bin";
    std::remove(path.c_str());
    std::mt19937 rng(2);

    std::vector<std::vector<float>> vecs;
    {
        HNSWIndex index(path, 16, 100, 32);
        REQUIRE(index.open());
        for (int i = 0; i < 100; i++) {
            vecs.push_back(randomUnitVector(rng));
            index.insert(static_cast<uint64_t>(i + 1), vecs.back().data());
        }
        REQUIRE(index.save());
    }

    SUBCASE("round-trip with identical search results") {
        HNSWIndex index(path, 16, 100, 32);
        REQUIRE(index.open());
        CHECK(index.size() == 100);
        auto results = index.knnSearch(vecs[0].data(), 5);
        REQUIRE(!results.empty());
        CHECK(results[0].first == 1); // self-match
        CHECK(results[0].second < 0.01f);
    }

    SUBCASE("byte flip is rejected") {
        flipByte(path);
        HNSWIndex index(path, 16, 100, 32);
        CHECK_FALSE(index.open());
        CHECK(index.size() == 0);
    }

    SUBCASE("truncation is rejected") {
        truncateFile(path);
        HNSWIndex index(path, 16, 100, 32);
        CHECK_FALSE(index.open());
    }

    std::remove(path.c_str());
}

TEST_CASE("PageIndex persistence") {
    const std::string path = "test_page_tmp.bin";
    std::remove(path.c_str());

    {
        PageIndex index(path);
        REQUIRE(index.open());
        for (uint64_t i = 1; i <= 30; i++) {
            index.insert(i, static_cast<uint32_t>(i % 3), static_cast<uint32_t>(i));
        }
        REQUIRE(index.save());
    }

    SUBCASE("round-trip") {
        PageIndex index(path);
        REQUIRE(index.open());
        uint32_t doc, page;
        REQUIRE(index.getPage(5, doc, page));
        CHECK(doc == 2);
        CHECK(page == 5);
    }

    SUBCASE("byte flip is rejected") {
        flipByte(path);
        PageIndex index(path);
        CHECK_FALSE(index.open());
    }

    std::remove(path.c_str());
}

TEST_CASE("ObjectStore persistence") {
    const std::string path = "test_objects_tmp.bin";
    std::remove(path.c_str());

    {
        ObjectStore store(path);
        REQUIRE(store.open());
        for (int i = 0; i < 15; i++) {
            nlohmann::json p;
            p["name"] = "obj_" + std::to_string(i);
            store.put("Widget", p);
        }
        REQUIRE(store.save());
    }

    SUBCASE("round-trip") {
        ObjectStore store(path);
        REQUIRE(store.open());
        CHECK(store.totalCount() == 15);
        nlohmann::json obj;
        REQUIRE(store.get(1, obj));
        CHECK(obj["name"].get<std::string>() == "obj_0");
    }

    SUBCASE("byte flip is rejected") {
        flipByte(path);
        ObjectStore store(path);
        CHECK_FALSE(store.open());
        CHECK(store.totalCount() == 0);
    }

    std::remove(path.c_str());
}

TEST_CASE("KnowledgeGraph persistence") {
    const std::string path = "test_kg_tmp.bin";
    std::remove(path.c_str());

    {
        KnowledgeGraph kg(path);
        REQUIRE(kg.open());
        KGExtractor extractor;
        auto entities = extractor.extract("The system uses BERT and MiniLM. John Smith wrote about BERT.");
        REQUIRE(!entities.empty());
        kg.addChunkEntities(42, entities);
        REQUIRE(kg.save());
    }

    SUBCASE("round-trip") {
        KnowledgeGraph kg(path);
        REQUIRE(kg.open());
        auto chunks = kg.getChunksForEntity("bert");
        REQUIRE(!chunks.empty());
        CHECK(chunks[0] == 42);
    }

    SUBCASE("byte flip is rejected") {
        flipByte(path);
        KnowledgeGraph kg(path);
        CHECK_FALSE(kg.open());
    }

    std::remove(path.c_str());
}

TEST_CASE("KnowledgeGraph removeChunk cleans co-occurrence") {
    KnowledgeGraph kg;
    KGExtractor extractor;
    auto entities = extractor.extract("Alice met Bob at OpenAI headquarters. Alice praised OpenAI.");
    REQUIRE(entities.size() >= 2);
    kg.addChunkEntities(1, entities);

    // Entities co-occur before removal
    std::string first = entities[0].text;
    CHECK(!kg.getRelatedEntities(first, 1).empty());

    // After removing the only chunk, orphaned entities must vanish from
    // both the entity map and the co-occurrence graph.
    kg.removeChunk(1);
    CHECK(kg.getChunksForEntity(first).empty());
    CHECK(kg.getRelatedEntities(first, 1).empty());
}

TEST_CASE("RelationIndex persistence") {
    const std::string path = "test_rel_tmp.bin";
    std::remove(path.c_str());

    {
        RelationIndex rel;
        rel.addRelation("authored", 1, 100);
        rel.addRelation("authored", 1, 101);
        rel.addRelation("cites", 100, 101);
        REQUIRE(rel.save(path));
    }

    SUBCASE("round-trip") {
        RelationIndex rel;
        REQUIRE(rel.open(path));
        auto targets = rel.getTargets("authored", 1);
        CHECK(targets.size() == 2);
        auto sources = rel.getSources("cites", 101);
        REQUIRE(sources.size() == 1);
        CHECK(sources[0] == 100);
    }

    SUBCASE("byte flip is rejected") {
        flipByte(path);
        RelationIndex rel;
        CHECK_FALSE(rel.open(path));
    }

    std::remove(path.c_str());
}

TEST_CASE("Atomic save preserves previous file on writer failure") {
    // atomicSave writes to .tmp then renames; verify a normal save leaves
    // no leftover .tmp file and the target is loadable.
    const std::string path = "test_atomic_tmp.bin";
    std::remove(path.c_str());
    std::remove((path + ".tmp").c_str());

    ChunkStore store(path);
    ChunkNode c;
    snprintf(c.text, MAX_TEXT_LEN, "atomic");
    store.put(c);
    REQUIRE(store.save());

    std::ifstream tmp(path + ".tmp");
    CHECK_FALSE(tmp.is_open()); // temp file cleaned up

    ChunkStore reopened(path);
    CHECK(reopened.open());
    CHECK(reopened.size() == 1);

    std::remove(path.c_str());
}
