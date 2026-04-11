/**
 * validate_index.cpp — Index integrity checker for EdgeVDB binary files.
 *
 * Usage: validate_index <chunks.bin> [hnsw.bin] [page.bin]
 */

#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>
#include <cstdint>
#include <cmath>

// Include schema for struct sizes
#include "schema.hpp"

using namespace edgevdb;

struct FileHeader {
    char magic[8];
    uint32_t version;
    uint64_t count;
};

bool validateChunkStore(const std::string& path) {
    std::cout << "=== Validating ChunkStore: " << path << " ===" << std::endl;

    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "  Cannot open file" << std::endl;
        return false;
    }

    // Get file size
    file.seekg(0, std::ios::end);
    auto file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::cout << "  File size: " << file_size << " bytes" << std::endl;

    // Read header
    char magic[8];
    file.read(magic, 8);
    const char expected[] = {'E','V','D','B','C','H','K','\0'};
    if (std::memcmp(magic, expected, 8) != 0) {
        std::cerr << "  FAIL: Invalid magic" << std::endl;
        return false;
    }
    std::cout << "  Magic: OK" << std::endl;

    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), 4);
    std::cout << "  Version: " << version << std::endl;

    uint64_t count;
    file.read(reinterpret_cast<char*>(&count), 8);
    std::cout << "  Record count: " << count << std::endl;

    // Validate record sizes
    size_t expected_size = 8 + 4 + 8 + count * sizeof(ChunkNode) + 4;
    if (static_cast<size_t>(file_size) < expected_size) {
        std::cerr << "  FAIL: File too small (expected " << expected_size << " bytes)" << std::endl;
        return false;
    }

    // Validate each record
    int valid = 0, invalid = 0;
    for (uint64_t i = 0; i < count; i++) {
        ChunkNode chunk;
        file.read(reinterpret_cast<char*>(&chunk), sizeof(ChunkNode));
        if (!file.good()) {
            std::cerr << "  FAIL: Read error at record " << i << std::endl;
            return false;
        }

        // Validate embedding norm
        float norm = 0.0f;
        for (size_t d = 0; d < EMBEDDING_DIM; d++) {
            norm += chunk.embedding[d] * chunk.embedding[d];
        }
        norm = std::sqrt(norm);

        if (std::abs(norm - 1.0f) > 0.01f && norm > 0.001f) {
            invalid++;
        } else {
            valid++;
        }
    }

    std::cout << "  Embeddings: " << valid << " valid, " << invalid << " non-unit-norm" << std::endl;
    std::cout << "  Status: " << (invalid == 0 ? "PASS ✓" : "WARN ⚠") << std::endl;
    return true;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: validate_index <chunks.bin> [hnsw.bin] [page.bin]" << std::endl;
        return 1;
    }

    bool all_ok = true;
    for (int i = 1; i < argc; i++) {
        if (!validateChunkStore(argv[i])) {
            all_ok = false;
        }
        std::cout << std::endl;
    }

    return all_ok ? 0 : 1;
}
