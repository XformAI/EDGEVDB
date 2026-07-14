#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "crc32.hpp"
#include <cstring>

using namespace edgevdb;

TEST_CASE("CRC32 golden value") {
    // Standard check value for CRC-32/ISO-HDLC.
    const char* data = "123456789";
    CHECK(crc32(data, 9) == 0xCBF43926u);
}

TEST_CASE("CRC32 empty input") {
    CHECK(crc32(nullptr, 0) == 0x00000000u);
}

TEST_CASE("CRC32 incremental equals one-shot") {
    const char* data = "The quick brown fox jumps over the lazy dog";
    size_t len = std::strlen(data);

    uint32_t oneshot = crc32(data, len);

    uint32_t state = CRC32_INIT;
    state = crc32Update(state, data, 10);
    state = crc32Update(state, data + 10, len - 10);
    CHECK(crc32Finalize(state) == oneshot);
}

TEST_CASE("CRC32 detects single-bit corruption") {
    unsigned char buf[64];
    for (int i = 0; i < 64; i++) buf[i] = static_cast<unsigned char>(i * 7);
    uint32_t before = crc32(buf, sizeof(buf));
    buf[33] ^= 0x01;
    CHECK(crc32(buf, sizeof(buf)) != before);
}
